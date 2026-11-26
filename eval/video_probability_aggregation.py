"""Video-level aggregation by *mean class probability*: a second evaluation-only metric.

Why this module exists (backlog B-6)
------------------------------------
`eval.video_level.aggregate_video_level_predictions` already gives one prediction
per source video by majority vote over the per-frame argmax. Its own docstring
records the weakness: a majority vote over argmax throws away confidence
entirely -- a video whose frames are all bare 51% guesses counts the same as one
whose frames are all 99% certain. `eval/video_level.py` explicitly defers the fix
("Mean-probability aggregation ... would be strictly better on this axis, but it
requires the engine to retain per-frame class probabilities") to a later change.
This module is that later change, kept separate so it can be reviewed on its own:
it consumes per-frame probability vectors, averages them within each source
video, and takes the argmax of the average.

    video_probability = mean_over_frames( softmax(frame_logits) )
    video_prediction  = argmax(video_probability)

Both video-level metrics stay side by side rather than one replacing the other:
they can disagree, and which one is higher is itself evidence about how
confident the frame predictions are.

Scope
-----
Evaluation-only. It never touches training, checkpoint selection, or the
per-frame metric. It shares `eval.metrics.classification_metrics` with every
other scorer in the project (rule of two), so its macro-F1, per-class table and
confusion matrix are directly comparable to `docs/TRAINING_LOG.md` without a
translation step. Grouping uses the composite ``(source_dataset,
source_video_id)`` key for the same reason the sibling module does: two datasets
could reuse a video id.

This does not confuse itself with `core.p1c_aggregation` (which averages metrics
across the three training seeds -- a different axis entirely) or with
`eval.video_level` (majority vote -- a different aggregation of the same axis).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from eval.metrics import classification_metrics


class _FrameRecord(Protocol):
    """The three fields this module needs from each frame.

    `core.p1c_engine.FramePrediction` satisfies this (it carries an extra
    `predicted_index`, unused here -- mean-probability recomputes its own
    per-video argmax from the averaged vector). A cached-feature head trainer
    can pass any lightweight object with the same three attributes.
    """

    source_dataset: str
    source_video_id: str
    target_index: int


@dataclass(frozen=True)
class VideoProbabilityPrediction:
    """One video's mean-probability prediction, with confidence audit fields.

    `mean_probability` is the full averaged vector, kept so a reviewer can see
    *how* a video was decided, not just the winner. `top_margin` is
    ``p(top-1) - p(top-2)`` of that averaged vector: a video with `top_margin`
    near zero was barely decided even though every frame contributed a full
    distribution, which the majority-vote metric's `vote_margin` cannot show.
    """

    source_dataset: str
    source_video_id: str
    target_index: int
    predicted_index: int
    frame_count: int
    mean_probability: tuple[float, ...]
    top_margin: float


@dataclass(frozen=True)
class VideoProbabilityResult:
    """Per-video mean-probability predictions plus metrics computed over them."""

    predictions: tuple[VideoProbabilityPrediction, ...]
    metrics: dict
    video_count: int
    low_margin_video_count: int
    low_margin_threshold: float


def aggregate_video_level_probabilities(
    frame_records: Sequence[_FrameRecord],
    frame_probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    low_margin_threshold: float | None = None,
) -> VideoProbabilityResult:
    """Average per-frame probability vectors within each source video, then argmax.

    `frame_records[i]` and `frame_probabilities[i]` describe the same frame and
    must be in the same order. `frame_probabilities` is ``(n_frames,
    n_classes)`` and is expected to hold per-frame class probabilities (softmax
    outputs); the argmax is invariant to whether the rows are normalised, but
    the reported `mean_probability`/`top_margin` are only meaningful if they
    are, so non-finite or negative entries are refused.

    Raises if any video's frames disagree about their own `target_index` -- the
    same manifest-defect guard `eval.video_level` applies, for the same reason
    (silently voting on the truth label would hide the defect, docs/PROJECT_RULES.md rule 4).

    `low_margin_threshold` defaults to ``1 / n_classes`` -- the probability mass
    a single class would hold under a uniform posterior. Videos whose averaged
    top-1-minus-top-2 margin falls below it are counted on the result so a
    shaky point estimate is visible rather than implicit.
    """

    if not isinstance(class_names, Sequence) or not class_names:
        raise ValueError("class_names must be a non-empty ordered sequence")
    n_classes = len(class_names)
    if len(set(class_names)) != n_classes:
        raise ValueError("class_names must be unique")

    probabilities = np.asarray(frame_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != n_classes:
        raise ValueError(
            f"frame_probabilities must be (n_frames, {n_classes}); got {probabilities.shape}"
        )
    if len(frame_records) != probabilities.shape[0]:
        raise ValueError(
            f"frame_records ({len(frame_records)}) and frame_probabilities "
            f"({probabilities.shape[0]}) disagree on frame count"
        )
    if probabilities.shape[0] == 0:
        raise ValueError("frame_records must be non-empty")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("frame_probabilities must be finite and non-negative")

    if low_margin_threshold is None:
        low_margin_threshold = 1.0 / n_classes

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(frame_records):
        groups[(record.source_dataset, record.source_video_id)].append(index)

    video_predictions: list[VideoProbabilityPrediction] = []
    low_margin_video_count = 0
    for (source_dataset, source_video_id), indices in groups.items():
        target_indices = {int(frame_records[i].target_index) for i in indices}
        if len(target_indices) != 1:
            raise ValueError(
                f"video {source_dataset!r}/{source_video_id!r} frames disagree on target_index: "
                f"{sorted(target_indices)}"
            )
        target_index = target_indices.pop()
        mean_vector = probabilities[indices].mean(axis=0)
        predicted_index = int(np.argmax(mean_vector))
        ordered = np.sort(mean_vector)[::-1]
        top_margin = float(ordered[0] - ordered[1]) if n_classes > 1 else float(ordered[0])
        if top_margin < low_margin_threshold:
            low_margin_video_count += 1
        video_predictions.append(
            VideoProbabilityPrediction(
                source_dataset=source_dataset,
                source_video_id=source_video_id,
                target_index=target_index,
                predicted_index=predicted_index,
                frame_count=len(indices),
                mean_probability=tuple(float(value) for value in mean_vector),
                top_margin=top_margin,
            )
        )

    video_predictions.sort(key=lambda item: (item.source_dataset, item.source_video_id))
    truth = [class_names[item.target_index] for item in video_predictions]
    predicted = [class_names[item.predicted_index] for item in video_predictions]
    return VideoProbabilityResult(
        predictions=tuple(video_predictions),
        metrics=classification_metrics(truth, predicted, class_names),
        video_count=len(video_predictions),
        low_margin_video_count=low_margin_video_count,
        low_margin_threshold=float(low_margin_threshold),
    )
