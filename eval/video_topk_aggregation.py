"""Video-level aggregation by *top-k mean class probability*: an event-sensitive metric.

Why this module exists
-----------------------
`docs/ERROR_ANALYSIS.md` measured that UCF-Crime anomaly events are a minority
of their video's frames -- Shooting is 9.9% of frames at the median and 1.0%
at worst -- and that in 9 of 9 annotated validation videos the event frames
cannot win a majority vote. `eval.video_level`'s majority vote over per-frame
argmax and `eval.video_probability_aggregation`'s mean over the full video
both average over the whole video, so both are structurally unable to let a
brief event decide the video-level prediction. This module provides the
event-sensitive alternative: instead of averaging every frame, it averages
only the k frames a class is most confident about.

The key design insight -- one rule, not three
----------------------------------------------
Maximum probability, top-k mean, and mean probability are the SAME rule at
different k. For each video and each class c, take that video's per-frame
probabilities for class c, keep the k largest, and average them; the
predicted class is the argmax over c of that per-class top-k score.

    k=1        -> maximum per-class probability (the single most confident
                  frame for each class wins that class's score)
    k=3/5/10   -> top-k mean (the event-sensitive middle ground)
    k=None,
    k >= frame_count -> mean over all frames, i.e. plain mean-probability
                  (`eval.video_probability_aggregation`'s rule, recovered
                  here as the k -> infinity limit of the same computation)

This does not replace either sibling module: majority vote, mean-probability,
and top-k mean are reported side by side, because which one is highest is
itself evidence about how a video's true class is distributed across its
frames.

Scope
-----
Evaluation-only. It never touches training, checkpoint selection, or the
per-frame metric. It shares `eval.metrics.classification_metrics` with every
other scorer in the project (rule of two), so its macro-F1, per-class table
and confusion matrix are directly comparable to `docs/TRAINING_LOG.md`
without a translation step. Grouping uses the composite ``(source_dataset,
source_video_id)`` key for the same reason the sibling modules do: two
datasets could reuse a video id.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from eval.metrics import classification_metrics


class _FrameRecord(Protocol):
    """The three fields this module needs from each frame.

    `core.p1c_engine.FramePrediction` satisfies this. A cached-feature head
    trainer can pass any lightweight object with the same three attributes.
    """

    source_dataset: str
    source_video_id: str
    target_index: int


@dataclass(frozen=True)
class VideoTopKPrediction:
    """One video's top-k-mean prediction, with confidence audit fields.

    `topk_probability` is the per-class top-k-mean vector (not a probability
    distribution -- each class's score comes from its own k frames, which
    need not be the same frames class-to-class -- kept so a reviewer can see
    *how* a video was decided). `top_margin` is the top-1-minus-top-2 margin
    of that vector, matching the sibling modules' audit field.
    """

    source_dataset: str
    source_video_id: str
    target_index: int
    predicted_index: int
    frame_count: int
    k_used: int
    topk_probability: tuple[float, ...]
    top_margin: float


@dataclass(frozen=True)
class VideoTopKResult:
    """Per-video top-k-mean predictions plus metrics computed over them."""

    predictions: tuple[VideoTopKPrediction, ...]
    metrics: dict
    video_count: int
    tie_broken_video_count: int
    k: int | None


def aggregate_video_level_topk(
    frame_records: Sequence[_FrameRecord],
    frame_probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    k: int | None,
) -> VideoTopKResult:
    """Per-class top-k-mean probability aggregation, then argmax.

    `frame_records[i]` and `frame_probabilities[i]` describe the same frame
    and must be in the same order. `frame_probabilities` is ``(n_frames,
    n_classes)`` and is expected to hold per-frame class probabilities
    (softmax outputs).

    `k` selects how many of a video's largest per-class probabilities are
    averaged to produce that class's score: `k=1` is maximum probability,
    `k=None` is the mean over every frame. A video with fewer frames than
    `k` uses all of its frames -- it is never padded and never an error,
    since a short clip is not a manifest defect.

    Raises if any video's frames disagree about their own `target_index` --
    the same manifest-defect guard the sibling modules apply (docs/PROJECT_RULES.md rule
    4: a defect must surface, not be voted away).

    Ties in the final per-video argmax break deterministically toward the
    lowest class index; the count of tie-broken videos is returned on the
    result rather than swallowed, matching `eval.video_level`'s contract.
    """

    if k is not None and (not isinstance(k, int) or isinstance(k, bool) or k < 1):
        raise ValueError(f"k must be a positive int or None; got {k!r}")

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

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(frame_records):
        groups[(record.source_dataset, record.source_video_id)].append(index)

    video_predictions: list[VideoTopKPrediction] = []
    tie_broken_video_count = 0
    for (source_dataset, source_video_id), indices in groups.items():
        target_indices = {int(frame_records[i].target_index) for i in indices}
        if len(target_indices) != 1:
            raise ValueError(
                f"video {source_dataset!r}/{source_video_id!r} frames disagree on target_index: "
                f"{sorted(target_indices)}"
            )
        target_index = target_indices.pop()

        frame_count = len(indices)
        k_used = frame_count if k is None else min(k, frame_count)
        video_probs = probabilities[indices]  # (frame_count, n_classes)
        # Per class, the k largest probabilities among this video's frames,
        # averaged -- np.sort ascending then take the last k columns.
        topk_vector = np.sort(video_probs, axis=0)[-k_used:, :].mean(axis=0)

        vote_counts = topk_vector
        highest_score = vote_counts.max()
        tied_classes = sorted(
            class_index
            for class_index in range(n_classes)
            if vote_counts[class_index] == highest_score
        )
        is_tie = len(tied_classes) > 1
        if is_tie:
            tie_broken_video_count += 1
        predicted_index = tied_classes[0]

        ordered = np.sort(topk_vector)[::-1]
        top_margin = float(ordered[0] - ordered[1]) if n_classes > 1 else float(ordered[0])

        video_predictions.append(
            VideoTopKPrediction(
                source_dataset=source_dataset,
                source_video_id=source_video_id,
                target_index=target_index,
                predicted_index=predicted_index,
                frame_count=frame_count,
                k_used=k_used,
                topk_probability=tuple(float(value) for value in topk_vector),
                top_margin=top_margin,
            )
        )

    video_predictions.sort(key=lambda item: (item.source_dataset, item.source_video_id))
    truth = [class_names[item.target_index] for item in video_predictions]
    predicted = [class_names[item.predicted_index] for item in video_predictions]
    return VideoTopKResult(
        predictions=tuple(video_predictions),
        metrics=classification_metrics(truth, predicted, class_names),
        video_count=len(video_predictions),
        tie_broken_video_count=tie_broken_video_count,
        k=k,
    )
