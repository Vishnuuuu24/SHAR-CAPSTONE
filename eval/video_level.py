"""Video-level aggregation of frame predictions: an additional, evaluation-only metric.

Why this metric exists
-----------------------
UCF-Crime frames inherit a video-level label (docs/PROJECT_RULES.md rule 1): the label is
genuinely a property of the video, not of any one frame. P1C-VIDEO-FRAME-
BASELINE-011 measured that frame *counts* are imbalanced up to 4.48x across
classes purely because videos differ in length, even though the videos
themselves are balanced by construction (34/class train, 4/class validation,
12/class test for AR fold 1). A frame-level macro-F1 is dominated by whichever
classes happen to have the longest videos; a per-video score is immune to that
by construction, because it counts each video once regardless of how many
frames it contributed. The owner decided (2026-08-27) to keep training and
frame-level checkpoint selection exactly as they are (`loss.class_balance:
none` stays; this run is the plain control) and to *additionally* report this
video-level score, at no training cost -- it is computed purely from the
frame predictions the engine already produces.

Method and its main limitation
-------------------------------
One predicted class per video is chosen by majority vote over that video's
per-frame argmax predictions. This is deliberately simple and has a known
weakness: majority vote over argmax discards confidence entirely. A video
where every frame is a bare 51% guess counts exactly the same as one where
every frame is 99% certain. Mean-probability aggregation (averaging softmax
outputs across a video's frames before taking the argmax) would be strictly
better on this axis, but it requires the engine to retain per-frame class
probabilities rather than only the argmax `predicted_index` it currently
records in `FramePrediction`. That is worth doing later; it is deliberately
not done here, so this module can be reviewed as a small, self-contained,
evaluation-only addition that touches neither training nor the engine.

Scope
-----
This is evaluation-only. It never changes training, and it never confuses
itself with `core.p1c_aggregation`, which aggregates metrics across the three
registered training seeds -- a different axis of aggregation entirely.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from core.p1c_engine import FramePrediction
from eval.metrics import classification_metrics


#: The only two values `evaluation.video_level_aggregation` may register.
#: `none` means this metric is not computed for the run; `majority_vote` is
#: the one aggregation method this module implements (see module docstring
#: for why mean-probability aggregation is not one of the choices yet).
VIDEO_LEVEL_AGGREGATION_SETTINGS = frozenset({"none", "majority_vote"})


@dataclass(frozen=True)
class VideoPrediction:
    """One video's aggregated prediction, audit fields retained.

    `frame_count` and `vote_margin` exist so a weak video-level prediction is
    visible rather than indistinguishable from a confident one: a video with
    `vote_margin` just above 1/num_classes was barely decided.
    """

    source_dataset: str
    source_video_id: str
    target_index: int
    predicted_index: int
    frame_count: int
    vote_margin: float
    tie_broken: bool


@dataclass(frozen=True)
class VideoLevelResult:
    """Per-video predictions plus metrics computed over them."""

    predictions: tuple[VideoPrediction, ...]
    metrics: dict
    video_count: int
    tie_broken_video_count: int


def aggregate_video_level_predictions(
    predictions: Sequence[FramePrediction], class_names: Sequence[str]
) -> VideoLevelResult:
    """Aggregate frame predictions to one prediction per (source_dataset, source_video_id).

    Grouping uses the composite key deliberately, never `source_video_id`
    alone: `core.p1c_engine.EvaluationResult.source_video_ids` already treats
    the composite as the globally unambiguous key, because two datasets could
    reuse the same video id.

    Raises if any video's frames disagree about their own `target_index` --
    that is a manifest defect (a video whose frames do not agree on their
    inherited label), and silently majority-voting the truth label would hide
    it rather than surface it (docs/PROJECT_RULES.md rule 4).

    Ties in the per-video frame vote break deterministically toward the
    lowest class index; the count of tie-broken videos is returned on the
    result rather than swallowed, because a high tie count means the metric
    is shakier than the point estimate alone would suggest.
    """

    if not isinstance(predictions, Sequence) or not predictions:
        raise ValueError("predictions must be a non-empty sequence of FramePrediction")
    if not isinstance(class_names, Sequence) or not class_names:
        raise ValueError("class_names must be a non-empty ordered sequence")

    groups: dict[tuple[str, str], list[FramePrediction]] = defaultdict(list)
    for prediction in predictions:
        groups[(prediction.source_dataset, prediction.source_video_id)].append(prediction)

    video_predictions: list[VideoPrediction] = []
    tie_broken_video_count = 0
    for (source_dataset, source_video_id), frames in groups.items():
        target_indices = {frame.target_index for frame in frames}
        if len(target_indices) != 1:
            raise ValueError(
                f"video {source_dataset!r}/{source_video_id!r} frames disagree on target_index: "
                f"{sorted(target_indices)}"
            )
        target_index = target_indices.pop()
        vote_counts = Counter(frame.predicted_index for frame in frames)
        highest_count = max(vote_counts.values())
        tied_classes = sorted(
            class_index for class_index, count in vote_counts.items() if count == highest_count
        )
        is_tie = len(tied_classes) > 1
        if is_tie:
            tie_broken_video_count += 1
        video_predictions.append(
            VideoPrediction(
                source_dataset=source_dataset,
                source_video_id=source_video_id,
                target_index=target_index,
                predicted_index=tied_classes[0],
                frame_count=len(frames),
                vote_margin=highest_count / len(frames),
                tie_broken=is_tie,
            )
        )

    # Deterministic ordering so a written report is stable run-to-run.
    video_predictions.sort(key=lambda item: (item.source_dataset, item.source_video_id))

    truth = [class_names[item.target_index] for item in video_predictions]
    predicted = [class_names[item.predicted_index] for item in video_predictions]
    return VideoLevelResult(
        predictions=tuple(video_predictions),
        metrics=classification_metrics(truth, predicted, class_names),
        video_count=len(video_predictions),
        tie_broken_video_count=tie_broken_video_count,
    )
