"""Clip-level event/background probes on official temporal intervals.

Why this module exists
-----------------------
`scripts/diagnose_p1c_event_separability.py` asked whether cached DINOv2
features separate Shooting EVENT frames from NON-EVENT frames of the same
video, one frame at a time. This module asks a related but different
question at the clip level: a short window of `clip_length` consecutive
cached rows, tiled non-overlapping across each annotated video, labelled
positive/negative/ambiguous from the same official (start, end) intervals
(`core.p1c_background_mask.load_intervals`). Clip-level evidence matters
because any eventual sequence or MIL model over these features consumes
clips, not single frames -- so the question "can a clip be told apart from
background" is closer to what such a model would need than the single-frame
diagnostic answers.

Intervals are used ONLY as evaluation labels here, exactly as in the
frame-level diagnostic. Nothing in the production training pipeline consumes
them, and no model receives an interval as an input feature (docs/PROJECT_RULES.md rule
2: no circular evaluation).

Registered choices, and why
----------------------------
- **Non-overlapping tiling (hop = clip_length * frame_stride).** Overlapping
  clips from one video share almost all of their rows and are near-duplicate
  evidence, not independent evidence -- using them would inflate the clip
  count and, worse, could put near-identical clips on both sides of a
  leave-one-video-out split were the grouping ever done at the clip level
  instead of the video level. Tiling gives one independent count per video
  segment instead.
- **Three-way positive/negative/ambiguous split**, not a plain
  overlap-threshold binary. A clip is a clean negative only when every one
  of its rows is both outside every span AND at least `negative_margin`
  cached rows from any span edge -- the margin exists because a clip sitting
  just outside a span very likely still shows lead-up or spill-over motion a
  hard binary label would call "background" incorrectly. Everything that is
  neither a confident positive nor a confident negative is dropped from
  training/eval rather than forced into one bucket, and its count is still
  reported (docs/PROJECT_RULES.md rule 4: keep negative/ambiguous outcomes visible, don't
  quietly drop them from the report).
- **`negative_margin` is in cached-row units**, not source-frame units, to
  match `clip_length`, `frame_stride` and the cache's own row granularity.
  Rows are cached at a fixed stride per video (5 source frames dataset-wide,
  per the verified facts this module was built against), so the margin is
  converted to source-frame units as `negative_margin * stride` (stride
  inferred from that video's own consecutive cached frames, falling back to
  5 for a single-row video) and compared directly against frame distance to
  each span edge -- simpler and safer than mapping an edge to a "nearest
  row" index, which would misbehave for a span entirely outside the video's
  frame range (every row would look "nearest" to the same clamped edge).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from core.p1c_temporal_features import build_temporal_features

#: The three registered clip representations (see `clip_features`).
CLIP_REPRESENTATIONS = ("single_frame", "dense_mean", "dense_mean_delta")


@dataclass(frozen=True)
class Clip:
    """One fixed-length, non-overlapping window of cached rows from one video."""

    video_id: str
    row_indices: tuple[int, ...]
    start_source_frame: int
    end_source_frame: int
    is_positive: bool
    event_overlap: float


@dataclass(frozen=True)
class EventClipBuild:
    """Everything `build_event_clips` produces: the two usable buckets, the
    excluded ambiguous bucket (kept so its count is reportable, never
    silently dropped -- docs/PROJECT_RULES.md rule 4), and a per-video breakdown.
    """

    positive: tuple[Clip, ...]
    negative: tuple[Clip, ...]
    ambiguous: tuple[Clip, ...]
    counts_by_video: dict[str, dict[str, int]]

    @property
    def usable(self) -> tuple[Clip, ...]:
        """Positive and negative clips, in build order -- what a probe trains on."""

        return self.positive + self.negative

    def report(self) -> dict[str, Any]:
        return {
            "positive": len(self.positive),
            "negative": len(self.negative),
            "ambiguous": len(self.ambiguous),
            "by_video": self.counts_by_video,
        }


def _video_spans(rows: Sequence[dict[str, Any]]) -> list[tuple[str, int, int]]:
    """`[(video_id, start_row, stop_row), ...]` contiguous per-video row spans,
    in row order. Relies on the same documented contiguity
    `core.p1c_temporal_features._video_row_spans` and
    `core.p1c_video_sampling._video_groups` rely on: rows are contiguous and
    ascending by `source_frame_index` within each video.
    """

    spans: list[tuple[str, int, int]] = []
    start = 0
    for index in range(1, len(rows) + 1):
        at_end = index == len(rows)
        boundary = at_end or rows[index]["source_video_id"] != rows[start]["source_video_id"]
        if boundary:
            spans.append((rows[start]["source_video_id"], start, index))
            start = index
    return spans


def _video_frame_stride(frames: Sequence[int]) -> int:
    """This video's own cached-row spacing in source frames (falls back to 5,
    the dataset-wide documented stride, for a single-row video)."""

    return frames[1] - frames[0] if len(frames) > 1 else 5


def _row_is_known_negative(frame: int, spans: list[tuple[int, int]], margin_frames: int) -> bool:
    """One row: outside every span AND >= `margin_frames` source frames from every span edge."""

    return all(frame < start - margin_frames or frame > end + margin_frames for start, end in spans)


def _window_starts(length: int, clip_length: int, frame_stride: int) -> tuple[list[int], int]:
    """Local (0-based, within-video) start row of each non-overlapping window.

    Window `i` occupies local rows `[start, start + (clip_length-1)*frame_stride]`
    (`span` rows of "territory"); consecutive windows are hopped by that full
    span so they never overlap -- see the module docstring's "registered
    choices" section. A trailing run shorter than `span` is dropped.
    """

    span = (clip_length - 1) * frame_stride + 1
    starts, start = [], 0
    while start + span <= length:
        starts.append(start)
        start += span
    return starts, span


def build_event_clips(
    rows: Sequence[dict[str, Any]],
    intervals: dict[str, list[tuple[int, int]]],
    *,
    label: str,
    clip_length: int = 16,
    frame_stride: int = 1,
    positive_overlap_threshold: float = 0.5,
    negative_margin: int = 16,
    seed: int = 0,
) -> EventClipBuild:
    """Tile every annotated `label` video into fixed-length clips and classify
    each as positive / negative / ambiguous against its official intervals.

    `seed` is registered but currently unused -- no sampling step in this
    function needs randomness; balancing or subsampling the returned clips is
    left to the caller (do not do it here, per the task's design note).
    Deterministic: identical inputs always give identical output, since
    tiling and classification are both pure functions of `rows`/`intervals`.
    """

    if clip_length < 1:
        raise ValueError(f"clip_length must be >= 1; got {clip_length}")
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1; got {frame_stride}")
    if negative_margin < 0:
        raise ValueError(f"negative_margin must be >= 0; got {negative_margin}")

    positives: list[Clip] = []
    negatives: list[Clip] = []
    ambiguous: list[Clip] = []
    counts_by_video: dict[str, dict[str, int]] = {}

    for video_id, start, stop in _video_spans(rows):
        video_rows = rows[start:stop]
        if video_rows[0]["label"] != label or video_id not in intervals:
            continue
        spans = intervals[video_id]
        frames = [int(row["source_frame_index"]) for row in video_rows]
        margin_frames = negative_margin * _video_frame_stride(frames)
        starts, _span_len = _window_starts(len(video_rows), clip_length, frame_stride)
        video_counts = {"positive": 0, "negative": 0, "ambiguous": 0}

        for local_start in starts:
            local_indices = [local_start + k * frame_stride for k in range(clip_length)]
            row_indices = tuple(start + i for i in local_indices)
            clip_frames = [frames[i] for i in local_indices]
            inside = sum(1 for f in clip_frames if any(a <= f <= b for a, b in spans))
            overlap = inside / clip_length
            is_positive = overlap >= positive_overlap_threshold

            clip = Clip(
                video_id=video_id,
                row_indices=row_indices,
                start_source_frame=clip_frames[0],
                end_source_frame=clip_frames[-1],
                is_positive=is_positive,
                event_overlap=overlap,
            )

            if is_positive:
                positives.append(clip)
                video_counts["positive"] += 1
                continue

            is_negative = all(_row_is_known_negative(f, spans, margin_frames) for f in clip_frames)
            if is_negative:
                negatives.append(clip)
                video_counts["negative"] += 1
            else:
                ambiguous.append(clip)
                video_counts["ambiguous"] += 1

        counts_by_video[video_id] = video_counts

    return EventClipBuild(
        positive=tuple(positives),
        negative=tuple(negatives),
        ambiguous=tuple(ambiguous),
        counts_by_video=counts_by_video,
    )


def clip_features(
    features: np.ndarray, rows: Sequence[dict[str, Any]], clips: Sequence[Clip], representation: str
) -> np.ndarray:
    """Build one `[n_clips, D']` representation of `clips` from cached `features`.

    See the module docstring's three registered variants. `rows`/`features`
    must be the full cache arrays a clip's `row_indices` index into -- not a
    pre-sliced subset -- so that `"dense_mean_delta"` can compute each row's
    delta from its true preceding cached row, including a clip's first row
    when that row is not the first row of its video.
    """

    if representation not in CLIP_REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}; expected one of {CLIP_REPRESENTATIONS}")
    if len(features) != len(rows):
        raise ValueError(f"features/rows disagree: {len(features)} vs {len(rows)}")

    if representation == "single_frame":
        centre = [clip.row_indices[len(clip.row_indices) // 2] for clip in clips]
        return np.ascontiguousarray(features[centre], dtype=np.float32)

    if representation == "dense_mean":
        return np.stack(
            [features[list(clip.row_indices)].astype(np.float32, copy=False).mean(axis=0) for clip in clips]
        )

    # dense_mean_delta: deltas computed on the FULL rows first, so a clip's
    # first row still gets a real delta from the cached row before it.
    full_deltas = build_temporal_features(features, rows, "diff")
    return np.stack(
        [full_deltas[list(clip.row_indices)].mean(axis=0) for clip in clips]
    ).astype(np.float32, copy=False)


def _train_test_masks(clips: Sequence[Clip], held_out_video_id: str) -> tuple[np.ndarray, np.ndarray]:
    test_mask = np.array([clip.video_id == held_out_video_id for clip in clips])
    return ~test_mask, test_mask


def _assert_no_leakage(clips: Sequence[Clip], train_mask: np.ndarray, held_out_video_id: str) -> None:
    """Fires if `train_mask` keeps any clip of `held_out_video_id` -- the
    invariant a leave-one-video-out split must never violate.
    """

    leaked = any(clip.video_id == held_out_video_id for clip, keep in zip(clips, train_mask) if keep)
    assert not leaked, f"leakage: held-out video {held_out_video_id!r} present in training clips"


def leave_one_video_out_clips(clips: Sequence[Clip], X: np.ndarray, *, seed: int) -> dict[str, Any]:
    """Fit one logistic regression per held-out video; score frame- and
    video-level. See the module docstring and the task spec for the full
    contract -- standardise on train only, `max_iter=1000`, `C=1.0`.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    if len(clips) != len(X):
        raise ValueError(f"clips/X disagree: {len(clips)} vs {len(X)}")

    y = np.array([1 if clip.is_positive else 0 for clip in clips], dtype=np.int64)
    video_ids = sorted({clip.video_id for clip in clips})

    per_video: dict[str, Any] = {}
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    pooled_prob: list[np.ndarray] = []
    mean_probability_by_video: dict[str, float] = {}

    for held_out in video_ids:
        train_mask, test_mask = _train_test_masks(clips, held_out)
        _assert_no_leakage(clips, train_mask, held_out)

        x_train, y_train = X[train_mask], y[train_mask]
        x_test, y_test = X[test_mask], y[test_mask]

        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        model.fit(scaler.transform(x_train), y_train)

        probability = model.predict_proba(scaler.transform(x_test))[:, 1]
        prediction = (probability >= 0.5).astype(np.int64)

        n_positive, n_negative = int(y_test.sum()), int((1 - y_test).sum())
        roc_auc = pr_auc = None
        if n_positive > 0 and n_negative > 0:
            roc_auc = float(roc_auc_score(y_test, probability))
            pr_auc = float(average_precision_score(y_test, probability))

        per_video[held_out] = {
            "n_positive": n_positive,
            "n_negative": n_negative,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "predictions": prediction.tolist(),
        }
        mean_probability_by_video[held_out] = float(probability.mean())

        pooled_true.append(y_test)
        pooled_pred.append(prediction)
        pooled_prob.append(probability)

    truth = np.concatenate(pooled_true)
    predicted = np.concatenate(pooled_pred)
    probability_all = np.concatenate(pooled_prob)

    pooled: dict[str, Any] = {
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "roc_auc": None,
        "pr_auc": None,
    }
    if truth.sum() > 0 and (1 - truth).sum() > 0:
        pooled["roc_auc"] = float(roc_auc_score(truth, probability_all))
        pooled["pr_auc"] = float(average_precision_score(truth, probability_all))

    # Video-level: every video here is an annotated `label` video, i.e.
    # positive by construction, so "predicted positive" (mean clip
    # probability > 0.5) is a recall measurement over these videos, not a
    # two-class video F1 -- see the module/task docstring.
    predicted_positive_videos = sum(1 for prob in mean_probability_by_video.values() if prob > 0.5)
    video_level = {
        "n_videos": len(video_ids),
        "n_predicted_positive": predicted_positive_videos,
        "recall": predicted_positive_videos / len(video_ids) if video_ids else None,
        "mean_clip_probability_by_video": mean_probability_by_video,
    }

    return {"per_video": per_video, "pooled": pooled, "video_level": video_level}
