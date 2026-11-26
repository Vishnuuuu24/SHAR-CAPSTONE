"""Per-video frame sampling to cut near-duplicate frames from long videos.

Why this module exists
-----------------------
Fold 1 train has 476 videos with 21 to 28,380 frames each (1351x imbalance,
median 446.5). Frames were extracted at stride 5, so consecutive rows are 5
source-frames apart -- a 28,380-frame video contributes an enormous, highly
redundant run of near-duplicate frames to every epoch, while a 21-frame video
contributes almost nothing. Uniform random subsampling over the whole training
set would still be dominated by the long videos in expectation; this module
instead samples a fixed *count* per video, so no source video can drown out
another regardless of its length.

Sampling policy: segment-based temporal coverage
--------------------------------------------------
For `frames_per_video` frames wanted from a video of length `L`, the video is
covered end to end rather than sampled from one contiguous block -- a
contiguous block would miss whatever the video does before or after that
block. This is the standard segment-based sampling approach used across video
action recognition (e.g. TSN): divide the video into `frames_per_video` equal
temporal segments and take one frame from each.

- `seed=None` (deterministic; validation/test): take the segment centre,
  i.e. evenly spaced positions `round(i * (L-1) / (n-1))` for `i` in
  `0..n-1`, deduplicated. Same video, same `n` -> same indices, always.
- `seed=<int>` (seeded; training): jitter uniformly *within* each segment
  instead of always taking its centre, so different seeds see different
  frames while every seed still covers the full video. This is what makes
  the sampler useful as a training-time augmentation rather than a fixed
  subsample repeated every epoch.

A video shorter than `frames_per_video` keeps every one of its frames --
this module only removes redundancy, it never upsamples or pads.

Per-class budgets: why a single global count is not enough
------------------------------------------------------------
Measured cross-fold on the DINOv2 cache, video-level macro-F1: Shooting's
F1 rises monotonically with frames per video -- 0.033 at 8, 0.090 at 16,
0.159 at 32, 0.289 using all frames -- because Shooting's anomaly occupies
only ~9.9% of its video on average (1.0% at worst), so uniform sampling
usually misses it entirely. Overall macro-F1 moves the opposite way -- 0.4068
at 8, 0.3953 at 16, 0.3825 at 32, 0.3433 at all frames -- because
near-duplicate frames from long videos come to dominate training as the
per-video budget grows. A single global budget cannot serve both goals at
once; `frames_per_video_by_label` lets a brief-event class (Shooting) be
sampled more densely while every other class stays at whichever budget is
best overall.

Honest caveat: raising one class's budget also raises that class's share of
the training rows (Shooting goes from 7.1% to 23.5% of train rows at 32
frames/video vs 8), so this is not a pure temporal-coverage change -- it
carries an implicit class-reweighting effect that this experiment design
cannot separate from the coverage effect.

Provenance
----------
This function returns indices into `rows`; it does not touch `rows` itself.
Video ids, labels and everything else in a row's provenance therefore follow
automatically through the returned indices, and the accompanying test asserts
that directly.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _validate_frames_per_video_by_label(frames_per_video_by_label: Mapping[str, int]) -> None:
    for label, budget in frames_per_video_by_label.items():
        if not isinstance(label, str):
            raise ValueError(f"frames_per_video_by_label keys must be strings; got {label!r}")
        if not isinstance(budget, int) or isinstance(budget, bool):
            raise ValueError(f"frames_per_video_by_label[{label!r}] must be an int; got {budget!r}")
        if budget < 1:
            raise ValueError(f"frames_per_video_by_label[{label!r}] must be >= 1; got {budget}")


def _video_groups(rows: Sequence[dict[str, Any]]) -> list[tuple[tuple[str, str], list[int]]]:
    """`[(source_dataset, source_video_id), [row indices]), ...]` in first-seen order.

    Rows are documented as contiguous and ascending by `source_frame_index`
    within each video group (see `core.p1c_probes.stride_rows_within_videos`,
    which relies on the same contiguity), so a single left-to-right pass is
    enough to recover each video's row-index run.
    """

    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (row["source_dataset"], row["source_video_id"])
        groups.setdefault(key, []).append(index)
    return list(groups.items())


def _segment_positions(length: int, frames_per_video: int, rng: np.random.Generator | None) -> np.ndarray:
    """Positions in `0..length-1` for one video, deterministic or seed-jittered."""

    if length <= frames_per_video:
        return np.arange(length)
    if frames_per_video == 1:
        centres = np.array([(length - 1) / 2.0])
    else:
        centres = np.linspace(0, length - 1, frames_per_video)
    if rng is None:
        positions = np.round(centres).astype(np.int64)
    else:
        # Segment half-width: half the gap to the next centre (first/last
        # segment reuses its one neighbour's gap), so jitter never reaches
        # into an adjacent segment's territory.
        if frames_per_video == 1:
            half_widths = np.array([length / 2.0])
        else:
            gaps = np.diff(centres)
            half_widths = np.empty(frames_per_video)
            half_widths[0] = gaps[0] / 2.0
            half_widths[-1] = gaps[-1] / 2.0
            half_widths[1:-1] = np.minimum(gaps[:-1], gaps[1:]) / 2.0
        jitter = rng.uniform(-half_widths, half_widths)
        positions = np.clip(np.round(centres + jitter), 0, length - 1).astype(np.int64)
    return np.unique(positions)


def sample_rows_per_video(
    rows: Sequence[dict[str, Any]],
    frames_per_video: int,
    *,
    seed: int | None = None,
    frames_per_video_by_label: Mapping[str, int] | None = None,
) -> np.ndarray:
    """Row indices keeping up to `frames_per_video` temporally-spread frames per video.

    See the module docstring for the deterministic-vs-seeded distinction.
    Always returns indices sorted ascending, so downstream row order (and
    hence the within-video contiguity other code relies on) is preserved.

    `frames_per_video_by_label` optionally overrides the budget for videos
    whose `label` appears as a key -- see the module docstring's "Per-class
    budgets" section for why. A video's label is read from its first row;
    a label absent from the mapping (or the mapping omitted entirely) falls
    back to `frames_per_video` unchanged, so every existing caller keeps its
    exact prior behaviour.
    """

    if frames_per_video < 1:
        raise ValueError(f"frames_per_video must be >= 1; got {frames_per_video}")
    if frames_per_video_by_label is not None:
        _validate_frames_per_video_by_label(frames_per_video_by_label)

    rng = None if seed is None else np.random.default_rng(seed)
    keep: list[int] = []
    for _key, indices in _video_groups(rows):
        budget = frames_per_video
        if frames_per_video_by_label is not None:
            label = rows[indices[0]]["label"]
            budget = frames_per_video_by_label.get(label, frames_per_video)
        positions = _segment_positions(len(indices), budget, rng)
        keep.extend(indices[position] for position in positions)
    return np.asarray(sorted(keep), dtype=np.int64)
