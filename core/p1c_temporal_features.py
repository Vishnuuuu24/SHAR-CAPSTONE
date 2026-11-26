"""Temporal window features built from cached per-frame features.

Why this module exists
-----------------------
`core/feature_cache.py` caches one pooled vector per frame, independent of
neighbouring frames. This module lets a probe or head see a short temporal
window around each frame instead of just the frame itself, without ever
mixing frames across a video boundary -- a window that reached into the next
video would attribute one video's motion context to another video's label,
which is exactly the kind of provenance leak docs/PROJECT_RULES.md's "preserve
source-video grouping ... end to end" rule exists to prevent.

Edge handling: clamp, never drop or wrap
------------------------------------------
For a window radius `r`, row `t`'s window is `[t-r, ..., t+r]` measured in
*rows*, which -- because rows are cached at stride 5 -- means neighbours 5
source-frames apart, not 1. At a video's first or last `r` rows, the window
would run off the video; this module clamps each out-of-range offset to the
nearest in-video row (repeating that edge frame) rather than dropping the row
(which would silently shrink the dataset and misalign it with `rows`/labels)
or wrapping into the next/previous video (which would leak that video's
frames into this one's window). Every variant therefore returns exactly one
output row per input row, aligned 1:1 with `rows`.

The five variants
-------------------
All operate on the `[N, D]` array `core.feature_cache.load_feature_cache`
returns and produce a new `[N, D']` float32 array:

- `"baseline"`: f_t unchanged, D' = D. A control -- this variant is here so a
  caller can compare temporal variants against "no temporal context at all"
  through the exact same code path, rather than falling back to the raw
  cache elsewhere.
- `"concat"`: [f(t-r), ..., f(t+r)] concatenated in order, D' = (2r+1) * D.
- `"mean"`: mean of the window's `2r+1` vectors, D' = D.
- `"mean_std"`: [mean, std] of the window concatenated, D' = 2D.
- `"diff"`: [f_t, f_t - f_prev] concatenated, D' = 2D, where `f_prev` is the
  previous cached row of the SAME video (edge-clamped to `f_t` itself at a
  video's first row, so that row's difference half is exactly zero). `radius`
  is ignored for this variant -- there is no window, only one fixed lookback
  of one cached row (5 source frames, stride 5, ~0.17 s at the dataset's
  nominal frame rate).

  This is a **feature-space** difference between DINOv2 global descriptors of
  two cached rows 5 source frames apart. It is NOT pixel differencing and NOT
  optical flow: it measures how much the pooled global descriptor moved
  between those two frames, which is at best a weak proxy for motion in the
  scene -- a descriptor can shift for reasons that have nothing to do with
  motion (lighting, compression, camera noise), and real motion confined to a
  small region of the frame can leave a global descriptor almost unchanged. A
  genuine pixel-difference experiment would need DINOv2 re-run on
  `|I_t - I_{t-1}|` difference images -- a full GPU cache rebuild -- and is
  out of scope here.

Memory cost at fold-1 train scale (391,014 rows x 1536-D, radius 2 -> 5 rows
per window; figures are the OUTPUT array only, float32, and assume the input
stays float16 as loaded from the cache):

- `baseline`: 391,014 x 1,536 x 4 B ~= 2.3 GB
- `mean` / `mean_std` intermediate mean: same shape as baseline ~= 2.3 GB;
  `mean_std`'s doubled width ~= 4.6 GB
- `concat`: 391,014 x 7,680 x 4 B ~= 11.5 GB -- by far the largest variant.
  `build_temporal_features` never materialises a `[N, 2r+1, D]` gather array
  before concatenating; it writes each of the `2r+1` blocks directly into its
  slice of a single preallocated `[N, D']` output array, so peak extra memory
  above the output array itself is one window's worth of row indices, not a
  second full-size copy. A caller working at this scale who cannot afford
  11.5 GB should combine `concat` with `core.p1c_video_sampling` to cut `N`
  first, exactly what `scripts/run_p1c_temporal_probe.py`'s `--frames-per-video`
  flag is for.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

#: The five registered temporal variants.
TEMPORAL_VARIANTS = ("baseline", "concat", "mean", "mean_std", "diff")


def _video_row_spans(rows: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    """`(start, stop)` row-index spans, one per video, in row order.

    Relies on the same documented contiguity `core.p1c_video_sampling` and
    `core.p1c_probes.stride_rows_within_videos` rely on: rows are contiguous
    and ascending by `source_frame_index` within each `(source_dataset,
    source_video_id)` group.
    """

    spans: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(rows) + 1):
        at_end = index == len(rows)
        boundary = at_end or (
            (rows[index]["source_dataset"], rows[index]["source_video_id"])
            != (rows[start]["source_dataset"], rows[start]["source_video_id"])
        )
        if boundary:
            spans.append((start, index))
            start = index
    return spans


def _clamped_offsets(span: tuple[int, int], radius: int) -> np.ndarray:
    """`[len, 2r+1]` absolute row indices for every row in `span`, edge-clamped."""

    start, stop = span
    length = stop - start
    local = np.arange(length)[:, None] + np.arange(-radius, radius + 1)[None, :]
    local = np.clip(local, 0, length - 1)
    return local + start


def build_temporal_features(
    features: np.ndarray, rows: Sequence[dict[str, Any]], variant: str, *, radius: int = 2
) -> np.ndarray:
    """Build one temporal-window variant, aligned 1:1 with `rows`.

    `features` is the `[N, D]` array `core.feature_cache.load_feature_cache`
    returns (any dtype; the output is always float32). See the module
    docstring for the five variants, edge clamping and memory cost.
    `radius` is ignored for `"diff"` (see the module docstring).
    """

    if variant not in TEMPORAL_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {TEMPORAL_VARIANTS}")
    if radius < 0:
        raise ValueError(f"radius must be >= 0; got {radius}")
    if len(features) != len(rows):
        raise ValueError(f"features/rows disagree: {len(features)} vs {len(rows)}")

    n, d = features.shape
    window = 2 * radius + 1

    if variant == "baseline":
        return np.ascontiguousarray(features, dtype=np.float32)

    if variant == "diff":
        out = np.empty((n, 2 * d), dtype=np.float32)
        for span in _video_row_spans(rows):
            start, stop = span
            length = stop - start
            current = np.arange(length) + start
            previous = np.clip(np.arange(length) - 1, 0, length - 1) + start
            current_features = features[current].astype(np.float32, copy=False)
            previous_features = features[previous].astype(np.float32, copy=False)
            out[start:stop, :d] = current_features
            out[start:stop, d:] = current_features - previous_features
        return out

    if variant == "concat":
        out = np.empty((n, window * d), dtype=np.float32)
    elif variant == "mean":
        out = np.empty((n, d), dtype=np.float32)
    else:  # mean_std
        out = np.empty((n, 2 * d), dtype=np.float32)

    for span in _video_row_spans(rows):
        start, stop = span
        offsets = _clamped_offsets(span, radius)  # [length, window], absolute row indices
        gathered = features[offsets].astype(np.float32, copy=False)  # [length, window, D]
        if variant == "concat":
            out[start:stop] = gathered.reshape(stop - start, window * d)
        elif variant == "mean":
            out[start:stop] = gathered.mean(axis=1)
        else:
            out[start:stop, :d] = gathered.mean(axis=1)
            out[start:stop, d:] = gathered.std(axis=1)

    return out
