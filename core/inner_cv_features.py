"""Per-video feature matrices for the Step 2 candidates of `core.inner_cv` (FINAL_PUSH_PLAN_2026-11-11).

Every builder returns a `[V, D]` float32 matrix aligned to `pool.video_ids`, so a candidate differs
from its control only in which matrix the same logistic-regression arm is given.

"Study frames" are the 82,679 frames shared by the DINOv2-B patch cache, the DINOv2-L cache and the
SigLIP patch cache (every video's 8-frame draws plus all frames of the validation videos). A
backbone contrast is computed on that common frame set, never all-frames against study-frames.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from core.inner_cv import Pool

P1C = "results/p1c"
DINOV2L = P1C + "/p1c-dinov2l-cache"
SIGLIP = P1C + "/p1c-siglip-patch-cache"
VJEPA2 = P1C + "/p1c-vjepa2-clip-cache-ar-fold{fold}"
BLOCKS = ("B_all", "B_study", "L_study", "S_study", "VJ")


def read_study_rows(cache_dir: Path) -> tuple[list[str], np.ndarray]:
    """`(video id per row, source frame index per row)` of a study-frame cache's `rows.csv`."""

    with (Path(cache_dir) / "rows.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return [r["source_video_id"] for r in rows], np.asarray([int(r["source_frame_index"]) for r in rows], dtype=np.int64)


def _row_groups(videos: Sequence[str]) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, video in enumerate(videos):
        groups.setdefault(video, []).append(index)
    return {v: np.asarray(ix, dtype=np.int64) for v, ix in groups.items()}


def study_matrix(pool: Pool, row_videos: Sequence[str], features: np.ndarray) -> np.ndarray:
    """Mean of `features[rows of the video]` for each pool video; `features` is `[N, D]` in study-row order."""

    if len(row_videos) != features.shape[0]:
        raise ValueError(f"{len(row_videos)} study rows vs {features.shape[0]} feature rows")
    groups = _row_groups(row_videos)
    missing = [v for v in pool.video_ids if v not in groups]
    if missing:
        raise ValueError(f"{len(missing)} pool videos have no study frames, e.g. {missing[:3]}")
    return np.stack([np.asarray(features[groups[v]], dtype=np.float32).mean(axis=0) for v in pool.video_ids])


def matched_pool_matrix(pool: Pool, row_videos: Sequence[str], row_frames: np.ndarray) -> np.ndarray:
    """The pool's own features averaged over exactly the study frames of each video (the matched control)."""

    groups = _row_groups(row_videos)
    out = []
    for video in pool.video_ids:
        wanted = row_frames[groups[video]]
        have = pool.frame_indices_of(video)
        positions = np.searchsorted(have, wanted)
        if (positions >= len(have)).any() or not np.array_equal(have[positions], wanted):
            raise ValueError(f"{video}: some study frames are not in the pool cache")
        out.append(pool.frames_of(video)[positions].mean(axis=0))
    return np.stack(out)


def pooled_tokens(cls_tokens: np.ndarray, patch_tokens: np.ndarray, *, chunk: int = 512) -> np.ndarray:
    """`[N, 2D]` float16 `[CLS ++ mean(patch)]` from a patch cache, read in chunks (the patch array is tens of GiB)."""

    from core.p1c_head_variants import pooled_vector_from_tokens

    out = np.empty((cls_tokens.shape[0], cls_tokens.shape[1] + patch_tokens.shape[2]), dtype=np.float16)
    for start in range(0, cls_tokens.shape[0], chunk):
        out[start:start + chunk] = pooled_vector_from_tokens(cls_tokens[start:start + chunk], patch_tokens[start:start + chunk])
    return out


def _clip_means(cache_root: Path, splits: Sequence[str]) -> dict[str, np.ndarray]:
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for split in splits:
        directory = Path(cache_root) / split
        features = np.load(directory / "features.float16.npy", mmap_mode="r")
        with (directory / "rows.csv").open() as handle:
            videos = [r["source_video_id"] for r in csv.DictReader(handle)]
        if len(videos) != features.shape[0]:
            raise ValueError(f"{directory}: {len(videos)} rows vs {features.shape[0]} feature rows")
        for video, rows in _row_groups(videos).items():
            block = np.asarray(features[rows], dtype=np.float32)
            sums[video] = sums.get(video, 0) + block.sum(axis=0)
            counts[video] = counts.get(video, 0) + len(rows)
    return {v: sums[v] / counts[v] for v in sums}


def clip_matrix(pool: Pool, cache_root: Path, splits: Sequence[str] = ("train", "validation"), fallback_roots: Sequence[Path] = ()) -> np.ndarray:
    """Mean clip feature per pool video from a fold's clip caches (V-JEPA 2 / VideoMAE layout).

    `fallback_roots` are other folds' caches, used only as a feature store for videos this fold's caches
    do not hold (its test videos, which are training videos of the other folds); the first root that has a video wins.
    """

    means = _clip_means(cache_root, splits)
    for root in fallback_roots:
        if all(v in means for v in pool.video_ids):
            break
        for video, mean in _clip_means(root, splits).items():
            means.setdefault(video, mean)
    missing = [v for v in pool.video_ids if v not in means]
    if missing:
        raise ValueError(f"{len(missing)} pool videos have no clips in {cache_root}, e.g. {missing[:3]}")
    return np.stack([means[v] for v in pool.video_ids])


def feature_blocks(fold: int, data_root: Path, pool: Pool, siglip_pooled: Path | None, wanted: set[str], *,
                   vjepa_fallback_folds: Sequence[int] = ()) -> dict[str, np.ndarray]:
    """The named per-video blocks for one pool. `siglip_pooled` is the `[CLS ++ mean(patch)]` array made by `pooled_tokens`."""

    data_root = Path(data_root)
    out = {}
    if "B_all" in wanted:
        out["B_all"] = pool.video_means()
    if wanted & {"B_study", "L_study", "S_study"}:
        row_videos, row_frames = read_study_rows(data_root / DINOV2L)
        if "B_study" in wanted:
            out["B_study"] = matched_pool_matrix(pool, row_videos, row_frames)
        if "L_study" in wanted:
            out["L_study"] = study_matrix(pool, row_videos, np.load(data_root / DINOV2L / "features.float16.npy", mmap_mode="r"))
        if "S_study" in wanted:
            s_videos, s_frames = read_study_rows(data_root / SIGLIP)
            if s_videos != row_videos or not np.array_equal(s_frames, row_frames):
                raise RuntimeError("the SigLIP and DINOv2-L caches do not hold the same study rows")
            out["S_study"] = study_matrix(pool, s_videos, np.load(siglip_pooled, mmap_mode="r"))
    if "VJ" in wanted:
        out["VJ"] = clip_matrix(pool, data_root / VJEPA2.format(fold=fold),
                                fallback_roots=[data_root / VJEPA2.format(fold=f) for f in vjepa_fallback_folds])
    return out


def span_matrix(pool: Pool, spans: Mapping[str, Sequence[tuple[int, int]]]) -> tuple[np.ndarray, int]:
    """Mean over only the frames inside a video's spans (GlanceVAD click windows); other videos keep their full mean.

    Returns `(matrix, videos that used a span)`. A video whose spans contain no cached frame falls back to its full mean.
    """

    from core.p1c_oracle_interval import frames_in_spans

    full = pool.video_means()
    out, used = full.copy(), 0
    for index, video in enumerate(pool.video_ids):
        if video in spans and spans[video]:
            mask = frames_in_spans(pool.frame_indices_of(video), spans[video])
            if mask.any():
                out[index] = pool.frames_of(video)[mask].mean(axis=0)
                used += 1
    return out, used


def extra_video_matrix(cache_dir: Path, class_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """`(X [n, D], y [n], video ids)`: one mean vector per extra-UCF video (its 8 cached frames)."""

    directory = Path(cache_dir)
    features = np.load(directory / "features.float16.npy", mmap_mode="r")
    with (directory / "rows.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    groups = _row_groups([r["source_video_id"] for r in rows])
    videos = sorted(groups)
    x = np.stack([np.asarray(features[groups[v]], dtype=np.float32).mean(axis=0) for v in videos])
    y = np.asarray([class_names.index(rows[int(groups[v][0])]["label"]) for v in videos], dtype=np.int64)
    return x, y, videos
