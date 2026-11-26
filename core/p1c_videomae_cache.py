"""Block-boundary hidden-state cache for partial VideoMAE fine-tuning.

One cache serves all four Action-Recognition folds: a clip is a pure function
of its video's pixels, and the 16-row tiling (`core.p1c_event_clips._window_starts`)
and the 8-clips-per-video temporal spread (`core.p1c_video_sampling`) are
functions of the video alone, so a video selected under fold 1 is selected
identically under fold 3. The cache holds the UNION of train+validation videos
over the folds; a fold's test videos are only ever read when they are another
fold's train/validation video, and a trainer addresses rows strictly through a
fold's own membership (`fold_row_indices`).

Layout: `hidden.float16.npy` is a preallocated `.npy` memmap
`[n_clips, tokens, dim]`; `rows.csv` carries provenance (video, first/last
source frame, target, label, fold roles); `manifest.json` pins backbone weight
digest, preprocessing, clip definition, code revision and the array digest.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.feature_cache_verify import _ordered_int
from core.p1c_video_sampling import sample_rows_per_video

TOKENS, DIM, ITEMSIZE = 1568, 768, 2
CLIPS_PER_VIDEO = 8
SPLITS = ("train", "validation")
ARRAY_NAME, ROWS_NAME, MANIFEST_NAME = "hidden.float16.npy", "rows.csv", "manifest.json"
ROW_FIELDS = ["row_index", "source_video_id", "start_source_frame", "end_source_frame", "target", "label", "fold_roles"]


def select_clips(tiled_rows: Sequence[Mapping[str, Any]], per_video: int = CLIPS_PER_VIDEO) -> list[Mapping[str, Any]]:
    """The `per_video` temporally-spread clips of each video, in tiling order.

    `tiled_rows` are one split's clip rows (`source_video_id`, `start_source_frame`,
    ...), contiguous per video and ascending, exactly as the clip caches store
    them. Deterministic: same video, same clip count -> same selection.
    """

    rows = [dict(r, source_dataset="ucf_crime_official_video_frames") for r in tiled_rows]
    return [tiled_rows[i] for i in sample_rows_per_video(rows, per_video)]


def union_plan(
    per_fold_split_rows: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]], per_video: int = CLIPS_PER_VIDEO
) -> list[dict[str, Any]]:
    """Union of selected clips over `{fold: {split: tiled rows}}`, one entry per distinct clip.

    Each entry: video, first/last source frame, target, label and `roles`
    ({fold: split}). Raises if two folds select different clips (or labels)
    for one video -- the premise that makes a shared cache valid.
    """

    by_video: dict[str, dict[str, Any]] = {}
    for fold in sorted(per_fold_split_rows):
        for split in SPLITS:
            chosen = select_clips(per_fold_split_rows[fold][split], per_video)
            videos: dict[str, list] = {}
            for r in chosen:
                videos.setdefault(r["source_video_id"], []).append(
                    (int(r["start_source_frame"]), int(r["end_source_frame"]), int(r["target"]), r["label"]))
            for video, clips in videos.items():
                entry = by_video.setdefault(video, {"clips": clips, "roles": {}})
                if entry["clips"] != clips:
                    raise ValueError(f"folds disagree on the selected clips of {video!r} (fold {fold} {split})")
                if fold in entry["roles"]:
                    raise ValueError(f"{video!r} is listed twice in fold {fold}")
                entry["roles"][fold] = split
    plan = []
    for video in sorted(by_video):
        for f0, f1, target, label in by_video[video]["clips"]:
            plan.append({"source_video_id": video, "start_source_frame": f0, "end_source_frame": f1,
                         "target": target, "label": label, "roles": dict(by_video[video]["roles"])})
    return plan


def cache_nbytes(n_clips: int, tokens: int = TOKENS, dim: int = DIM) -> int:
    return n_clips * tokens * dim * ITEMSIZE


def encode_roles(roles: Mapping[int, str]) -> str:
    return ";".join(f"{fold}:{roles[fold]}" for fold in sorted(roles))


def decode_roles(text: str) -> dict[int, str]:
    return {int(a): b for a, b in (item.split(":") for item in text.split(";") if item)}


def write_rows(path: Path, plan: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ROW_FIELDS)
        for i, c in enumerate(plan):
            writer.writerow([i, c["source_video_id"], c["start_source_frame"], c["end_source_frame"],
                             c["target"], c["label"], encode_roles(c["roles"])])


def read_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as handle:
        records = list(csv.DictReader(handle))
    return [{"row_index": int(r["row_index"]), "source_video_id": r["source_video_id"],
             "start_source_frame": int(r["start_source_frame"]), "end_source_frame": int(r["end_source_frame"]),
             "target": int(r["target"]), "label": r["label"], "roles": decode_roles(r["fold_roles"])}
            for r in records]


def fold_row_indices(rows: Sequence[Mapping[str, Any]], fold: int, split: str) -> list[int]:
    """Cache rows whose video the fold lists under `split` -- the ONLY way a trainer addresses rows."""

    return [r["row_index"] for r in rows if r["roles"].get(fold) == split]


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def ulp_gate(cached: np.ndarray, fresh: np.ndarray, *, max_ulps: int = 8, floor_magnitude: float = 1.0) -> dict[str, float]:
    """K-26 gate on float16 arrays: each element may differ by `max_ulps` steps, each step floored at `floor_magnitude`.

    Same arithmetic as `core.feature_cache_verify.verify_feature_cache`;
    returns the worst |delta|, worst ULP distance and whether the gate holds.
    """

    a, b = cached.astype(np.float32).ravel(), fresh.astype(np.float32).ravel()
    delta = np.abs(a - b)
    step = np.spacing(np.maximum(np.abs(b), floor_magnitude).astype(np.float16)).astype(np.float32)
    ulps = np.abs(_ordered_int(np.ascontiguousarray(cached, dtype=np.float16).ravel())
                  - _ordered_int(np.ascontiguousarray(fresh, dtype=np.float16).ravel()))
    return {"max_abs_delta": float(delta.max()), "max_ulps": int(ulps.max()),
            "ok": bool(np.all(delta <= max_ulps * step))}


def publish_cache(source: Path, destination: Path) -> str:
    """Copy a built cache directory to durable storage and verify the array digest; returns it."""

    destination.mkdir(parents=True, exist_ok=True)
    for name in (ROWS_NAME, MANIFEST_NAME, ARRAY_NAME):
        shutil.copyfile(source / name, destination / name)
    src, dst = sha256_file(source / ARRAY_NAME), sha256_file(destination / ARRAY_NAME)
    if src != dst:
        raise RuntimeError(f"published copy differs: {src} != {dst}")
    return dst


def git_revision(repo_root: Path) -> dict[str, Any]:
    """HEAD commit and whether importable code differs from it (rule 5: code revision)."""

    import subprocess

    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": run("rev-parse", "HEAD") or "unknown",
            "dirty_tracked_files": [l[3:] for l in run("status", "--porcelain", "--untracked-files=no").splitlines()]}
