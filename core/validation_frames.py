"""Evenly spaced raw frames from the AR validation videos, read from the D-34 frame blob (no preprocessing)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from core.p1c_video_sampling import sample_rows_per_video

MANIFEST = "results/p1b/p1b-video-ar-fold{fold}/validation_manifest.csv"
CLOSURE = "results/p1c/p1c-video-frames/extract_closure.json"


def select_validation_frames(repo_root: Path, folds: Sequence[int], classes: Sequence[str], frames: int) -> dict[str, dict]:
    """video id -> {label, fold, rows, frames, filepath, file_digest, dataset}.

    The first fold whose validation set holds a video provides its rows; `frames` rows per video are chosen by
    `sample_rows_per_video(seed=None)` (deterministic, temporally spread). `dataset.raw_jpeg(row)` returns the bytes.
    """

    from data.p1c_loader import FramePreprocessing
    from data.video_dataset import P1CVideoFrameDataset

    pre = FramePreprocessing.from_config({"resize_height": 224, "resize_width": 224, "interpolation": "bilinear",
                                          "color_mode": "RGB", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    out, datasets = {}, {}
    for fold in folds:
        path = Path(repo_root) / MANIFEST.format(fold=fold)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        datasets[fold] = P1CVideoFrameDataset(path, closure_path=Path(repo_root) / CLOSURE, preprocessing=pre,
                                              class_names=list(classes), verify_digest=False, split="validation")
        by_video = defaultdict(list)
        for i in sample_rows_per_video(rows, frames, seed=None):
            by_video[rows[i]["source_video_id"]].append(i)
        for vid, idx in by_video.items():
            if vid not in out:
                out[vid] = {"label": rows[idx[0]]["label"], "fold": fold, "rows": idx,
                            "frames": [int(rows[i]["source_frame_index"]) for i in idx],
                            "filepath": rows[idx[0]]["filepath"], "file_digest": rows[idx[0]]["file_digest"]}
    for v in out.values():
        v["dataset"] = datasets[v["fold"]]
    return out
