"""UCF-Crime videos outside the 700-video action-recognition protocol, as extra training data (D-64).

The frame blob (D-34) covers all 1,950 official videos; the AR folds use 50 per class. Burglary, RoadAccidents,
Robbery and Stealing have 300 more anomaly videos between them. They are disjoint from every fold's train,
validation and test videos by file name, and they carry only their directory name as a video-level label (D-40).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

EXTRA_CLASSES = ("Burglary", "RoadAccidents", "Robbery", "Stealing")
AR_MANIFEST = "results/p1b/p1b-video-ar-fold{fold}/{split}_manifest.csv"
VIDEOS_CSV = "results/p1c/p1c-video-frames/videos.csv"
FOLDS, SPLITS = (1, 2, 3, 4), ("train", "validation", "test")


def protocol_video_ids(repo_root: Path) -> set[str]:
    out: set[str] = set()
    for fold in FOLDS:
        for split in SPLITS:
            with (Path(repo_root) / AR_MANIFEST.format(fold=fold, split=split)).open(newline="", encoding="utf-8") as h:
                out |= {r["source_video_id"] for r in csv.DictReader(h)}
    return out


def select_extra_videos(videos: Iterable[Mapping[str, str]], protocol: set[str], classes: Sequence[str] = EXTRA_CLASSES) -> list[dict]:
    """Rows of videos.csv whose directory is one of `classes` and whose file stem is not in the protocol, sorted by name."""

    picked = []
    for r in videos:
        stem = r["file_name"].rsplit(".", 1)[0]
        if r["source_directory"] in classes and stem not in protocol:
            if r["decode_status"] != "ok":
                raise ValueError(f"{stem}: decode_status {r['decode_status']!r}")
            picked.append({"video": stem, "label": r["source_directory"], "first_row": int(r["first_row_index"]), "frames": int(r["kept_frame_count"])})
    return sorted(picked, key=lambda v: v["video"])


def blob_rows(video: Mapping, frame_index: np.ndarray) -> list[dict]:
    """One manifest-style row per kept frame of a video, with `blob_row` pointing into the frame blob."""

    a, n = video["first_row"], video["frames"]
    seg = frame_index[a:a + n]
    return [{"source_dataset": "ucf", "source_video_id": video["video"], "source_frame_index": int(f), "label": video["label"],
             "blob_row": a + i} for i, f in enumerate(seg["source_frame_index"])]


def balanced_extra_subset(extra_labels: Mapping[str, str], per_class: int, seed: int) -> set[str]:
    """`per_class` videos of every class chosen by a seeded shuffle (all of them if the class has fewer)."""

    rng = np.random.default_rng(seed)
    by: dict[str, list[str]] = {}
    for v, c in sorted(extra_labels.items()):
        by.setdefault(c, []).append(v)
    keep: set[str] = set()
    for c in sorted(by):
        vids = by[c]
        keep |= {vids[i] for i in rng.permutation(len(vids))[:per_class]}
    return keep
