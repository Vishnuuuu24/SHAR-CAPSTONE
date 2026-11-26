"""Official UCF-Crime Anomaly-Detection split for the live detector (FINAL_PUSH_PLAN_2026-11-11 Step 4, D-71).

A separate task from the 14-class classifier (docs/PROJECT_RULES.md rule 3): binary "is something abnormal happening
at this moment", trained with VIDEO-level labels only and tested on the official frame intervals.

D-71 label mapping (K-12 refuses to invent one): a video is `normal` when its source directory is one
of the Normal directories and `anomaly` otherwise. The 13 crime folders keep their name as `category`,
which is reported per category but never used as a training target here.

Frames come from the D-34 blob, which keeps every 5th source frame; `every` subsamples the kept frames
again (3 -> one frame per 15 source frames, 0.5 s at 30 fps, close to the one-in-16 sampling of the
CLIP-feature detectors in the literature).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

SPLIT_DIR = "Datasets/UCF Crime Official/raw/UCF_Crimes/Anomaly_Detection_splits"
ANNOTATION = "Datasets/UCF Crime Official/annotations/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt"
VIDEOS_CSV = "results/p1c/p1c-video-frames/videos.csv"
FRAME_INDEX = "results/p1c/p1c-video-frames/frame_index.npy"
NORMAL_DIRECTORIES = frozenset({"Training_Normal_Videos_Anomaly", "Testing_Normal_Videos_Anomaly", "Normal_Videos_event", "z_Normal_Videos_event"})
SPLIT_FILES = {"train": "Anomaly_Train.txt", "test": "Anomaly_Test.txt"}


@dataclass(frozen=True)
class DetectVideo:
    relative_path: str          # "<source_directory>/<file_name>", the unique key (K-11: a stem is not unique)
    split: str                  # "train" | "test"
    anomaly: bool
    category: str               # crime folder, or "Normal"
    fps: float
    first_blob_row: int
    kept_frames: int            # frames the blob holds for this video (every 5th source frame)
    decoded_frames: int         # every source frame the decoder produced (the frame-level metric's denominator)

    @property
    def stem(self) -> str:
        return self.relative_path.rsplit("/", 1)[1].rsplit(".", 1)[0]


def category_of(relative_path: str) -> tuple[bool, str]:
    directory = relative_path.split("/", 1)[0]
    return (False, "Normal") if directory in NORMAL_DIRECTORIES else (True, directory)


def load_split(repo_root: Path) -> list[DetectVideo]:
    """Every video of the official train and test lists, joined to the blob's video table; raises on any gap or overlap."""

    repo_root = Path(repo_root)
    with (repo_root / VIDEOS_CSV).open(newline="", encoding="utf-8") as handle:
        table = {r["relative_path"]: r for r in csv.DictReader(handle)}
    videos, seen = [], set()
    for split, file_name in SPLIT_FILES.items():
        for line in (repo_root / SPLIT_DIR / file_name).read_text().splitlines():
            path = line.strip()
            if not path:
                continue
            if path in seen:
                raise ValueError(f"{path} is listed twice across the official train/test lists")
            if path not in table:
                raise ValueError(f"{path} is in {file_name} but not in the frame blob's videos.csv")
            seen.add(path)
            row = table[path]
            anomaly, category = category_of(path)
            videos.append(DetectVideo(path, split, anomaly, category, float(row["reported_fps"]),
                                      int(row["first_row_index"]), int(row["kept_frame_count"]), int(row["decoded_frame_count"])))
    return videos


def frame_plan(videos: Sequence[DetectVideo], frame_index: np.ndarray, every: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(blob_row, video_position, source_frame_index)` for every `every`-th kept frame of each video, in video order."""

    if every < 1:
        raise ValueError("every must be >= 1")
    rows, owner = [], []
    for position, video in enumerate(videos):
        mine = np.arange(video.first_blob_row, video.first_blob_row + video.kept_frames, every, dtype=np.int64)
        rows.append(mine)
        owner.append(np.full(len(mine), position, dtype=np.int64))
    blob_rows = np.concatenate(rows)
    return blob_rows, np.concatenate(owner), np.asarray(frame_index["source_frame_index"][blob_rows], dtype=np.int64)


def load_test_intervals(repo_root: Path) -> dict[str, list[tuple[int, int]]]:
    """Official anomalous source-frame spans per test video stem (inclusive); normal videos map to an empty list."""

    out: dict[str, list[tuple[int, int]]] = {}
    for line in (Path(repo_root) / ANNOTATION).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        spans = [(int(parts[2]), int(parts[3])), (int(parts[4]), int(parts[5]))]
        out[parts[0].rsplit(".", 1)[0]] = [(a, b) for a, b in spans if a >= 0 and b > a]
    return out


def frame_labels(source_frames: np.ndarray, spans: Sequence[tuple[int, int]]) -> np.ndarray:
    """1 where a sampled source frame lies inside any official span, else 0."""

    labels = np.zeros(len(source_frames), dtype=np.int8)
    for start, end in spans:
        labels[(source_frames >= start) & (source_frames <= end)] = 1
    return labels
