"""Frame bookkeeping for re-extracting the fold-cache frames with another frozen backbone (D-65).

The experiment uses, per fold, the 8-frame seed-0 training draw and every cached validation frame (as loaded by
`scripts.run_subset7_search.load_fold_features`). These helpers take the union of those frames over folds, map each
(video, source frame) to its row in the D-34 frame blob, and realign a cache built on that union to any fold's rows.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

Key = tuple[str, int]


def frame_key(row: Mapping) -> Key:
    return (str(row["source_video_id"]), int(row["source_frame_index"]))


def union_keys(row_lists: Iterable[Sequence[Mapping]]) -> list[Key]:
    """Sorted unique (video, frame) keys over several row lists."""

    return sorted({frame_key(r) for rows in row_lists for r in rows})


def blob_rows_for(keys: Sequence[Key], videos_csv: Path, frame_index: np.ndarray) -> np.ndarray:
    """Frame-blob row of every key; raises if a video or frame is not in the blob."""

    with Path(videos_csv).open(newline="") as h:
        first = {r["file_name"].rsplit(".", 1)[0]: (int(r["first_row_index"]), int(r["kept_frame_count"])) for r in csv.DictReader(h)}
    lookup: dict[str, dict[int, int]] = {}
    out = np.empty(len(keys), dtype=np.int64)
    for i, (video, frame) in enumerate(keys):
        if video not in lookup:
            a, n = first[video]
            lookup[video] = {int(f): a + j for j, f in enumerate(frame_index["source_frame_index"][a:a + n])}
        out[i] = lookup[video][frame]
    return out


def align(features: np.ndarray, keys: Sequence[Key], rows: Sequence[Mapping]) -> np.ndarray:
    """Rows of `features` (indexed like `keys`) in the order of `rows`; raises on any missing frame."""

    position = {k: i for i, k in enumerate(keys)}
    return features[np.array([position[frame_key(r)] for r in rows], dtype=np.int64)]
