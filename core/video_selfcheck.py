"""End-to-end check of the video-testing path: real MP4s through decoding, preprocessing, DINOv2 and the exported heads.

Scores one fold's validation videos with that fold's heads and compares the macro-F1 with the value recorded when the
heads were exported, alongside the same heads on the registered feature cache. Validation only; the test view is not read.
A pass means this code path behaves like the evaluated pipeline, not that the classifier is good.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

UCF_VIDEOS = "Datasets/UCF Crime Official/raw/UCF_Crimes/Videos"
MIN_COSINE = 0.999  # earlier pixel-path checks measured 0.9999 or better (D-65/D-67)
F1_TOLERANCE = 0.02  # the head export's own tolerance (scripts/export_demo_heads.py)


def frame_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def decoded_features(model, rows: list[dict], root: Path) -> np.ndarray:
    """Features for `rows`, decoded from the MP4s through the same calls `DemoModel.predict` makes."""

    from core.demo_inference import decode_frames

    paths = {p.stem: p for p in root.rglob("*.mp4")}
    positions: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        positions[row["source_video_id"]].append(position)
    out: np.ndarray | None = None
    for video, where in positions.items():
        frames, indices, _fps = decode_frames(paths[video], max_frames=10**9)
        features = model.features(frames)
        lookup = {source: k for k, source in enumerate(indices)}
        if out is None:
            out = np.zeros((len(rows), features.shape[1]), dtype=np.float32)
        out[where] = features[[lookup[int(rows[p]["source_frame_index"])] for p in where]]
    return out


def run_self_check(model, repo_root: Path, fold: int = 1) -> dict:
    from core.demo_inference import FRAMES
    from core.p1c_oracle_interval import predict_video_probs
    from core.specialist import GroupMap, remap_sequences
    from scripts.run_subset7_search import load_fold_features, score, sequences

    data = load_fold_features(fold, ("validation",))
    cached, rows = data["validation"]
    decoded = decoded_features(model, rows, repo_root / UCF_VIDEOS)
    cosines = frame_cosines(cached, decoded)
    recorded = {(c["fold"], c["seed"]): c["val_macro_f1"] for c in json.loads((repo_root / "results/demo/export.json").read_text())["checks"]}
    groups = GroupMap(data["class_names"], data["class_names"])
    cells = []
    for entry in (h for h in model.heads if h["fold"] == fold):
        seed, predictions, scores = entry["seed"], {}, {}
        for name, features in (("cache", cached), ("decoded", decoded)):
            seqs = remap_sequences(sequences({"validation": (features, rows)}, "validation", seed), groups)
            predictions[name] = {k: predict_video_probs(entry["head"], seq, FRAMES) for k, (seq, _row) in seqs.items()}
            scores[name] = score(groups, seqs, predictions[name])[0]
        same = np.mean([int(np.argmax(predictions["cache"][k])) == int(np.argmax(predictions["decoded"][k])) for k in predictions["cache"]])
        cells.append({"fold": fold, "seed": seed, "recorded_macro_f1": recorded[(fold, seed)], "cache_macro_f1": scores["cache"],
                      "decoded_macro_f1": scores["decoded"], "same_prediction_fraction": float(same)})
    passed = bool(cosines.min() >= MIN_COSINE and all(
        abs(c["decoded_macro_f1"] - c["recorded_macro_f1"]) <= F1_TOLERANCE and abs(c["cache_macro_f1"] - c["recorded_macro_f1"]) <= F1_TOLERANCE
        for c in cells))
    return {"passed": passed, "videos": len({r["source_video_id"] for r in rows}), "frames": len(rows),
            "min_frame_cosine": float(cosines.min()), "median_frame_cosine": float(np.median(cosines)),
            "thresholds": {"min_cosine": MIN_COSINE, "f1_tolerance": F1_TOLERANCE}, "cells": cells, "test_view_read": False}
