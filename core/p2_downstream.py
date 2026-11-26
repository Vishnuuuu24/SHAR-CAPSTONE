"""RESULT 1 (P2C) downstream half shared by the classical and learned runners: feature check, classifier, table."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from core.confusion_groups import macro_f1


def clean_feature_check(repo_root: Path, order, videos, feats, folds) -> dict:
    """Condition 0 (identity, identity) must reproduce the fold-cache validation features of the same frames."""

    diffs, cos = [], []
    for fold in folds:
        root = Path(repo_root) / f"results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}/validation"
        rows = list(csv.DictReader((root / "rows.csv").open()))
        pos = {(r["source_video_id"], int(r["source_frame_index"])): i for i, r in enumerate(rows)}
        ref = np.load(root / "features.float16.npy", mmap_mode="r")
        for vi, v in enumerate(order):
            if videos[v]["fold"] != fold:
                continue
            a = np.asarray(feats[0, vi], dtype=np.float32)
            b = np.asarray(ref[[pos[(v, f)] for f in videos[v]["frames"]]], dtype=np.float32)
            diffs.append(float(np.abs(a - b).max()))
            cos.append(float(np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))))
    return {"max_abs_diff": max(diffs), "median_max_abs_diff": float(np.median(diffs)), "min_mean_cosine": min(cos), "videos": len(diffs)}


def classify(conds, videos, order, feats, classes, *, folds: Sequence[int], seeds: Sequence[int], threads: int,
             epochs: int | None) -> list[dict]:
    """D-53 head retrained per fold x seed on fold-cache train features, applied to every condition's sequences."""

    import torch
    from core.p1c_head_variants import group_pooled_sequences_by_video
    from core.p1c_oracle_interval import predict_video_probs
    from scripts.run_p1c_xd_extra_probe import FRAMES, train_head
    from scripts.run_subset7_search import load_fold_features, sequences
    torch.set_num_threads(threads)

    position = {v: i for i, v in enumerate(order)}
    runs = []
    for fold in folds:
        fd = load_fold_features(fold, ("train", "validation"))
        fold_videos = sorted({r["source_video_id"] for r in fd["validation"][1]} & set(position))
        truth = np.array([classes.index(videos[v]["label"]) for v in fold_videos])
        for seed in seeds:
            result, head = train_head(seed, sequences(fd, "train", seed), sequences(fd, "validation", seed), classes, epochs)
            for c in conds:
                pooled, rows = [], []
                for v in fold_videos:
                    x = np.asarray(feats[c.index, position[v]], dtype=np.float32)
                    for j, f in enumerate(videos[v]["frames"]):
                        pooled.append(x[j])
                        rows.append({"source_dataset": "ucf", "source_video_id": v, "source_frame_index": f,
                                     "target": classes.index(videos[v]["label"]), "label": videos[v]["label"]})
                seqs = group_pooled_sequences_by_video(np.stack(pooled), rows, permute_frame_order=True, permute_seed=2000 + seed)
                pred = np.array([int(np.argmax(predict_video_probs(head, seqs[("ucf", v)][0], FRAMES))) for v in fold_videos])
                runs.append({"condition": c.name, "fold": fold, "seed": seed, "macro_f1": macro_f1(truth, pred, len(classes)),
                             "accuracy": float((truth == pred).mean()), "videos": len(fold_videos),
                             "full_frame_validation_macro_f1": result.best_validation_macro_f1})
            print(json.dumps({"fold": fold, "seed": seed, "clean_32frame_macro_f1": next(r["macro_f1"] for r in runs[::-1] if r["condition"] == conds[0].name and r["fold"] == fold and r["seed"] == seed)}), flush=True)
        del fd
    return runs


def condition_table(conds, quality: list[dict], runs: list[dict], folds: Sequence[int]) -> dict[str, dict]:
    """Per condition: mean PSNR (finite frames) and SSIM over videos, fold-averaged macro-F1/accuracy."""

    table = {}
    for c in conds:
        q = [r for r in quality if r["condition"] == c.name]
        rs = [r for r in runs if r["condition"] == c.name]
        per_fold = {f: {k: float(np.mean([r[k] for r in rs if r["fold"] == f])) for k in ("macro_f1", "accuracy")} for f in folds}
        finite = [float(r["psnr_mean"]) for r in q if math.isfinite(float(r["psnr_mean"]))]
        table[c.name] = {"degradation": c.degradation_id, "restoration": c.restoration,
                         "psnr_db": float(np.mean(finite)) if finite else None, "ssim": float(np.mean([float(r["ssim_mean"]) for r in q])),
                         "macro_f1": float(np.mean([v["macro_f1"] for v in per_fold.values()])),
                         "accuracy": float(np.mean([v["accuracy"] for v in per_fold.values()])), "per_fold": per_fold}
    return table
