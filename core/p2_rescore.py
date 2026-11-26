"""RESULT 1 re-score with a downstream head that is not selected on the scored videos (FINAL_PUSH_PLAN Step 3).

`core.p2_downstream.classify` trains the D-53 head on a fold's training videos and picks its epoch on
that fold's clean validation videos -- the same videos every degradation/restoration condition is then
scored on. The clean reference therefore carries the epoch-selection optimism (D-66) that no perturbed
condition shares, which biases every "restoring clean frames" delta downwards.

Here the stored per-condition features are re-scored with either rule:

- `best_validation` reproduces the registered numbers (the check that this module reads the stored
  arrays the way the original runners wrote them);
- `last_epoch` trains a fixed number of epochs and keeps the final weights (no selection).

Nothing is re-extracted; the feature arrays, PSNR and SSIM are those of the original runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from core import inner_cv
from core.confusion_groups import macro_f1
from core.p1c_head_variants import CHECKPOINT_BEST_VALIDATION, CHECKPOINT_LAST_EPOCH

CACHE = "results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}"
RUN_DIRS = ("results/p2/p2-classical", "results/p2/p2b-learned", "results/p2/p2b-nafnet", "results/p2/p2b-extra")


def condition_names(run_dir: Path) -> list[str]:
    """Condition names in feature-array order: `summary.json` keeps the runner's condition order."""

    return list(json.loads((Path(run_dir) / "summary.json").read_text()))


def rescore(data_root: Path, run_dirs: Sequence[Path], *, folds: Sequence[int], seeds: Sequence[int], rule: str,
            epochs: int | None, threads: int = 4) -> dict[str, list[dict]]:
    """`{run_dir name: [{condition, fold, seed, macro_f1, accuracy, videos}]}` for every stored condition."""

    import torch

    from core.p1c_head_variants import group_pooled_sequences_by_video
    from core.p1c_oracle_interval import predict_video_probs

    if rule not in (CHECKPOINT_BEST_VALIDATION, CHECKPOINT_LAST_EPOCH):
        raise ValueError(f"unknown rule {rule!r}")
    torch.set_num_threads(threads)
    data_root = Path(data_root)
    pools = {fold: inner_cv.load_pool(data_root / CACHE.format(fold=fold), fold) for fold in folds}
    order = sorted({v for pool in pools.values() for v in pool.videos_of_split("validation")})
    position = {v: i for i, v in enumerate(order)}
    stored = {}
    for run_dir in run_dirs:
        names = condition_names(run_dir)
        feats = np.load(Path(run_dir) / "features.float16.npy", mmap_mode="r")
        if feats.shape[:2] != (len(names), len(order)):
            raise ValueError(f"{run_dir}: features {feats.shape} do not match {len(names)} conditions x {len(order)} videos")
        stored[Path(run_dir).name] = (names, feats)

    trainer = {"bootstrap_iterations": 100}
    if epochs is not None:
        trainer["epochs"] = epochs
    if rule == CHECKPOINT_LAST_EPOCH:
        trainer["checkpoint_rule"] = CHECKPOINT_LAST_EPOCH
    out: dict[str, list[dict]] = {name: [] for name in stored}
    for fold, pool in pools.items():
        fold_videos = sorted(pool.videos_of_split("validation"))
        truth = np.array([int(pool.targets[pool.video_ids.index(v)]) for v in fold_videos])
        for seed in seeds:
            head, _monitor, result = inner_cv.fit_attention_head(
                pool, pool.videos_of_split("train"), pool.videos_of_split("validation"), seed, **trainer)
            for run_name, (names, feats) in stored.items():
                for index, name in enumerate(names):
                    parts = [np.asarray(feats[index, position[v]], dtype=np.float32) for v in fold_videos]
                    rows = [row for v, part in zip(fold_videos, parts) for row in inner_cv._rows(pool, v, len(part))]
                    seqs = group_pooled_sequences_by_video(np.concatenate(parts), rows, permute_frame_order=True, permute_seed=2000 + seed)
                    pred = np.array([int(np.argmax(predict_video_probs(head, seqs[(inner_cv.SOURCE_DATASET, v)][0], inner_cv.FRAMES)))
                                     for v in fold_videos])
                    out[run_name].append({"condition": name, "fold": fold, "seed": seed,
                                          "macro_f1": macro_f1(truth, pred, len(pool.class_names)),
                                          "accuracy": float((truth == pred).mean()), "videos": len(fold_videos),
                                          "head_epochs_trained": len(result.epochs), "head_epoch_kept": result.best_epoch})
            print(json.dumps({"fold": fold, "seed": seed, "rule": rule, "epoch_kept": result.best_epoch,
                              "clean": next(r["macro_f1"] for r in out[next(iter(stored))][::-1]
                                            if r["condition"].startswith("identity__identity") and r["fold"] == fold and r["seed"] == seed)}), flush=True)
    return out


def difference_from_stored(run_dir: Path, runs: Sequence[dict]) -> dict[str, float]:
    """How far a re-score is from the registered numbers: the largest single (condition, fold, seed) cell
    difference and the largest difference of a condition's mean over its cells.

    The training loop does not promise bit-exactness across thread counts (K-24), so the gate is on the
    condition means, which are what the registered table reports; the cell maximum is reported beside it.
    """

    registered = {(r["condition"], r["fold"], r["seed"]): r["macro_f1"]
                  for r in json.loads((Path(run_dir) / "classification_runs.json").read_text())}
    cells = [(r["condition"], r["macro_f1"] - registered[(r["condition"], r["fold"], r["seed"])]) for r in runs]
    by_condition: dict[str, list[float]] = {}
    for name, delta in cells:
        by_condition.setdefault(name, []).append(delta)
    return {"max_cell": max(abs(d) for _n, d in cells), "max_condition_mean": max(abs(float(np.mean(v))) for v in by_condition.values()),
            "cells_differing": sum(abs(d) > 1e-9 for _n, d in cells), "cells": len(cells)}


def write_run(run_dir: Path, output: Path, runs: Sequence[dict], folds: Sequence[int]) -> None:
    """A copy of the run's `summary.json` with the classifier columns replaced, plus the new per-cell runs.

    PSNR and SSIM are carried over unchanged; the layout is what `scripts.build_result1_table.rows_for` reads.
    """

    old = json.loads((Path(run_dir) / "summary.json").read_text())
    summary = {}
    for name, row in old.items():
        mine = [r for r in runs if r["condition"] == name]
        per_fold = {str(f): {k: float(np.mean([r[k] for r in mine if r["fold"] == f])) for k in ("macro_f1", "accuracy")} for f in folds}
        summary[name] = {"degradation": row["degradation"], "restoration": row["restoration"], "psnr_db": row["psnr_db"], "ssim": row["ssim"],
                         "macro_f1": float(np.mean([v["macro_f1"] for v in per_fold.values()])),
                         "accuracy": float(np.mean([v["accuracy"] for v in per_fold.values()])), "per_fold": per_fold}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (output / "classification_runs.json").write_text(json.dumps(list(runs), indent=1) + "\n")
