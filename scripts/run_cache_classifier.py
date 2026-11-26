#!/usr/bin/env python3
"""Tip-Adapter-style memory classifier fused with the D-53 DINOv2 head (validation only; test untouched).

Per fold: one embedding per video (mean of L2-normalised fold-cache DINOv2 frame features, all cached frames), a kernel
vote exp(-beta(1-cos)) over that fold's training videos, fused with the stored head probabilities.
Primary (fixed before any run): beta=5, weight=0.5. Secondary: (beta, weight) chosen per scored fold on the OTHER folds'
validation runs minus every video shared with the scored fold (leave-one-fold-out, no scored-fold labels).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.cache_classifier import fuse, kernel_vote, video_embeddings  # noqa: E402
from core.confusion_groups import load_runs, macro_f1  # noqa: E402
from core.specialist_eval import evaluate_system  # noqa: E402

CACHE = "results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}/{split}"
BASELINE = "results/p1c/p1c-late-fusion/dinov2_probs.csv"
BETAS = (1.0, 2.0, 5.0, 10.0, 20.0)
WEIGHTS = (0.25, 0.5, 0.75)
PRIMARY = (5.0, 0.5)


def _split(fold: int, split: str):
    root = REPO_ROOT / CACHE.format(fold=fold, split=split)
    rows = list(csv.DictReader((root / "rows.csv").open(newline="", encoding="utf-8")))
    if any(int(r["row_index"]) != i for i, r in enumerate(rows)):
        raise RuntimeError(f"{root}: rows.csv row_index is not positional")
    return video_embeddings(np.load(root / "features.float16.npy", mmap_mode="r"), rows)


def build(folds, classes, out: Path) -> dict:
    sha = {}
    for fold in folds:
        tr_ids, tr_lab, tr_emb = _split(fold, "train")
        va_ids, va_lab, va_emb = _split(fold, "validation")
        if set(tr_ids) & set(va_ids):
            raise RuntimeError(f"fold {fold}: train/validation overlap")
        targets = [classes.index(x) for x in tr_lab]
        probs = np.stack([kernel_vote(va_emb, tr_emb, targets, len(classes), b) for b in BETAS])
        np.savez(out / f"pcache_fold{fold}.npz", betas=np.array(BETAS), video_ids=np.array(va_ids), labels=np.array(va_lab), probs=probs)
        sha[fold] = {s: hashlib.sha256((REPO_ROOT / CACHE.format(fold=fold, split=s) / "manifest.json").read_bytes()).hexdigest() for s in ("train", "validation")}
        print(json.dumps({"fold": fold, "train_videos": len(tr_ids), "val_videos": len(va_ids)}), flush=True)
    return sha


def _cache_for(out: Path, fold: int, videos) -> dict[float, np.ndarray]:
    z = np.load(out / f"pcache_fold{fold}.npz")
    pos = {v: i for i, v in enumerate(z["video_ids"].tolist())}
    idx = [pos[v] for v in videos]
    return {float(b): z["probs"][k][idx] for k, b in enumerate(z["betas"])}


def evaluate(classes, runs, out: Path) -> dict:
    n = len(classes)
    folds = sorted({f for f, _ in runs})
    cache = {key: _cache_for(out, key[0], r["videos"]) for key, r in runs.items()}
    valset = {f: set(runs[next(k for k in runs if k[0] == f)]["videos"]) for f in folds}
    base = {k: (r["truth"], r["probs"].argmax(1)) for k, r in runs.items()}
    systems = {f"cache_alone_b{PRIMARY[0]}": {k: (r["truth"], cache[k][PRIMARY[0]].argmax(1)) for k, r in runs.items()},
               f"fusion_b{PRIMARY[0]}_w{PRIMARY[1]}": {k: (r["truth"], fuse(r["probs"], cache[k][PRIMARY[0]], PRIMARY[1]).argmax(1)) for k, r in runs.items()}}
    chosen, lofo = {}, {}
    for f in folds:
        def score(beta, w):
            vals = []
            for k, r in runs.items():
                if k[0] == f:
                    continue
                keep = np.array([v not in valset[f] for v in r["videos"]])
                if keep.sum() == 0:
                    continue
                pred = fuse(r["probs"], cache[k][beta], w).argmax(1)
                vals.append(macro_f1(r["truth"][keep], pred[keep], n))
            return float(np.mean(vals))
        best = max(itertools.product(BETAS, WEIGHTS), key=lambda bw: (score(*bw), -bw[1]))
        chosen[f] = {"beta": best[0], "weight": best[1], "selection_score": score(*best)}
        for k, r in runs.items():
            if k[0] == f:
                lofo[k] = (r["truth"], fuse(r["probs"], cache[k][best[0]], best[1]).argmax(1))
    systems["fusion_lofo_tuned"] = lofo
    return {"systems": {name: evaluate_system(classes, base, cand) for name, cand in systems.items()}, "lofo_choice": chosen}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--output", type=Path, default=Path("results/p1c/cache-classifier"))
    ap.add_argument("--evaluate-only", action="store_true")
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    classes, runs = load_runs(REPO_ROOT / BASELINE)
    sha = None if args.evaluate_only else build(args.folds, classes, args.output)
    summary = evaluate(classes, runs, args.output)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    if sha is not None:
        (args.output / "config.json").write_text(json.dumps({
            "git_revision": rev, "folds": args.folds, "betas": BETAS, "weights": WEIGHTS, "primary": PRIMARY,
            "embedding": "mean of L2-normalised fold-cache DINOv2 cls_plus_mean_patch frame features (all cached frames), L2-normalised",
            "fold_cache_manifest_sha256": sha, "baseline_probs_sha256": hashlib.sha256((REPO_ROOT / BASELINE).read_bytes()).hexdigest(),
            "head_probs": "D-53 step_permuted, 4 folds x 3 seeds (stored)", "test_split": "untouched"}, indent=2) + "\n")
    for name, s in summary["systems"].items():
        print(name, json.dumps({k: s[k] for k in ("candidate_mean", "baseline_mean", "mean_delta", "per_fold_delta", "folds_meeting_gate", "passes_gate")}, default=float))
    print("lofo_choice", json.dumps(summary["lofo_choice"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
