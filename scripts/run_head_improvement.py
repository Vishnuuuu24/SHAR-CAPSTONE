#!/usr/bin/env python3
"""D-63: training-data variants of the D-53 head (resampling, label smoothing, learning curve) on the four AR folds.

Validation only (the test split is spent, D-57). Arms and the +0.04-on-every-fold gate are fixed in
core/head_experiment.py before any result is seen. The control must reproduce the stored D-53 validation numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.head_experiment import ARMS, GATED, run_fold, summarise_arms  # noqa: E402


def _worker(job):
    fold, seeds, arms, epochs, threads = job
    import torch
    torch.set_num_threads(threads)
    t0 = time.time()
    rec, probs = run_fold(fold, seeds, arms, epochs_override=epochs)
    print(json.dumps({"fold": fold, "runs": len(rec), "seconds": round(time.time() - t0)}), flush=True)
    return rec, probs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None, help="epoch ceiling override (smoke only)")
    ap.add_argument("--output", type=Path, default=Path("results/p1c/head-improvement"))
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(f, args.seeds, args.arms, args.epochs, args.threads) for f in args.folds]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(_worker, jobs, chunksize=1)
    records = [r for rec, _ in parts for r in rec]
    probs = [r for _, p in parts for r in p]
    summary = summarise_arms(records)
    (args.output / "runs.json").write_text(json.dumps(records, indent=1) + "\n")
    classes = [k[2:] for k in probs[0] if k.startswith("p_")]
    with (args.output / "probs.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(probs[0]))
        w.writeheader()
        w.writerows(probs)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    (args.output / "summary.json").write_text(json.dumps({"git_revision": rev, "folds": args.folds, "seeds": args.seeds, "arms": {a: ARMS[a] for a in args.arms},
                                                          "gated_arms": list(GATED), "gate": "macro-F1 delta vs control >= 0.04 on every fold",
                                                          "split": "validation only", "classes": classes, "summary": summary}, indent=1) + "\n")
    for arm, s in summary.items():
        print(arm, json.dumps({k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v] if isinstance(v, list) else v)
                               for k, v in s.items() if k in ("macro_f1", "per_fold_delta_vs_control", "mean_delta", "folds_positive", "gate_passed", "median_best_epoch")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
