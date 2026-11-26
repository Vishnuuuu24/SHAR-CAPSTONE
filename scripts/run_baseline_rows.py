#!/usr/bin/env python3
"""D-69: the registered baseline rows (GAP+linear, MLP, CNN-LSTM on frozen ResNet50; LSTM and D-53 heads on DINOv2-B)
under the D-65 protocol, on cached features (CPU).

Writes <output>/{runs.json, probs.csv, summary.json}. Validation only; the test split is not read. Baselines are
reported with a video-clustered 95% interval each and a descriptive paired difference against the D-53 head; no claim
rule applies (nothing here is a candidate).
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

from core.baseline_rows import REFERENCE, ROWS  # noqa: E402


def _worker(job):
    fold, seeds, rows, epochs, threads = job
    import torch

    from core.baseline_rows import run_fold
    torch.set_num_threads(threads)
    t0 = time.time()
    out = run_fold(fold, seeds, rows, epochs=epochs, repo_root=REPO_ROOT)
    print(json.dumps({"fold": fold, "runs": len(out[0]), "seconds": round(time.time() - t0)}), flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rows", nargs="+", default=list(ROWS), choices=list(ROWS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None, help="smoke only")
    ap.add_argument("--output", type=Path, default=Path("results/p1c/baseline-rows-d65"))
    args = ap.parse_args(argv)
    from core import failure_analysis as fa
    from core.d65_claim import contrast
    from core.head_experiment import summarise_arms

    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(_worker, [(f, args.seeds, args.rows, args.epochs, args.threads) for f in args.folds], chunksize=1)
    records = [r for rec, _, _ in parts for r in rec]
    probs = [r for _, p, _ in parts for r in p]
    caches = [c for _, _, c in parts]
    (args.output / "runs.json").write_text(json.dumps(records, indent=1) + "\n")
    with (args.output / "probs.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(probs[0]))
        w.writeheader()
        w.writerows(probs)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip())
    summary = {"git_revision": rev, "working_tree_dirty": dirty, "decision": "D-69", "protocol": "D-65", "folds": args.folds,
               "seeds": args.seeds, "split": "validation only", "label_view": "inherited video-level labels (D-40), not event-only",
               "seconds": round(time.time() - t0), "caches": caches,
               "arms": summarise_arms(records, control=REFERENCE) if REFERENCE in args.rows else {}, "intervals": {}, "vs_reference": {}}
    if args.epochs is None:
        for row in args.rows:
            boot = fa.paired_cluster_bootstrap(fa.load_arm_predictions(args.output / "probs.csv", row), draws=2000)
            summary["intervals"][row] = boot["base"]
            if row != REFERENCE and REFERENCE in args.rows:
                c = contrast(args.output / "probs.csv", row, REFERENCE)
                summary["vs_reference"][row] = {"difference": c["paired_bootstrap"]["difference"], "per_class_f1": c["per_class_f1"],
                                                "violence_mean_f1": c["violence_mean_f1"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for row, s in summary["arms"].items():
        ci = summary["intervals"].get(row, {}).get("ci95")
        print(json.dumps({"row": row, "macro_f1": round(s["macro_f1"], 4), "ci95": [round(x, 4) for x in ci] if ci else None,
                          "per_fold": [round(x, 4) for x in s["per_fold_macro_f1"]]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
