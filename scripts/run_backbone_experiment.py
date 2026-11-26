#!/usr/bin/env python3
"""D-65 round 1: DINOv2 ViT-L vs ViT-B under the prospective protocol (inner-monitor epoch selection, paired claim rule).

Writes results/p1c/backbone-l-vs-b/{runs.json, probs.csv, summary.json}. Validation only; the test split is not read.
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

from core.backbone_experiment import ARMS, CONTRAST, MONITOR_PER_CLASS, run_fold  # noqa: E402

def _worker(job):
    fold, seeds, arms, epochs, threads = job
    import torch
    torch.set_num_threads(threads)
    t0 = time.time()
    out = run_fold(fold, seeds, arms, epochs=epochs)
    print(json.dumps({"fold": fold, "runs": len(out[0]), "seconds": round(time.time() - t0)}), flush=True)
    return out


def claim(probs_path: Path, candidate: str, control: str) -> dict:
    """D-66's single-contrast verdict (95% CI lower bound > 0 and gain >= +0.02) on the shared D-65 contrast report."""

    from core.confusion_groups import GATE_DELTA
    from core.d65_claim import contrast

    c = contrast(probs_path, candidate, control)
    lo, obs = c["paired_bootstrap"]["difference"]["ci95"][0], c["paired_bootstrap"]["difference"]["observed"]
    return {**c, "claim_passed": bool(lo > 0 and obs >= 0.02),
            "rule": "paired 95% CI lower bound > 0 (Holm over 1 registered contrast) and point gain >= +0.02",
            "historical_gate_threshold": GATE_DELTA}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None, help="smoke only")
    ap.add_argument("--output", type=Path, default=Path("results/p1c/backbone-l-vs-b"))
    args = ap.parse_args(argv)
    from core.head_experiment import summarise_arms

    args.output.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(_worker, [(f, args.seeds, args.arms, args.epochs, args.threads) for f in args.folds], chunksize=1)
    records = [r for rec, _ in parts for r in rec]
    probs = [r for _, p in parts for r in p]
    (args.output / "runs.json").write_text(json.dumps(records, indent=1) + "\n")
    with (args.output / "probs.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(probs[0]))
        w.writeheader()
        w.writerows(probs)
    summary = {"arms": summarise_arms(records, control="B_proj") if "B_proj" in args.arms else {}}
    if set(CONTRAST) <= set(args.arms):
        summary["registered_contrast"] = claim(args.output / "probs.csv", *CONTRAST)
    if {"B_ref", "B_proj"} <= set(args.arms):
        summary["noise_calibration_B_ref_vs_B_proj"] = claim(args.output / "probs.csv", "B_ref", "B_proj")["paired_bootstrap"]["difference"]
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    (args.output / "summary.json").write_text(json.dumps({"git_revision": rev, "decision": "D-65", "folds": args.folds, "seeds": args.seeds,
                                                          "monitor_per_class": MONITOR_PER_CLASS, "split": "validation only", **summary}, indent=1) + "\n")
    for arm, s in summary["arms"].items():
        print(arm, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k in ("macro_f1", "accuracy", "mean_delta", "folds_positive")}))
    if "registered_contrast" in summary:
        c = summary["registered_contrast"]
        print("CONTRAST", json.dumps({"diff": round(c["paired_bootstrap"]["difference"]["observed"], 4), "ci95": [round(x, 4) for x in c["paired_bootstrap"]["difference"]["ci95"]],
                                      "claim_passed": c["claim_passed"], "violence": {k: round(v, 3) for k, v in c["violence_mean_f1"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
