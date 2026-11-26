#!/usr/bin/env python3
"""D-67: DINOv2-B partial fine-tuning (last 2 / last 4 blocks) vs the frozen control, under the D-65 protocol.

Writes <output>/{runs.json, probs.csv, checks.json, summary.json}. Validation only; the test split is not read.
`--smoke` runs fold 1, seed 0, 2 epochs, one LR and 2,000 validation frames, to measure throughput and memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.finetune_experiment import ARMS, CONTRASTS, LR_GRID  # noqa: E402


def _log(obj) -> None:
    print(json.dumps(obj, default=str), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("results/p1c/finetune-b-last2-last4"))
    args = ap.parse_args(argv)
    import torch

    from core.d65_claim import claims
    from core.finetune_experiment import run_fold
    from core.head_experiment import summarise_arms

    torch.set_num_threads(args.threads)
    if args.smoke:
        args.folds, args.seeds, args.output = [1], [0], args.output.with_name(args.output.name + "-smoke")
    opts = {"epochs": 2, "lr_grid": (LR_GRID[1],), "max_validation_frames": 2000} if args.smoke else {}
    args.output.mkdir(parents=True, exist_ok=True)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip())
    t0 = time.time()
    records, probs, checks = [], [], []
    for fold in args.folds:
        r, p, c = run_fold(fold, args.seeds, args.arms, device=args.device, threads=args.threads, log=_log, **opts)
        records += r
        probs += p
        checks.append(c)
        if torch.cuda.is_available():
            c["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        (args.output / "runs.json").write_text(json.dumps(records, indent=1) + "\n")  # rewritten after every fold
        (args.output / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    with (args.output / "probs.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(probs[0]))
        w.writeheader()
        w.writerows(probs)
    scored = [r for r in records if r.get("scored", True) and r["arm"] in ARMS]
    summary = {"git_revision": rev, "working_tree_dirty": dirty, "decision": "D-67", "protocol": "D-65",
               "folds": args.folds, "seeds": args.seeds, "smoke": args.smoke, "lr_grid": list(opts.get("lr_grid", LR_GRID)),
               "split": "validation only", "label_view": "inherited video-level labels (D-40), not event-only",
               "seconds": round(time.time() - t0), "arms": summarise_arms(scored, control="B_frozen") if "B_frozen" in args.arms else {}}
    registered = [c for c in CONTRASTS if set(c) <= set(args.arms)]
    if registered and not args.smoke:
        summary["registered_contrasts"] = claims(args.output / "probs.csv", registered)
    if any(r["arm"] == "B_ft2_lr0" for r in records):
        lr0 = [r for r in records if r["arm"] == "B_ft2_lr0"][0]
        ref = [r for r in records if r["arm"] == "B_frozen" and r["fold"] == lr0["fold"] and r["seed"] == lr0["seed"]]
        summary["pipeline_check_lr0"] = {"lr0_macro_f1": lr0["macro_f1"], "control_macro_f1": ref[0]["macro_f1"] if ref else None,
                                         "lr0_best_epoch": lr0["best_epoch"], "control_best_epoch": ref[0]["best_epoch"] if ref else None}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    for arm, s in summary["arms"].items():
        _log({"arm": arm, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k in ("macro_f1", "mean_delta", "folds_positive")}})
    for c in summary.get("registered_contrasts", []):
        d = c["paired_bootstrap"]["difference"]
        _log({"contrast": c["contrast"], "diff": round(d["observed"], 4), "ci95": [round(x, 4) for x in d["ci95"]],
              "p_one_sided": round(c["one_sided_p"], 4), "claim_passed": c["claim_passed"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
