#!/usr/bin/env python3
"""Inner cross-validation of classifier arms on each official fold's 532 training videos (FINAL_PUSH_PLAN Step 1).

Every arm is trained for a length fixed in advance and scored on held-out predictions for all pool
videos (`core.inner_cv`). Test videos are never loaded. Writes `probs.csv`, `runs.json`, `summary.json`.

Pass rule for a contrast (D-65, kept): the paired source-video bootstrap 95% interval of the macro-F1
difference excludes zero AND the gain is at least +0.02.

CPU only; the pod still claims the MIG slice (Kyverno). Run from a code snapshot with `--data-root`
pointing at the checkout that holds the caches.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CACHES = {
    "dinov2b": "results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}",
    "resnet50": "results/p1c/p1c-frozen-feature-cache-ar-fold{fold}",
}
# Fixed training lengths, declared before any run: the median historical best epoch (0-indexed) plus one,
# rounded up -- attention 6.5 (results/p1c/headline-14class-test), linear 3.5 (p1c-sampling-ablation n8).
# The half/double rows are sensitivity checks, not candidates.
ARMS = {
    "R0_meanpool_logistic": ("dinov2b", "meanpool", {}),
    "R1_linear_vote_e5": ("dinov2b", "linear", {"epochs": 5}),
    "R2_attention_e8": ("dinov2b", "attention", {"epochs": 8}),
    "R3_meanpool_logistic_resnet50": ("resnet50", "meanpool", {}),
    "S_linear_vote_e3": ("dinov2b", "linear", {"epochs": 3}),
    "S_linear_vote_e10": ("dinov2b", "linear", {"epochs": 10}),
    "S_attention_e4": ("dinov2b", "attention", {"epochs": 4}),
    "S_attention_e16": ("dinov2b", "attention", {"epochs": 16}),
}
CONTRASTS = [  # (candidate, base)
    ("R1_linear_vote_e5", "R0_meanpool_logistic"),
    ("R2_attention_e8", "R0_meanpool_logistic"),
    ("R2_attention_e8", "R1_linear_vote_e5"),
    ("R0_meanpool_logistic", "R3_meanpool_logistic_resnet50"),
    ("S_attention_e4", "R2_attention_e8"),
    ("S_attention_e16", "R2_attention_e8"),
    ("S_linear_vote_e3", "R1_linear_vote_e5"),
    ("S_linear_vote_e10", "R1_linear_vote_e5"),
]


def _worker(job: tuple) -> dict:
    fold, data_root, arm_names, parts, repeats, split_seed, threads = job
    import torch

    from core import inner_cv

    torch.set_num_threads(threads)
    data_root = Path(data_root)
    pools = {"dinov2b": inner_cv.load_pool(data_root / CACHES["dinov2b"].format(fold=fold), fold)}
    main = pools["dinov2b"]
    splits = inner_cv.inner_splits(main.targets, parts=parts, repeats=repeats, seed=split_seed)
    out = {"fold": fold, "videos": len(main.video_ids), "cache_sha256": {"dinov2b": main.cache_sha256}, "arms": {}, "tables": {}}
    for name in arm_names:
        family, kind, kwargs = ARMS[name]
        started = time.perf_counter()
        if family not in pools:
            pools[family] = inner_cv.load_pool(data_root / CACHES[family].format(fold=fold), fold)
            if pools[family].video_ids != main.video_ids or not np.array_equal(pools[family].targets, main.targets):
                raise RuntimeError(f"fold {fold}: {family} pool videos differ from the dinov2b pool")
            out["cache_sha256"][family] = pools[family].cache_sha256
        pool = pools[family]
        if kind == "meanpool":
            arm = inner_cv.meanpool_logistic_arm(**kwargs)
        elif kind == "linear":
            arm = inner_cv.linear_vote_arm(**kwargs)
        elif kind == "attention":
            arm = inner_cv.attention_arm(**kwargs)
        else:
            raise ValueError(f"unknown arm kind {kind!r}")
        tables = inner_cv.run_arm(pool, arm, splits)
        scores = inner_cv.cell_scores(pool, tables)
        out["arms"][name] = {**scores, "seconds": round(time.perf_counter() - started, 1)}
        out["tables"][name] = tables
        print(json.dumps({"fold": fold, "arm": name, "macro_f1": [round(x, 4) for x in scores["macro_f1"]],
                          "seconds": out["arms"][name]["seconds"]}), flush=True)
    out["pool"] = main.light()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path.cwd(), help="checkout holding results/p1c/*feature-cache*")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--parts", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--code-revision", default="unrecorded", help="git SHA of the code snapshot being run")
    args = ap.parse_args(argv)

    from core import inner_cv
    from core import inner_cv_report as report

    args.output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    jobs = [(f, str(args.data_root), args.arms, args.parts, args.repeats, args.split_seed, args.threads) for f in args.folds]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        results = sorted(pool.map(_worker, jobs), key=lambda r: r["fold"])

    probs_path = report.write_probabilities(args.output, results, args.arms)
    summary_arms = report.arm_summaries(probs_path, results, args.arms, args.draws)
    contrasts = report.paired_contrasts(probs_path, CONTRASTS, args.arms, args.draws)

    provenance = {
        "plan": "docs/FINAL_PUSH_PLAN_2026-11-11.md Step 1", "code_revision": args.code_revision, "host": socket.gethostname(),
        "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "label_view": "inherited video-level labels (D-40), not event-only", "test_videos_read": False,
        "pool": "official fold train + validation caches (532 videos, 38 per class)",
        "config": {"folds": args.folds, "arms": {n: ARMS[n] for n in args.arms}, "parts": args.parts, "repeats": args.repeats,
                   "split_seed": args.split_seed, "sampling_seed": 0, "frames": inner_cv.FRAMES, "bootstrap_draws": args.draws,
                   "pass_rule": f"paired 95% interval excludes 0 and gain >= {report.MIN_GAIN}"},
        "cache_sha256": {str(r["fold"]): r["cache_sha256"] for r in results},
    }
    report.write_outputs(args.output, provenance, results, summary_arms, contrasts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
