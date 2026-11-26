#!/usr/bin/env python3
"""Re-test of the near-miss classifier candidates under inner cross-validation (FINAL_PUSH_PLAN Step 2).

Every arm is the reference classifier of Step 1 (one vector per video, standardised, logistic regression
with nested choice of C) given a different feature matrix or training set, so each contrast changes one
factor. Scored on held-out predictions for each fold's 532 training videos; test videos are never
scored and their labels are never read.

Registered candidates (Holm over the five; one-sided p <= 0.025/(m - rank) AND gain >= +0.02):
  L_study        vs B_study    DINOv2-L instead of DINOv2-B, on the common study frames
  B+S_study      vs B_study    DINOv2-B ++ SigLIP-base fusion, study frames
  B+VJ           vs B_all      DINOv2-B ++ V-JEPA 2 clip features
  glance_train   vs B_all      training vectors from GlanceVAD click windows (+/- 2 s) and official intervals
  extra_balanced vs B_all      300 extra UCF videos as training rows, inverse-frequency class weights
Everything else is reported for context only.

`--stack NAME...` instead evaluates the fusion of the named surviving feature blocks (second round).
CPU only; the pod still claims the MIG slice (Kyverno).
"""

from __future__ import annotations

import argparse
import csv
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

from core.inner_cv_features import BLOCKS, DINOV2L, P1C, SIGLIP, VJEPA2, feature_blocks  # noqa: E402

POOL_CACHE = P1C + "/p1c-dinov2-feature-cache-ar-fold{fold}"
EXTRA_UCF = P1C + "/p1c-extra-ucf-cache"
GLANCE = P1C + "/glance-annotations/ucf_glance_annotations.csv"
VIDEOS_CSV = P1C + "/p1c-video-frames/videos.csv"
ANNOTATION = "Datasets/UCF Crime Official/annotations/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt"
GLANCE_WINDOW_S = 2.0

CANDIDATES = [  # (candidate, base): the registered Holm family
    ("L_study", "B_study"), ("B+S_study", "B_study"), ("B+VJ", "B_all"), ("glance_train", "B_all"), ("extra_balanced", "B_all"),
]
CONTEXT = [
    ("S_study", "B_study"), ("B+L_study", "B_study"), ("VJ", "B_all"), ("balanced_only", "B_all"), ("extra_only", "B_all"),
    ("B_study", "B_all"),
]


def _worker(job: tuple) -> dict:
    fold, data_root, siglip_pooled, parts, repeats, split_seed, threads, stack = job
    import torch

    from core import inner_cv
    from core import inner_cv_features as feats
    from core.p1c_background_mask import load_intervals
    from core.p1c_oracle_interval import glance_spans, load_glances

    torch.set_num_threads(threads)
    data_root = Path(data_root)
    pool = inner_cv.load_pool(data_root / POOL_CACHE.format(fold=fold), fold)
    splits = inner_cv.inner_splits(pool.targets, parts=parts, repeats=repeats, seed=split_seed)
    info: dict = {}

    def fused(*names: str):
        matrix = np.concatenate([blocks[n] for n in names], axis=1)
        return lambda _pool: matrix

    if stack:
        blocks = feature_blocks(fold, data_root, pool, siglip_pooled, set(stack) | {"B_all"})
        arms = {"B_all": inner_cv.meanpool_logistic_arm(), "stack:" + "+".join(stack): inner_cv.meanpool_logistic_arm(matrix=fused(*stack))}
    else:
        blocks = feature_blocks(fold, data_root, pool, siglip_pooled, set(BLOCKS))
        with (data_root / VIDEOS_CSV).open(newline="", encoding="utf-8") as handle:
            fps = {r["file_name"].rsplit(".", 1)[0]: float(r["reported_fps"]) for r in csv.DictReader(handle)}
        glances = {v: c for v, c in load_glances(data_root / GLANCE).items() if v in fps}
        spans = {**glance_spans(glances, fps, GLANCE_WINDOW_S), **load_intervals(data_root / ANNOTATION)}
        glance_matrix, info["glance_videos_using_a_span"] = feats.span_matrix(pool, spans)
        extra_x, extra_y, extra_videos = feats.extra_video_matrix(data_root / EXTRA_UCF, pool.class_names)
        if set(extra_videos) & set(pool.video_ids):
            raise RuntimeError("extra UCF videos overlap the pool")
        info["extra_videos"] = len(extra_videos)
        extra = lambda _pool: (extra_x, extra_y)  # noqa: E731
        arms = {
            "B_all": inner_cv.meanpool_logistic_arm(),
            "B_study": inner_cv.meanpool_logistic_arm(matrix=fused("B_study")),
            "L_study": inner_cv.meanpool_logistic_arm(matrix=fused("L_study")),
            "S_study": inner_cv.meanpool_logistic_arm(matrix=fused("S_study")),
            "B+S_study": inner_cv.meanpool_logistic_arm(matrix=fused("B_study", "S_study")),
            "B+L_study": inner_cv.meanpool_logistic_arm(matrix=fused("B_study", "L_study")),
            "VJ": inner_cv.meanpool_logistic_arm(matrix=fused("VJ")),
            "B+VJ": inner_cv.meanpool_logistic_arm(matrix=fused("B_all", "VJ")),
            "glance_train": inner_cv.meanpool_logistic_arm(train_matrix=lambda _pool: glance_matrix),
            "extra_only": inner_cv.meanpool_logistic_arm(extra=extra),
            "balanced_only": inner_cv.meanpool_logistic_arm(class_weight="balanced"),
            "extra_balanced": inner_cv.meanpool_logistic_arm(extra=extra, class_weight="balanced"),
        }
    out = {"fold": fold, "videos": len(pool.video_ids), "cache_sha256": pool.cache_sha256, "info": info, "arms": {}, "tables": {},
           "block_dims": {n: int(b.shape[1]) for n, b in blocks.items()}}
    for name, arm in arms.items():
        started = time.perf_counter()
        tables = inner_cv.run_arm(pool, arm, splits)
        out["arms"][name] = {**inner_cv.cell_scores(pool, tables), "seconds": round(time.perf_counter() - started, 1)}
        out["tables"][name] = tables
        print(json.dumps({"fold": fold, "arm": name, "macro_f1": [round(x, 4) for x in out["arms"][name]["macro_f1"]],
                          "seconds": out["arms"][name]["seconds"]}), flush=True)
    out["pool"] = pool.light()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--parts", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--stack", nargs="+", default=None, choices=list(BLOCKS), help="second round: fuse these surviving blocks")
    ap.add_argument("--code-revision", default="unrecorded")
    args = ap.parse_args(argv)

    from core import inner_cv
    from core import inner_cv_features as feats
    from core import inner_cv_report as report

    args.output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    siglip_pooled = None
    if args.stack is None or "S_study" in args.stack:
        siglip_pooled = args.output / "siglip_pooled.float16.npy"
        if not siglip_pooled.exists():  # 25 GiB of patch tokens -> 254 MiB of [CLS ++ mean(patch)], once
            source = args.data_root / SIGLIP
            pooled = feats.pooled_tokens(np.load(source / "cls.float16.npy", mmap_mode="r"), np.load(source / "features.float16.npy", mmap_mode="r"))
            np.save(siglip_pooled, pooled)
            print(json.dumps({"siglip_pooled": list(pooled.shape), "seconds": round((datetime.now(timezone.utc) - started).total_seconds())}), flush=True)

    jobs = [(f, str(args.data_root), siglip_pooled, args.parts, args.repeats, args.split_seed, args.threads, args.stack) for f in args.folds]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        results = sorted(pool.map(_worker, jobs), key=lambda r: r["fold"])

    arm_names = list(results[0]["arms"])
    probs_path = report.write_probabilities(args.output, results, arm_names)
    arms = report.arm_summaries(probs_path, results, arm_names, args.draws)
    pairs = [(n, "B_all") for n in arm_names if n.startswith("stack:")] if args.stack else CANDIDATES + CONTEXT
    contrasts = report.paired_contrasts(probs_path, pairs, arm_names, args.draws)
    holm = [] if args.stack else report.holm_family(contrasts, CANDIDATES)
    for row in holm:
        print(json.dumps({"holm": row}), flush=True)
    provenance = {
        "plan": "docs/FINAL_PUSH_PLAN_2026-11-11.md Step 2", "code_revision": args.code_revision, "host": socket.gethostname(),
        "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "label_view": "inherited video-level labels (D-40), not event-only", "test_videos_scored": False,
        "pool": "official fold train + validation caches (532 videos, 38 per class)",
        "config": {"folds": args.folds, "parts": args.parts, "repeats": args.repeats, "split_seed": args.split_seed, "sampling_seed": 0,
                   "bootstrap_draws": args.draws, "stack": args.stack, "glance_window_s": GLANCE_WINDOW_S,
                   "classifier": "StandardScaler + LogisticRegression, C from (0.003, 0.01, 0.03, 0.1) by 4-part nested CV on the training side",
                   "family_rule": f"Holm, one-sided p <= {report.ALPHA_ONE_SIDED}/(m - rank) and gain >= {report.MIN_GAIN}"},
        "cache_sha256": {str(r["fold"]): r["cache_sha256"] for r in results},
        "block_dims": results[0]["block_dims"], "info": {str(r["fold"]): r["info"] for r in results},
        "feature_sources": {"L_study": DINOV2L, "S_study": SIGLIP, "VJ": VJEPA2, "extra": EXTRA_UCF, "glance": GLANCE},
    }
    report.write_outputs(args.output, provenance, results, arms, contrasts, extra={"holm_family": holm})
    return 0


if __name__ == "__main__":
    sys.exit(main())
