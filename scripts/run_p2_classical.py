#!/usr/bin/env python3
"""RESULT 1, classical stage: P2A on real UCF frames + P2C downstream through the frozen D-53 classifier (D-59).

For the 108 AR validation videos x 32 evenly spaced native 320x240 frames: every registered degradation x every
registered classical restoration (configs/p2a_conventions.yaml, 10 x 5 = 50 conditions) -> PSNR/SSIM against the clean
frame (float32, before any quantization) -> rounded to uint8 at the classifier boundary -> the registered DINOv2
preprocessing -> pooled features. Classifier: the D-53 step_permuted head retrained per fold x seed on the fold-cache
training features (best epoch on the fold's full validation set), then applied to each condition's 32-frame sequences.
Validation videos only; the test split is untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p2_downstream import classify, clean_feature_check, condition_table  # noqa: E402
from core.p2_pipeline import conditions, decode_rgb_float, process_frames, sample_key, summarise  # noqa: E402
from core.p2a_conventions import load_frozen_p2a_conventions  # noqa: E402
from core.subset7 import FAMILY  # noqa: E402
from core.validation_frames import CLOSURE, select_validation_frames  # noqa: E402

CONFIG = REPO_ROOT / "configs/p2a_conventions.yaml"
DIM = 1536
_W: dict = {}


def _init() -> None:
    import cv2
    cv2.setNumThreads(1)
    _W["conv"] = load_frozen_p2a_conventions(CONFIG)
    _W["conds"] = conditions(_W["conv"])


def _work(task):
    vi, ci, jpegs, frame_idx, filepath, digest = task
    clean = [decode_rgb_float(j) for j in jpegs]
    keys = [sample_key(filepath, f, digest, 0) for f in frame_idx]
    u8, ps, ss = process_frames(clean, keys, _W["conds"][ci], _W["conv"])
    return vi, ci, u8, ps, ss


def extract(args, conds, videos, order, out: Path) -> dict:
    import torch
    from scripts.build_combined_patch_cache import _load_backbone_and_preprocessing
    from scripts.build_p1c_patch_cache import _preprocess_batch

    device = torch.device(args.device)
    backbone, pre, weight_sha = _load_backbone_and_preprocessing(device)
    feats = np.lib.format.open_memmap(out / "features.float16.npy", mode="w+", dtype=np.float16,
                                      shape=(len(conds), len(order), args.frames, DIM))
    jpegs = {v: [videos[v]["dataset"].raw_jpeg(i) for i in videos[v]["rows"]] for v in order}
    tasks = ((vi, c.index, jpegs[v], videos[v]["frames"], videos[v]["filepath"], videos[v]["file_digest"])
             for vi, v in enumerate(order) for c in conds)
    quality, t0 = [], time.time()
    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one thread per worker; spawned children inherit
        os.environ[var] = "1"
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        for n, (vi, ci, u8, ps, ss) in enumerate(pool.imap(_work, tasks, chunksize=1), 1):
            with torch.no_grad():
                f = backbone.pooled_features(_preprocess_batch(u8, pre).to(device))
            feats[ci, vi] = f.to(torch.float16).cpu().numpy()
            finite = [p for p in ps if math.isfinite(p)]
            quality.append({"condition": conds[ci].name, "video": order[vi],
                            "psnr_mean": float(np.mean(finite)) if finite else float("inf"),
                            "psnr_infinite_frames": len(ps) - len(finite), "ssim_mean": float(np.mean(ss))})
            if n % (10 * len(conds)) == 0:
                print(json.dumps({"videos_done": n // len(conds), "of": len(order), "elapsed_s": round(time.time() - t0)}), flush=True)
    feats.flush()
    with (out / "quality.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(quality[0]))
        w.writeheader()
        w.writerows(quality)
    return {"backbone_weight_sha256": weight_sha, "extract_seconds": round(time.time() - t0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-videos", type=int, default=None, help="smoke only")
    ap.add_argument("--epochs", type=int, default=None, help="smoke only")
    ap.add_argument("--output", type=Path, default=Path("results/p2/p2-classical"))
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    conv = load_frozen_p2a_conventions(CONFIG)
    conds = conditions(conv)
    classes = sorted(FAMILY)
    videos = select_validation_frames(REPO_ROOT, args.folds, classes, args.frames)
    order = sorted(videos)[: args.limit_videos]
    info = extract(args, conds, videos, order, args.output)
    feats = np.load(args.output / "features.float16.npy", mmap_mode="r")
    check = clean_feature_check(REPO_ROOT, order, videos, feats, args.folds)
    print(json.dumps({"clean_feature_check": check}), flush=True)
    runs = classify(conds, videos, order, feats, classes, folds=args.folds, seeds=args.seeds, threads=args.threads, epochs=args.epochs)
    quality = list(csv.DictReader((args.output / "quality.csv").open()))
    table = condition_table(conds, quality, runs, args.folds)
    summary = summarise(table, conds[0].name)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (args.output / "classification_runs.json").write_text(json.dumps(runs, indent=1) + "\n")
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    (args.output / "config.json").write_text(json.dumps({
        "git_revision": rev, "conventions": str(CONFIG.relative_to(REPO_ROOT)), "conventions_sha256": conv.source_sha256,
        "conditions": len(conds), "videos": len(order), "frames_per_video": args.frames, "folds": args.folds, "seeds": args.seeds,
        "frame_rule": "sample_rows_per_video(seed=None) over each video's validation manifest rows; native 320x240 JPEG from the D-34 blob",
        "noise_seed": "data.degradations SampleKey(relative_path=<filepath>/frame<index>, file_digest, global_seed=0)",
        "quality_metrics": "PSNR/SSIM on float32 [0,1] after restoration (clipped), before quantization; PSNR inf for identity/identity",
        "classifier_input": "restored frames rounded to uint8, then the registered 224x224 bilinear preprocessing (preprocess_crop)",
        "classifier": "D-53 step_permuted head retrained per fold x seed on fold-cache train features (D-59)",
        "frame_blob_sha256": json.loads((REPO_ROOT / CLOSURE).read_text())["blob_sha256"], "clean_feature_check": check,
        "limit_videos": args.limit_videos, "epochs_override": args.epochs, "test_split": "untouched", **info}, indent=1) + "\n")
    for name, r in summary.items():
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("psnr_db", "ssim", "macro_f1", "macro_f1_delta_vs_clean", "macro_f1_delta_vs_unrestored")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
