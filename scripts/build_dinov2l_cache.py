#!/usr/bin/env python3
"""D-65: frozen DINOv2 ViT-L/14-reg4 pooled features (2048-d) for exactly the frames the B-vs-L experiment uses.

Frames = union over folds 1-4 of the 8-frame seed-0 training draws and all cached validation frames, read from the
D-34 blob and put through the registered preprocessing (224 bilinear, ImageNet normalisation). `--check-b N` first
re-extracts N of those frames with the registered ViT-B through this same path and compares them with the B fold
cache (a check on frame mapping and preprocessing, independent of L). `--limit-frames` is the calibration mode:
it reports throughput, GPU and host memory and projected size/time for the full set, and writes nothing else.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.backbone_cache import align, union_keys  # noqa: E402
from core.validation_frames import CLOSURE  # noqa: E402


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--check-b", type=int, default=128)
    ap.add_argument("--limit-frames", type=int, default=None, help="calibration only: process this many frames, write a report")
    ap.add_argument("--output", type=Path, default=Path("results/p1c/p1c-dinov2l-cache"))
    args = ap.parse_args(argv)
    import torch

    from core.frame_batches import FrameBatcher
    from models.dinov2_large_backbone import WEIGHTS_SHA256, DINOv2ViTL14Reg
    from scripts.build_combined_patch_cache import _load_backbone_and_preprocessing
    from scripts.run_subset7_search import load_fold_features

    t_start = time.time()
    fold_rows, fold_b = {}, {}
    for f in args.folds:
        fd = load_fold_features(f, ("train", "validation"))
        fold_rows[f] = {s: fd[s][1] for s in ("train", "validation")}
        fold_b[f] = {s: fd[s][0] for s in ("train", "validation")}
        del fd
    keys = union_keys(rows for f in args.folds for rows in fold_rows[f].values())
    batch_tensor = FrameBatcher(args.threads)
    blob = batch_tensor.blob_rows(keys)
    print(json.dumps({"frames_needed": len(keys), "videos": len({k[0] for k in keys}), "load_s": round(time.time() - t_start)}), flush=True)
    device = torch.device(args.device)

    def extract(model, rows: np.ndarray, dim: int, report_every: int = 0) -> np.ndarray:
        out = np.empty((len(rows), dim), dtype=np.float16)
        t0 = time.time()
        for s in range(0, len(rows), args.batch):
            x = batch_tensor(rows[s:s + args.batch]).to(device)
            with torch.no_grad():
                out[s:s + len(x)] = model.pooled_features(x).to(torch.float16).cpu().numpy()
            if report_every and (s // args.batch) % report_every == 0:
                done = s + len(x)
                print(json.dumps({"frames_done": done, "of": len(rows), "fps": round(done / (time.time() - t0), 1)}), flush=True)
        return out

    # 1. B through this path vs the B fold cache, on frames spread over the key list
    check = {}
    if args.check_b:
        b, _pre_b, _sha = _load_backbone_and_preprocessing(device)
        pick = np.linspace(0, len(keys) - 1, args.check_b).astype(int)
        fb = extract(b, blob[pick], 1536).astype(np.float32)
        ref_keys, ref = [], []
        for f in args.folds:
            for s in ("train", "validation"):
                ref_keys += [(str(r["source_video_id"]), int(r["source_frame_index"])) for r in fold_rows[f][s]]
                ref.append(fold_b[f][s])
        pos = {k: i for i, k in enumerate(ref_keys)}
        refm = np.concatenate(ref)[[pos[keys[i]] for i in pick]].astype(np.float32)
        cos = (fb * refm).sum(1) / (np.linalg.norm(fb, axis=1) * np.linalg.norm(refm, axis=1))
        check = {"frames": int(len(pick)), "min_cosine": float(cos.min()), "max_abs_diff": float(np.abs(fb - refm).max())}
        print(json.dumps({"b_path_check": check}), flush=True)
        if check["min_cosine"] < 0.9999:
            raise RuntimeError(f"B features through this path do not match the B fold cache: {check}")
        del b
        torch.cuda.empty_cache()
    del fold_b

    # 2. L
    model = DINOv2ViTL14Reg.from_official_checkpoint(REPO_ROOT).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    n = len(keys) if args.limit_frames is None else min(args.limit_frames, len(keys))
    t0 = time.time()
    feats = extract(model, blob[:n], 2048, report_every=50)
    secs = time.time() - t0
    stats = {"frames": n, "seconds": round(secs, 1), "frames_per_s": round(n / secs, 1),
             "gpu_peak_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2), "host_peak_rss_gib": round(_rss_gib(), 2),
             "projected_full_minutes": round(len(keys) / (n / secs) / 60, 1), "projected_cache_gib": round(len(keys) * 2048 * 2 / 2**30, 2)}
    print(json.dumps({"calibration" if args.limit_frames else "extraction": stats}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    if args.limit_frames is not None:
        (args.output / "calibration.json").write_text(json.dumps({"git_revision": rev, "b_path_check": check, **stats, "frames_needed": len(keys)}, indent=1) + "\n")
        return 0
    np.save(args.output / "features.float16.npy", feats)
    with (args.output / "rows.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["source_video_id", "source_frame_index", "blob_row"])
        w.writerows((k[0], k[1], int(r)) for k, r in zip(keys, blob))
    for f in args.folds:  # every fold's rows must realign without a gap
        for s in ("train", "validation"):
            align(feats, keys, fold_rows[f][s])
    (args.output / "manifest.json").write_text(json.dumps({
        "git_revision": rev, "backbone": "DINOv2 ViT-L/14-reg4 (timm vit_large_patch14_reg4_dinov2, official weights)",
        "weights_sha256": WEIGHTS_SHA256, "feature": "CLS ++ mean(patch), registers excluded, 2048-d float16",
        "frames": "union over folds of 8-frame seed-0 training draws and all cached validation frames", "shape": list(feats.shape),
        "features_sha256": hashlib.sha256(feats.tobytes()).hexdigest(), "folds": args.folds,
        "frame_blob_sha256": json.loads((REPO_ROOT / CLOSURE).read_text())["blob_sha256"], "b_path_check": check, **stats}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
