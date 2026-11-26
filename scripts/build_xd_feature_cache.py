#!/usr/bin/env python3
"""DINOv2 pooled features (cls_plus_mean_patch, 1536-D, the UCF fold-cache feature) for every XD-Violence video.

32 frames per video at the centres of 32 equal segments of the DECODED frame count (K-06), each decoded BGR->RGB,
resized to 320x240 (INTER_AREA, the UCF frame size), then the registered 224x224 bilinear preprocessing.
Output: results/xd/xd-dinov2-pooled32/{features.float16.npy [N,32,1536], rows.csv, failures.csv, manifest.json}.
"""

from __future__ import annotations

import argparse
import csv
import functools
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.combined_features import FRAME_SIZE, decode_uniform_frames, extract_cache, select_rows  # noqa: E402
from core.p1c_extraction_io import finalize_streamed_array, write_manifest  # noqa: E402
from core.p1c_readiness import sha256_file  # noqa: E402
from scripts.build_combined_patch_cache import HEADLINE_CONFIG, _load_backbone_and_preprocessing  # noqa: E402
from scripts.build_p1c_patch_cache import _preprocess_batch  # noqa: E402

FRAMES = 32
DIM = 1536


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("results/xd/manifest.csv"))
    ap.add_argument("--output-root", type=Path, default=Path("results/xd/xd-dinov2-pooled32"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-videos", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    with args.manifest.open(newline="", encoding="utf-8") as fh:
        rows = select_rows(list(csv.DictReader(fh)), args.limit)
    print(f"{len(rows)} videos x {FRAMES} frames", flush=True)
    device = torch.device(args.device)
    backbone, preprocessing, weight_sha = _load_backbone_and_preprocessing(device)

    def forward(frames: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            pooled = backbone.pooled_features(_preprocess_batch(frames, preprocessing).to(device))
            return pooled.to(torch.float16).cpu().numpy()

    result = extract_cache(rows, args.output_root, forward, token_shape=(DIM,), frames=FRAMES,
                           batch_videos=args.batch_videos, num_workers=args.num_workers,
                           decoder=functools.partial(decode_uniform_frames, k=FRAMES))
    shape, features_sha = finalize_streamed_array(args.output_root, result["memmap"])
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    write_manifest(args.output_root, {
        "feature": "cls_plus_mean_patch (DINOv2ViTB14Reg.pooled_features, registers excluded)", "shape": shape, "dtype": "float16",
        "features_sha256": features_sha, "xd_manifest_path": str(args.manifest), "xd_manifest_sha256": sha256_file(args.manifest),
        "git_revision": rev, "seed": 0, "limit": args.limit, "frames_per_video": FRAMES,
        "frame_position_rule": "centres of 32 equal segments over the DECODED frame count (K-06)",
        "resize_rule": f"BGR->RGB, cv2.resize to {FRAME_SIZE[0]}x{FRAME_SIZE[1]} INTER_AREA, then registered 224x224 bilinear preprocessing",
        "backbone": "timm_dinov2_vitb14_reg4", "backbone_weight_sha256": weight_sha,
        "preprocessing_from": str(HEADLINE_CONFIG.relative_to(REPO_ROOT)),
        "n_videos": len(rows), "n_ok": result["n_ok"], "n_failed": result["n_failed"], "device": str(device),
        "source": "HF mirror jherng/xd-violence rev 8eba22ca, verified results/xd/download-verify.json",
    })
    print(f"wrote {args.output_root}: {result['n_ok']} ok, {result['n_failed']} failed, shape {shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
