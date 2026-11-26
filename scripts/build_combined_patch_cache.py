#!/usr/bin/env python3
"""DINOv2 token cache for every SHAR-Combined manifest row (GPU, pod-intended; features only, test split included).

8 frames/video at the centres of 8 equal segments, every frame resized to 320x240 (INTER_AREA) first,
then the headline backbone and preprocessing (`scripts/build_p1c_patch_cache.py`) unchanged.

    ./.venv/bin/python scripts/build_combined_patch_cache.py --device cuda:0 --num-workers 6 --batch-videos 4
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.combined_features import FRAME_SIZE, FRAMES, cls_plus_patches, extract_cache, select_rows
from core.p1c_extraction_io import finalize_streamed_array, write_manifest
from core.p1c_readiness import sha256_file
from data.p1c_loader import FramePreprocessing
from scripts.build_p1c_feature_cache import _local_dinov2
from scripts.build_p1c_patch_cache import _preprocess_batch

HEADLINE_CONFIG = REPO_ROOT / "configs/experiments/p1c/P1C-VIDEO-DINOV2-B2-AR-FOLD1.yaml"
SEED = 0  # nothing here is random; recorded so the manifest carries the four reproducibility fields


def _parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("Datasets/SHAR-Combined/manifest.csv"))
    ap.add_argument("--output-root", type=Path, default=Path("results/p1c/combined-patch-cache"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-videos", type=int, default=4, help="videos per forward (x8 frames)")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N rows, round-robin across sources")
    return ap.parse_args(argv)


def _load_backbone_and_preprocessing(device: torch.device):
    protocol = yaml.safe_load(HEADLINE_CONFIG.read_text(encoding="utf-8"))["fixed_protocol"]
    weights = protocol["backbone"]["weights"]
    if sha256_file(REPO_ROOT / weights["path"]) != weights["sha256"]:
        raise RuntimeError("registered DINOv2 weight file does not match its registered sha256")
    backbone = _local_dinov2(weights).to(device).eval()
    return backbone, FramePreprocessing.from_config(protocol["preprocessing"]), weights["sha256"]


def main(argv=None) -> int:
    args = _parse_args(argv)
    with args.manifest.open(newline="", encoding="utf-8") as fh:
        rows = select_rows(list(csv.DictReader(fh)), args.limit)
    print(f"{len(rows)} videos x {FRAMES} frames", flush=True)
    device = torch.device(args.device)
    backbone, preprocessing, weight_sha = _load_backbone_and_preprocessing(device)

    def forward(frames: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tokens = backbone(_preprocess_batch(frames, preprocessing).to(device))
            return cls_plus_patches(tokens, backbone.num_prefix_tokens).to(torch.float16).cpu().numpy()

    result = extract_cache(rows, args.output_root, forward, batch_videos=args.batch_videos, num_workers=args.num_workers)
    shape, features_sha = finalize_streamed_array(args.output_root, result["memmap"])
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    write_manifest(args.output_root, {
        "feature": "cls_plus_16x16_patch_tokens_registers_excluded", "token_axis": "[video, frame, token, dim]; token 0 = CLS",
        "shape": shape, "dtype": "float16", "features_sha256": features_sha,
        "combined_manifest_path": str(args.manifest), "combined_manifest_sha256": sha256_file(args.manifest),
        "git_revision": rev, "seed": SEED, "limit": args.limit,
        "frames_per_video": FRAMES,
        "frame_position_rule": "centres of 8 equal segments, floor((i+0.5)*n/8) over the DECODED frame count (K-06); videos shorter than 8 frames repeat frames",
        "resize_rule": f"decoded BGR->RGB (K-05), cv2.resize to {FRAME_SIZE[0]}x{FRAME_SIZE[1]} INTER_AREA for every source, then the registered 224x224 bilinear preprocessing (core.p1c_roi.preprocess_crop); no JPEG round trip",
        "backbone": "timm_dinov2_vitb14_reg4", "backbone_weight_sha256": weight_sha,
        "preprocessing_from": str(HEADLINE_CONFIG.relative_to(REPO_ROOT)),
        "n_videos": len(rows), "n_ok": result["n_ok"], "n_failed": result["n_failed"],
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "device": str(device),
        "batch_videos": args.batch_videos,
    })
    print(f"wrote {args.output_root}: {result['n_ok']} ok, {result['n_failed']} failed, shape {shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
