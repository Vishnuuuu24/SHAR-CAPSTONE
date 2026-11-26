#!/usr/bin/env python3
"""Build the XD-Violence extra-training feature cache on CPU (<= 4 threads). Training-only; see core/p1c_xd_extra.py."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch  # noqa: E402

torch.set_num_threads(4)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_readiness import sha256_file  # noqa: E402
from core.p1c_xd_extra import (  # noqa: E402
    CANDIDATE_STRIDE, FRAMES_PER_VIDEO, JPEG_QUALITY, assemble, discover, file_list_sha256, process_chunk,
)
from data.p1c_loader import FramePreprocessing  # noqa: E402
from scripts.build_p1c_feature_cache import _load_backbone  # noqa: E402
from scripts.run_p1c_frame_baseline import _load_config  # noqa: E402

CONFIG = "configs/experiments/p1c/P1C-VIDEO-DINOV2-B2-AR-FOLD1.yaml"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xd-root", type=Path, default=Path("Datasets/XD-Violence/train"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/p1c/p1c-xd-extra-cache"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-videos", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N videos of the sorted list")
    args = ap.parse_args(argv)

    videos = discover(args.xd_root)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"no mp4 files under {args.xd_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts = args.output_dir / "parts"
    parts.mkdir(exist_ok=True)

    _cfg, protocol, _cls = _load_config(Path(CONFIG))
    preprocessing = FramePreprocessing.from_config(protocol["preprocessing"])
    backbone = _load_backbone(protocol["backbone"]).eval()

    chunks = [videos[i : i + args.chunk_videos] for i in range(0, len(videos), args.chunk_videos)]
    for i, chunk in enumerate(chunks):
        if (parts / f"part-{i:05d}.json").exists():
            continue
        process_chunk(chunk, parts, i, backbone, preprocessing, args.seed)
        print(f"chunk {i + 1}/{len(chunks)} done", flush=True)

    rows, failed = assemble(parts, args.output_dir)
    shutil.rmtree(parts)
    per_class = Counter()
    for vid in {r["video_id"]: r["class_name"] for r in rows}.items():
        per_class[vid[1]] += 1
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    manifest = {
        "purpose": "XD-Violence extra TRAINING videos only; never evaluation; labels mapped by name to UCF classes",
        "device": "cpu", "torch_version": torch.__version__, "torch_num_threads": torch.get_num_threads(),
        "platform": platform.platform(), "git_revision": rev,
        "features_sha256": sha256_file(args.output_dir / "features.npy"),
        "rows_sha256": sha256_file(args.output_dir / "rows.jsonl"),
        "source_file_list_sha256": file_list_sha256(videos, args.xd_root), "source_files": len(videos),
        "preprocessing": protocol["preprocessing"], "config": CONFIG,
        "backbone_weight_sha256": protocol["backbone"]["weights"]["sha256"],
        "pooling": "models.dinov2_backbone.pool_cls_plus_mean_patch (CLS ++ mean patch, registers excluded), fp32 compute, fp16 stored",
        "seed": args.seed, "seed_rule": "per-video default_rng(first 4 bytes of sha256('<seed>:<file name>'))",
        "sampling_rule": f"candidates = every {CANDIDATE_STRIDE}th decoded frame; {FRAMES_PER_VIDEO} distinct seeded picks, "
                         "ascending; repeat-cycle if fewer; frames JPEG round-tripped at q%d like the UCF extraction" % JPEG_QUALITY,
        "videos_per_class": dict(sorted(per_class.items())), "rows": len(rows), "failed_videos": failed,
        "limit": args.limit,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "videos_per_class", "failed_videos")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
