#!/usr/bin/env python3
"""Verify an existing B-1 feature cache against the live backbone.

Thin CLI. `scripts/build_p1c_feature_cache.py` already verifies every cache it
builds, so this exists for the cases where rebuilding would be wasteful: a
verification tolerance was revised, a cache built earlier needs re-checking
before it is trusted by a new experiment, or a cache is being adopted by a
config other than the one that built it.

Rebuilding the train cache costs a full forward pass over 391,014 frames;
verifying a sample of it costs a few hundred. That difference is the whole
reason this script is separate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.feature_cache_verify import verify_feature_cache
from models.classifiers import apply_backbone_trainable_mode
from scripts.build_p1c_feature_cache import _SPLITS, _load_backbone, _load_kwargs_for_config
from scripts.run_p1c_frame_baseline import _load_config, _make_datasets


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=_SPLITS, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64, help="match the batch size the cache was built at")
    parser.add_argument(
        "--max-ulps", type=int, default=8,
        help="allowed float16 ULPs, each step floored at --ulp-floor-magnitude (see K-26)",
    )
    parser.add_argument(
        "--ulp-floor-magnitude", type=float, default=1.0,
        help="magnitude below which the ULP step no longer shrinks",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=None, help="optional path to write the JSON report to")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _config, protocol, _run_class = _load_config(args.config)

    backbone_block = protocol["backbone"]
    trainable = backbone_block.get("trainable", "full")
    # Dispatch on backbone.family through the builder's single registry -- never a
    # hardcoded ResNet50 loader. This script verified a DINOv2 cache by loading its
    # weights into a ResNet50 and died on the state-dict mismatch, twice, because the
    # B-2 family dispatch was added to the builder alone.
    backbone = apply_backbone_trainable_mode(_load_backbone(backbone_block), trainable)
    train, validation = _make_datasets(protocol, seed=0)
    dataset = train if args.split == "train" else validation

    report = verify_feature_cache(
        dataset,
        backbone,
        args.cache_dir,
        device=torch.device(args.device),
        sample_size=args.sample_size,
        seed=args.seed,
        batch_size=args.batch_size,
        max_ulps=args.max_ulps,
        ulp_floor_magnitude=args.ulp_floor_magnitude,
        **_load_kwargs_for_config(protocol, args.split, trainable),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
