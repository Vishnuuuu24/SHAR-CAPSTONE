#!/usr/bin/env python3
"""Targeted DINOv2 extraction for the A2/A3 representation comparison.

For ONLY the annotated Shooting videos (21 videos, ~14,400 frames across the
four fold caches, deduplicated), this writes two artifacts the pooled caches
cannot provide:

  patch224/   raw patch tokens at the registered 224x224 -- [N, 256, 768] fp16
  pooled448/  CLS + mean-patch at 448x448                  -- [N, 1536]     fp16

This is deliberately NOT a cache rebuild. Patch tokens for a full fold would be
~154 GB; for these frames they are ~5.7 GB. Each artifact carries its own
rows.csv aligned 1:1 with the array, the source-video provenance every other
artifact in this project carries, and a manifest naming the backbone digest,
preprocessing and interval-annotation source, so a later loader can refuse a
mismatch the same way `core.feature_cache` does.

Same backbone weights, same normalisation, same bilinear resize as the
registered cache. The 448 pass changes exactly one thing -- resolution -- so
the resolution ablation stays single-factor. 448 is 32 x 14, so the patch grid
is exactly 32x32 = 1024 tokens with no cropping.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_background_mask import load_intervals
from core.p1c_extraction_io import finalize_streamed_array, select_rows_for_videos, write_manifest
from data.p1c_loader import FramePreprocessing
from scripts.build_p1c_feature_cache import _load_backbone
from scripts.diagnose_p1c_event_separability import ANNOTATION
from scripts.run_p1c_frame_baseline import _load_config, _make_datasets

CONFIGS = [f"configs/experiments/p1c/P1C-VIDEO-DINOV2-B2-AR-FOLD{f}.yaml" for f in (1, 2, 3, 4)]


def _check_against_cache(patches: np.ndarray, rows: list[dict]) -> dict:
    """Compare mean(patch tokens) to the registered pooled cache for the same frames."""
    lookup: dict[tuple[str, int], tuple[Path, int]] = {}
    for fold in (1, 2, 3, 4):
        root = Path(f"results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}/train")
        if not (root / "rows.csv").exists():
            continue
        for i, r in enumerate(csv.DictReader(open(root / "rows.csv"))):
            lookup.setdefault((r["source_video_id"], int(r["source_frame_index"])), (root, i))
    arrays: dict[Path, np.ndarray] = {}
    worst, n = 0.0, 0
    for i, r in enumerate(rows):
        hit = lookup.get((r["source_video_id"], int(r["source_frame_index"])))
        if hit is None:
            continue
        root, j = hit
        if root not in arrays:
            arrays[root] = np.load(root / "features.float16.npy", mmap_mode="r")
        cached = np.asarray(arrays[root][j][768:], dtype=np.float32)
        worst = max(worst, float(np.abs(patches[i].astype(np.float32).mean(0) - cached).max()))
        n += 1
    return {"max_abs_delta": worst, "rows_checked": n}


def _finalize(out_dir: Path, memmap, rows: list[dict], manifest: dict) -> None:
    """Flush a streamed array, write provenance beside it, and digest the file."""
    shape, digest = finalize_streamed_array(out_dir, memmap)
    with open(out_dir / "rows.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_manifest(out_dir, {
        **manifest, "row_count": len(rows), "shape": shape, "dtype": "float16", "cache_sha256": digest,
    })
    print(f"wrote {out_dir}: shape {shape}, sha256 {digest[:16]}", flush=True)


def main(argv=None) -> int:
    from torch.utils.data import DataLoader, Subset

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="Shooting")
    parser.add_argument("--output-root", type=Path, default=Path("results/p1c/p1c-shooting-tokens"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--extract-closure-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug: cap frames per config")
    parser.add_argument("--skip", choices=["patch224", "pooled448"], action="append", default=[],
                        help="skip an artifact that already exists (resume after a partial run)")
    args = parser.parse_args(argv)
    do_patch, do_448 = "patch224" not in args.skip, "pooled448" not in args.skip

    intervals = load_intervals(ANNOTATION)
    device = torch.device(args.device)

    # Pass 1 -- plan from metadata only. No JPEG is decoded here, so the total
    # row count is known before any array is allocated.
    plan, seen, weight_sha, class_names = [], set(), None, None
    for config in CONFIGS:
        _c, protocol, _r = _load_config(Path(config))
        if args.extract_closure_path is not None:
            protocol["data"]["video_frames"]["extract_closure_path"] = str(args.extract_closure_path)
        weight_sha = protocol["backbone"]["weights"]["sha256"]; class_names = protocol["model"]["class_names"]
        train = _make_datasets(protocol, seed=0)[0]
        selected = [s for s in select_rows_for_videos(train, set(intervals), seen) if class_names[s[3]] == args.label]
        if args.limit:
            selected = selected[: args.limit]
        if not selected:
            continue
        pre448 = {**protocol["preprocessing"], "resize_height": 448, "resize_width": 448}
        train448 = _make_datasets({**protocol, "preprocessing": pre448}, seed=0)[0] if do_448 else None
        plan.append((config, train, train448, selected, protocol["backbone"]))
        seen |= {s[1] for s in selected}
        print(f"{config}: {len(selected)} frames from {len({s[1] for s in selected})} new videos", flush=True)
    total = sum(len(p[3]) for p in plan)
    print(f"plan: {total} frames, {len(seen)} videos", flush=True)

    # Allocate once, on disk. Accumulating chunks in RAM and concatenating
    # doubled 5.7 GB inside a 16 GiB pod and was OOM-killed.
    patch_out = pooled_out = None
    if do_patch:
        (args.output_root / "patch224").mkdir(parents=True, exist_ok=True)
        patch_out = np.lib.format.open_memmap(args.output_root / "patch224" / "features.float16.npy",
                                              mode="w+", dtype=np.float16, shape=(total, 256, 768))
    if do_448:
        (args.output_root / "pooled448").mkdir(parents=True, exist_ok=True)
        pooled_out = np.lib.format.open_memmap(args.output_root / "pooled448" / "features.float16.npy",
                                               mode="w+", dtype=np.float16, shape=(total, 1536))

    backbone = _load_backbone(plan[0][4]).to(device).eval()
    rows, written = [], 0
    mk = lambda ds, idx: DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    # Pass 2 -- stream. Two loaders in lockstep (shuffle=False) so batch k of
    # each holds the same frames; six workers because a bare __getitem__ loop
    # decoded one JPEG at a time at ~2.4 frames/s on the slice.
    with torch.no_grad():
        for config, train, train448, selected, _b in plan:
            idx = [s[0] for s in selected]
            loaders = [mk(train, idx)] + ([mk(train448, idx)] if do_448 else [])
            pos = 0
            for batches in zip(*loaders):
                n = batches[0]["image"].shape[0]
                if do_patch:
                    tok = backbone(batches[0]["image"].to(device, non_blocking=True))
                    patch_out[written:written + n] = tok[:, backbone.num_prefix_tokens:, :].to(torch.float16).cpu().numpy()
                if do_448:
                    x448 = batches[-1]["image"].to(device, non_blocking=True)
                    pooled_out[written:written + n] = backbone.pooled_features(x448).to(torch.float16).cpu().numpy()
                for _, video, frame, target in selected[pos:pos + n]:
                    rows.append({"source_video_id": video, "source_frame_index": frame,
                                 "target": target, "label": class_names[target]})
                pos += n; written += n
                if written % 1024 < n:
                    print(f"  {written} frames", flush=True)
    assert written == total == len(rows), (written, total, len(rows))

    base = {"backbone_family": "timm_dinov2_vitb14_reg4", "backbone_weight_sha256": weight_sha,
            "label": args.label, "annotation_source": str(ANNOTATION), "videos": sorted(seen),
            "interpolation": "bilinear", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "batch_size": args.batch_size}
    if do_patch:
        # GPU against GPU: on the same device this sits at fp16 tolerance for ~95%
        # of rows (median ~2e-3) with a cuDNN batch-composition tail (K-24); on CPU
        # it reads up to 4e-2 because the cache's patch-embed conv ran under TF32.
        check = _check_against_cache(patch_out, rows)
        print(f"cache consistency: max |mean(patch) - cached mean-patch| = {check['max_abs_delta']:.3e} "
              f"over {check['rows_checked']} rows", flush=True)
        _finalize(args.output_root / "patch224", patch_out, rows,
                  {**base, "feature": "patch_tokens_pre_pool", "resolution": 224, "patch_grid": [16, 16],
                   "cache_consistency": check})
    if do_448:
        _finalize(args.output_root / "pooled448", pooled_out, rows,
                  {**base, "feature": "cls_plus_mean_patch_pre_head", "resolution": 448, "patch_grid": [32, 32]})
    print(f"done: {total} frames, {len(seen)} videos", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
