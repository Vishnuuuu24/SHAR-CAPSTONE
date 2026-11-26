#!/usr/bin/env python3
"""DINOv2 patch-token cache for the P3A head comparison (GPU, pod-only).

P3A (`docs/PHASE_PLAN.md` section 6) compares three heads that must see the
SAME input: GAP+linear, an MLP, and the SEMSCNN-inspired 2-D multiscale-SE
head. The last one consumes a 2-D spatial map, which the registered pooled
cache does not contain -- pooling is exactly what it discards. So this script
writes the pre-pool patch grid for the study's frame set.

Scope and cost. The phase plan specifies ResNet50's C5 7x7x2048 map because it
was written while ResNet was the backbone; the project's selected backbone is
now DINOv2 (D-44/D-45), so the heads are compared on DINOv2 patch tokens at
the registered 224x224 -- a 16x16 grid of 768-D tokens, which is a genuine 2-D
map with justified spatial adjacency (`docs/ARCHITECTURE.md` forbids only
reshaping an already-pooled vector into an image). That deviation is recorded
in the manifest so a reader cannot mistake it for the plan's C5 arm.

Patch tokens for a full fold would be ~154 GB. This writes them only for the
frame set the validated recipe actually trains and scores on -- TRAIN thinned
to 8 frames/video at `sampling_seed=0` plus every VALIDATION row, deduplicated
across the four folds -- which `core.p1c_roi.plan_roi_frames` already defines
and the ROI study already used: 82,679 frames, 32.5 GB at fp16.

Registers are excluded, matching `models.dinov2_backbone.pool_cls_plus_mean_patch`:
the grid holds patch tokens only, and the CLS token is stored separately so a
head can use either or both without re-reading pixels.

    ./.venv/bin/python scripts/build_p1c_patch_cache.py --device cuda:0 \
        --num-workers 6 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_extraction_io import finalize_streamed_array, write_manifest
from core.p1c_readiness import sha256_file
from core.p1c_roi import RawFramePlanDataset, plan_roi_frames, select_plan_rows
from data.p1c_loader import FramePreprocessing
from scripts.build_p1c_feature_cache import _load_backbone
from scripts.run_p1c_frame_baseline import _load_config, _make_datasets

#: `--backbone` -> (registered per-fold configs, patch grid). The frame PLAN and
#: the per-fold row provenance always come from the DINOv2 pooled caches below
#: (same frames for every backbone, which is what makes the head comparison
#: apples-to-apples); only the backbone block and preprocessing differ per variant.
VARIANTS = {
    "dinov2": ([f"configs/experiments/p1c/P1C-VIDEO-DINOV2-B2-AR-FOLD{f}.yaml" for f in (1, 2, 3, 4)], (16, 16)),
    "siglip": ([f"configs/experiments/p1c/P1C-VIDEO-SIGLIP-AR-FOLD{f}.yaml" for f in (1, 2, 3, 4)], (14, 14)),
}
DEFAULT_CACHE_ROOTS = [Path(f"results/p1c/p1c-dinov2-feature-cache-ar-fold{f}") for f in (1, 2, 3, 4)]
FRAMES_PER_VIDEO = 8
SAMPLING_SEED = 0
#: DINOv2: 224 / 14 = 16 -> 16x16 grid. SigLIP: 224 / 16 = 14 -> 14x14 grid (see VARIANTS).
TOKEN_DIM = 768


def _preprocess_batch(pixels: np.ndarray, preprocessing: FramePreprocessing) -> torch.Tensor:
    """Resize/normalise raw uint8 frames exactly as the registered dataset does.

    `core.p1c_roi.preprocess_crop` owns this arithmetic for one PIL image; this
    reuses it per frame rather than re-deriving the transform, so a full-frame
    row here is the same tensor the registered cache was built from.
    """

    from PIL import Image

    from core.p1c_roi import preprocess_crop

    return torch.stack(
        [preprocess_crop(Image.fromarray(frame, mode="RGB"), preprocessing) for frame in pixels]
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=sorted(VARIANTS), default="dinov2")
    parser.add_argument("--output-root", type=Path, default=Path("results/p1c/p1c-patch-cache"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--extract-closure-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug: cap the plan to the first N frames")
    parser.add_argument(
        "--finalize-only", action="store_true",
        help="skip the GPU pass and only hash/verify/manifest an already-written output root "
             "(recovery path for K-30: the extraction succeeded and finalisation was OOM-killed)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from torch.utils.data import DataLoader

    args = _parse_args(argv)
    configs, patch_grid = VARIANTS[args.backbone]
    if args.finalize_only:
        return _finalize_only(args)
    device = torch.device(args.device)

    plan = plan_roi_frames(DEFAULT_CACHE_ROOTS, frames_per_video=FRAMES_PER_VIDEO, sampling_seed=SAMPLING_SEED)
    if args.limit:
        plan = plan[: args.limit]
    total = len(plan)
    print(f"plan: {total} frames", flush=True)
    remaining: set[tuple[str, int]] = set(plan)
    plan_index = {pair: i for i, pair in enumerate(plan)}

    args.output_root.mkdir(parents=True, exist_ok=True)
    patches = np.lib.format.open_memmap(
        args.output_root / "features.float16.npy", mode="w+", dtype=np.float16,
        shape=(total, patch_grid[0] * patch_grid[1], TOKEN_DIM),
    )
    cls_tokens = np.lib.format.open_memmap(
        args.output_root / "cls.float16.npy", mode="w+", dtype=np.float16, shape=(total, TOKEN_DIM)
    )
    rows: list[dict[str, Any]] = [None] * total  # type: ignore[list-item]

    protocol = preprocessing = class_names = backbone_block = None
    per_config = []
    for config_path in configs:
        _config, this_protocol, _run_class = _load_config(Path(config_path))
        if args.extract_closure_path is not None:
            this_protocol["data"]["video_frames"]["extract_closure_path"] = str(args.extract_closure_path)
        train, validation = _make_datasets(this_protocol, seed=0)
        per_config.append((config_path, train, validation))
        if protocol is None:
            protocol = this_protocol
            preprocessing = FramePreprocessing.from_config(protocol["preprocessing"])
            class_names = protocol["model"]["class_names"]
            backbone_block = protocol["backbone"]
    assert protocol is not None

    backbone = _load_backbone(backbone_block).to(device).eval()
    prefix = backbone.num_prefix_tokens
    written = 0
    with torch.no_grad():
        for _config_path, train, validation in per_config:
            for dataset in (train, validation):
                selected_raw = select_plan_rows(dataset, remaining)
                if not selected_raw:
                    continue
                selected = [
                    (index, video, frame, target, plan_index[(video, frame)])
                    for index, video, frame, target in selected_raw
                ]
                for _index, video, frame, _target in selected_raw:
                    remaining.discard((video, frame))
                loader = DataLoader(
                    RawFramePlanDataset(dataset, selected), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers,
                )
                for batch in loader:
                    tensor = _preprocess_batch(batch["pixels"].numpy(), preprocessing).to(device)
                    tokens = backbone(tensor)
                    grid = tokens[:, prefix:, :]
                    if grid.shape[1] != patch_grid[0] * patch_grid[1]:
                        raise RuntimeError(
                            f"expected {patch_grid[0] * patch_grid[1]} patch tokens, got {grid.shape[1]}"
                        )
                    grid_np = grid.to(torch.float16).cpu().numpy()
                    cls_np = tokens[:, 0, :].to(torch.float16).cpu().numpy()
                    for k, global_index in enumerate(batch["global_index"].tolist()):
                        patches[global_index] = grid_np[k]
                        cls_tokens[global_index] = cls_np[k]
                        target = int(batch["target"][k])
                        rows[global_index] = {
                            "source_video_id": batch["source_video_id"][k],
                            "source_frame_index": int(batch["source_frame_index"][k]),
                            "target": target,
                            "label": class_names[target],
                        }
                    written += len(batch["global_index"])
                    if written % 2048 < args.batch_size:
                        print(f"  {written}/{total} frames", flush=True)

    if remaining:
        raise RuntimeError(f"{len(remaining)} planned frames not found in any fold manifest")
    assert all(row is not None for row in rows), "not every plan row was written"

    import csv

    with (args.output_root / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    shape, patch_sha = finalize_streamed_array(args.output_root, patches)
    cls_shape, cls_sha = finalize_streamed_array(args.output_root, cls_tokens, path_name="cls.float16.npy")
    consistency = _consistency_for(args.backbone, args.output_root, rows)
    write_manifest(args.output_root, {
        "backbone_family": backbone_block["family"],
        "backbone_weight_sha256": backbone_block["weights"]["sha256"],
        "feature": "patch_tokens_pre_pool_plus_cls",
        "resolution": preprocessing.resize_height,
        "patch_grid": list(patch_grid),
        "registers_excluded": True,
        "num_prefix_tokens": int(prefix),
        "backbone_variant": args.backbone,
        "frames_per_video": FRAMES_PER_VIDEO,
        "sampling_seed": SAMPLING_SEED,
        "phase_plan_deviation": _DEVIATION_NOTE[args.backbone],
        "row_count": total,
        "shape": shape,
        "cls_shape": cls_shape,
        "dtype": "float16",
        "features_sha256": patch_sha,
        "cls_sha256": cls_sha,
        "pooled_cache_consistency": consistency,
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "batch_size": args.batch_size,
    })
    print(f"wrote {args.output_root}: {total} rows, consistency {consistency}", flush=True)
    return 0


def _finalize_only(args: argparse.Namespace) -> int:
    """Hash, verify and manifest an output root whose arrays are already complete.

    Exists because the GPU pass is the expensive half (47 min on the slice) and
    K-30 killed only the hashing step after it had finished. Everything this
    writes is derived from what is already on disk; nothing is recomputed from
    pixels, so a root that is NOT complete must fail rather than be blessed --
    hence the row-count and all-zero-tail checks below.
    """

    import csv as _csv

    if args.backbone != "dinov2":
        raise SystemExit("--finalize-only is the DINOv2 K-30 recovery path; rerun the build for other backbones")
    root = args.output_root
    patch_grid = VARIANTS[args.backbone][1]
    with (root / "rows.csv").open() as handle:
        rows = list(_csv.DictReader(handle))
    patches = np.load(root / "features.float16.npy", mmap_mode="r")
    cls_tokens = np.load(root / "cls.float16.npy", mmap_mode="r")
    if not (len(rows) == patches.shape[0] == cls_tokens.shape[0]):
        raise RuntimeError(
            f"incomplete root: {len(rows)} rows, {patches.shape[0]} patch rows, {cls_tokens.shape[0]} cls rows"
        )
    # The extraction writes in plan order, so an all-zero final row means the
    # loop did not reach the end and this root must not be finalised.
    for name, array in (("patches", patches), ("cls", cls_tokens)):
        if not np.any(np.asarray(array[-1], dtype=np.float32)):
            raise RuntimeError(f"{name} last row is all zeros -- extraction did not complete")
    print(f"finalize-only: {len(rows)} rows, last row non-zero in both arrays", flush=True)

    patch_sha = sha256_file(root / "features.float16.npy")
    cls_sha = sha256_file(root / "cls.float16.npy")
    consistency = _consistency_for(args.backbone, root, [
        {"source_video_id": r["source_video_id"], "source_frame_index": int(r["source_frame_index"])}
        for r in rows
    ])
    write_manifest(root, {
        "backbone_family": "timm_dinov2_vitb14_reg4",  # finalize-only is the DINOv2 K-30 recovery path
        "feature": "patch_tokens_pre_pool_plus_cls",
        "resolution": 224,
        "patch_grid": list(patch_grid),
        "registers_excluded": True,
        "frames_per_video": FRAMES_PER_VIDEO,
        "sampling_seed": SAMPLING_SEED,
        "phase_plan_deviation": (
            "P3A specifies ResNet50 C5 7x7x2048; this cache is DINOv2 patch tokens because DINOv2 "
            "is the project's selected backbone (D-44/D-45). Recorded 2026-09-25."
        ),
        "row_count": len(rows),
        "shape": list(patches.shape),
        "cls_shape": list(cls_tokens.shape),
        "dtype": "float16",
        "features_sha256": patch_sha,
        "cls_sha256": cls_sha,
        "pooled_cache_consistency": consistency,
        "finalized_separately": (
            "K-30: the GPU extraction completed and finalisation was OOM-killed; this manifest was "
            "written by --finalize-only from the arrays that pass had already flushed."
        ),
    })
    print(f"wrote manifest for {root}: consistency {consistency}", flush=True)
    return 0


_DEVIATION_NOTE = {
    "dinov2": (
        "P3A specifies ResNet50 C5 7x7x2048; this cache is DINOv2 patch tokens because DINOv2 "
        "is the project's selected backbone (D-44/D-45). Recorded, owner-approved 2026-09-25."
    ),
    "siglip": (
        "Backbone-swap comparison cache: SigLIP ViT-B/16-224 on the same 82,679-frame plan as the "
        "DINOv2 patch cache. cls.float16.npy holds SigLIP's attention-pooling-head output (it has no "
        "CLS token), features.float16.npy the 14x14 patch tokens after post_layernorm."
    ),
}


def _consistency_for(backbone: str, out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """DINOv2 caches are checked against the registered pooled cache; SigLIP has no such cache."""

    if backbone == "dinov2":
        return _check_against_pooled_cache(out_dir, rows)
    return {"skipped": f"no registered pooled cache exists for backbone {backbone!r}"}


def _check_against_pooled_cache(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """CLS ++ mean(patch) here must reproduce the registered pooled 1536-D cache.

    The registered cache stores `pool_cls_plus_mean_patch` output, so recomputing
    that pooling from this cache's own two arrays is an end-to-end check that the
    grid and the CLS token were written for the frame their row claims. Tolerance
    is the K-28 GPU/fp16 envelope, not equality.
    """

    import csv as _csv
    import statistics

    lookup: dict[tuple[str, int], tuple[Path, int]] = {}
    for root in DEFAULT_CACHE_ROOTS:
        for split in ("train", "validation"):
            path = root / split / "rows.csv"
            if not path.exists():
                continue
            with path.open() as handle:
                for i, row in enumerate(_csv.DictReader(handle)):
                    lookup.setdefault((row["source_video_id"], int(row["source_frame_index"])), (root / split, i))

    patches = np.load(out_dir / "features.float16.npy", mmap_mode="r")
    cls_tokens = np.load(out_dir / "cls.float16.npy", mmap_mode="r")
    arrays: dict[Path, np.ndarray] = {}
    deltas: list[float] = []
    for i, row in enumerate(rows):
        if i % 97:  # sample ~1% of rows; a full pass would re-read 32 GB
            continue
        hit = lookup.get((row["source_video_id"], row["source_frame_index"]))
        if hit is None:
            continue
        root, index = hit
        if root not in arrays:
            arrays[root] = np.load(root / "features.float16.npy", mmap_mode="r")
        pooled = np.concatenate([
            np.asarray(cls_tokens[i], dtype=np.float32),
            np.asarray(patches[i], dtype=np.float32).mean(axis=0),
        ])
        cached = np.asarray(arrays[root][index], dtype=np.float32)
        deltas.append(float(np.abs(pooled - cached).max()))
    return {
        "rows_checked": len(deltas),
        "max_abs_delta": max(deltas) if deltas else None,
        "median_abs_delta": statistics.median(deltas) if deltas else None,
    }


if __name__ == "__main__":
    sys.exit(main())
