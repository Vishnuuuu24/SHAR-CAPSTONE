#!/usr/bin/env python3
"""Build the block-8 hidden-state cache that partial VideoMAE fine-tuning trains from.

Thin CLI. Planning, row provenance, the ULP gate and the publish step live in
`core.p1c_videomae_cache`; the encoder split lives in `core.p1c_videomae_split`.

For every video in the union of train+validation over the Action-Recognition
folds (test videos are never planned), the 8 temporally-spread 16-row clips
(`select_clips`) are run through VideoMAE-B blocks 0-7 in float32 and the
hidden state entering block 8 is stored as float16 `[1568, 768]` in a
preallocated memmap (K-28: streamed, worker-backed loading). The plan is read
from the registered fold clip caches' rows.csv (identical tiling), and every
clip the datasets actually produce is asserted equal to the plan.

Verification recomputes a sample live at the build batch size and applies the
K-26 ULP gate; it also reports the max |delta| between the pooled readout of
the full backbone and of the cache-split path (blocks 8-11 + fc_norm).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_videomae_cache import (ARRAY_NAME, DIM, MANIFEST_NAME, ROWS_NAME, SPLITS, TOKENS, cache_nbytes,
                                     git_revision, publish_cache, sha256_file, ulp_gate, union_plan, write_rows)
from core.p1c_videomae_split import CACHE_BLOCKS, TopBlocksClassifier, encode_to_cache_point
from scripts.build_p1c_clip_cache import _ClipDataset, _tile
from scripts.build_p1c_feature_cache import _load_backbone
from scripts.run_p1c_frame_baseline import _load_config, _make_datasets

CONFIG = "configs/experiments/p1c/P1C-VIDEO-VIDEOMAE-AR-FOLD{fold}.yaml"
TILING_ROOT = "results/p1c/p1c-videomae-clip-cache-ar-fold{fold}"


def load_plan(folds):
    tiling = {f: {s: list(csv.DictReader(open(REPO_ROOT / TILING_ROOT.format(fold=f) / s / "rows.csv"))) for s in SPLITS}
              for f in folds}
    return union_plan(tiling)


def _hidden(backbone, x, device):
    return encode_to_cache_point(backbone, x.to(device, non_blocking=True))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--output-dir", type=Path, required=True, help="node-local scratch while building")
    p.add_argument("--publish-dir", type=Path, default=None, help="durable copy (repo) written after verification")
    p.add_argument("--extract-closure-path", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8, help="clips per batch")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--verify-sample", type=int, default=64)
    p.add_argument("--plan-only", action="store_true", help="print the plan and byte total, then exit")
    args = p.parse_args(argv)

    plan = load_plan(args.folds)
    n = len(plan)
    print(f"plan: {n} clips over {len({c['source_video_id'] for c in plan})} videos; "
          f"{TOKENS}x{DIM} fp16 -> {cache_nbytes(n):,} bytes ({cache_nbytes(n) / 2**30:.2f} GiB)", flush=True)
    if args.plan_only:
        return 0

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proto0 = _load_config(REPO_ROOT / CONFIG.format(fold=args.folds[0]))[1]
    backbone = _load_backbone(proto0["backbone"]).to(device).eval()
    class_names = proto0["model"]["class_names"]
    row_of = {(c["source_video_id"], c["start_source_frame"], c["end_source_frame"]): i for i, c in enumerate(plan)}
    out = np.lib.format.open_memmap(args.output_dir / ARRAY_NAME, mode="w+", dtype=np.float16, shape=(n, TOKENS, DIM))
    written = np.zeros(n, dtype=bool)
    live: dict[int, tuple] = {}  # row -> (dataset, 16 dataset indices): kept for verification
    max_abs, started = 0.0, time.time()

    for fold in args.folds:
        _c, proto, _r = _load_config(REPO_ROOT / CONFIG.format(fold=fold))
        if args.extract_closure_path is not None:
            proto["data"]["video_frames"]["extract_closure_path"] = str(args.extract_closure_path)
        assert proto["backbone"]["weights"]["sha256"] == proto0["backbone"]["weights"]["sha256"]
        datasets = dict(zip(SPLITS, _make_datasets(proto, seed=0)))
        for split in SPLITS:
            todo = []
            for idx16, video, f0, f1, target in _tile(datasets[split]):
                row = row_of.get((video, f0, f1))
                if row is None or written[row] or plan[row]["roles"].get(fold) != split:
                    continue
                assert plan[row]["target"] == target and plan[row]["label"] == class_names[target], (video, f0)
                todo.append((row, idx16))
            print(f"fold {fold} {split}: {len(todo)} new clips", flush=True)
            if not todo:
                continue
            loader = DataLoader(_ClipDataset(datasets[split], [(i16,) for _r, i16 in todo]), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
            cursor = 0
            with torch.no_grad():
                for x in loader:
                    rows = [r for r, _ in todo[cursor:cursor + len(x)]]
                    h = _hidden(backbone, x, device).to(torch.float16).cpu().numpy()
                    if not np.isfinite(h).all():
                        raise FloatingPointError(f"non-finite or fp16-overflowing hidden state in rows {rows}")
                    max_abs = max(max_abs, float(np.abs(h).max()))
                    out[rows] = h
                    written[rows] = True
                    for r, i16 in todo[cursor:cursor + len(x)]:
                        live[r] = (datasets[split], i16)
                    cursor += len(x)
                    if cursor % 256 < args.batch_size:
                        print(f"  {int(written.sum())}/{n} clips, {time.time() - started:.0f}s", flush=True)
    if not written.all():
        raise RuntimeError(f"{int((~written).sum())} planned clips were never produced by any fold dataset")
    out.flush()
    del out

    write_rows(args.output_dir / ROWS_NAME, plan)
    cache = np.load(args.output_dir / ARRAY_NAME, mmap_mode="r")
    sample = sorted(np.random.default_rng(0).choice(n, size=min(args.verify_sample, n), replace=False).tolist())
    reports, pooled_delta, exact = [], 0.0, 0
    top = TopBlocksClassifier(backbone, num_classes=len(class_names), trainable_blocks=0, seed=0).to(device).eval()
    with torch.no_grad():
        for s in range(0, len(sample), args.batch_size):
            ids = sample[s:s + args.batch_size]
            x = torch.stack([torch.stack([live[i][0][j]["image"] for j in live[i][1]]) for i in ids])
            fresh = _hidden(backbone, x, device)
            full = backbone.pooled_features(x.to(device))
            for k, i in enumerate(ids):
                f16 = fresh[k].to(torch.float16).cpu().numpy()
                exact += int(np.array_equal(f16, cache[i]))
                reports.append(ulp_gate(np.asarray(cache[i]), f16))
            hidden = torch.from_numpy(np.stack([np.asarray(cache[i]) for i in ids])).to(device)
            h = hidden.float()
            for layer in top.layers:
                h = layer(h)
            pooled_delta = max(pooled_delta, float((top.fc_norm(h.mean(1)) - full).abs().max()))
    if not all(r["ok"] for r in reports):
        raise RuntimeError(f"cache failed the ULP gate: worst {max(r['max_ulps'] for r in reports)} ULPs")
    digest = sha256_file(args.output_dir / ARRAY_NAME)
    manifest = {
        "kind": "videomae_block_hidden_cache", "backbone_family": proto0["backbone"]["family"],
        "backbone_weight_sha256": proto0["backbone"]["weights"]["sha256"], "cache_blocks": CACHE_BLOCKS,
        "cached_tensor": "hidden state entering transformer block 8 (output of block index 7)",
        "dtype": "float16", "shape": [n, TOKENS, DIM], "total_bytes": cache_nbytes(n),
        "clip_length": 16, "frame_stride": 1, "hop": 16, "clips_per_video": 8,
        "clip_selection": "core.p1c_video_sampling.sample_rows_per_video over core.p1c_event_clips._window_starts tiling",
        "folds": args.folds, "videos": len({c["source_video_id"] for c in plan}), "class_names": class_names,
        "preprocessing": proto0["preprocessing"], "build_precision": "float32 (no autocast)",
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "max_abs_value": max_abs,
        "plan_source_rows_sha256": {str(f): {s: sha256_file(REPO_ROOT / TILING_ROOT.format(fold=f) / s / "rows.csv")
                                             for s in SPLITS} for f in args.folds},
        "source_manifest_sha256": {str(f): {s: _load_config(REPO_ROOT / CONFIG.format(fold=f))[1]["data"][s]["sha256"]
                                            for s in SPLITS} for f in args.folds},
        "git": git_revision(REPO_ROOT), "cache_sha256": digest, "rows_sha256": sha256_file(args.output_dir / ROWS_NAME),
        "verification": {"rows_sampled": len(sample), "bit_identical_rows": exact,
                         "max_abs_delta": max(r["max_abs_delta"] for r in reports),
                         "max_ulps": max(r["max_ulps"] for r in reports), "ulp_gate": {"max_ulps": 8, "floor_magnitude": 1.0},
                         "batch_size": args.batch_size, "pooled_readout_split_vs_full_max_abs_delta": pooled_delta},
        "created_at": datetime.now(timezone.utc).isoformat(), "build_seconds": time.time() - started,
    }
    (args.output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"built {args.output_dir}: {n} clips, sha256 {digest[:16]}, verify {manifest['verification']}", flush=True)
    if args.publish_dir is not None:
        print(f"published to {args.publish_dir}, digest {publish_cache(args.output_dir, args.publish_dir)[:16]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
