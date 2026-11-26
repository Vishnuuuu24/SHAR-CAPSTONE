#!/usr/bin/env python3
"""Build a CLIP-level VideoMAE feature cache for one fold and split.

The frame-level cache (`scripts/build_p1c_feature_cache.py`) stores one vector
per frame. A video backbone consumes 16 frames at once and emits one vector per
clip, so this is a separate artifact with its own row unit: one row per
non-overlapping 16-row window of a video, tiled exactly as
`core.p1c_event_clips._window_starts` tiles them (hop = clip length, trailing
partial window dropped), so the clips are the same ones every A2/A3 result
used and the DINOv2 control can be built from the frame cache over identical
windows.

Rows record video, first and last source frame, target and label -- the
provenance every other artifact carries -- and the manifest pins the backbone
digest, preprocessing and clip definition so a loader can refuse a mismatch.

Streams to a preallocated memmap with worker-backed loading (K-28: the two
things that cost slice time last time). Verifies a sample against a live
recompute at the build's batch size before reporting success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_event_clips import _window_starts
from scripts.build_p1c_feature_cache import _load_backbone
from scripts.run_p1c_frame_baseline import _load_config, _make_datasets

CLIP_LENGTH, FRAME_STRIDE = 16, 1


class _ClipDataset(Dataset):
    def __init__(self, ds, clips):  # clips: [(dataset indices x16, video, f0, f1, target)]
        self.ds, self.clips = ds, clips
    def __len__(self): return len(self.clips)
    def __getitem__(self, k):
        idx = self.clips[k][0]
        return torch.stack([self.ds[i]["image"] for i in idx])


def _tile(ds):
    """Every non-overlapping 16-row clip of every video in `ds`, metadata only."""
    vids, ref, fr, tg = ds._videos, ds._video_ref, ds._frame_index, ds._target   # noqa: SLF001
    by_video: dict[str, list[int]] = {}
    for i in range(len(ds)):
        by_video.setdefault(vids[int(ref[i])]["source_video_id"], []).append(i)
    clips = []
    for video, idx in by_video.items():
        idx.sort(key=lambda i: int(fr[i]))
        starts, _span = _window_starts(len(idx), CLIP_LENGTH, FRAME_STRIDE)
        for s in starts:
            w = [idx[s + k * FRAME_STRIDE] for k in range(CLIP_LENGTH)]
            clips.append((w, video, int(fr[w[0]]), int(fr[w[-1]]), int(tg[w[0]])))
    return clips


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--split", choices=["train", "validation"], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8, help="clips per batch (= 128 frames)")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--verify-sample", type=int, default=256)
    p.add_argument("--extract-closure-path", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="debug: cap clips")
    args = p.parse_args(argv)

    _c, proto, _r = _load_config(args.config)
    if args.extract_closure_path is not None:
        proto["data"]["video_frames"]["extract_closure_path"] = str(args.extract_closure_path)
    device = torch.device(args.device)
    backbone = _load_backbone(proto["backbone"]).to(device).eval()
    class_names = proto["model"]["class_names"]
    train, validation = _make_datasets(proto, seed=0)
    ds = train if args.split == "train" else validation
    clips = _tile(ds)
    if args.limit: clips = clips[: args.limit]
    n, d = len(clips), backbone.out_features
    print(f"{args.split}: {n} clips of {CLIP_LENGTH} rows over {len({c[1] for c in clips})} videos -> dim {d}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(args.output_dir / "features.float16.npy", mode="w+", dtype=np.float16, shape=(n, d))
    loader = DataLoader(_ClipDataset(ds, clips), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    written = 0
    with torch.no_grad():
        for x in loader:
            f = backbone.pooled_features(x.to(device, non_blocking=True)).to(torch.float16).cpu().numpy()
            out[written:written + len(f)] = f; written += len(f)
            if written % 1024 < args.batch_size: print(f"  {written} clips", flush=True)
    assert written == n
    out.flush(); del out

    with open(args.output_dir / "rows.csv", "w", newline="") as h:
        w = csv.writer(h); w.writerow(["row_index", "source_video_id", "start_source_frame", "end_source_frame", "target", "label"])
        for i, (_, v, f0, f1, t) in enumerate(clips): w.writerow([i, v, f0, f1, t, class_names[t]])

    # verify: recompute a deterministic sample at the build's batch size
    feats = np.load(args.output_dir / "features.float16.npy", mmap_mode="r")
    rng = np.random.default_rng(0); sample = sorted(rng.choice(n, size=min(args.verify_sample, n), replace=False).tolist())
    worst = 0.0
    with torch.no_grad():
        for s in range(0, len(sample), args.batch_size):
            ids = sample[s:s + args.batch_size]
            x = torch.stack([torch.stack([ds[i]["image"] for i in clips[k][0]]) for k in ids]).to(device)
            f = backbone.pooled_features(x).to(torch.float16).cpu().numpy().astype(np.float32)
            worst = max(worst, float(np.abs(f - np.asarray(feats[ids], dtype=np.float32)).max()))
    digest = hashlib.sha256((args.output_dir / "features.float16.npy").read_bytes()).hexdigest()
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "kind": "clip_feature_cache", "backbone_family": proto["backbone"]["family"],
        "backbone_weight_sha256": proto["backbone"]["weights"]["sha256"], "feature_dim": d, "dtype": "float16",
        "clip_length": CLIP_LENGTH, "frame_stride": FRAME_STRIDE, "hop": CLIP_LENGTH * FRAME_STRIDE,
        "row_count": n, "split": args.split, "class_names": class_names, "preprocessing": proto["preprocessing"],
        "source_manifest_sha256": proto["data"][args.split]["sha256"], "cache_sha256": digest,
        "verification": {"rows_sampled": len(sample), "max_abs_delta": worst, "batch_size": args.batch_size},
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output_dir}: {n} x {d}, sha256 {digest[:16]}, verify max|delta| {worst:.3e} over {len(sample)} clips", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
