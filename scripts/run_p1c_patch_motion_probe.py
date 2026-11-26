#!/usr/bin/env python3
"""A3 representation 3 -- patch-token motion, and the 448 resolution ablation.

Both consume the targeted extraction in `results/p1c/p1c-shooting-tokens/` and
score through the SAME clip set, LOVO harness and classifier as
`scripts/run_p1c_event_clip_probe.py`, so each row of the final table differs
from the others in representation only.

Representation 3, "patch_motion": per frame, the absolute per-patch change
`|P_t - P_{t-1}|` over the 16x16 grid is pooled over patches, then averaged
over the clip, and concatenated with the mean-patch appearance:

    [ mean_clip( mean_p(P_t) ),  mean_clip( mean_p(|P_t - P_{t-1}|) ) ]   -> 1536-D

The first half equals the registered cache's mean-patch half exactly. The
second is what R2's global delta cannot express: R2 takes
`mean_p(P_t) - mean_p(P_{t-1})`, a SIGNED mean where opposite motions cancel;
here the absolute value is taken per patch BEFORE pooling, so nothing cancels.
That is the one thing this row tests. `P_{t-1}` is the previous cached row of
the same video (0.167 s); the first row of a video gets a zero delta.

Resolution ablation, "pooled448": the same `dense_mean` representation as R1,
computed from CLS+mean-patch features extracted at 448x448 instead of 224x224.
Same clips, same splits, same classifier -- resolution is the only change.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_background_mask import load_intervals
from core.p1c_event_clips import build_event_clips, clip_features, leave_one_video_out_clips
from scripts.diagnose_p1c_event_separability import ANNOTATION


def _load(root: Path):
    rows = [{"source_dataset": "ucf_crime_official_video_frames", **r} for r in csv.DictReader(open(root / "rows.csv"))]
    return np.load(root / "features.float16.npy", mmap_mode="r"), rows


def patch_motion_features(patches, rows, clips) -> np.ndarray:
    """[n_clips, 1536]: mean-patch appearance ++ mean |per-patch delta|, both clip-averaged."""
    starts = {}
    for i, r in enumerate(rows):
        starts.setdefault(r["source_video_id"], i)   # rows are contiguous per video
    out = np.zeros((len(clips), 1536), dtype=np.float32)
    for k, clip in enumerate(clips):
        idx = list(clip.row_indices)
        first = starts[clip.video_id]
        P = np.asarray(patches[idx], dtype=np.float32)                       # [L, 256, 768]
        prev_idx = [i - 1 if i > first else i for i in idx]                  # clamp at video start
        Pprev = np.asarray(patches[prev_idx], dtype=np.float32)
        out[k, :768] = P.mean(axis=(0, 1))
        out[k, 768:] = np.abs(P - Pprev).mean(axis=(0, 1))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens-root", type=Path, default=Path("results/p1c/p1c-shooting-tokens"))
    parser.add_argument("--label", default="Shooting")
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--positive-overlap-threshold", type=float, default=0.5)
    parser.add_argument("--negative-margin", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", choices=["patch_motion", "pooled448"], action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    intervals = load_intervals(ANNOTATION)
    results = {}
    for arm in args.arm:
        root = args.tokens_root / ("patch224" if arm == "patch_motion" else "pooled448")
        feats, rows = _load(root)
        build = build_event_clips(rows, intervals, label=args.label, clip_length=args.clip_length,
                                  frame_stride=args.frame_stride,
                                  positive_overlap_threshold=args.positive_overlap_threshold,
                                  negative_margin=args.negative_margin, seed=args.seed)
        clips = build.usable
        print(f"[{arm}] clips: positive={len(build.positive)} negative={len(build.negative)} "
              f"ambiguous={len(build.ambiguous)} videos={len({c.video_id for c in clips})}", flush=True)
        if arm == "patch_motion":
            X = patch_motion_features(feats, rows, clips)
        else:
            X = clip_features(np.asarray(feats, dtype=np.float32), rows, clips, "dense_mean")
        res = leave_one_video_out_clips(clips, X, seed=args.seed)
        po = res["pooled"]
        print(f"[{arm}] D={X.shape[1]} pooled f1={po['f1']:.3f} precision={po['precision']:.3f} "
              f"recall={po['recall']:.3f} roc_auc={po['roc_auc']:.3f} pr_auc={po['pr_auc']:.3f} "
              f"video-level recall={res['video_level']['recall']:.3f}", flush=True)
        results[arm] = {**res, "feature_dim": int(X.shape[1]), "resolution": 224 if arm == "patch_motion" else 448,
                        "clip_report": build.report() if hasattr(build, "report") else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"label": args.label, "clip_length": args.clip_length,
        "frame_stride": args.frame_stride, "positive_overlap_threshold": args.positive_overlap_threshold,
        "negative_margin": args.negative_margin, "seed": args.seed,
        "results_by_representation": results}, indent=2, default=str) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
