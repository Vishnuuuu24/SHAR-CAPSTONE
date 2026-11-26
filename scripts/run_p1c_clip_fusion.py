#!/usr/bin/env python3
"""Phase 2 of the temporal plan: does a motion backbone help the motion classes?

Four arms on IDENTICAL clips, folds, seeds, head recipe and evaluation; the
representation is the only factor:

    dinov2_raw      mean of the 16 DINOv2 frame vectors over each clip window,
                    raw -- the registered pipeline on clip units (ties to 0.4068)
    dinov2          the same, per-block standardised (matched control for fusion)
    videomae        the VideoMAE-B clip vector, standardised
    fusion          [dinov2 ++ videomae], each block standardised on TRAIN

Clip windows are the ones `scripts/build_p1c_clip_cache.py` tiled; the DINOv2
clip vector is computed here from the frame cache over exactly those windows,
asserted to be 16 rows each. Training samples 8 clips per video with the same
deterministic temporal spread `core.p1c_video_sampling` uses for frames; the
validation split is scored in full and majority-voted per video, so the
video-level number means the same thing as every earlier one.

Standardisation is fitted on the training split only and applied to
validation. It is a deviation from the registered recipe, which is why the raw
DINOv2 arm is kept: the fusion result must be read against `dinov2`, and
`dinov2` against `dinov2_raw` shows what standardisation alone does.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_head_training import train_cached_head
from core.p1c_video_sampling import sample_rows_per_video

ARMS = ("dinov2_raw", "dinov2", "videomae", "fusion")


def _load_clip_cache(root: Path):
    rows = [{"source_dataset": "ucf_crime_official_video_frames", "source_video_id": r["source_video_id"],
             "source_frame_index": int(r["start_source_frame"]), "end_source_frame": int(r["end_source_frame"]),
             "target": int(r["target"]), "label": r["label"]} for r in csv.DictReader(open(root / "rows.csv"))]
    return np.asarray(np.load(root / "features.float16.npy", mmap_mode="r"), dtype=np.float32), rows


def _dinov2_clip_means(frame_root: Path, clip_rows):
    """Mean DINOv2 frame vector over each clip's exact 16-row window."""
    F = np.load(frame_root / "features.float16.npy", mmap_mode="r")
    index = defaultdict(dict)
    for i, r in enumerate(csv.DictReader(open(frame_root / "rows.csv"))):
        index[r["source_video_id"]][int(r["source_frame_index"])] = i
    out = np.zeros((len(clip_rows), F.shape[1]), dtype=np.float32)
    for k, c in enumerate(clip_rows):
        frames = [f for f in index[c["source_video_id"]] if c["source_frame_index"] <= f <= c["end_source_frame"]]
        assert len(frames) == 16, (c["source_video_id"], c["source_frame_index"], len(frames))
        out[k] = np.asarray(F[sorted(index[c["source_video_id"]][f] for f in frames)], dtype=np.float32).mean(0)
    return out


def _standardise(train: np.ndarray, val: np.ndarray):
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return (train - mu) / sd, (val - mu) / sd


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--clips-per-video", type=int, default=8)
    p.add_argument("--output", type=Path, default=Path("results/p1c/p1c-exp-P2-fusion"))
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    summary = defaultdict(dict)
    for fold in args.folds:
        vroot = Path(f"results/p1c/p1c-videomae-clip-cache-ar-fold{fold}")
        droot = Path(f"results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}")
        Vtr, rtr = _load_clip_cache(vroot / "train"); Vva, rva = _load_clip_cache(vroot / "validation")
        class_names = json.load(open(vroot / "train" / "manifest.json"))["class_names"]
        Dtr = _dinov2_clip_means(droot / "train", rtr); Dva = _dinov2_clip_means(droot / "validation", rva)
        keep = sample_rows_per_video(rtr, args.clips_per_video)   # deterministic, temporally spread
        rtr_s = [rtr[i] for i in keep]
        print(f"fold{fold}: train {len(rtr)} clips -> {len(keep)} sampled ({args.clips_per_video}/video); "
              f"validation {len(rva)} clips over {len({r['source_video_id'] for r in rva})} videos | "
              f"|dinov2| {np.abs(Dtr).mean():.3f} |videomae| {np.abs(Vtr).mean():.3f}", flush=True)
        Dtr_s, Vtr_s = Dtr[keep], Vtr[keep]
        Dz_tr, Dz_va = _standardise(Dtr_s, Dva); Vz_tr, Vz_va = _standardise(Vtr_s, Vva)
        feats = {"dinov2_raw": (Dtr_s, Dva), "dinov2": (Dz_tr, Dz_va), "videomae": (Vz_tr, Vz_va),
                 "fusion": (np.concatenate([Dz_tr, Vz_tr], 1), np.concatenate([Dz_va, Vz_va], 1))}
        for arm in args.arms:
            Xtr, Xva = feats[arm]; per_seed = []
            for seed in args.seeds:
                res = train_cached_head(Xtr, rtr_s, Xva, rva, class_names, seed=seed, bootstrap_iterations=100)
                pr = res.probe_result
                per_seed.append({"seed": seed, "best_epoch": res.best_epoch,
                                 "frame_macro_f1": pr.frame_metrics["macro_f1"],
                                 "video_macro_f1": pr.video_level_metrics["macro_f1"],
                                 "per_class_video_f1": {c: pr.video_level_metrics["per_class"][c]["f1"] for c in class_names},
                                 "confusion": pr.video_level_metrics["confusion_matrix"]})
            vm = [s["video_macro_f1"] for s in per_seed]
            summary[arm][fold] = {"video_macro_f1_mean": st.mean(vm), "video_macro_f1_sd": st.stdev(vm) if len(vm) > 1 else 0.0,
                                  "frame_macro_f1_mean": st.mean(s["frame_macro_f1"] for s in per_seed),
                                  "dim": int(Xtr.shape[1]), "seeds": per_seed}
            print(f"  {arm:<11} D={Xtr.shape[1]:<5} video macro-F1 {st.mean(vm):.4f} +/- {summary[arm][fold]['video_macro_f1_sd']:.4f}  "
                  f"epochs {[s['best_epoch'] for s in per_seed]}", flush=True)
    (args.output / "fusion-results.json").write_text(json.dumps({"clips_per_video": args.clips_per_video,
        "seeds": args.seeds, "class_names": class_names, "results": summary}, indent=1) + "\n")
    print("\n=== CROSS-FOLD video-level macro-F1 (mean of fold means; sd across folds) ===")
    for arm in args.arms:
        m = [summary[arm][f]["video_macro_f1_mean"] for f in args.folds]
        sd = st.stdev(m) if len(m) > 1 else 0.0
        print(f"  {arm:<11} {st.mean(m):.4f} +/- {sd:.4f}   folds {[f'{x:.4f}' for x in m]}")
    print(f"wrote {args.output/'fusion-results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
