#!/usr/bin/env python3
"""Does frozen V-JEPA 2 ViT-L add anything over DINOv2 on UCF-Crime clips? Same protocol as `run_p1c_clip_fusion.py`.

Arms, all on IDENTICAL clips, folds, seeds, head recipe and evaluation (the representation is the only factor):

    dinov2          standardised mean of the DINOv2 frame vectors over each clip window (the matched control)
    videomae        the cached VideoMAE-B clip vector (reference: the N9 arm, rebuilt through this same code path)
    vjepa2          the V-JEPA 2 clip vector, standardised
    vjepa2_shuffled the same encoder fed each clip's frames in a fixed random order (needs the `--frame-order shuffled`
                    cache); if it scores like `vjepa2`, the features are not using temporal order
    fusion          [dinov2 ++ vjepa2], each block standardised on TRAIN

Prospective comparison rule (D-65 style, fixed before any V-JEPA 2 number exists): a V-JEPA 2 arm beats its reference only if
the cross-fold paired difference in video macro-F1 is >= +0.04 on 4/4 folds (the bar N29 used). Anything else is recorded
as neutral or negative. All scores are inherited-video-label validation scores (D-40), never event-only.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_head_training import train_cached_head
from core.p1c_video_sampling import sample_rows_per_video
from scripts.run_p1c_clip_fusion import _dinov2_clip_means, _load_clip_cache, _standardise

ARMS = ("dinov2", "videomae", "vjepa2", "vjepa2_shuffled", "fusion")
BAR = 0.04
CACHES = {"videomae": "results/p1c/p1c-videomae-clip-cache-ar-fold{fold}",
          "vjepa2": "results/p1c/p1c-vjepa2-clip-cache-ar-fold{fold}",
          "vjepa2_shuffled": "results/p1c/p1c-vjepa2-shuffled-clip-cache-ar-fold{fold}"}


def _key(r):
    return (r["source_video_id"], r["source_frame_index"])


def _align_train(feats, rows, wanted_rows, what):
    """Rows of a (possibly subsampled) train cache, in the order of `wanted_rows`; every wanted clip must be present."""

    index = {_key(r): i for i, r in enumerate(rows)}
    missing = [_key(r) for r in wanted_rows if _key(r) not in index]
    if missing:
        raise ValueError(f"{what}: {len(missing)} clips the VideoMAE tiling selects are absent, e.g. {missing[0]}")
    return feats[[index[_key(r)] for r in wanted_rows]]


def _same_clips(a, b, what):
    if [_key(r) for r in a] != [_key(r) for r in b]:
        raise ValueError(f"{what}: clip rows differ from the VideoMAE tiling, so the arms are not on identical clips")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--clips-per-video", type=int, default=8)
    p.add_argument("--output", type=Path, default=Path("results/p1c/p1c-exp-vjepa2-probe"))
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    summary: dict = defaultdict(dict)
    for fold in args.folds:
        base = Path(CACHES["videomae"].format(fold=fold))
        droot = Path(f"results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}")
        loaded = {s: _load_clip_cache(base / s) for s in ("train", "validation")}
        (Vtr, rtr), (Vva, rva) = loaded["train"], loaded["validation"]
        class_names = json.load(open(base / "train" / "manifest.json"))["class_names"]
        keep = sample_rows_per_video(rtr, args.clips_per_video)
        rtr_s = [rtr[i] for i in keep]
        X = {"videomae": (Vtr[keep], Vva),
             "dinov2": (_dinov2_clip_means(droot / "train", rtr_s), _dinov2_clip_means(droot / "validation", rva))}
        for arm in ("vjepa2", "vjepa2_shuffled"):
            if arm in args.arms or arm == "vjepa2" and "fusion" in args.arms:
                root = Path(CACHES[arm].format(fold=fold))
                (tr, rt), (va, rv) = _load_clip_cache(root / "train"), _load_clip_cache(root / "validation")
                _same_clips(rv, rva, f"{arm} fold {fold} validation")
                X[arm] = (_align_train(tr, rt, rtr_s, f"{arm} fold {fold} train"), va)
        Z = {arm: _standardise(tr, va) for arm, (tr, va) in X.items()}
        Z["fusion"] = tuple(np.concatenate([Z["dinov2"][i], Z["vjepa2"][i]], 1) for i in (0, 1)) if "vjepa2" in Z else None
        for arm in args.arms:
            Xtr, Xva = Z[arm]
            per_seed = []
            for seed in args.seeds:
                res = train_cached_head(Xtr, rtr_s, Xva, rva, class_names, seed=seed, bootstrap_iterations=100)
                m = res.probe_result.video_level_metrics
                per_seed.append({"seed": seed, "best_epoch": res.best_epoch, "video_macro_f1": m["macro_f1"],
                                 "per_class_video_f1": {c: m["per_class"][c]["f1"] for c in class_names}})
            vm = [s["video_macro_f1"] for s in per_seed]
            summary[arm][fold] = {"video_macro_f1_mean": st.mean(vm), "dim": int(Xtr.shape[1]), "seeds": per_seed}
            print(f"fold{fold} {arm:<16} D={Xtr.shape[1]:<5} video macro-F1 {st.mean(vm):.4f}", flush=True)

    out: dict = {"seeds": args.seeds, "clips_per_video": args.clips_per_video, "bar": BAR, "results": summary, "paired": {}}
    print("\n=== cross-fold video macro-F1 (inherited-label validation, D-40) ===")
    for arm in args.arms:
        m = [summary[arm][f]["video_macro_f1_mean"] for f in args.folds]
        print(f"  {arm:<16} {st.mean(m):.4f} +/- {st.stdev(m) if len(m) > 1 else 0:.4f}  folds {[round(x, 4) for x in m]}")
    for arm, ref in (("vjepa2", "dinov2"), ("vjepa2", "videomae"), ("fusion", "dinov2"), ("vjepa2_shuffled", "vjepa2")):
        if arm in args.arms and ref in args.arms:
            d = [summary[arm][f]["video_macro_f1_mean"] - summary[ref][f]["video_macro_f1_mean"] for f in args.folds]
            out["paired"][f"{arm}-{ref}"] = {"mean": st.mean(d), "per_fold": d, "meets_bar_all_folds": all(x >= BAR for x in d)}
            print(f"  paired {arm} - {ref}: {st.mean(d):+.4f}  folds {[f'{x:+.4f}' for x in d]}  bar +{BAR} on all folds: {all(x >= BAR for x in d)}")
    (args.output / "vjepa2-probe-results.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
