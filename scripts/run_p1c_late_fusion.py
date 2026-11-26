#!/usr/bin/env python3
"""Pre-registered equal-weight late fusion of DINOv2 head (A) with VideoMAE arms k0/k4 on the same UCF validation videos.

Validation only (test untouched). Fusion weight fixed at 0.5; nothing is selected on these numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import p1c_late_fusion as lf  # noqa: E402
from core.p1c_head_variants import group_pooled_sequences_by_video, load_pooled_features  # noqa: E402
from scripts.run_p1c_head_comparison import _load_fold, _load_patch_cache, _patch_indices_for  # noqa: E402
from scripts.run_p1c_xd_extra_probe import FRAMES, _train_eval  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dinov2_probs(folds, seeds, patch_cache_root):
    """Baseline arm of run_p1c_xd_extra_probe (no XD extras): full 14-class probabilities per validation video."""

    cls_mmap, feat_mmap, key_to_index = _load_patch_cache(patch_cache_root)
    load_args = SimpleNamespace(frames=FRAMES, sampling_seed=0,
                                extract_closure_path=Path("results/p1c/p1c-video-frames/extract_closure.json"))
    probs, f1s, classes = {}, {}, None
    for fold in folds:
        fd = _load_fold(fold, load_args)
        classes, val_rows, train_rows = fd["class_names"], fd["validation_rows"], fd["train_rows"]
        train_pooled = load_pooled_features(cls_mmap, feat_mmap, _patch_indices_for(train_rows, key_to_index))
        val_pooled = load_pooled_features(cls_mmap, feat_mmap, _patch_indices_for(val_rows, key_to_index))
        for seed in seeds:
            tr = group_pooled_sequences_by_video(train_pooled, train_rows, permute_frame_order=True, permute_seed=1000 + seed)
            va = group_pooled_sequences_by_video(val_pooled, val_rows, permute_frame_order=True, permute_seed=2000 + seed)
            result, preds = _train_eval(seed, tr, va, classes, None)
            for k, (_seq, row) in va.items():
                probs[(fold, seed, k[1])] = (classes[int(row["target"])], np.asarray(preds[k], dtype=float))
            print(json.dumps({"fold": fold, "seed": seed, "best_epoch": result.best_epoch}), flush=True)
    return classes, probs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--videomae-csv", type=Path, default=Path("results/p1c/p1c-videomae-finetune/per_video.csv"))
    ap.add_argument("--reference-summary", type=Path, default=Path("results/p1c/p1c-xd-extra-probe/summary.json"))
    ap.add_argument("--patch-cache-root", type=Path, default=Path("results/p1c/p1c-patch-cache"))
    ap.add_argument("--output", type=Path, default=Path("results/p1c/p1c-late-fusion"))
    ap.add_argument("--reuse-dinov2", action="store_true", help="read saved dinov2_probs.csv instead of re-training")
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    a_csv = args.output / "dinov2_probs.csv"
    if args.reuse_dinov2:
        a = lf.load_dinov2_probs(a_csv)
    else:
        a = dinov2_probs(args.folds, args.seeds, args.patch_cache_root)
        lf.write_probs_csv(a_csv, *a)
    classes = a[0]
    systems = {"A": a}
    for arm in ("k0", "k4"):
        b = lf.load_videomae_probs(args.videomae_csv, arm)
        keys = lf.align(a, b)
        systems[arm] = (classes, {k: b[1][k] for k in keys})
        systems[f"A+{arm}"] = (classes, lf.fused_probs(a, b, keys))
    ev = lf.evaluate(systems, "A", ["k0", "k4", "A+k0", "A+k4"])
    ref = json.loads(args.reference_summary.read_text())["arms"]["baseline"]
    ref_fm = {int(f): v for f, v in ref["fold_mean_macro_f1"].items()}
    ev["sanity_vs_registered_baseline"] = {
        "reference_overall_mean": ref["macro_f1_over_runs"]["mean"], "rerun_overall_mean": ev["systems"]["A"]["macro_f1_mean"],
        "per_fold_abs_diff": {f: abs(ev["systems"]["A"]["fold_mean_macro_f1"][f] - ref_fm[f]) for f in ref_fm}}
    ev["purpose"] = "validation-only pre-registered equal-weight late fusion; no tuning; test untouched"
    (args.output / "summary.json").write_text(json.dumps(ev, indent=2, sort_keys=True) + "\n")
    for name in ("k0", "k4", "A+k0", "A+k4"):
        lf.write_probs_csv(args.output / f"_tmp_{name}.csv", classes, systems[name][1], system=name)
    with (args.output / "per_video.csv").open("w") as out:
        for i, name in enumerate(("k0", "k4", "A+k0", "A+k4")):
            lines = (args.output / f"_tmp_{name}.csv").read_text().splitlines(keepends=True)
            out.writelines(lines if i == 0 else lines[1:])
            (args.output / f"_tmp_{name}.csv").unlink()
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    config = {"folds": args.folds, "seeds": args.seeds, "fusion_weight": lf.FUSION_WEIGHT, "git_revision": rev,
              "primary": "A+k4", "control": "A+k0", "gate": "+0.04 fold-mean macro-F1 on 4/4 folds vs A",
              "videomae_per_video_sha256": _sha(args.videomae_csv), "dinov2_probs_sha256": _sha(a_csv),
              "reference_summary_sha256": _sha(args.reference_summary), "device": "cpu", "num_threads": 4,
              "runtime_s": round(time.time() - started), "reused_dinov2_csv": args.reuse_dinov2}
    (args.output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"systems": {n: {"mean": s["macro_f1_mean"], "folds": s["fold_mean_macro_f1"]} for n, s in ev["systems"].items()},
                      "sanity": ev["sanity_vs_registered_baseline"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
