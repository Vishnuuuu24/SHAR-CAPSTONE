#!/usr/bin/env python3
"""E3-E5: train confusion-guided specialists on the cached DINOv2 features and compare routing / fusion with the general model.

CPU. Validation only (test untouched). Baseline = registered DINOv2 step_permuted head probabilities (dinov2_probs.csv).
Specialist epoch selection uses a video-level hold-out of the TRAINING videos (leak-free); the `valmon` mode additionally
mimics the baseline's own convention (best epoch picked on validation labels) as a labelled sensitivity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import p1c_late_fusion as lf  # noqa: E402
from core.p1c_head_variants import group_pooled_sequences_by_video, load_pooled_features  # noqa: E402
from core.p1c_xd_extra import stable_seed  # noqa: E402
from core.specialist import GroupMap, fuse_probs, oracle_substitute, remap_sequences, route_hard, split_holdout  # noqa: E402
from core.specialist_eval import evaluate_system, within_group_accuracy  # noqa: E402
from scripts.run_p1c_head_comparison import _load_fold, _load_patch_cache, _patch_indices_for  # noqa: E402  (sets 4-thread limits)
from scripts.run_p1c_xd_extra_probe import FRAMES, _train_eval  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/divide_and_conquer/specialists.json"))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--patch-cache-root", type=Path, default=Path("results/p1c/p1c-patch-cache"))
    ap.add_argument("--output", type=Path, default=Path("results/divide-and-conquer/e3-e5"))
    ap.add_argument("--repro-check", action="store_true", help="retrain the general model for the first (fold, seed) and compare to the CSV")
    ap.add_argument("--epochs", type=int, default=None, help="epoch ceiling override (smoke only)")
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    classes, base_rows = lf.load_dinov2_probs(Path(cfg["baseline_probs"]))
    classes = list(classes)
    cls_mmap, feat_mmap, key_to_index = _load_patch_cache(args.patch_cache_root)
    load_args = SimpleNamespace(frames=FRAMES, sampling_seed=0, extract_closure_path=Path("results/p1c/p1c-video-frames/extract_closure.json"))
    w, mon_modes = cfg["fusion_weight"], ("holdout", "valmon")
    runs: dict[str, dict] = defaultdict(dict)          # system -> {(fold, seed): (truth, pred)}
    support, within, spec_csv, repro = [], [], [], None
    for fold in args.folds:
        fd = _load_fold(fold, load_args)
        train_rows, val_rows = fd["train_rows"], fd["validation_rows"]
        train_pooled = load_pooled_features(cls_mmap, feat_mmap, _patch_indices_for(train_rows, key_to_index))
        val_pooled = load_pooled_features(cls_mmap, feat_mmap, _patch_indices_for(val_rows, key_to_index))
        for seed in args.seeds:
            tr = group_pooled_sequences_by_video(train_pooled, train_rows, permute_frame_order=True, permute_seed=1000 + seed)
            va = group_pooled_sequences_by_video(val_pooled, val_rows, permute_frame_order=True, permute_seed=2000 + seed)
            vkeys = sorted(va, key=lambda k: k[1])
            truth = np.array([classes.index(va[k][1]["label"]) for k in vkeys])
            for k, t in zip(vkeys, truth):
                if base_rows[(fold, seed, k[1])][0] != classes[t]:
                    raise RuntimeError(f"label mismatch vs baseline CSV for {k}")
            base = np.stack([base_rows[(fold, seed, k[1])][1] for k in vkeys])
            runs["baseline"][(fold, seed)] = (truth, base.argmax(1))
            if args.repro_check and repro is None:
                _res, preds = _train_eval(seed, tr, va, classes, args.epochs)
                again = np.stack([preds[k] for k in vkeys])
                repro = {"fold": fold, "seed": seed, "max_abs_prob_diff": float(np.abs(again - base).max()),
                         "argmax_agreement": float((again.argmax(1) == base.argmax(1)).mean())}
                print(json.dumps({"repro_check": repro}), flush=True)
            for gname, gs in cfg["group_sets"].items():
                gm = GroupMap(classes, gs["per_fold"][str(fold)])
                counts = {c: sum(1 for _k, (_s, r) in tr.items() if r["label"] == c) for c in gm.names}
                support.append({"group_set": gname, "fold": fold, "seed": seed, "train_videos": counts})
                if min(counts.values()) < cfg["min_train_videos_per_class"]:
                    raise RuntimeError(f"insufficient support for {gname} fold {fold}: {counts}")
                by_cls = defaultdict(list)
                for k, (_s, r) in tr.items():
                    if r["label"] in gm.names:
                        by_cls[r["label"]].append(k[1])
                fit_ids, hold_ids = split_holdout(by_cls, cfg["holdout_fraction"], lambda c: stable_seed(c, 7000 + fold))
                sets = {"holdout": (remap_sequences(tr, gm, fit_ids), remap_sequences(tr, gm, hold_ids)),
                        "valmon": (remap_sequences(tr, gm), remap_sequences(va, gm))}
                for mode in mon_modes:
                    spec_train, monitor = sets[mode]
                    result, preds = _train_eval(seed, spec_train, va, gm.names, args.epochs, monitor_seq=monitor)
                    spec = np.stack([preds[k] for k in vkeys])
                    system = f"{gname}/{mode}"
                    runs[f"{system}/hard_route"][(fold, seed)] = (truth, route_hard(base, spec, gm))
                    runs[f"{system}/fusion_w{w}"][(fold, seed)] = (truth, fuse_probs(base, spec, gm, w).argmax(1))
                    runs[f"{system}/oracle_route_trained"][(fold, seed)] = (truth, oracle_substitute(base, spec, truth, gm))
                    within.append({"group_set": gname, "mode": mode, "fold": fold, "seed": seed, "best_epoch": result.best_epoch,
                                   "spec_train_videos": len(spec_train), "monitor_videos": len(monitor),
                                   **within_group_accuracy(base, spec, truth, gm.idx)})
                    spec_csv += [[gname, mode, fold, seed, k[1], classes[t]] + [f"{x:.6f}" for x in row] for k, t, row in zip(vkeys, truth, spec)]
                    print(json.dumps({"fold": fold, "seed": seed, "set": gname, "mode": mode, "best_epoch": result.best_epoch,
                                      **{k: round(v, 3) for k, v in within[-1].items() if k in ("specialist", "baseline_restricted")}}), flush=True)
    summary = {name: evaluate_system(classes, runs["baseline"], r) for name, r in runs.items() if name != "baseline"}
    def mean_within(gname, mode):
        rows = [r for r in within if r["group_set"] == gname and r["mode"] == mode]
        return {k: float(np.mean([r[k] for r in rows])) for k in ("specialist", "baseline_restricted", "chance")}
    out = {"systems": summary, "within_group_accuracy": {f"{g}/{m}": mean_within(g, m) for g in cfg["group_sets"] for m in mon_modes},
           "within_group_runs": within, "support": support, "repro_check": repro}
    (args.output / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    with (args.output / "specialist_probs.csv").open("w", newline="") as h:
        wr = csv.writer(h); wr.writerow(["group_set", "mode", "fold", "seed", "video", "true_label", "p_local_in_group_order"])
        wr.writerows(spec_csv)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    config = {"git_revision": rev, "folds": args.folds, "seeds": args.seeds, "fusion_weight": w, "gate_delta": 0.04, "device": "cpu",
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "baseline_probs_sha256": hashlib.sha256(Path(cfg["baseline_probs"]).read_bytes()).hexdigest(),
              "runtime_s": round(time.time() - started), "epochs_override": args.epochs}
    (args.output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({n: {"mean_delta": round(s["mean_delta"], 4), "gate_folds": s["folds_meeting_gate"]} for n, s in summary.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
