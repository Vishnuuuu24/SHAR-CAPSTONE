#!/usr/bin/env python3
"""SHAR-Combined controlled experiment: does adding XD-Violence + CUVA training videos help the UCF 14-class head?

Arms `ucf_only` / `all_sources`; folds 1..4 of the combined split; seeds 0..2; head = headline `step_permuted`.
Validation = the fold's train_pool rows (never the sealed test split). Headline = UCF-part macro-F1, video level.
Class balancing: inverse-frequency class weights in the cross-entropy loss (same rule in both arms).

    ./.venv/bin/python scripts/run_combined_probe.py --threads 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.combined_dataset import UCF  # noqa: E402
from core.combined_probe import (ARMS, CLASS_NAMES, evaluate_parts, fold_rows, frame_rows_and_pooled,  # noqa: E402
                                 inverse_frequency_weights, load_cache, pool_videos, summarize)
from core.combined_split import N_FOLDS  # noqa: E402
from core.p1c_head_training import (DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE,  # noqa: E402
                                    DEFAULT_MIN_DELTA, DEFAULT_PATIENCE, DEFAULT_WEIGHT_DECAY)
from core.p1c_head_variants import group_pooled_sequences_by_video  # noqa: E402
from core.p1c_readiness import sha256_file  # noqa: E402
from scripts.run_p1c_xd_extra_probe import FRAMES, _train_eval  # noqa: E402


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)  # the imported probe script defaults to 4
    features, index, cache_manifest = load_cache(args.cache_root, args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)

    # Pool every needed video once (trainable folds only -- test rows are never selected).
    needed, fold_data = {}, {}
    for fold in args.folds:
        for arm in args.arms:
            tr, va, counts = fold_rows(args.manifest, fold, arm, index)
            fold_data[(fold, arm)] = (tr, va, counts)
            for r in tr + va:
                needed[(r["source_dataset"], r["source_video_id"])] = index[(r["source_dataset"], r["source_video_id"])]
    keys = sorted(needed)
    pooled = dict(zip(keys, pool_videos(features, [needed[k] for k in keys])))
    print(f"pooled {len(keys)} videos", flush=True)

    runs, per_video, started = [], [], time.time()
    for fold in args.folds:
        for seed in args.seeds:
            for arm in args.arms:
                tr, va, _ = fold_data[(fold, arm)]
                tr_pooled, tr_frames = frame_rows_and_pooled(tr, pooled)
                va_pooled, va_frames = frame_rows_and_pooled(va, pooled)
                train_seq = group_pooled_sequences_by_video(tr_pooled, tr_frames, permute_frame_order=True, permute_seed=1000 + seed)
                val_seq = group_pooled_sequences_by_video(va_pooled, va_frames, permute_frame_order=True, permute_seed=2000 + seed)
                monitor = {k: v for k, v in val_seq.items() if k[0] == UCF}
                weights = inverse_frequency_weights([CLASS_NAMES.index(r["unified_label"]) for r in tr], len(CLASS_NAMES))
                result, preds = _train_eval(seed, train_seq, val_seq, CLASS_NAMES, args.epochs, class_weights=weights, monitor_seq=monitor)
                parts = evaluate_parts(preds, va)
                runs.append({"arm": arm, "fold": fold, "seed": seed, "parts": parts, "best_epoch": result.best_epoch,
                             "train_videos": len(tr), "val_videos": len(va)})
                for r in va:
                    p = preds[(r["source_dataset"], r["source_video_id"])]
                    per_video.append([arm, fold, seed, r["source_dataset"], r["source_video_id"], r["unified_label"],
                                      CLASS_NAMES[int(np.argmax(p))]] + [f"{x:.6f}" for x in p])
                print(json.dumps({"arm": arm, "fold": fold, "seed": seed, "ucf_macro_f1": parts["ucf"]["macro_f1"],
                                  "best_epoch": result.best_epoch, "train_videos": len(tr),
                                  "elapsed_s": round(time.time() - started)}), flush=True)

    summary = summarize(runs)
    summary["runs"] = [{k: v for k, v in r.items() if k != "parts"} | {"macro_f1": {p: m["macro_f1"] for p, m in r["parts"].items()}} for r in runs]
    summary["purpose"] = ("SHAR-Combined arm comparison on fold validation rows (no sealed-test scoring). Headline = UCF-part macro-F1; "
                          "XD/CUVA parts are separate-domain reports, never pooled into it. Best epoch is chosen on the UCF-part validation "
                          "macro-F1 (headline convention), identically in both arms.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.output / "per_video.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "fold", "seed", "source_dataset", "source_video_id", "true_label", "pred_label"] + ["p_" + c for c in CLASS_NAMES])
        w.writerows(per_video)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    config = {"folds": args.folds, "seeds": args.seeds, "arms": {a: list(ARMS[a]) for a in args.arms}, "git_revision": rev,
              "class_names": CLASS_NAMES, "epochs_override": args.epochs, "device": "cpu", "threads": args.threads,
              "head": "step_permuted (StepHead token_dim=1536 frames=8 depth=1); torch.manual_seed(seed) before construction; "
                      "train frame order permuted with seed 1000+seed, validation 2000+seed",
              "head_hparams": {"optimizer": "AdamW", "learning_rate": DEFAULT_LEARNING_RATE, "weight_decay": DEFAULT_WEIGHT_DECAY,
                               "batch_size": DEFAULT_BATCH_SIZE, "epochs_ceiling": DEFAULT_EPOCHS, "patience": DEFAULT_PATIENCE,
                               "min_delta": DEFAULT_MIN_DELTA},
              "class_balancing": "inverse-frequency class weights n/(K*n_c) in the cross-entropy loss, computed from each arm's own training videos; "
                                 "same rule in both arms (the headline recipe is unweighted CE)",
              "video_level": "8 cached frames = one chunk; softmax of the head's logits; macro-F1 over classes with support in the part",
              "checkpoint_selection": "best epoch by UCF-part validation macro-F1 (headline convention; optimistic, identical in both arms)",
              "gate": f"UCF-part macro-F1 gain >= {summary['gate_threshold']} on 4/4 folds, seed-mean per fold",
              "fold_counts": {f"{k[0]}:{k[1]}": {"train": len(v[0]), "val": len(v[1]), **v[2]} for k, v in fold_data.items()},
              "cache_features_sha256": cache_manifest["features_sha256"], "cache_manifest_sha256": sha256_file(args.cache_root / "manifest.json"),
              "combined_manifest_sha256": sha256_file(args.manifest)}
    (args.output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("Datasets/SHAR-Combined/manifest.csv"))
    ap.add_argument("--cache-root", type=Path, default=Path("results/p1c/combined-patch-cache"))
    ap.add_argument("--output", type=Path, default=Path("results/p1c/combined-probe"))
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(1, N_FOLDS + 1)))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--epochs", type=int, default=None, help="epoch ceiling override (smoke only)")
    ap.add_argument("--threads", type=int, default=4)
    summary = run(ap.parse_args(argv))
    cmp = summary.get("comparison_all_sources_minus_ucf_only")
    for arm, e in summary["arms"].items():
        print(arm, json.dumps({p: round(v["macro_f1_mean_over_folds"], 4) for p, v in e["parts"].items()}))
    if cmp:
        print("gate", json.dumps(cmp["gate_on_ucf_part"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
