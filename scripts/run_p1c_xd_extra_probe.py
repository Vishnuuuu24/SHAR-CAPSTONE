#!/usr/bin/env python3
"""Cross-dataset TRAINING-ONLY probe: do extra XD-Violence Shooting/Fighting/Abuse videos help the UCF 14-class head?

Validation is 100% UCF (headline protocol). Not a UCF headline number: train is no longer class-balanced and uses outside data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
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

from core.p1c_head_variants import group_pooled_sequences_by_video, load_pooled_features, train_sequence_head  # noqa: E402
from core.p1c_oracle_interval import predict_video_probs  # noqa: E402
from core.p1c_xd_extra import load_cache  # noqa: E402
from core.p1c_xd_probe import ARM_K, aggregate, extra_sequences, leakage_check, run_metrics, select_extra_videos  # noqa: E402
from scripts.run_p1c_head_comparison import _build_head, _load_fold, _load_patch_cache, _patch_indices_for  # noqa: E402

FRAMES = 8


def train_head(seed, train_seq, monitor_seq, class_names, epochs=None, class_weights=None, **trainer_kwargs):
    """Seeded step_permuted head trained on `train_seq`, best epoch chosen on `monitor_seq`; returns (result, head in eval mode).

    `trainer_kwargs` go straight to `train_sequence_head` (e.g. `patience`, `epoch_resampler`, `label_smoothing`)."""
    torch.manual_seed(seed)  # seeded head init, identical across arms (K-31)
    head = _build_head("step_permuted", len(class_names), FRAMES)
    kwargs = {} if epochs is None else {"epochs": epochs}
    if class_weights is not None:
        kwargs["class_weights"] = class_weights
    result = train_sequence_head(head, train_seq, monitor_seq, class_names, frames=FRAMES, seed=seed, **kwargs, **trainer_kwargs)
    head.eval()
    return result, head


def _train_eval(seed, train_seq, val_seq, class_names, epochs, class_weights=None, monitor_seq=None):
    """`monitor_seq` (default `val_seq`) drives best-epoch selection; predictions are returned for every `val_seq` video."""
    result, head = train_head(seed, train_seq, val_seq if monitor_seq is None else monitor_seq, class_names, epochs, class_weights)
    preds = {k: predict_video_probs(head, seq, FRAMES) for k, (seq, _row) in val_seq.items()}
    return result, preds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=list(ARM_K), choices=list(ARM_K))
    ap.add_argument("--xd-cache", type=Path, default=Path("results/p1c/p1c-xd-extra-cache"))
    ap.add_argument("--patch-cache-root", type=Path, default=Path("results/p1c/p1c-patch-cache"))
    ap.add_argument("--output", type=Path, default=Path("results/p1c/p1c-xd-extra-probe"))
    ap.add_argument("--epochs", type=int, default=None, help="epoch ceiling override (smoke only)")
    ap.add_argument("--smoke", action="store_true", help="fold 1, seed 0, 1 epoch")
    args = ap.parse_args(argv)
    if args.smoke:
        args.folds, args.seeds, args.epochs = [1], [0], 1
    args.output.mkdir(parents=True, exist_ok=True)

    xd_feats, xd_rows = load_cache(args.xd_cache)
    xd_manifest = json.loads((args.xd_cache / "manifest.json").read_text())
    videos_by_class: dict[str, list[str]] = defaultdict(list)
    for vid, cls in sorted({(r["video_id"], r["class_name"]) for r in xd_rows}):
        videos_by_class[cls].append(vid)
    selected = {arm: select_extra_videos(videos_by_class, arm) for arm in args.arms}
    cls_mmap, feat_mmap, key_to_index = _load_patch_cache(args.patch_cache_root)
    load_args = SimpleNamespace(frames=FRAMES, sampling_seed=0,
                                extract_closure_path=Path("results/p1c/p1c-video-frames/extract_closure.json"))

    runs, per_video, fold_cache_sha, val_means, val_labels, val_ids, class_names = [], [], {}, [], [], [], None
    started = time.time()
    for fold in args.folds:
        fd = _load_fold(fold, load_args)
        class_names, val_rows, train_rows = fd["class_names"], fd["validation_rows"], fd["train_rows"]
        fold_cache_sha[str(fold)] = fd["cache_sha256"]
        train_pooled = load_pooled_features(cls_mmap, feat_mmap, _patch_indices_for(train_rows, key_to_index))
        val_pooled = load_pooled_features(cls_mmap, feat_mmap, _patch_indices_for(val_rows, key_to_index))
        for seed in args.seeds:
            ucf_train = group_pooled_sequences_by_video(train_pooled, train_rows, permute_frame_order=True, permute_seed=1000 + seed)
            val_seq = group_pooled_sequences_by_video(val_pooled, val_rows, permute_frame_order=True, permute_seed=2000 + seed)
            for arm in args.arms:
                train_seq = dict(ucf_train)  # UCF part identical across arms; extras appended to TRAIN only
                if selected[arm]:
                    train_seq.update(extra_sequences(xd_feats, xd_rows, selected[arm], class_names, 3000 + seed))
                result, preds = _train_eval(seed, train_seq, val_seq, class_names, args.epochs)
                keys = sorted(val_seq)
                truth = [int(val_seq[k][1]["target"]) for k in keys]
                pred = [int(np.argmax(preds[k])) for k in keys]
                metrics = run_metrics(truth, pred, class_names)
                runs.append({"arm": arm, "fold": fold, "seed": seed, "metrics": metrics, "best_epoch": result.best_epoch,
                             "train_videos": len(train_seq)})
                for k, t, p in zip(keys, truth, pred):
                    per_video.append([arm, fold, seed, k[1], class_names[t], class_names[p], int(t == p), f"{preds[k][t]:.6f}"])
                print(json.dumps({"arm": arm, "fold": fold, "seed": seed, "macro_f1": metrics["macro_f1"],
                                  "best_epoch": result.best_epoch, "train_videos": len(train_seq),
                                  "elapsed_s": round(time.time() - started)}), flush=True)
        for (_ds, vid), (seq, row) in group_pooled_sequences_by_video(val_pooled, val_rows).items():
            val_means.append(seq.mean(axis=0)); val_labels.append(row["label"]); val_ids.append(f"fold{fold}:{vid}")

    used = sorted({v for ids in selected.values() for v in ids})
    xd_means = np.stack([np.asarray(xd_feats[[i for i, r in enumerate(xd_rows) if r["video_id"] == v]], np.float32).mean(0) for v in used]) if used else None
    summary = {"purpose": "cross-dataset training-only probe; validation is 100% UCF; NOT a UCF headline; train no longer class-balanced; plain CE",
               "arms": aggregate(runs, class_names), "runs": [{k: r[k] for k in ("arm", "fold", "seed", "best_epoch", "train_videos")} | {"macro_f1": r["metrics"]["macro_f1"]} for r in runs],
               "selected_extra_videos": {a: {c: sum(v in set(videos_by_class[c]) for v in ids) for c in videos_by_class} for a, ids in selected.items()},
               "leakage_check": leakage_check(xd_means, used, np.stack(val_means), val_labels, val_ids) if used else None}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.output / "per_video.csv").open("w", newline="") as h:
        w = csv.writer(h); w.writerow(["arm", "fold", "seed", "video", "label", "pred", "correct", "p_true"]); w.writerows(per_video)
    patch_manifest = json.loads((args.patch_cache_root / "manifest.json").read_text())
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    config = {"folds": args.folds, "seeds": args.seeds, "arms": {a: ARM_K[a] for a in args.arms}, "git_revision": rev,
              "smoke": args.smoke, "epochs_override": args.epochs, "device": "cpu", "num_threads": 4,
              "xd_cache_dir": str(args.xd_cache), "xd_features_sha256": xd_manifest["features_sha256"], "xd_rows_sha256": xd_manifest["rows_sha256"],
              "xd_source_file_list_sha256": xd_manifest["source_file_list_sha256"],
              "ucf_patch_cache_features_sha256": patch_manifest["features_sha256"], "ucf_patch_cache_cls_sha256": patch_manifest["cls_sha256"],
              "ucf_fold_cache_sha256": fold_cache_sha, "extra_selection": "per-class fixed shuffle default_rng(stable_seed(class, 0)); prefix k; same in every fold",
              "head": "step_permuted; torch.manual_seed(seed) before construction; train permute 1000+seed (UCF) / 3000+seed (XD); val permute 2000+seed",
              "loss": "plain cross-entropy, unweighted", "validation": "UCF fold validation, all cached frames, 8-frame chunks, mean logits, video-level",
              "gate": "+0.04 macro-F1 on 4/4 folds, fold-mean over seeds, vs baseline from this same seeded path"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    for arm, e in summary["arms"].items():
        print(arm, json.dumps({"macro_f1": e["macro_f1_over_runs"], "gate": e.get("gate")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
