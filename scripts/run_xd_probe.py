#!/usr/bin/env python3
"""XD-Violence external table (P4C): the final SHAR pipeline trained and tested on XD, plus a UCF->XD transfer test.

In-domain: frozen DINOv2 pooled features (32 frames/video), step_permuted head (D-53), trained on the official XD train
split minus a grouped validation hold-out (best epoch), targets = each video's FIRST filename label. Arms fixed before
any run: `unweighted` (primary, pipeline unchanged) and `class_weighted` (inverse frequency; XD is imbalanced).
Scored ONCE on the official test split: video-level per-class AP / mAP against all labels (289 videos are multi-label),
accuracy and macro-F1 on single-label test videos, violence-vs-normal video AUC.

Transfer: the D-53 head trained on each UCF AR fold (fold-cache features, 3 seeds), applied to XD test videos whose
labels all have a UCF counterpart (Riot excluded), argmax over the 6 shared classes.
Not comparable with the UCF headline or with XD's official frame-level AP (different task/metric, rule 3).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.confusion_groups import macro_f1, per_class_f1  # noqa: E402
from core.p1c_head_variants import group_pooled_sequences_by_video  # noqa: E402
from core.xd_dataset import XD_CLASSES  # noqa: E402

CACHE = Path("results/xd/xd-dinov2-pooled32")
TRAIN_POSITIONS = list(range(2, 32, 4))  # 8 of the 32 segment centres ~ centres of 8 equal segments
UCF_OF = {"Normal": "Normal_Videos_event", "Fighting": "Fighting", "Shooting": "Shooting", "Abuse": "Abuse",
          "Car accident": "RoadAccidents", "Explosion": "Explosion"}
SHARED = [c for c in XD_CLASSES if c in UCF_OF]


def load_xd():
    feats = np.load(CACHE / "features.float16.npy", mmap_mode="r")
    cache_rows = list(csv.DictReader((CACHE / "rows.csv").open()))
    man = {r["source_video_id"]: r for r in csv.DictReader(Path("results/xd/manifest.csv").open())}
    videos = []
    for i, r in enumerate(cache_rows):
        if r["ok"] != "1":
            continue
        m = man[r["source_video_id"]]
        videos.append({"i": i, "id": r["source_video_id"], "split": m["split"], "labels": m["unified_label"].split("|"),
                       "primary": m["primary_label"]})
    return feats, videos


def xd_sequences(feats, videos, positions, seed_offset, seed):
    """{(dataset, video): (seq, row)} via the headline grouping/permutation code."""

    pooled, rows = [], []
    for v in videos:
        x = np.asarray(feats[v["i"]], dtype=np.float32)[positions]
        for j, p in enumerate(positions):
            pooled.append(x[j])
            rows.append({"source_dataset": "xd_violence", "source_video_id": v["id"], "source_frame_index": p,
                         "target": XD_CLASSES.index(v["primary"]), "label": v["primary"]})
    return group_pooled_sequences_by_video(np.stack(pooled), rows, permute_frame_order=True, permute_seed=seed_offset + seed)


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if y.any() else float("nan")


def in_domain(feats, videos, seeds, arm, train_eval, epochs=None, dry_run=False):
    by = {s: [v for v in videos if v["split"] == s] for s in ("train", "validation", "test")}
    if dry_run:  # exercise every code path without reading the test split: validation stands in for test
        by["test"] = by["validation"]
    counts = np.bincount([XD_CLASSES.index(v["primary"]) for v in by["train"]], minlength=len(XD_CLASSES))
    weights = (counts.sum() / (len(counts) * np.maximum(counts, 1))).tolist() if arm == "class_weighted" else None
    test_ids = [v["id"] for v in by["test"]]
    multi_hot = np.array([[c in v["labels"] for c in XD_CLASSES] for v in by["test"]], dtype=int)
    single = np.array([len(v["labels"]) == 1 for v in by["test"]])
    truth1 = np.array([XD_CLASSES.index(v["primary"]) for v in by["test"]])
    runs = []
    for seed in seeds:
        tr = xd_sequences(feats, by["train"], TRAIN_POSITIONS, 1000, seed)
        va = xd_sequences(feats, by["validation"], list(range(32)), 2000, seed)
        te = xd_sequences(feats, by["test"], list(range(32)), 3000, seed)
        result, preds = train_eval(seed, tr, {**va, **te}, list(XD_CLASSES), epochs, class_weights=weights, monitor_seq=va)
        P = np.stack([preds[("xd_violence", i)] for i in test_ids])
        pred = P.argmax(1)
        ap = {c: average_precision(multi_hot[:, k], P[:, k]) for k, c in enumerate(XD_CLASSES)}
        from sklearn.metrics import roc_auc_score
        runs.append({"seed": seed, "best_epoch": result.best_epoch,
                     "map": float(np.nanmean(list(ap.values()))), "ap": ap,
                     "single_label_accuracy": float((pred[single] == truth1[single]).mean()),
                     "single_label_macro_f1": macro_f1(truth1[single], pred[single], len(XD_CLASSES)),
                     "single_label_per_class_f1": dict(zip(XD_CLASSES, per_class_f1(truth1[single], pred[single], len(XD_CLASSES)).tolist())),
                     "top1_in_any_label": float(np.mean([XD_CLASSES[p] in v["labels"] for p, v in zip(pred, by["test"])])),
                     "violence_auc": float(roc_auc_score(multi_hot[:, 0] == 0, 1.0 - P[:, 0]))})
        print(json.dumps({"arm": arm, **{k: v for k, v in runs[-1].items() if not isinstance(v, dict)}}), flush=True)
    return runs, {"train_counts": dict(zip(XD_CLASSES, counts.tolist())), "test_videos": len(test_ids),
                  "single_label_test_videos": int(single.sum()), "class_weights": weights}


def transfer(feats, videos, folds, seeds, train_eval, epochs=None, dry_run=False):
    from scripts.run_subset7_search import load_fold_features, sequences
    split = "validation" if dry_run else "test"
    test = [v for v in videos if v["split"] == split and all(l in UCF_OF for l in v["labels"])]
    single = np.array([len(v["labels"]) == 1 for v in test])
    truth = np.array([SHARED.index(v["primary"]) for v in test])
    runs = []
    for fold in folds:
        fd = load_fold_features(fold, ("train", "validation"))
        ucf = fd["class_names"]
        cols = [ucf.index(UCF_OF[c]) for c in SHARED]
        for seed in seeds:
            tr, va = sequences(fd, "train", seed), sequences(fd, "validation", seed)
            te = xd_sequences(feats, test, list(range(32)), 3000, seed)
            result, preds = train_eval(seed, tr, {**va, **te}, ucf, epochs, monitor_seq=va)
            P = np.stack([preds[("xd_violence", v["id"])] for v in test])[:, cols]
            pred = P.argmax(1)
            runs.append({"fold": fold, "seed": seed, "accuracy_single": float((pred[single] == truth[single]).mean()),
                         "macro_f1_single": macro_f1(truth[single], pred[single], len(SHARED)),
                         "per_class_f1_single": dict(zip(SHARED, per_class_f1(truth[single], pred[single], len(SHARED)).tolist())),
                         "top1_in_any_label": float(np.mean([SHARED[p] in v["labels"] for p, v in zip(pred, test)]))})
            print(json.dumps({"transfer": True, **{k: v for k, v in runs[-1].items() if not isinstance(v, dict)}}), flush=True)
        del fd
    return runs, {"test_videos": len(test), "single_label": int(single.sum()), "shared_classes": SHARED,
                  "mapping": UCF_OF, "excluded": "test videos carrying Riot (no UCF counterpart)"}


def _mean(runs, key):
    return float(np.mean([r[key] for r in runs]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--output", type=Path, default=Path("results/xd/xd-probe"))
    ap.add_argument("--dry-run", action="store_true", help="1 epoch, validation stands in for test (never reads the test split)")
    args = ap.parse_args(argv)
    epochs = 1 if args.dry_run else None
    out = args.output / "test.json"
    if out.exists():
        raise SystemExit(f"{out} exists: the XD test split has already been spent")
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    from scripts.run_p1c_xd_extra_probe import _train_eval
    torch.set_num_threads(args.threads)
    feats, videos = load_xd()
    result = {}
    for arm in ("unweighted", "class_weighted"):
        runs, info = in_domain(feats, videos, args.seeds, arm, _train_eval, epochs, args.dry_run)
        result[arm] = {"runs": runs, "info": info, **{k: _mean(runs, k) for k in ("map", "single_label_accuracy", "single_label_macro_f1", "top1_in_any_label", "violence_auc")},
                       "ap_mean": {c: float(np.mean([r["ap"][c] for r in runs])) for c in XD_CLASSES}}
    truns, tinfo = transfer(feats, videos, args.folds, args.seeds, _train_eval, epochs, args.dry_run)
    result["ucf_to_xd_transfer"] = {"runs": truns, "info": tinfo, **{k: _mean(truns, k) for k in ("accuracy_single", "macro_f1_single", "top1_in_any_label")}}
    manifest = json.loads((CACHE / "manifest.json").read_text())
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    result["provenance"] = {"git_revision": rev, "seeds": args.seeds, "folds": args.folds, "features_sha256": manifest["features_sha256"],
                            "xd_manifest_sha256": hashlib.sha256(Path("results/xd/manifest.csv").read_bytes()).hexdigest(),
                            "train_positions": TRAIN_POSITIONS, "eval_positions": "all 32 (4 chunks of 8, mean logits)",
                            "split_note": "official XD split; 184 source groups appear in both official train and test", "dry_run": args.dry_run}
    out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))} for k, v in result.items() if k != "provenance"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
