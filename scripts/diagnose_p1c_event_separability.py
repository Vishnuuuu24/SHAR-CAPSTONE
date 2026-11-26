#!/usr/bin/env python3
"""Diagnostic: can a linear classifier separate Shooting EVENT frames from
NON-EVENT frames of the same videos, using cached DINOv2 features?

This is a precondition test, not a model. Every intervention tried so far has
failed to move Shooting -- aggregation (0.033 under six rules), exposure
(precision halved), temporal windows (+0.0037). The open question is whether the
frozen representation encodes the event/non-event distinction at all. If it does
not, then supervised event learning, MIL, and any sequence model over these same
features are all blocked by the representation rather than by their design.

Method. Official temporal intervals (the UCF-Crime distribution's
`Temporal_Anomaly_Annotation_For_Testing_Videos`) label each cached frame of an
annotated Shooting video as event or non-event. A logistic regression is fitted
on cached DINOv2 features and evaluated **leave-one-video-out**: every fit holds
out one whole video, so no video ever contributes frames to both sides. Frames
within a video are near-duplicates, so a frame-level split would leak almost
perfectly and report a meaningless AUC -- the grouping is the experiment.

Intervals are used ONLY as evaluation labels here. Nothing in the production
pipeline consumes them, and no model receives an interval as an input feature.

Reported against two baselines: ROC-AUC chance is 0.5 by construction, PR-AUC
chance is the positive rate, and a label-shuffled control confirms the harness
itself returns chance when the signal is destroyed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_background_mask import load_intervals  # noqa: E402

ANNOTATION = Path(
    "Datasets/UCF Crime Official/annotations/"
    "Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt"
)


def collect_frames(cache_roots, intervals, label):
    """Per annotated `label` video: feature rows and event/non-event targets.

    Deduplicated by video id across fold caches -- a video's features do not
    depend on which fold's cache holds it, so taking the first occurrence avoids
    counting one video several times and inflating the sample.
    """
    per_video: dict[str, dict] = {}
    for root in cache_roots:
        rows_path, feat_path = root / "rows.csv", root / "features.float16.npy"
        if not rows_path.exists():
            continue
        features = np.load(feat_path, mmap_mode="r")
        grouped = defaultdict(list)
        for index, row in enumerate(csv.DictReader(open(rows_path))):
            grouped[(row["source_video_id"], row["label"])].append((index, int(row["source_frame_index"])))
        for (video, video_label), entries in grouped.items():
            if video_label != label or video in per_video or video not in intervals:
                continue
            spans = intervals[video]
            idx = [i for i, _ in entries]
            y = np.array([1 if any(a <= f <= b for a, b in spans) else 0 for _, f in entries], dtype=np.int64)
            if y.sum() < 5 or (1 - y).sum() < 5:
                continue  # too few of one side for a meaningful held-out score
            per_video[video] = {"x": np.asarray(features[idx], dtype=np.float32), "y": y}
    return per_video


def leave_one_video_out(per_video, *, seed: int, shuffle_labels: bool = False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    names = sorted(per_video)
    results, pooled_scores, pooled_truth = [], [], []
    for held_out in names:
        train_names = [n for n in names if n != held_out]
        assert held_out not in train_names, "leakage: held-out video is in training"
        x_tr = np.concatenate([per_video[n]["x"] for n in train_names])
        y_tr = np.concatenate([per_video[n]["y"] for n in train_names])
        if shuffle_labels:
            y_tr = rng.permutation(y_tr)
        scaler = StandardScaler().fit(x_tr)
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        model.fit(scaler.transform(x_tr), y_tr)
        x_te, y_te = per_video[held_out]["x"], per_video[held_out]["y"]
        score = model.predict_proba(scaler.transform(x_te))[:, 1]
        pooled_scores.append(score)
        pooled_truth.append(y_te)
        results.append({
            "video": held_out,
            "frames": int(len(y_te)),
            "event_frames": int(y_te.sum()),
            "positive_rate": float(y_te.mean()),
            "roc_auc": float(roc_auc_score(y_te, score)),
            "pr_auc": float(average_precision_score(y_te, score)),
            "train_videos": len(train_names),
            "train_frames": int(len(y_tr)),
        })
    truth = np.concatenate(pooled_truth)
    scores = np.concatenate(pooled_scores)
    return results, {
        "pooled_roc_auc": float(roc_auc_score(truth, scores)),
        "pooled_pr_auc": float(average_precision_score(truth, scores)),
        "pooled_positive_rate": float(truth.mean()),
        "total_frames": int(len(truth)),
        "total_event_frames": int(truth.sum()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-glob", default="results/p1c/p1c-dinov2-feature-cache-ar-fold*/train")
    parser.add_argument("--label", default="Shooting")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/p1c/event-separability-diagnostic.json"))
    args = parser.parse_args(argv)

    intervals = load_intervals(ANNOTATION)
    roots = sorted(Path().glob(args.cache_glob))
    per_video = collect_frames(roots, intervals, args.label)
    if len(per_video) < 3:
        raise SystemExit(f"only {len(per_video)} usable annotated {args.label} videos; not enough for leave-one-out")

    results, pooled = leave_one_video_out(per_video, seed=args.seed)
    control, control_pooled = leave_one_video_out(per_video, seed=args.seed, shuffle_labels=True)

    print(f"label={args.label}  usable annotated videos={len(per_video)}  "
          f"frames={pooled['total_frames']:,}  event frames={pooled['total_event_frames']:,} "
          f"({pooled['pooled_positive_rate']*100:.1f}%)\n")
    print(f"{'held-out video':<24}{'frames':>8}{'event':>7}{'pos rate':>10}{'ROC-AUC':>10}{'PR-AUC':>9}")
    for r in results:
        print(f"{r['video']:<24}{r['frames']:>8}{r['event_frames']:>7}{r['positive_rate']:>9.3f}"
              f"{r['roc_auc']:>10.3f}{r['pr_auc']:>9.3f}")
    roc = [r["roc_auc"] for r in results]
    pr = [r["pr_auc"] for r in results]
    base = [r["positive_rate"] for r in results]
    croc = [r["roc_auc"] for r in control]
    print(f"\n  per-video ROC-AUC  mean {np.mean(roc):.3f} +/- {np.std(roc, ddof=1):.3f}   (chance 0.500)")
    print(f"  per-video PR-AUC   mean {np.mean(pr):.3f} +/- {np.std(pr, ddof=1):.3f}   (chance = positive rate, mean {np.mean(base):.3f})")
    print(f"  pooled ROC-AUC {pooled['pooled_roc_auc']:.3f} | pooled PR-AUC {pooled['pooled_pr_auc']:.3f} "
          f"(chance {pooled['pooled_positive_rate']:.3f})")
    print(f"  shuffled-label control: per-video ROC-AUC mean {np.mean(croc):.3f}, pooled {control_pooled['pooled_roc_auc']:.3f}")
    print(f"  videos above chance (ROC>0.5): {sum(1 for v in roc if v > 0.5)}/{len(roc)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "label": args.label,
        "seed": args.seed,
        "usable_videos": len(per_video),
        "per_video": results,
        "pooled": pooled,
        "shuffled_control_per_video": control,
        "shuffled_control_pooled": control_pooled,
        "annotation_source": str(ANNOTATION),
        "cache_roots": [str(r) for r in roots],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
