#!/usr/bin/env python3
"""Post-processing diagnostic: does per-video score normalisation recover
cross-video Shooting classification?

Every representation tried so far ranks event clips above background clips
WITHIN a held-out video (per-video ROC 0.65-0.81) while the same scores are at
chance when pooled ACROSS videos (0.35-0.44). One explanation is score-offset /
scale drift between videos: each camera sits at its own baseline
"Shooting-ness", so a single global threshold fails even though the ordering
inside each video is right. This tests that explanation directly.

The LOVO fit is replicated exactly from `core.p1c_event_clips.leave_one_video_out_clips`
(same scaler-on-train, same LogisticRegression, same seed); this script only
adds score capture and three post-hoc transforms of the HELD-OUT video's scores:

  raw       s_t
  z-score   (s_t - mean_v) / std_v      using that video's OWN score distribution
  rank      percentile of s_t within the video's scores

**This is unsupervised test-time calibration, not training.** mean_v, std_v and
the ranks are computed from the held-out video's unlabelled prediction scores
only. No event label from the held-out video touches the normalisation, and the
classifier never sees the held-out video. The transforms are strictly monotonic
within a video, so within-video ROC is provably unchanged; the only thing that
can move is the POOLED ranking, which is exactly what a global threshold needs.

Video-level threshold: to avoid tuning on test labels, the pooled threshold is
the value that a Shooting video's mean clip score must exceed, fixed at 0 for
z-scores (the video's own mean, i.e. "more than half its clips are above its
own average" is vacuous -- so instead the video-level rule is the fraction of
clips above the TRAINING videos' pooled 90th-percentile normalised score). The
90th percentile is registered here as the choice, not tuned.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_background_mask import load_intervals
from core.p1c_event_clips import _assert_no_leakage, _train_test_masks, build_event_clips, clip_features
from scripts.diagnose_p1c_event_separability import ANNOTATION
from scripts.run_p1c_event_clip_probe import load_deduplicated_cache
from scripts.run_p1c_patch_motion_probe import _load, patch_motion_features


def lovo_scores(clips, X, *, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y = np.array([1 if c.is_positive else 0 for c in clips], dtype=np.int64)
    out = {}
    for held_out in sorted({c.video_id for c in clips}):
        tr, te = _train_test_masks(clips, held_out)
        _assert_no_leakage(clips, tr, held_out)
        scaler = StandardScaler().fit(X[tr])
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed).fit(scaler.transform(X[tr]), y[tr])
        s_test = model.predict_proba(scaler.transform(X[te]))[:, 1]
        s_train = model.predict_proba(scaler.transform(X[tr]))[:, 1]
        # per-TRAINING-video normalisation stats, for the video-level threshold
        train_vids = np.array([c.video_id for c, m in zip(clips, tr) if m])
        z_train = np.concatenate([(s_train[train_vids == v] - s_train[train_vids == v].mean()) / (s_train[train_vids == v].std() + 1e-9)
                                  for v in np.unique(train_vids)])
        out[held_out] = {"scores": s_test, "y": y[te], "train_z_p90": float(np.percentile(z_train, 90))}
    return out


def transforms(s: np.ndarray) -> dict[str, np.ndarray]:
    z = (s - s.mean()) / (s.std() + 1e-9)
    rank = (np.argsort(np.argsort(s)) + 0.5) / len(s)
    return {"raw": s, "zscore": z, "rank": rank}


def evaluate(per_video):
    from sklearn.metrics import average_precision_score, roc_auc_score
    report = {}
    for name in ("raw", "zscore", "rank"):
        within_roc, within_pr, pooled_s, pooled_y, video_hits = [], [], [], [], 0
        for v, d in per_video.items():
            s = transforms(d["scores"])[name]; y = d["y"]
            pooled_s.append(s); pooled_y.append(y)
            if y.sum() and (1 - y).sum():
                within_roc.append(roc_auc_score(y, s)); within_pr.append(average_precision_score(y, s))
            if name == "zscore":
                video_hits += int((s > d["train_z_p90"]).mean() > 0.1)   # >10% of clips above training p90
        S, Y = np.concatenate(pooled_s), np.concatenate(pooled_y)
        report[name] = {
            "within_roc_mean": float(np.mean(within_roc)), "within_roc_median": float(np.median(within_roc)),
            "within_roc_sd": float(np.std(within_roc, ddof=1)), "within_pr_mean": float(np.mean(within_pr)),
            "pooled_roc": float(roc_auc_score(Y, S)), "pooled_pr": float(average_precision_score(Y, S)),
            "pooled_base_rate": float(Y.mean()), "n_videos_auc": len(within_roc),
            **({"video_recall_at_train_p90": video_hits / len(per_video)} if name == "zscore" else {}),
        }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/p1c/p1c-exp-A4-normalization/normalization.json"))
    args = parser.parse_args(argv)
    intervals = load_intervals(ANNOTATION)
    kw = dict(label="Shooting", clip_length=16, frame_stride=1, positive_overlap_threshold=0.5, negative_margin=16, seed=args.seed)

    results = {}
    # global representations from the deduplicated pooled caches
    rows, feats = load_deduplicated_cache(sorted(Path().glob("results/p1c/p1c-dinov2-feature-cache-ar-fold*/train")))
    build = build_event_clips(rows, intervals, **kw); clips = build.usable
    for rep in ("single_frame", "dense_mean"):
        X = clip_features(np.asarray(feats, dtype=np.float32), rows, clips, rep)
        results[rep] = evaluate(lovo_scores(clips, X, seed=args.seed))
    # patch motion from the targeted extraction
    P, prow = _load(Path("results/p1c/p1c-shooting-tokens/patch224"))
    build_p = build_event_clips(prow, intervals, **kw); clips_p = build_p.usable
    results["patch_motion"] = evaluate(lovo_scores(clips_p, patch_motion_features(P, prow, clips_p), seed=args.seed))

    print(f"{'representation':<14}{'transform':<8}{'within ROC mean':>16}{'median':>8}{'within PR':>10}{'POOLED ROC':>12}{'POOLED PR':>10}")
    for rep, R in results.items():
        for t in ("raw", "zscore", "rank"):
            r = R[t]
            print(f"{rep:<14}{t:<8}{r['within_roc_mean']:>10.3f}+/-{r['within_roc_sd']:.3f}{r['within_roc_median']:>8.3f}"
                  f"{r['within_pr_mean']:>10.3f}{r['pooled_roc']:>12.3f}{r['pooled_pr']:>10.3f}")
        print()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed": args.seed, "clip_params": kw, "results": results}, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
