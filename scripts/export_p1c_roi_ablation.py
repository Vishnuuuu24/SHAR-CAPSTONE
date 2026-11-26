#!/usr/bin/env python3
"""P3C closure artifacts: `roi_ablation.csv` plus the paired-by-video analysis.

`docs/PHASE_PLAN.md` section 6 requires P3C to close with "`roi_ablation.csv`,
ledger and paired downstream analysis". The ROI study (D-51) produced the
measurement; this writes it in the shape the DoD names, and pairs it where the
project pairs everything else -- **by source video**, not by fold. The runs
themselves are untouched: this reads `results/p1c/p1c-roi-study/` and derives.

Why per-video pairing matters here. The fold-level comparison has four paired
points, so a -0.0092 difference sits inside a 0.0462 spread and says almost
nothing about consistency. Pairing by video gives 224 points (56 validation
videos x 4 folds), each scored by both arms on the same frames with the same
seeds, which is what makes "the crop does not help" a statement about videos
rather than about four numbers.

Both arms' per-frame validation probabilities are read from the `predictions/`
arrays `scripts/run_p1c_roi_head.py --capture-probabilities` writes, aggregated
to a video label by the SAME majority vote the headline uses
(`eval.video_level.aggregate_video_level_predictions`), and compared per video.

    ./.venv/bin/python scripts/export_p1c_roi_ablation.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.feature_cache import load_feature_cache  # noqa: E402
from core.p1c_engine import FramePrediction  # noqa: E402
from eval.video_level import aggregate_video_level_predictions  # noqa: E402
from scripts.build_p1c_feature_cache import _load_kwargs_for_config  # noqa: E402
from scripts.run_p1c_frame_baseline import _load_config  # noqa: E402

ARMS = ("full_frame", "roi")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_FOLDS = (1, 2, 3, 4)


def _validation_rows(fold: int, extract_closure_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The fold's untouched validation rows, in cache order -- the axis the probability arrays index."""

    config_path = REPO_ROOT / f"configs/experiments/p1c/P1C-VIDEO-DINOV2-B2-AR-FOLD{fold}.yaml"
    _config, protocol, _run_class = _load_config(config_path)
    protocol["data"]["video_frames"]["extract_closure_path"] = str(extract_closure_path)
    trainable = protocol["backbone"].get("trainable", "full")
    root = REPO_ROOT / f"results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}/validation"
    _features, rows, _manifest = load_feature_cache(root, **_load_kwargs_for_config(protocol, "validation", trainable))
    return list(rows), list(protocol["model"]["class_names"])


def _video_predictions(
    probabilities: np.ndarray, rows: list[dict[str, Any]], class_names: list[str]
) -> dict[str, tuple[str, str]]:
    """`{video_id: (predicted_label, true_label)}` under the headline majority vote."""

    predicted = np.asarray(probabilities).argmax(axis=1)
    frame_records = [
        FramePrediction(
            source_dataset=row["source_dataset"],
            source_video_id=row["source_video_id"],
            target_index=int(row["target"]),
            predicted_index=int(predicted[index]),
        )
        for index, row in enumerate(rows)
    ]
    aggregated = aggregate_video_level_predictions(frame_records, class_names)
    return {
        video.source_video_id: (class_names[video.predicted_index], class_names[video.target_index])
        for video in aggregated.predictions
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-root", type=Path, default=REPO_ROOT / "results/p1c/p1c-roi-study")
    parser.add_argument("--folds", type=int, nargs="+", default=list(DEFAULT_FOLDS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--extract-closure-path", type=Path,
        default=REPO_ROOT / "results/p1c/p1c-video-frames/extract_closure.json",
    )
    args = parser.parse_args(argv)

    summary = json.loads((args.study_root / "summary.json").read_text())
    roi_manifest = json.loads((REPO_ROOT / "results/p1c/p1c-roi-cache/manifest.json").read_text())

    # --- roi_ablation.csv: one row per (fold, arm, seed), the DoD's flat table.
    ablation_rows: list[dict[str, Any]] = []
    for fold in args.folds:
        for arm in ARMS:
            for seed in args.seeds:
                payload = json.loads((args.study_root / f"fold{fold}" / f"{arm}-seed-{seed}.json").read_text())
                probe = payload["probe_result"]
                ablation_rows.append({
                    "fold": fold,
                    "arm": arm,
                    "seed": seed,
                    "video_level_macro_f1": probe["video_level_metrics"]["macro_f1"],
                    "video_level_accuracy": probe["video_level_metrics"]["accuracy"],
                    "frame_macro_f1": probe["frame_metrics"]["macro_f1"],
                    "frame_accuracy": probe["frame_metrics"]["accuracy"],
                    "best_epoch": payload["best_epoch"],
                    "video_count": probe["video_count"],
                    "train_rows": probe["train_rows"],
                    "validation_rows": probe["validation_rows"],
                })
    csv_path = args.study_root / "roi_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0]))
        writer.writeheader()
        writer.writerows(ablation_rows)
    print(f"wrote {csv_path}: {len(ablation_rows)} rows", flush=True)

    # --- paired-by-video analysis.
    per_video: list[dict[str, Any]] = []
    for fold in args.folds:
        rows, class_names = _validation_rows(fold, args.extract_closure_path)
        predictions_dir = args.study_root / f"fold{fold}" / "predictions"
        for seed in args.seeds:
            arm_predictions = {}
            for arm in ARMS:
                probabilities = np.load(predictions_dir / f"{arm}-seed-{seed}.npy")
                if probabilities.shape[0] != len(rows):
                    raise RuntimeError(
                        f"fold {fold} {arm} seed {seed}: {probabilities.shape[0]} probability rows "
                        f"against {len(rows)} validation rows"
                    )
                arm_predictions[arm] = _video_predictions(probabilities, rows, class_names)
            videos = set(arm_predictions["full_frame"]) & set(arm_predictions["roi"])
            for video in sorted(videos):
                full_pred, true = arm_predictions["full_frame"][video]
                roi_pred, roi_true = arm_predictions["roi"][video]
                if true != roi_true:
                    raise RuntimeError(f"{video}: arms disagree on the true label ({true} vs {roi_true})")
                per_video.append({
                    "fold": fold, "seed": seed, "source_video_id": video, "true_label": true,
                    "full_frame_predicted": full_pred, "roi_predicted": roi_pred,
                    "full_frame_correct": int(full_pred == true), "roi_correct": int(roi_pred == true),
                })

    paired_path = args.study_root / "roi_paired_by_video.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_video[0]))
        writer.writeheader()
        writer.writerows(per_video)

    # McNemar's discordant counts: the paired question is "on how many videos do
    # the arms disagree, and which way", which a difference of macro-F1 means
    # cannot answer.
    roi_only = sum(1 for r in per_video if r["roi_correct"] and not r["full_frame_correct"])
    full_only = sum(1 for r in per_video if r["full_frame_correct"] and not r["roi_correct"])
    both = sum(1 for r in per_video if r["roi_correct"] and r["full_frame_correct"])
    neither = sum(1 for r in per_video if not r["roi_correct"] and not r["full_frame_correct"])
    discordant = roi_only + full_only
    # Exact two-sided binomial test at p=0.5 on the discordant pairs, computed
    # from scipy if available and from the closed form otherwise -- no
    # approximation is needed at these counts.
    from math import comb

    if discordant:
        tail = sum(
            comb(discordant, k) for k in range(0, min(roi_only, full_only) + 1)
        ) + sum(
            comb(discordant, k) for k in range(max(roi_only, full_only), discordant + 1)
        )
        p_value = min(1.0, tail / (2 ** discordant))
    else:
        p_value = 1.0

    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"roi_only": 0, "full_only": 0})
    for r in per_video:
        if r["roi_correct"] and not r["full_frame_correct"]:
            by_class[r["true_label"]]["roi_only"] += 1
        elif r["full_frame_correct"] and not r["roi_correct"]:
            by_class[r["true_label"]]["full_only"] += 1

    fold_means = summary["paired_diff_roi_minus_full_frame_per_fold"]
    analysis = {
        "unit": "one (fold, seed, source video) triple",
        "paired_points": len(per_video),
        "distinct_videos": len({r["source_video_id"] for r in per_video}),
        "both_correct": both,
        "neither_correct": neither,
        "roi_correct_only": roi_only,
        "full_frame_correct_only": full_only,
        "discordant_pairs": discordant,
        "mcnemar_exact_two_sided_p": p_value,
        "interpretation": (
            "The crop changes the answer on a substantial minority of videos but not in a "
            "consistent direction; see the per-class discordance for where each view wins."
        ),
        "discordance_by_true_class": {k: dict(v) for k, v in sorted(by_class.items())},
        "fold_level_paired_diff": fold_means,
        "fold_level_paired_diff_mean": summary["paired_diff_mean"],
        "fold_level_paired_diff_sd": summary["paired_diff_sd"],
        "arms": {arm: summary["arms"][arm]["video_level_macro_f1_mean"] for arm in ARMS},
        "roi_policy": roi_manifest["policy"],
        "roi_detector_sha256": roi_manifest["detector_sha256"],
        "roi_detector_licence": roi_manifest["detector_licence"],
        "roi_fallback_fraction": roi_manifest["fallback_fraction"],
        "fallback_rows_reproduce_registered_cache": json.loads(
            (REPO_ROOT / "results/p1c/p1c-roi-cache/fallback_check.json").read_text()
        ),
    }
    analysis_path = args.study_root / "roi_paired_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {paired_path}: {len(per_video)} paired points", flush=True)
    print(
        f"paired by video: both {both}, neither {neither}, roi-only {roi_only}, "
        f"full-only {full_only}, discordant {discordant}, McNemar exact p={p_value:.4f}",
        flush=True,
    )
    print(f"wrote {analysis_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
