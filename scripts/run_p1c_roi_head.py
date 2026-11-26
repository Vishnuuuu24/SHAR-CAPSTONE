#!/usr/bin/env python3
"""ROI ablation: train the registered linear head on full-frame vs ROI-crop features (CPU).

Single-factor comparison for the ROI study. For each of the four AR folds:
load the registered DINOv2 feature cache's train/validation rows exactly as
`scripts/run_p1c_sampled_head.py` does, sample TRAIN 8 frames/video with
`sample_rows_per_video(rows, 8, seed=0)`, then build TWO feature matrices for
the SAME rows (train sample + full validation): `full_frame` is the
registered cache's own features, `roi` is the matching person-crop features
from the ROI cache `scripts/build_p1c_roi_cache.py` wrote. Both arms are
trained with `core.p1c_head_training.train_cached_head` defaults, at seeds
0, 1, 2 -- everything held equal except which feature matrix is fed in, so
any difference is attributable to the crop alone.

The ROI store is looked up by `(source_video_id, source_frame_index)`, never
by row position -- the registered cache and the ROI cache were built in
different row orders. A missing key is a hard failure: docs/PROJECT_RULES.md's
reproducibility rule means every video-level majority vote must be a
majority over the SAME frames on both arms, so a silently-dropped row would
make the two arms' training and validation sets disagree without saying so.

No GPU. Run under the same CPU pinning as `run_p1c_sampled_head.py`:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
    NUMEXPR_NUM_THREADS=4 taskset -c 0-3 \\
    ./.venv/bin/python scripts/run_p1c_roi_head.py --roi-cache-root results/p1c/p1c-roi-cache
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.feature_cache import load_feature_cache  # noqa: E402
from core.p1c_head_training import train_cached_head  # noqa: E402
from core.p1c_video_sampling import sample_rows_per_video  # noqa: E402
from scripts.build_p1c_feature_cache import _load_kwargs_for_config  # noqa: E402
from scripts.run_p1c_frame_baseline import _load_config  # noqa: E402

DEFAULT_FOLDS = (1, 2, 3, 4)
DEFAULT_SEEDS = (0, 1, 2)
FRAMES_PER_VIDEO = 8
SAMPLING_SEED = 0
ARMS = ("full_frame", "roi")


def _config_path(fold: int) -> Path:
    return Path(f"configs/experiments/p1c/P1C-VIDEO-DINOV2-B2-AR-FOLD{fold}.yaml")


def _feature_cache_root(fold: int) -> Path:
    return Path(f"results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}")


def _load_roi_features(roi_cache_root: Path) -> tuple[np.ndarray, dict[tuple[str, int], int], list[dict[str, Any]]]:
    """`(features, key -> roi row index, roi rows)` for the ROI cache at `roi_cache_root`."""

    features = np.load(roi_cache_root / "features.float16.npy", mmap_mode="r")
    with (roi_cache_root / "rows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    key_to_index = {
        (row["source_video_id"], int(row["source_frame_index"])): index for index, row in enumerate(rows)
    }
    if len(key_to_index) != len(rows):
        raise ValueError(f"{roi_cache_root}/rows.csv has duplicate (video, frame) keys")
    return features, key_to_index, rows


def _roi_matrix_for(
    rows: list[dict[str, Any]],
    roi_features: np.ndarray,
    key_to_index: dict[tuple[str, int], int],
) -> tuple[np.ndarray, float]:
    """The ROI feature matrix aligned 1:1 with `rows`, plus the fallback fraction among them.

    Hard-fails (per the ROI brief) if any row's `(source_video_id,
    source_frame_index)` is missing from the ROI store.
    """

    missing = [
        (row["source_video_id"], int(row["source_frame_index"])) for row in rows
        if (row["source_video_id"], int(row["source_frame_index"])) not in key_to_index
    ]
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(rows)} rows are missing from the ROI store, e.g. {missing[:5]}"
        )
    indices = [key_to_index[(row["source_video_id"], int(row["source_frame_index"]))] for row in rows]
    return np.asarray(roi_features[indices], dtype=np.float16), float(len(indices))


def _run_fold(
    fold: int,
    *,
    roi_cache_root: Path,
    roi_features: np.ndarray,
    roi_key_to_index: dict[tuple[str, int], int],
    roi_rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
    output_root: Path,
    extract_closure_path: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    _config, protocol, _run_class = _load_config(_config_path(fold))
    if extract_closure_path is not None:
        # Same escape hatch as run_p1c_sampled_head.py: the config registers a
        # pod emptyDir path; off-pod only the closure's blob SHA-256 is read.
        protocol["data"]["video_frames"]["extract_closure_path"] = str(extract_closure_path)
    trainable = protocol["backbone"].get("trainable", "full")
    class_names = protocol["model"]["class_names"]

    cache_root = _feature_cache_root(fold)
    train_features, train_rows, _ = load_feature_cache(
        cache_root / "train", **_load_kwargs_for_config(protocol, "train", trainable)
    )
    validation_features, validation_rows, _ = load_feature_cache(
        cache_root / "validation", **_load_kwargs_for_config(protocol, "validation", trainable)
    )

    keep = sample_rows_per_video(train_rows, FRAMES_PER_VIDEO, seed=SAMPLING_SEED)
    sampled_train_rows = [train_rows[i] for i in keep]
    full_frame_train = train_features[keep]
    full_frame_validation = validation_features

    roi_train, roi_train_fallback_n = _roi_matrix_for(sampled_train_rows, roi_features, roi_key_to_index)
    roi_validation, roi_validation_fallback_n = _roi_matrix_for(validation_rows, roi_features, roi_key_to_index)
    fallback_lookup = {
        (row["source_video_id"], int(row["source_frame_index"])): int(row["roi_fallback"])
        for row in roi_rows
    }
    fallback_count = sum(
        fallback_lookup[(row["source_video_id"], int(row["source_frame_index"]))]
        for row in sampled_train_rows + validation_rows
    )
    fallback_fraction = fallback_count / (len(sampled_train_rows) + len(validation_rows))

    arm_features = {
        "full_frame": (full_frame_train, full_frame_validation),
        "roi": (roi_train, roi_validation),
    }

    fold_output = output_root / f"fold{fold}"
    fold_output.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    # Per-video predictions are what make P3C's DoD "paired downstream analysis"
    # a per-video paired test (224 videos) rather than a per-fold one (12 cells),
    # so they are written beside each run when captured.
    predictions_dir = fold_output / "predictions"
    for arm in ARMS:
        train_x, validation_x = arm_features[arm]
        for seed in seeds:
            result = train_cached_head(
                train_x, sampled_train_rows, validation_x, validation_rows, class_names, seed=seed,
                capture_validation_probabilities=args.capture_probabilities,
            )
            payload = result.to_json() | {
                "fold": fold, "arm": arm, "config": str(_config_path(fold)),
                "roi_cache_root": str(roi_cache_root),
                "frames_per_video": FRAMES_PER_VIDEO, "sampling_seed": SAMPLING_SEED,
            }
            if args.capture_probabilities and result.best_validation_probabilities is not None:
                predictions_dir.mkdir(parents=True, exist_ok=True)
                np.save(
                    predictions_dir / f"{arm}-seed-{seed}.npy",
                    np.asarray(result.best_validation_probabilities, dtype=np.float32),
                )
            path = fold_output / f"{arm}-seed-{seed}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            results[arm][seed] = payload
            print(
                f"fold {fold} arm {arm} seed {seed}: video-level macro-F1 "
                f"{payload['probe_result']['video_level_metrics']['macro_f1']:.4f}, "
                f"frame macro-F1 {payload['probe_result']['frame_metrics']['macro_f1']:.4f}",
                flush=True,
            )

    return {"results": results, "fallback_fraction": fallback_fraction}


def _summarize(
    fold_reports: dict[int, dict[str, Any]], class_names: list[str], seeds: tuple[int, ...]
) -> dict[str, Any]:
    def _metric(fold: int, arm: str, seed: int, kind: str) -> float:
        payload = fold_reports[fold]["results"][arm][seed]
        return payload["probe_result"][f"{kind}_metrics"]["macro_f1"]

    per_arm_summary: dict[str, Any] = {}
    for arm in ARMS:
        per_fold_video = {
            fold: statistics.mean(_metric(fold, arm, seed, "video_level") for seed in seeds)
            for fold in fold_reports
        }
        per_fold_frame = {
            fold: statistics.mean(_metric(fold, arm, seed, "frame") for seed in seeds)
            for fold in fold_reports
        }
        per_class: dict[str, float] = {}
        for label in class_names:
            values = [
                fold_reports[fold]["results"][arm][seed]["probe_result"]["video_level_metrics"]["per_class"]
                .get(label, {}).get("f1", float("nan"))
                for fold in fold_reports for seed in seeds
            ]
            values = [v for v in values if v == v]  # drop NaN (label absent from a fold)
            per_class[label] = statistics.mean(values) if values else None
        per_arm_summary[arm] = {
            "video_level_macro_f1_mean_per_fold": per_fold_video,
            "frame_macro_f1_mean_per_fold": per_fold_frame,
            "video_level_macro_f1_mean": statistics.mean(per_fold_video.values()),
            "video_level_macro_f1_sd": statistics.stdev(per_fold_video.values()) if len(per_fold_video) > 1 else 0.0,
            "per_class_video_level_f1_mean_over_seeds": per_class,
        }

    paired_diff = {
        fold: per_arm_summary["roi"]["video_level_macro_f1_mean_per_fold"][fold]
        - per_arm_summary["full_frame"]["video_level_macro_f1_mean_per_fold"][fold]
        for fold in fold_reports
    }
    fallback_fractions = {fold: fold_reports[fold]["fallback_fraction"] for fold in fold_reports}

    return {
        "arms": per_arm_summary,
        "paired_diff_roi_minus_full_frame_per_fold": paired_diff,
        "paired_diff_mean": statistics.mean(paired_diff.values()),
        "paired_diff_sd": statistics.stdev(paired_diff.values()) if len(paired_diff) > 1 else 0.0,
        "fallback_fraction_per_fold": fallback_fractions,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roi-cache-root", type=Path, default=Path("results/p1c/p1c-roi-cache"))
    parser.add_argument("--folds", type=int, nargs="+", default=list(DEFAULT_FOLDS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=Path("results/p1c/p1c-roi-study"))
    parser.add_argument(
        "--capture-probabilities", action="store_true",
        help="save per-frame validation probabilities per (arm, seed) so the P3C paired analysis "
             "can be computed per source video instead of per fold",
    )
    parser.add_argument("--extract-closure-path", type=Path,
                        default=Path("results/p1c/p1c-video-frames/extract_closure.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    seeds = tuple(args.seeds)

    roi_features, roi_key_to_index, roi_rows = _load_roi_features(args.roi_cache_root)
    print(f"ROI store: {len(roi_rows)} rows from {args.roi_cache_root}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    fold_reports: dict[int, dict[str, Any]] = {}
    class_names: list[str] = []
    for fold in args.folds:
        _config, protocol, _run_class = _load_config(_config_path(fold))
        class_names = protocol["model"]["class_names"]
        fold_reports[fold] = _run_fold(
            fold, roi_cache_root=args.roi_cache_root, roi_features=roi_features,
            roi_key_to_index=roi_key_to_index, roi_rows=roi_rows, seeds=seeds, output_root=args.output,
            extract_closure_path=args.extract_closure_path, args=args,
        )

    summary = _summarize(fold_reports, class_names, seeds)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"summary: full_frame {summary['arms']['full_frame']['video_level_macro_f1_mean']:.4f}, "
        f"roi {summary['arms']['roi']['video_level_macro_f1_mean']:.4f}, "
        f"paired diff {summary['paired_diff_mean']:+.4f} (sd {summary['paired_diff_sd']:.4f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
