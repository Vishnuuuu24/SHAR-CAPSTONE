#!/usr/bin/env python3
"""Fix #1 (backlog B-6): frame vs video-level evaluation of the cached linear head.

This is an *evaluation* addition, not a new model. It trains the registered
CONFIRM-015 linear head on a B-1/B-2 feature cache exactly as
`scripts/run_p1c_cached_head.py` does -- same recipe, same
`core.p1c_head_training.train_cached_head` -- but additionally captures the
best epoch's per-frame validation probabilities and reports THREE scores side
by side:

    frame-level            : every validation frame scored independently
    video majority vote    : one class per video, majority of per-frame argmax
    video mean probability  : one class per video, argmax of the mean softmax

Frame-level and majority-vote come straight from `train_cached_head`'s existing
`probe_result` (it already runs `eval.metrics` and `eval.video_level`); the
mean-probability score is `eval.video_probability_aggregation`, new in B-6.

Every video-level number here counts each source video once, so it is immune to
the 4.48x frame-count imbalance that videos-differ-in-length produces. Scope is
inherited from the cache and not negotiable: whichever AR fold/split the cache
was built for, inherited video-level labels (D-40), Test never touched.

CPU only. Run exactly as `scripts/run_p1c_cached_head.py` documents:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
    NUMEXPR_NUM_THREADS=4 taskset -c 4-7 \\
    ./.venv/bin/python scripts/eval_p1c_video_aggregation.py --config ... --cache-root ...
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.feature_cache import load_feature_cache
from core.p1c_engine import FramePrediction
from core.p1c_head_training import train_cached_head
from eval.video_probability_aggregation import aggregate_video_level_probabilities
from scripts.build_p1c_feature_cache import _load_kwargs_for_config
from scripts.run_p1c_frame_baseline import _load_config

DEFAULT_SEEDS = (0, 1, 2)
_SUMMARY_KEYS = ("frame", "video_majority_vote", "video_mean_probability")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True, help="dir with train/ and validation/ caches")
    parser.add_argument("--output", type=Path, required=True, help="dir for per-seed JSON, npy, confusion matrices")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--extract-closure-path", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser.parse_args(argv)


def _frame_records(rows: list[dict[str, Any]]) -> list[FramePrediction]:
    # predicted_index is filled with the target here; mean-prob recomputes its
    # own per-video argmax and never reads this field.
    return [
        FramePrediction(
            source_dataset=row["source_dataset"],
            source_video_id=row["source_video_id"],
            target_index=int(row["target"]),
            predicted_index=int(row["target"]),
        )
        for row in rows
    ]


def _render_confusion(matrix: list[list[int]], labels: list[str]) -> str:
    width = max(len(label) for label in labels)
    header = " " * (width + 2) + " ".join(f"{i:>5}" for i in range(len(labels)))
    lines = [header, f"{'':>{width}}  (columns = predicted index)"]
    for i, (label, row) in enumerate(zip(labels, matrix, strict=True)):
        lines.append(f"{label:>{width}} {i:>1} " + " ".join(f"{value:>5}" for value in row))
    lines.append("")
    lines.append("index -> label: " + ", ".join(f"{i}={label}" for i, label in enumerate(labels)))
    return "\n".join(lines)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _config, protocol, _run_class = _load_config(args.config)
    trainable = protocol["backbone"].get("trainable", "full")
    class_names = list(protocol["model"]["class_names"])
    if args.extract_closure_path is not None:
        protocol["data"]["video_frames"]["extract_closure_path"] = str(args.extract_closure_path)

    train_features, train_rows, train_manifest = load_feature_cache(
        args.cache_root / "train", **_load_kwargs_for_config(protocol, "train", trainable)
    )
    validation_features, validation_rows, _ = load_feature_cache(
        args.cache_root / "validation", **_load_kwargs_for_config(protocol, "validation", trainable)
    )
    validation_records = _frame_records(validation_rows)
    print(
        f"train {len(train_rows)} rows, validation {len(validation_rows)} rows, "
        f"dim {train_manifest.feature_dim}, cache {train_manifest.cache_sha256[:12]}, "
        f"backbone {train_manifest.backbone_family}",
        flush=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict[str, Any]] = []

    for seed in args.seeds:
        print(f"\n--- seed {seed} ---", flush=True)
        result = train_cached_head(
            train_features, train_rows, validation_features, validation_rows, class_names,
            seed=seed, bootstrap_iterations=args.bootstrap_iterations,
            capture_validation_probabilities=True,
        )
        probs = np.asarray(result.best_validation_probabilities, dtype=np.float64)
        np.save(args.output / f"seed-{seed}-validation-probabilities.npy", probs.astype(np.float32))

        frame_metrics = result.probe_result.frame_metrics
        majority = result.probe_result.video_level_metrics
        mean_prob = aggregate_video_level_probabilities(validation_records, probs, class_names)

        seed_payload = {
            "seed": seed,
            "cache_sha256": train_manifest.cache_sha256,
            "backbone_family": train_manifest.backbone_family,
            "config": str(args.config),
            "best_epoch": result.best_epoch,
            "stopped_early": result.stopped_early,
            "frame": frame_metrics,
            "video_majority_vote": majority,
            "video_mean_probability": mean_prob.metrics
            | {
                "video_count": mean_prob.video_count,
                "low_margin_video_count": mean_prob.low_margin_video_count,
                "low_margin_threshold": mean_prob.low_margin_threshold,
            },
        }
        (args.output / f"seed-{seed}.json").write_text(
            json.dumps(seed_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output / f"seed-{seed}-confusion-frame.txt").write_text(
            _render_confusion(frame_metrics["confusion_matrix"], class_names), encoding="utf-8"
        )
        (args.output / f"seed-{seed}-confusion-video-meanprob.txt").write_text(
            _render_confusion(mean_prob.metrics["confusion_matrix"], class_names), encoding="utf-8"
        )
        (args.output / f"seed-{seed}-confusion-video-majorityvote.txt").write_text(
            _render_confusion(majority["confusion_matrix"], class_names), encoding="utf-8"
        )
        per_seed.append(seed_payload)
        print(
            f"  frame macro-F1 {frame_metrics['macro_f1']:.4f} | "
            f"video majority-vote {majority['macro_f1']:.4f} | "
            f"video mean-prob {mean_prob.metrics['macro_f1']:.4f} "
            f"(low-margin videos {mean_prob.low_margin_video_count}/{mean_prob.video_count})",
            flush=True,
        )

    aggregate: dict[str, Any] = {"seeds": [p["seed"] for p in per_seed], "class_names": class_names}
    for key in _SUMMARY_KEYS:
        aggregate[key] = {
            "macro_f1": _summary([p[key]["macro_f1"] for p in per_seed]),
            "accuracy": _summary([p[key]["accuracy"] for p in per_seed]),
            "per_class_f1": {
                label: _summary([p[key]["per_class"][label]["f1"] for p in per_seed])
                for label in class_names
            },
        }
    (args.output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if len(per_seed) > 1:
        print("\n=== aggregate over seeds", aggregate["seeds"], "===", flush=True)
        for key in _SUMMARY_KEYS:
            s = aggregate[key]["macro_f1"]
            print(f"  {key:<24} macro-F1 {s['mean']:.4f} +/- {s['sample_std']:.4f} "
                  f"(min {s['minimum']:.4f}, max {s['maximum']:.4f})", flush=True)
        motion = [c for c in ("Fighting", "Assault", "Shooting", "Vandalism") if c in class_names]
        print("  motion classes (frame -> video mean-prob), per-class F1 mean:", flush=True)
        for label in motion:
            f = aggregate["frame"]["per_class_f1"][label]["mean"]
            v = aggregate["video_mean_probability"]["per_class_f1"][label]["mean"]
            print(f"    {label:<14} {f:.3f} -> {v:.3f}", flush=True)
    print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
