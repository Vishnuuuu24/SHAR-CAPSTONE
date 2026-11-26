#!/usr/bin/env python3
"""Train the registered linear head on a B-1/B-2 feature cache, on CPU.

Thin CLI. It resolves the cache identity through
`scripts.build_p1c_feature_cache._load_kwargs_for_config` -- the same single
definition the builder, the verifier and `scripts/run_p1c_linear_probes.py`
all use, so this trainer cannot be served a cache built under a different
backbone, preprocessing or manifest -- and hands the loaded features to
`core.p1c_head_training.train_cached_head`, which trains and scores the
registered CONFIRM-015 recipe.

No GPU. Run under `taskset -c 0-3` with the BLAS thread environment pinned to
match, exactly as `scripts/run_p1c_linear_probes.py` documents:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
    NUMEXPR_NUM_THREADS=4 taskset -c 0-3 \\
    ./.venv/bin/python scripts/run_p1c_cached_head.py --config ... --cache-root ...

`--extract-closure-path` exists for the same reason it does on
`run_p1c_linear_probes.py`: a config registers the pod emptyDir path a
*training* run reads pixels from, and that path does not exist off-pod. This
trainer reads no pixels either -- it needs the closure only for the blob
SHA-256 that identifies the cache -- and the durable copy lives at
`results/p1c/p1c-video-frames/extract_closure.json`.

Scope is inherited from the cache and is not negotiable here: whichever AR
fold and split the cache was built for, inherited video-level labels (D-40,
docs/PROJECT_RULES.md rule 1). Test is never touched.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.feature_cache import load_feature_cache
from core.p1c_head_training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_DELTA,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    train_cached_head,
)
from scripts.build_p1c_feature_cache import _load_kwargs_for_config
from scripts.run_p1c_frame_baseline import _load_config

DEFAULT_SEEDS = (0, 1, 2)
_SUMMARY_METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the registered config the cache was built from")
    parser.add_argument("--cache-root", type=Path, required=True, help="directory holding train/ and validation/ caches")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=None, help="directory to write one JSON per seed plus an aggregate into")
    parser.add_argument(
        "--extract-closure-path", type=Path, default=None,
        help="read the D-34/D-35 video-blob closure from here instead of the path the config "
             "registers (see module docstring)",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args(argv)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def _aggregate(results: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    """Mean/sd across seeds. A separate, lighter definition from
    `core.p1c_aggregation.aggregate_validation_seeds`: that function requires
    a `selected_checkpoint` file path from a lifecycle-managed pixel run, which
    this CPU cached-feature trainer never writes -- there is no torch
    checkpoint on disk to name."""

    frame_metrics = [result["probe_result"]["frame_metrics"] for result in results]
    video_metrics = [result["probe_result"]["video_level_metrics"] for result in results]
    metric_summary = {name: _summary([m[name] for m in frame_metrics]) for name in _SUMMARY_METRICS}
    video_macro_f1 = _summary([m["macro_f1"] for m in video_metrics])
    per_class: dict[str, dict[str, float]] = {}
    for label in class_names:
        per_class[label] = _summary([m["per_class"][label]["f1"] for m in frame_metrics])
    return {
        "seeds": [result["seed"] for result in results],
        "frame_macro_f1": metric_summary["macro_f1"],
        "frame_metrics": metric_summary,
        "video_level_macro_f1": video_macro_f1,
        "per_class_frame_f1": per_class,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _config, protocol, _run_class = _load_config(args.config)
    trainable = protocol["backbone"].get("trainable", "full")
    class_names = protocol["model"]["class_names"]
    if args.extract_closure_path is not None:
        protocol["data"]["video_frames"]["extract_closure_path"] = str(args.extract_closure_path)

    train_features, train_rows, train_manifest = load_feature_cache(
        args.cache_root / "train", **_load_kwargs_for_config(protocol, "train", trainable)
    )
    validation_features, validation_rows, _ = load_feature_cache(
        args.cache_root / "validation", **_load_kwargs_for_config(protocol, "validation", trainable)
    )
    print(
        f"train {len(train_rows)} rows, validation {len(validation_rows)} rows, "
        f"dim {train_manifest.feature_dim}, cache {train_manifest.cache_sha256[:12]}",
        flush=True,
    )

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)

    seed_reports: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"\n--- seed {seed} ---", flush=True)
        result = train_cached_head(
            train_features, train_rows, validation_features, validation_rows, class_names,
            seed=seed, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            patience=args.patience, min_delta=args.min_delta,
            bootstrap_iterations=args.bootstrap_iterations, bootstrap_seed=args.bootstrap_seed,
        )
        payload = result.to_json() | {"cache_sha256": train_manifest.cache_sha256, "config": str(args.config)}
        seed_reports.append(payload)
        probe = payload["probe_result"]
        ci = probe["video_clustered_macro_f1_ci"]
        print(
            f"best epoch {result.best_epoch} ({'early-stopped' if result.stopped_early else 'epoch ceiling'}), "
            f"validation macro-F1 {result.best_validation_macro_f1:.4f} "
            f"(clustered 95% CI {ci['lower_95']:.4f}-{ci['upper_95']:.4f}), "
            f"video-level macro-F1 {probe['video_level_metrics']['macro_f1']:.4f} "
            f"over {probe['video_count']} videos, train {result.train_seconds:.1f}s",
            flush=True,
        )
        if args.output is not None:
            path = args.output / f"seed-{seed}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote {path}", flush=True)

    if len(seed_reports) > 1:
        aggregate = _aggregate(seed_reports, class_names)
        summary = aggregate["frame_macro_f1"]
        print(
            f"\n=== aggregate over seeds {aggregate['seeds']} ===\n"
            f"frame macro-F1 {summary['mean']:.4f} +/- {summary['sample_std']:.4f} "
            f"(min {summary['minimum']:.4f}, max {summary['maximum']:.4f})",
            flush=True,
        )
        if args.output is not None:
            path = args.output / "aggregate.json"
            path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
