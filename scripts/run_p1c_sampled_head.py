#!/usr/bin/env python3
"""Train the registered linear head on a per-video-sampled feature cache, on CPU.

Thin CLI, identical to `scripts/run_p1c_cached_head.py` except that TRAIN rows
are first thinned with `core.p1c_video_sampling.sample_rows_per_video`
(`--frames-per-video`, `--sampling-seed`). VALIDATION is deliberately left
full and deterministic and is never sampled: changing which validation rows
get scored would change what the reported macro-F1 *means*, and every earlier
number in `docs/TRAINING_LOG.md` was measured against the full validation
split -- sampling it would silently break comparability with all of them.
Sampling here is a training-set design choice (does capping per-video frame
count help or hurt against a fixed, unmoving yardstick), not an evaluation
change.

Resolves cache identity through
`scripts.build_p1c_feature_cache._load_kwargs_for_config`, same as
`run_p1c_cached_head.py`, so this cannot be served a cache built under a
different backbone, preprocessing or manifest, and shares its
`--extract-closure-path` escape hatch for the same reason (a training config
registers a pod emptyDir pixel path that does not exist off-pod; this script
reads no pixels, only the closure's blob SHA-256).

No GPU. Run under `taskset -c 0-3` with the BLAS thread environment pinned:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
    NUMEXPR_NUM_THREADS=4 taskset -c 0-3 \\
    ./.venv/bin/python scripts/run_p1c_sampled_head.py --config ... --cache-root ... \\
        --frames-per-video 50

Scope is inherited from the cache and is not negotiable here: whichever AR
fold and split the cache was built for, inherited video-level labels (D-40,
docs/PROJECT_RULES.md rule 1). Test is never touched.

`--background-mask-label LABEL` (default off) loads the official temporal
intervals and zero-weights known-background rows of that label in the
already-sampled TRAIN rows via `core.p1c_background_mask.background_weights`
-- see that module's docstring for exactly what "known background" means and
its stated limitation (no negative signal, only removed positive signal).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.feature_cache import load_feature_cache
from core.p1c_background_mask import background_weight_counts, background_weights, load_intervals
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
from core.p1c_video_sampling import sample_rows_per_video
from scripts.build_p1c_feature_cache import _load_kwargs_for_config
from scripts.run_p1c_frame_baseline import _load_config

DEFAULT_SEEDS = (0, 1, 2)

#: Same official annotation file `scripts/diagnose_p1c_event_separability.py` reads.
ANNOTATION = Path(
    "Datasets/UCF Crime Official/annotations/"
    "Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt"
)


def _parse_frames_per_video_for(entries: list[str]) -> dict[str, int]:
    """`["Shooting=32", ...]` -> `{"Shooting": 32}`, for `--frames-per-video-for`."""

    mapping: dict[str, int] = {}
    for entry in entries:
        label, sep, value = entry.partition("=")
        if not sep or not label:
            raise ValueError(f"--frames-per-video-for expects LABEL=N; got {entry!r}")
        mapping[label] = int(value)
    return mapping


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the registered config the cache was built from")
    parser.add_argument("--cache-root", type=Path, required=True, help="directory holding train/ and validation/ caches")
    parser.add_argument("--frames-per-video", type=int, required=True,
                        help="cap TRAIN rows per source video (core.p1c_video_sampling); validation is untouched")
    parser.add_argument("--sampling-seed", type=int, default=None,
                        help="seed the TRAIN sampler's within-segment jitter; omit for deterministic evenly-spaced frames")
    parser.add_argument(
        "--frames-per-video-for", action="append", default=[], metavar="LABEL=N",
        help="override --frames-per-video for one label (core.p1c_video_sampling); repeatable",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="training seeds for the head itself (independent of --sampling-seed)")
    parser.add_argument("--output", type=Path, default=None, help="directory to write one JSON per seed into")
    parser.add_argument(
        "--extract-closure-path", type=Path, default=None,
        help="read the D-34/D-35 video-blob closure from here instead of the path the config registers",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--capture-probabilities", action="store_true",
        help="also capture the best epoch's per-frame validation probabilities and write them to "
             "--output/seed-<s>-validation-probabilities.npy, as scripts/eval_p1c_video_aggregation.py does",
    )
    parser.add_argument(
        "--background-mask-label", default=None, metavar="LABEL",
        help="zero-weight known-background TRAIN rows of this label using the official temporal "
             "intervals (core.p1c_background_mask); default off, every row keeps weight 1",
    )
    return parser.parse_args(argv)


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

    frames_per_video_by_label = _parse_frames_per_video_for(args.frames_per_video_for)
    keep = sample_rows_per_video(
        train_rows, args.frames_per_video, seed=args.sampling_seed,
        frames_per_video_by_label=frames_per_video_by_label or None,
    )
    train_features = train_features[keep]
    train_rows = [train_rows[i] for i in keep]
    print(
        f"train {len(train_rows)} rows sampled ({args.frames_per_video}/video, "
        f"per-label overrides {frames_per_video_by_label or 'none'}, "
        f"sampling_seed={args.sampling_seed}), validation {len(validation_rows)} rows (full, untouched), "
        f"dim {train_manifest.feature_dim}, cache {train_manifest.cache_sha256[:12]}",
        flush=True,
    )

    train_weights = None
    background_mask_counts = None
    if args.background_mask_label is not None:
        intervals = load_intervals(ANNOTATION)
        train_weights = background_weights(train_rows, intervals, label=args.background_mask_label)
        background_mask_counts = background_weight_counts(train_rows, intervals, label=args.background_mask_label)
        print(
            f"background mask label={args.background_mask_label}: "
            f"{background_mask_counts['annotated_videos']} annotated videos, "
            f"{background_mask_counts['rows_zeroed']} rows zeroed, "
            f"{background_mask_counts['rows_kept']} rows kept (of {len(train_rows)} sampled train rows)",
            flush=True,
        )

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        print(f"\n--- seed {seed} ---", flush=True)
        result = train_cached_head(
            train_features, train_rows, validation_features, validation_rows, class_names,
            seed=seed, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            patience=args.patience, min_delta=args.min_delta,
            bootstrap_iterations=args.bootstrap_iterations, bootstrap_seed=args.bootstrap_seed,
            capture_validation_probabilities=args.capture_probabilities,
            train_weights=train_weights,
        )
        payload: dict[str, Any] = result.to_json() | {
            "cache_sha256": train_manifest.cache_sha256,
            "config": str(args.config),
            "frames_per_video": args.frames_per_video,
            "frames_per_video_by_label": frames_per_video_by_label,
            "sampling_seed": args.sampling_seed,
            "sampled_train_rows": len(train_rows),
            "background_mask_label": args.background_mask_label,
            "background_mask_counts": background_mask_counts,
        }
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
            if args.capture_probabilities:
                probs_path = args.output / f"seed-{seed}-validation-probabilities.npy"
                np.save(probs_path, np.asarray(result.best_validation_probabilities, dtype=np.float32))
                print(f"wrote {probs_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
