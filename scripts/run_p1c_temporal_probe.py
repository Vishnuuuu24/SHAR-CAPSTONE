#!/usr/bin/env python3
"""Train the registered linear head on temporal-window features, on CPU.

Thin CLI. Builds `core.p1c_temporal_features.build_temporal_features` on top
of a loaded B-1/B-2 feature cache (`--variant`, `--radius`) and trains the
same CONFIRM-015 head `scripts/run_p1c_cached_head.py` trains, via
`core.p1c_head_training.train_cached_head`. That function already scores
through `core.p1c_probes.score_predictions`, which reports BOTH frame-level
metrics (`probe_result.frame_metrics`) and the video-level majority-vote
metrics (`probe_result.video_level_metrics`) -- this script prints both and
writes no new metric code, per the spec.

`--frames-per-video` (optional) applies `core.p1c_video_sampling` to TRAIN
rows before building temporal windows, exactly as `run_p1c_sampled_head.py`
does and for the same reason: VALIDATION stays full and deterministic so the
reported metric keeps meaning what every earlier number in
`docs/TRAINING_LOG.md` meant. Sampling happens on rows/features BEFORE the
temporal window is built, not after -- windows are built per surviving video
span from the sampled cache, so a window still only ever looks at frames that
belong to the same video (never at a different video's frames, and never at a
frame excluded from that video's own sample).

Resolves cache identity through
`scripts.build_p1c_feature_cache._load_kwargs_for_config`, same as
`run_p1c_cached_head.py`, and shares its `--extract-closure-path` escape
hatch for the same reason.

No GPU. Run under `taskset -c 0-3` with the BLAS thread environment pinned:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
    NUMEXPR_NUM_THREADS=4 taskset -c 0-3 \\
    ./.venv/bin/python scripts/run_p1c_temporal_probe.py --config ... --cache-root ... \\
        --variant concat --radius 2

Mind the memory cost documented in `core/p1c_temporal_features`'s module
docstring before choosing `concat` at full fold-1 scale -- combine with
`--frames-per-video` to cut `N` first if it does not fit.

Scope is inherited from the cache and is not negotiable here: whichever AR
fold and split the cache was built for, inherited video-level labels (D-40,
docs/PROJECT_RULES.md rule 1). Test is never touched.
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
from core.p1c_temporal_features import TEMPORAL_VARIANTS, build_temporal_features
from core.p1c_video_sampling import sample_rows_per_video
from scripts.build_p1c_feature_cache import _load_kwargs_for_config
from scripts.run_p1c_frame_baseline import _load_config

DEFAULT_SEEDS = (0, 1, 2)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the registered config the cache was built from")
    parser.add_argument("--cache-root", type=Path, required=True, help="directory holding train/ and validation/ caches")
    parser.add_argument("--variant", choices=TEMPORAL_VARIANTS, required=True)
    parser.add_argument(
        "--window-before-sampling", action="store_true",
        help="cut temporal windows on the full cache BEFORE thinning rows, so windows stay local "
             "(see the comment in main); default keeps the original sample-then-window order",
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--frames-per-video", type=int, default=None,
                        help="optionally cap TRAIN rows per source video before building windows; validation is untouched")
    parser.add_argument("--sampling-seed", type=int, default=None,
                        help="seed the TRAIN sampler's within-segment jitter (only used with --frames-per-video)")
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

    # ORDER MATTERS, and the first temporal probe got it wrong. Sampling first
    # thins each video before windows are cut, so `t-1` and `t+1` are no longer
    # the neighbouring frames -- at 16 frames from a median 446-frame video the
    # retained rows sit ~28 apart, and a radius-2 "window" then spans most of the
    # video instead of ~0.7 s of it. That tests a different hypothesis than the
    # one B-12 asks about.
    #
    # `--window-before-sampling` builds windows on the FULL cache first, so every
    # window is genuinely local, and only then thins the rows -- keeping the
    # cost saving without destroying adjacency. The old order is retained as the
    # default so the already-recorded negative result stays reproducible.
    def _thin(features, rows):
        if args.frames_per_video is None:
            return features, rows
        keep = sample_rows_per_video(rows, args.frames_per_video, seed=args.sampling_seed)
        return features[keep], [rows[i] for i in keep]

    if args.window_before_sampling:
        train_windowed = build_temporal_features(train_features, train_rows, args.variant, radius=args.radius)
        train_windowed, train_rows = _thin(train_windowed, train_rows)
    else:
        train_features, train_rows = _thin(train_features, train_rows)
        train_windowed = build_temporal_features(train_features, train_rows, args.variant, radius=args.radius)
    validation_windowed = build_temporal_features(
        validation_features, validation_rows, args.variant, radius=args.radius
    )
    print(
        f"train {len(train_rows)} rows (variant={args.variant}, radius={args.radius}"
        + (f", sampled {args.frames_per_video}/video" if args.frames_per_video is not None else "")
        + f") -> dim {train_windowed.shape[1]}, validation {len(validation_rows)} rows (full, untouched), "
        f"cache {train_manifest.cache_sha256[:12]}",
        flush=True,
    )

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        print(f"\n--- seed {seed} ---", flush=True)
        result = train_cached_head(
            train_windowed, train_rows, validation_windowed, validation_rows, class_names,
            seed=seed, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            patience=args.patience, min_delta=args.min_delta,
            bootstrap_iterations=args.bootstrap_iterations, bootstrap_seed=args.bootstrap_seed,
            capture_validation_probabilities=args.capture_probabilities,
        )
        payload: dict[str, Any] = result.to_json() | {
            "cache_sha256": train_manifest.cache_sha256,
            "config": str(args.config),
            "temporal_variant": args.variant,
            "temporal_radius": args.radius,
            "frames_per_video": args.frames_per_video,
            "sampling_seed": args.sampling_seed,
        }
        probe = payload["probe_result"]
        ci = probe["video_clustered_macro_f1_ci"]
        print(
            f"best epoch {result.best_epoch} ({'early-stopped' if result.stopped_early else 'epoch ceiling'}), "
            f"frame macro-F1 {probe['frame_metrics']['macro_f1']:.4f} "
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
