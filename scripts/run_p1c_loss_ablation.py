#!/usr/bin/env python3
"""P3B: cross-entropy vs. three imbalance-oriented losses on the validated pipeline, on CPU.

Model of `scripts/run_p1c_sampled_head.py`: same cache loading
(`core.feature_cache.load_feature_cache` through
`scripts.build_p1c_feature_cache._load_kwargs_for_config`), same fixed 8
frames/video TRAIN sampling (`core.p1c_video_sampling.sample_rows_per_video`,
`seed=0` -- the registered D-46 pipeline), VALIDATION left full and untouched.
The one axis this script adds is the loss function
(`core.p1c_losses.LOSS_NAMES`, plumbed through `core.p1c_head_training.
train_cached_head(..., loss=..., focal_gamma=...)`).

Why two of the four arms are pre-registered as vacuous
--------------------------------------------------------
Under `sample_rows_per_video(rows, 8, seed=0)` fold 1 train is EXACTLY
class-balanced (272 rows/class, 14 classes) -- see `core/p1c_losses.py`'s
module docstring for the algebra. `weighted_cross_entropy` and
`balanced_softmax` are therefore provably identical to `cross_entropy` on
that fold; `vacuity_check.json` asserts this empirically rather than only in
prose. `focal` is not vacuous -- see `core/p1c_losses.py`.

No GPU. Run under `taskset -c 0-1` with `OMP_NUM_THREADS=2` pinned:

    OMP_NUM_THREADS=2 taskset -c 0-1 ./.venv/bin/python scripts/run_p1c_loss_ablation.py \\
        --folds 1 2 3 4 --losses cross_entropy weighted_cross_entropy focal balanced_softmax \\
        --seeds 0 1 2 --output results/p1c/p1c-loss-ablation
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

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
from core.p1c_losses import LOSS_NAMES
from core.p1c_video_sampling import sample_rows_per_video
from scripts.build_p1c_feature_cache import _load_kwargs_for_config
from scripts.run_p1c_frame_baseline import _load_config

DEFAULT_FOLDS = (1, 2, 3, 4)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_FRAMES_PER_VIDEO = 8
DEFAULT_SAMPLING_SEED = 0
DEFAULT_FOCAL_GAMMA = 2.0
CONTROL_LOSS = "cross_entropy"
#: The pre-registered tail set (docs/PROJECT_RULES.md rule 4: keep negative results
#: visible for the classes this pipeline is weakest on -- D-46, D-50).
TAIL_CLASSES = ("Shooting", "Fighting", "Assault", "Vandalism")
DEFAULT_EXTRACT_CLOSURE_PATH = Path("results/p1c/p1c-video-frames/extract_closure.json")

CACHE_ROOT_TEMPLATE = "results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}"
CONFIG_TEMPLATE = "configs/experiments/p1c/P1C-VIDEO-DINOV2-B2-AR-FOLD{fold}.yaml"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, nargs="+", default=list(DEFAULT_FOLDS))
    parser.add_argument("--losses", choices=LOSS_NAMES, nargs="+", default=list(LOSS_NAMES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--frames-per-video", type=int, default=DEFAULT_FRAMES_PER_VIDEO)
    parser.add_argument("--sampling-seed", type=int, default=DEFAULT_SAMPLING_SEED)
    parser.add_argument("--focal-gamma", type=float, default=DEFAULT_FOCAL_GAMMA)
    parser.add_argument("--output", type=Path, default=Path("results/p1c/p1c-loss-ablation"))
    parser.add_argument(
        "--extract-closure-path", type=Path, default=DEFAULT_EXTRACT_CLOSURE_PATH,
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
    return parser.parse_args(argv)


def _load_fold(fold: int, args: argparse.Namespace) -> dict[str, Any]:
    """Load and sample one fold's train/validation cache, once per fold."""

    config_path = Path(CONFIG_TEMPLATE.format(fold=fold))
    cache_root = Path(CACHE_ROOT_TEMPLATE.format(fold=fold))
    _config, protocol, _run_class = _load_config(config_path)
    trainable = protocol["backbone"].get("trainable", "full")
    class_names = protocol["model"]["class_names"]
    if args.extract_closure_path is not None:
        protocol["data"]["video_frames"]["extract_closure_path"] = str(args.extract_closure_path)

    train_features, train_rows, train_manifest = load_feature_cache(
        cache_root / "train", **_load_kwargs_for_config(protocol, "train", trainable)
    )
    validation_features, validation_rows, _ = load_feature_cache(
        cache_root / "validation", **_load_kwargs_for_config(protocol, "validation", trainable)
    )

    keep = sample_rows_per_video(train_rows, args.frames_per_video, seed=args.sampling_seed)
    train_features = train_features[keep]
    train_rows = [train_rows[i] for i in keep]

    return {
        "config_path": config_path,
        "class_names": class_names,
        "train_features": train_features,
        "train_rows": train_rows,
        "validation_features": validation_features,
        "validation_rows": validation_rows,
        "cache_sha256": train_manifest.cache_sha256,
    }


def _run_one(fold_data: dict[str, Any], loss: str, seed: int, args: argparse.Namespace, *, fold: int) -> tuple[dict[str, Any], Any]:
    # `capture_validation_probabilities=True` is inference-only after training
    # completes (see `core.p1c_head_training`'s docstring) and does not
    # perturb the selected checkpoint or any metric; it is turned on
    # unconditionally here so `_vacuity_check` can compare fold-1/seed-0
    # arms' actual output distributions without a second training run.
    result = train_cached_head(
        fold_data["train_features"], fold_data["train_rows"],
        fold_data["validation_features"], fold_data["validation_rows"], fold_data["class_names"],
        seed=seed, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        patience=args.patience, min_delta=args.min_delta,
        bootstrap_iterations=args.bootstrap_iterations, bootstrap_seed=args.bootstrap_seed,
        loss=loss, focal_gamma=args.focal_gamma,
        capture_validation_probabilities=(fold == 1 and seed == 0),
    )
    payload = result.to_json() | {
        "cache_sha256": fold_data["cache_sha256"],
        "config": str(fold_data["config_path"]),
        "frames_per_video": args.frames_per_video,
        "sampling_seed": args.sampling_seed,
        "loss": loss,
        "focal_gamma": args.focal_gamma,
    }
    probe = payload["probe_result"]
    ci = probe["video_clustered_macro_f1_ci"]
    print(
        f"fold {fold_data['config_path'].name} loss={loss} seed={seed}: "
        f"best epoch {result.best_epoch} ({'early-stopped' if result.stopped_early else 'epoch ceiling'}), "
        f"validation macro-F1 {result.best_validation_macro_f1:.4f} "
        f"(clustered 95% CI {ci['lower_95']:.4f}-{ci['upper_95']:.4f}), "
        f"video-level macro-F1 {probe['video_level_metrics']['macro_f1']:.4f} "
        f"over {probe['video_count']} videos, train {result.train_seconds:.1f}s",
        flush=True,
    )
    return payload, result


def _mean_sd(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _per_class_video_f1(payload: dict[str, Any], class_names: Sequence[str]) -> dict[str, float]:
    per_class = payload["probe_result"]["video_level_metrics"]["per_class"]
    return {name: float(per_class[name]["f1"]) for name in class_names}


def _build_summary(
    runs: dict[tuple[int, str, int], dict[str, Any]],
    folds: Sequence[int],
    losses: Sequence[str],
    seeds: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    per_fold_per_loss: dict[int, dict[str, Any]] = {}
    for fold in folds:
        per_fold_per_loss[fold] = {}
        for loss in losses:
            video_f1s = [runs[(fold, loss, seed)]["probe_result"]["video_level_metrics"]["macro_f1"] for seed in seeds]
            frame_f1s = [runs[(fold, loss, seed)]["probe_result"]["frame_metrics"]["macro_f1"] for seed in seeds]
            per_class_by_seed = [_per_class_video_f1(runs[(fold, loss, seed)], class_names) for seed in seeds]
            per_class_mean = {
                name: statistics.fmean(values[name] for values in per_class_by_seed) for name in class_names
            }
            per_fold_per_loss[fold][loss] = {
                "video_level_macro_f1_mean": statistics.fmean(video_f1s),
                "frame_macro_f1_mean": statistics.fmean(frame_f1s),
                "per_class_video_level_f1_mean": per_class_mean,
                "tail_classes_mean_f1": statistics.fmean(per_class_mean[name] for name in TAIL_CLASSES),
            }

    cross_fold: dict[str, Any] = {}
    paired_diff_vs_control: dict[str, Any] = {}
    for loss in losses:
        fold_means = [per_fold_per_loss[fold][loss]["video_level_macro_f1_mean"] for fold in folds]
        cross_fold[loss] = _mean_sd(fold_means)
        if loss == CONTROL_LOSS:
            continue
        diffs = [
            per_fold_per_loss[fold][loss]["video_level_macro_f1_mean"]
            - per_fold_per_loss[fold][CONTROL_LOSS]["video_level_macro_f1_mean"]
            for fold in folds
        ]
        paired_diff_vs_control[loss] = {
            "per_fold": dict(zip(folds, diffs, strict=True)),
            **_mean_sd(diffs),
            "folds_positive": sum(1 for d in diffs if d > 0),
            "folds_total": len(diffs),
        }

    return {
        "control_loss": CONTROL_LOSS,
        "tail_classes": list(TAIL_CLASSES),
        "folds": list(folds),
        "losses": list(losses),
        "seeds": list(seeds),
        "per_fold_per_loss": per_fold_per_loss,
        "cross_fold_video_level_macro_f1": cross_fold,
        "paired_diff_vs_control": paired_diff_vs_control,
    }


def _max_abs_logit_delta(control_probabilities: np.ndarray, arm_probabilities: np.ndarray) -> float:
    """Max absolute per-class logit delta between two models' validation outputs.

    Only the softmax probabilities are available here (`HeadTrainingResult.
    best_validation_probabilities`), and softmax is invariant to a per-row
    additive shift of the logits it came from -- an unrecoverable degree of
    freedom, and exactly the shift Balanced Softmax's `log(prior)` term
    exploits. Row-centring `log(p)` (`log(p) - mean_c(log(p))`) removes that
    same unrecoverable shift from both sides before differencing, so what
    remains is the genuinely comparable part of the logit vector.
    """

    def _centred_log(probabilities: np.ndarray) -> np.ndarray:
        log_p = np.log(np.clip(probabilities, 1e-12, 1.0))
        return log_p - log_p.mean(axis=1, keepdims=True)

    return float(np.max(np.abs(_centred_log(control_probabilities) - _centred_log(arm_probabilities))))


def _vacuity_check(
    results: dict[tuple[int, str, int], Any], runs: dict[tuple[int, str, int], dict[str, Any]], losses: Sequence[str]
) -> dict[str, Any] | None:
    """Fold 1, seed 0: assert weighted_cross_entropy/balanced_softmax == cross_entropy exactly.

    Under `sample_rows_per_video(rows, 8, seed=0)` fold 1 train is exactly
    class-balanced (`core/p1c_losses.py`'s module docstring), so these two
    arms are provably the same computation as `cross_entropy`. This checks
    that empirically rather than asserting it only in prose; if it is ever
    NOT exact, the delta is recorded rather than hidden (docs/PROJECT_RULES.md rule 4).
    """

    control_key = (1, CONTROL_LOSS, 0)
    if control_key not in runs:
        return None
    control_payload = runs[control_key]
    control_result = results[control_key]
    control_macro_f1 = control_payload["probe_result"]["frame_metrics"]["macro_f1"]
    checks: dict[str, Any] = {}
    for loss in ("weighted_cross_entropy", "balanced_softmax"):
        key = (1, loss, 0)
        if key not in runs or loss not in losses:
            continue
        arm_payload = runs[key]
        arm_result = results[key]
        arm_macro_f1 = arm_payload["probe_result"]["frame_metrics"]["macro_f1"]
        exact = arm_macro_f1 == control_macro_f1
        max_abs_logit_delta = _max_abs_logit_delta(
            control_result.best_validation_probabilities, arm_result.best_validation_probabilities
        )
        checks[loss] = {
            "validation_macro_f1_exactly_equal_to_cross_entropy": exact,
            "cross_entropy_validation_macro_f1": control_macro_f1,
            f"{loss}_validation_macro_f1": arm_macro_f1,
            "max_abs_logit_delta": max_abs_logit_delta,
        }
        if not exact:
            checks[loss]["note"] = (
                "validation macro-F1 differed from cross_entropy; not hidden, see the values above"
            )
    return {"fold": 1, "seed": 0, "control_loss": CONTROL_LOSS, "checks": checks}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    fold_data_by_fold: dict[int, dict[str, Any]] = {}
    runs: dict[tuple[int, str, int], dict[str, Any]] = {}
    results: dict[tuple[int, str, int], Any] = {}
    class_names: Sequence[str] | None = None

    for fold in args.folds:
        fold_data = _load_fold(fold, args)
        fold_data_by_fold[fold] = fold_data
        class_names = fold_data["class_names"]
        print(
            f"fold {fold}: train {len(fold_data['train_rows'])} rows sampled "
            f"({args.frames_per_video}/video, seed={args.sampling_seed}), "
            f"validation {len(fold_data['validation_rows'])} rows (full, untouched), "
            f"cache {fold_data['cache_sha256'][:12]}",
            flush=True,
        )
        fold_dir = args.output / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for loss in args.losses:
            for seed in args.seeds:
                payload, result = _run_one(fold_data, loss, seed, args, fold=fold)
                runs[(fold, loss, seed)] = payload
                results[(fold, loss, seed)] = result
                path = fold_dir / f"{loss}-seed-{seed}.json"
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if class_names is None:
        raise RuntimeError("no fold was processed; --folds was empty")

    summary = _build_summary(runs, args.folds, args.losses, args.seeds, class_names)
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)

    vacuity = _vacuity_check(results, runs, args.losses)
    if vacuity is not None:
        vacuity_path = args.output / "vacuity_check.json"
        vacuity_path.write_text(json.dumps(vacuity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {vacuity_path}", flush=True)
        for loss, check in vacuity["checks"].items():
            status = "EXACT" if check["validation_macro_f1_exactly_equal_to_cross_entropy"] else "NOT EXACT"
            print(f"vacuity check {loss} vs cross_entropy: {status}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
