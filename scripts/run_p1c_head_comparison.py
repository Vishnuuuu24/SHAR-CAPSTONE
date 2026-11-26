#!/usr/bin/env python3
"""P3A: compare four visual heads on the SAME cached DINOv2 input, matched budgets (CPU).

Four arms, one factor -- the head architecture (`core.p1c_heads`):
`gap_linear` (the registered control), `mlp`, `multiscale_se`, and `step`
(arXiv 2503.24298). Everything else is held fixed: the frame set (TRAIN
thinned to 8 frames/video at `sampling_seed=0`, VALIDATION full and
untouched, per AR fold, D-40/D-46), the optimizer/schedule/early-stopping
recipe (`core.p1c_head_variants`, mirroring `core.p1c_head_training`), and
the scoring (`core.p1c_probes.score_predictions`).

Input and the join it requires
--------------------------------
The 82,679-row patch cache (`results/p1c/p1c-patch-cache/`) does not itself
carry `source_dataset` or fold membership -- only `source_video_id`,
`source_frame_index`, `target`, `label`. Per-fold train/validation row
provenance instead comes from the REGISTERED DINOv2 pooled-feature cache
(`results/p1c/p1c-dinov2-feature-cache-ar-fold{N}/`), loaded and sampled
exactly as `scripts/run_p1c_loss_ablation.py` does; the patch cache is then
looked up by `(source_video_id, source_frame_index)` to find which physical
row of `features.float16.npy` / `cls.float16.npy` each selected row maps to.
This mirrors `scripts/run_p1c_roi_head.py`'s `_load_roi_features`/
`_roi_matrix_for` pattern exactly (same brief), including hard-failing on any
missing key -- a silently dropped row would make the video-level majority
vote (or, for STEP, the per-video sequence) score a different set of frames
than the registered cache's own numbers do.

Memory discipline
-------------------
`features.float16.npy` (32.5 GB) is opened once with `mmap_mode="r"` and
NEVER fancy-indexed with a whole fold's worth of rows at once (a fold needs
up to ~49,719 validation rows; at float32 that is ~42 GB, per the brief).
`gap_linear`/`mlp`/`step` only need the small pooled `[CLS ++ mean(patch)]`
vector, computed by streaming the memmap in chunks
(`core.p1c_head_variants.load_pooled_features`); `multiscale_se` trains
directly off the memmap in per-batch minibatches
(`core.p1c_head_variants.train_grid_head`) and never materialises a whole
split's grid.

Self-check
-----------
`gap_linear` on this cache is required to reproduce the registered pipeline's
0.4068 +/- 0.0353 cross-fold video-level macro-F1 (D-46) to within the K-28
feature-cache tolerance, because it is defined to be the exact same
computation on what should be the exact same features, just resolved through
a different cache. `--registered-gap-linear-mean`/`--registered-gap-linear-sd`
name that comparison; the check is printed loudly, not silently passed over,
if it does not hold.

Flow-fused arms
----------------
`gap_linear_flow` and `step_permuted_flow` add explicit RAFT optical-flow
descriptors (`core.p1c_flow`, `scripts/build_p1c_flow_cache.py`,
`--flow-cache-root`) as EXTRA input dimensions on top of the same pooled
DINOv2 vector every other arm sees -- `gap_linear_flow` is `gap_linear`'s own
architecture on `[CLS ++ mean(patch) ++ flow]`, and `step_permuted_flow` is
the D-53 winner (`step_permuted`: StepHead with `permute_frame_order=True`)
on the same fused per-frame vector. Each therefore changes exactly ONE thing
against its own existing control (`gap_linear` 0.4116 +/- 0.0367,
`step_permuted` 0.4599 +/- 0.0468, both already registered at
`results/p1c/p1c-head-comparison{,-stepctl}/summary.json` and NOT re-run
here): the input. The DINOv2 block is already on a comparable, roughly
unit-scale range (it comes out of the backbone's own normalisation) and is
NEVER re-standardised; only the flow block is standardised, on TRAIN-fold
statistics (mean/std per dimension, `+1e-6` floor, computed from the fold's
own sampled train rows and applied unchanged to validation) exactly the way
`scripts/run_p1c_clip_fusion.py::_standardise` already does it for its
DINOv2/VideoMAE fusion -- the two blocks have unrelated units (frozen
appearance features vs. pixel-displacement statistics), so treating them
identically would be arbitrary, not principled. The fold's flow
standardisation mean/std are recorded in each fused run's JSON. Both fused
arms hard-fail (`KeyError`) if any selected row's `(source_video_id,
source_frame_index)` is missing from the flow cache -- a silently skipped
row would score a different, undocumented set of frames.

No GPU. Run under the pinned CPU policy:

    OMP_NUM_THREADS=2 taskset -c 0-1 ./.venv/bin/python scripts/run_p1c_head_comparison.py \\
        --heads gap_linear mlp multiscale_se step --folds 1 2 3 4 --seeds 0 1 2 \\
        --patch-cache-root results/p1c/p1c-patch-cache --output results/p1c/p1c-head-comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.feature_cache import load_feature_cache  # noqa: E402
from core.p1c_head_training import DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_BOOTSTRAP_SEED  # noqa: E402
from core.p1c_head_variants import (  # noqa: E402
    group_pooled_sequences_by_video,
    load_pooled_features,
    train_grid_head,
    train_sequence_head,
    train_vector_head,
)
from core.p1c_flow import DESCRIPTOR_LENGTH as FLOW_DESCRIPTOR_LENGTH  # noqa: E402
from core.p1c_heads import GapLinearHead, MlpHead, MultiScaleSeHead, StepHead, VideoMeanMlpHead  # noqa: E402
from core.p1c_video_sampling import sample_rows_per_video  # noqa: E402
from scripts.build_p1c_feature_cache import _load_kwargs_for_config  # noqa: E402
from scripts.run_p1c_frame_baseline import _load_config  # noqa: E402

DEFAULT_FOLDS = (1, 2, 3, 4)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_FRAMES_PER_VIDEO = 8
DEFAULT_SAMPLING_SEED = 0
HEAD_NAMES = (
    "gap_linear", "mlp", "multiscale_se", "step", "step_permuted", "video_mean_mlp",
    "gap_linear_flow", "step_permuted_flow",
)
#: Arms whose input is the pooled DINOv2 vector fused with a standardised flow
#: descriptor block, rather than the pooled vector alone.
FLOW_FUSED_HEADS = ("gap_linear_flow", "step_permuted_flow")
#: `[fused_dim]`'s pooled-vector half; the flow half is `FLOW_DESCRIPTOR_LENGTH`.
POOLED_VECTOR_DIM = 1536

#: `step_permuted` is STEP's own control, not a fifth idea: identical architecture,
#: identical capacity, identical video-level prediction, with each video's frame
#: order destroyed by a deterministic per-video permutation. STEP differs from
#: `gap_linear` in TWO ways at once -- order-sensitive attention AND predicting a
#: video directly instead of voting over its frames -- so a gain over `gap_linear`
#: alone cannot be attributed to temporal modelling. Only the gap between `step`
#: and `step_permuted` isolates order-sensitivity, which is the entire premise of
#: arXiv 2503.24298.
CONTROL_HEAD = "gap_linear"
TAIL_CLASSES = ("Shooting", "Fighting", "Assault", "Vandalism")

CACHE_ROOT_TEMPLATE = "results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}"
CONFIG_TEMPLATE = "configs/experiments/p1c/P1C-VIDEO-DINOV2-B2-AR-FOLD{fold}.yaml"

#: D-46's registered cross-fold figure for the exact computation `gap_linear`
#: reproduces here, and the K-28 feature tolerance's practical envelope for
#: "close enough to call it the same result" (see the module docstring's
#: pooled-cache-consistency numbers: median 3.8e-3, max 4.4e-2 absolute).
REGISTERED_GAP_LINEAR_MEAN = 0.4068
REGISTERED_GAP_LINEAR_SD = 0.0353
SELF_CHECK_TOLERANCE = 0.05


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads", choices=HEAD_NAMES, nargs="+", default=list(HEAD_NAMES))
    parser.add_argument("--folds", type=int, nargs="+", default=list(DEFAULT_FOLDS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--patch-cache-root", type=Path, default=Path("results/p1c/p1c-patch-cache"))
    parser.add_argument("--flow-cache-root", type=Path, default=Path("results/p1c/p1c-flow-cache"))
    parser.add_argument("--output", type=Path, default=Path("results/p1c/p1c-head-comparison"))
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES_PER_VIDEO, help="STEP window / TRAIN sampling budget")
    parser.add_argument("--sampling-seed", type=int, default=DEFAULT_SAMPLING_SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--extract-closure-path", type=Path,
        default=Path("results/p1c/p1c-video-frames/extract_closure.json"),
        help="read the D-34/D-35 video-blob closure from here instead of the path the registered config names",
    )
    parser.add_argument("--registered-gap-linear-mean", type=float, default=REGISTERED_GAP_LINEAR_MEAN)
    parser.add_argument("--registered-gap-linear-sd", type=float, default=REGISTERED_GAP_LINEAR_SD)
    return parser.parse_args(argv)


def _load_patch_cache(patch_cache_root: Path) -> tuple[np.ndarray, np.ndarray, dict[tuple[str, int], int]]:
    """`(cls_mmap, features_mmap, key -> patch-cache row index)`; hard-fails on duplicate keys.

    Mirrors `scripts/run_p1c_roi_head.py::_load_roi_features` exactly -- same
    join problem (a store keyed by `(source_video_id, source_frame_index)`
    built in a different row order than the registered cache), same
    memmap-only load of the large array.
    """

    cls = np.load(patch_cache_root / "cls.float16.npy", mmap_mode="r")
    features = np.load(patch_cache_root / "features.float16.npy", mmap_mode="r")
    with (patch_cache_root / "rows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    key_to_index = {(row["source_video_id"], int(row["source_frame_index"])): index for index, row in enumerate(rows)}
    if len(key_to_index) != len(rows):
        raise ValueError(f"{patch_cache_root}/rows.csv has duplicate (video, frame) keys")
    if cls.shape[0] != len(rows) or features.shape[0] != len(rows):
        raise ValueError(f"{patch_cache_root} array/row-count mismatch: {cls.shape[0]}/{features.shape[0]} vs {len(rows)} rows")
    return cls, features, key_to_index


def _patch_indices_for(rows: Sequence[dict[str, Any]], key_to_index: dict[tuple[str, int], int]) -> np.ndarray:
    """Patch-cache row positions aligned 1:1 with `rows`; hard-fails on a missing key."""

    missing = [
        (row["source_video_id"], int(row["source_frame_index"])) for row in rows
        if (row["source_video_id"], int(row["source_frame_index"])) not in key_to_index
    ]
    if missing:
        raise KeyError(f"{len(missing)} of {len(rows)} rows are missing from the patch cache, e.g. {missing[:5]}")
    return np.asarray(
        [key_to_index[(row["source_video_id"], int(row["source_frame_index"]))] for row in rows], dtype=np.int64
    )


def _load_flow_cache(flow_cache_root: Path) -> tuple[np.ndarray, dict[tuple[str, int], int]]:
    """`(features_mmap, key -> flow-cache row index)`; hard-fails on duplicate keys or the wrong width.

    Same join shape as `_load_patch_cache`: a store keyed by
    `(source_video_id, source_frame_index)`, built by
    `scripts/build_p1c_flow_cache.py` in its own row order, memmap-only
    (the flow cache is small -- 82,679 x 162 float16, ~27 MB -- but reusing
    the mmap-then-index pattern keeps this arm's code identical in shape to
    the patch-cache join it sits next to).
    """

    features = np.load(flow_cache_root / "features.float16.npy", mmap_mode="r")
    if features.shape[1] != FLOW_DESCRIPTOR_LENGTH:
        raise ValueError(
            f"{flow_cache_root}/features.float16.npy has width {features.shape[1]}, "
            f"expected {FLOW_DESCRIPTOR_LENGTH} (core.p1c_flow.DESCRIPTOR_LENGTH)"
        )
    with (flow_cache_root / "rows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    key_to_index = {(row["source_video_id"], int(row["source_frame_index"])): index for index, row in enumerate(rows)}
    if len(key_to_index) != len(rows):
        raise ValueError(f"{flow_cache_root}/rows.csv has duplicate (video, frame) keys")
    if features.shape[0] != len(rows):
        raise ValueError(f"{flow_cache_root} array/row-count mismatch: {features.shape[0]} vs {len(rows)} rows")
    return features, key_to_index


def _flow_indices_for(rows: Sequence[dict[str, Any]], key_to_index: dict[tuple[str, int], int]) -> np.ndarray:
    """Flow-cache row positions aligned 1:1 with `rows`; hard-fails on a missing key.

    Mirrors `_patch_indices_for` exactly (rule of two): a fused arm must
    score the same set of frames the registered cache defines, and a
    silently dropped row would do otherwise.
    """

    missing = [
        (row["source_video_id"], int(row["source_frame_index"])) for row in rows
        if (row["source_video_id"], int(row["source_frame_index"])) not in key_to_index
    ]
    if missing:
        raise KeyError(f"{len(missing)} of {len(rows)} rows are missing from the flow cache, e.g. {missing[:5]}")
    return np.asarray(
        [key_to_index[(row["source_video_id"], int(row["source_frame_index"]))] for row in rows], dtype=np.int64
    )


def _standardise_flow(train: np.ndarray, validation: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """`(train_z, validation_z, stats)`, TRAIN-fold mean/std only -- never re-fit on validation.

    Same convention as `scripts/run_p1c_clip_fusion.py::_standardise`
    (rule of two): `mu, sd = train.mean(0), train.std(0) + 1e-6`, both splits
    transformed by the TRAIN statistics. Applied to the flow block only --
    the pooled DINOv2 block is already on a comparable, roughly unit scale
    coming out of the backbone and must not be re-standardised (see the
    module docstring's "Flow-fused arms" section for why the two blocks are
    treated differently). `stats` is recorded verbatim in the run payload so
    a reader can see exactly what was fit.
    """

    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-6
    stats = {"mean": mean.tolist(), "std": std.tolist(), "dim": int(mean.shape[0]), "fit_on": "train_fold_rows_only"}
    return (train - mean) / std, (validation - mean) / std, stats


def _load_fold(fold: int, args: argparse.Namespace) -> dict[str, Any]:
    """Fold provenance (train sampled + validation full) from the registered DINOv2 cache.

    Only `train_rows`/`validation_rows`/`class_names` are kept; the
    registered cache's own pooled feature ARRAYS are discarded immediately
    -- this run's features come from the patch cache instead, joined by
    `(source_video_id, source_frame_index)`.
    """

    config_path = Path(CONFIG_TEMPLATE.format(fold=fold))
    cache_root = Path(CACHE_ROOT_TEMPLATE.format(fold=fold))
    _config, protocol, _run_class = _load_config(config_path)
    trainable = protocol["backbone"].get("trainable", "full")
    class_names = protocol["model"]["class_names"]
    if args.extract_closure_path is not None:
        protocol["data"]["video_frames"]["extract_closure_path"] = str(args.extract_closure_path)

    _train_features, train_rows, train_manifest = load_feature_cache(
        cache_root / "train", **_load_kwargs_for_config(protocol, "train", trainable)
    )
    _validation_features, validation_rows, _ = load_feature_cache(
        cache_root / "validation", **_load_kwargs_for_config(protocol, "validation", trainable)
    )
    del _train_features, _validation_features

    keep = sample_rows_per_video(train_rows, args.frames, seed=args.sampling_seed)
    train_rows = [train_rows[i] for i in keep]

    return {
        "config_path": config_path,
        "class_names": class_names,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "cache_sha256": train_manifest.cache_sha256,
    }


def _build_head(head_name: str, num_classes: int, frames: int):
    if head_name == "gap_linear":
        return GapLinearHead(in_dim=1536, num_classes=num_classes)
    if head_name == "mlp":
        return MlpHead(in_dim=1536, hidden=32, num_classes=num_classes)
    if head_name == "multiscale_se":
        return MultiScaleSeHead(token_dim=768, grid=(16, 16), width=32, num_classes=num_classes)
    if head_name == "video_mean_mlp":
        return VideoMeanMlpHead(token_dim=1536, depth=3, num_classes=num_classes)
    if head_name in ("step", "step_permuted"):
        return StepHead(token_dim=1536, frames=frames, depth=1, num_classes=num_classes)
    if head_name == "gap_linear_flow":
        return GapLinearHead(in_dim=POOLED_VECTOR_DIM + FLOW_DESCRIPTOR_LENGTH, num_classes=num_classes)
    if head_name == "step_permuted_flow":
        return StepHead(token_dim=POOLED_VECTOR_DIM + FLOW_DESCRIPTOR_LENGTH, frames=frames, depth=1, num_classes=num_classes)
    raise ValueError(f"unknown head {head_name!r}")


def _run_one(
    head_name: str, fold: int, seed: int, fold_data: dict[str, Any], patch_cache: tuple, args: argparse.Namespace,
    flow_cache: tuple | None = None,
) -> dict[str, Any]:
    cls_mmap, features_mmap, key_to_index = patch_cache
    class_names = fold_data["class_names"]
    train_rows, validation_rows = fold_data["train_rows"], fold_data["validation_rows"]
    train_index = _patch_indices_for(train_rows, key_to_index)
    validation_index = _patch_indices_for(validation_rows, key_to_index)

    head = _build_head(head_name, len(class_names), args.frames)
    common = dict(
        seed=seed, bootstrap_iterations=args.bootstrap_iterations, bootstrap_seed=args.bootstrap_seed,
    )

    flow_stats: dict[str, Any] | None = None
    started = time.perf_counter()
    if head_name in ("gap_linear", "mlp"):
        train_pooled = load_pooled_features(cls_mmap, features_mmap, train_index)
        validation_pooled = load_pooled_features(cls_mmap, features_mmap, validation_index)
        result = train_vector_head(head, train_pooled, train_rows, validation_pooled, validation_rows, class_names, **common)
    elif head_name == "multiscale_se":
        result = train_grid_head(
            head, features_mmap, train_index, train_rows, validation_index, validation_rows, class_names, **common
        )
    elif head_name in ("step", "step_permuted", "video_mean_mlp"):
        permute = head_name == "step_permuted"
        train_pooled = load_pooled_features(cls_mmap, features_mmap, train_index)
        validation_pooled = load_pooled_features(cls_mmap, features_mmap, validation_index)
        # The permutation seed is folded with the run seed so the three seeds of
        # the control arm see three different shuffles, not one repeated.
        train_sequences = group_pooled_sequences_by_video(
            train_pooled, train_rows, permute_frame_order=permute, permute_seed=1000 + seed
        )
        validation_sequences = group_pooled_sequences_by_video(
            validation_pooled, validation_rows, permute_frame_order=permute, permute_seed=2000 + seed
        )
        result = train_sequence_head(head, train_sequences, validation_sequences, class_names, frames=args.frames, **common)
    elif head_name in FLOW_FUSED_HEADS:
        if flow_cache is None:
            raise RuntimeError(f"{head_name} requires --flow-cache-root but none was loaded")
        flow_features_mmap, flow_key_to_index = flow_cache
        flow_train_index = _flow_indices_for(train_rows, flow_key_to_index)
        flow_validation_index = _flow_indices_for(validation_rows, flow_key_to_index)
        flow_train_raw = np.asarray(flow_features_mmap[flow_train_index], dtype=np.float32)
        flow_validation_raw = np.asarray(flow_features_mmap[flow_validation_index], dtype=np.float32)
        flow_train_z, flow_validation_z, flow_stats = _standardise_flow(flow_train_raw, flow_validation_raw)

        train_pooled = load_pooled_features(cls_mmap, features_mmap, train_index)
        validation_pooled = load_pooled_features(cls_mmap, features_mmap, validation_index)
        fused_train = np.concatenate([train_pooled, flow_train_z], axis=1)
        fused_validation = np.concatenate([validation_pooled, flow_validation_z], axis=1)

        if head_name == "gap_linear_flow":
            result = train_vector_head(head, fused_train, train_rows, fused_validation, validation_rows, class_names, **common)
        else:  # step_permuted_flow
            train_sequences = group_pooled_sequences_by_video(
                fused_train, train_rows, permute_frame_order=True, permute_seed=1000 + seed
            )
            validation_sequences = group_pooled_sequences_by_video(
                fused_validation, validation_rows, permute_frame_order=True, permute_seed=2000 + seed
            )
            result = train_sequence_head(head, train_sequences, validation_sequences, class_names, frames=args.frames, **common)
    else:
        raise ValueError(f"unknown head {head_name!r}")
    wall_seconds = time.perf_counter() - started

    payload = result.to_json() | {
        "fold": fold, "head": head_name, "seed": seed,
        "cache_sha256": fold_data["cache_sha256"],
        "config": str(fold_data["config_path"]),
        "patch_cache_root": str(args.patch_cache_root),
        "frames_per_video": args.frames,
        "sampling_seed": args.sampling_seed,
        "parameter_count": head.parameter_count(),
        "wall_clock_seconds": wall_seconds,
    }
    if flow_stats is not None:
        payload["flow_cache_root"] = str(args.flow_cache_root)
        payload["flow_standardisation"] = flow_stats
    probe = payload["probe_result"]
    print(
        f"fold {fold} head {head_name} seed {seed}: best epoch {result.best_epoch} "
        f"({'early-stopped' if result.stopped_early else 'epoch ceiling'}), "
        f"validation macro-F1 {result.best_validation_macro_f1:.4f}, "
        f"video-level macro-F1 {probe['video_level_metrics']['macro_f1']:.4f} "
        f"over {probe['video_count']} videos, params {head.parameter_count()}, wall {wall_seconds:.1f}s",
        flush=True,
    )
    return payload


def _mean_sd(values: Sequence[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def _per_class_video_f1(payload: dict[str, Any], class_names: Sequence[str]) -> dict[str, float]:
    per_class = payload["probe_result"]["video_level_metrics"]["per_class"]
    return {name: float(per_class[name]["f1"]) for name in class_names if name in per_class}


def _build_summary(
    runs: dict[tuple[int, str, int], dict[str, Any]], folds: Sequence[int], heads: Sequence[str],
    seeds: Sequence[int], class_names: Sequence[str],
) -> dict[str, Any]:
    per_fold_per_head: dict[int, dict[str, Any]] = {}
    for fold in folds:
        per_fold_per_head[fold] = {}
        for head in heads:
            keys = [(fold, head, seed) for seed in seeds if (fold, head, seed) in runs]
            if not keys:
                continue
            video_f1s = [runs[key]["probe_result"]["video_level_metrics"]["macro_f1"] for key in keys]
            frame_f1s = [runs[key]["probe_result"]["frame_metrics"]["macro_f1"] for key in keys]
            per_class_by_seed = [_per_class_video_f1(runs[key], class_names) for key in keys]
            per_class_mean = {
                name: statistics.fmean(values[name] for values in per_class_by_seed if name in values)
                for name in class_names if any(name in values for values in per_class_by_seed)
            }
            wall_clock = [runs[key]["wall_clock_seconds"] for key in keys]
            per_fold_per_head[fold][head] = {
                "video_level_macro_f1_mean": statistics.fmean(video_f1s),
                "frame_macro_f1_mean": statistics.fmean(frame_f1s),
                "per_class_video_level_f1_mean": per_class_mean,
                "tail_classes_mean_f1": statistics.fmean(per_class_mean[name] for name in TAIL_CLASSES if name in per_class_mean),
                "parameter_count": runs[keys[0]]["parameter_count"],
                "video_level_aggregation": runs[keys[0]]["recipe"]["video_level_aggregation"],
                "wall_clock_seconds_mean": statistics.fmean(wall_clock),
            }

    cross_fold: dict[str, Any] = {}
    paired_diff_vs_control: dict[str, Any] = {}
    for head in heads:
        fold_means = [per_fold_per_head[fold][head]["video_level_macro_f1_mean"] for fold in folds if head in per_fold_per_head[fold]]
        if not fold_means:
            continue
        cross_fold[head] = _mean_sd(fold_means)
        if head == CONTROL_HEAD:
            continue
        diffs = [
            per_fold_per_head[fold][head]["video_level_macro_f1_mean"] - per_fold_per_head[fold][CONTROL_HEAD]["video_level_macro_f1_mean"]
            for fold in folds if head in per_fold_per_head[fold] and CONTROL_HEAD in per_fold_per_head[fold]
        ]
        if diffs:
            paired_diff_vs_control[head] = {
                "per_fold": {fold: diff for fold, diff in zip(folds, diffs, strict=False)},
                **_mean_sd(diffs),
                "folds_positive": sum(1 for d in diffs if d > 0),
                "folds_total": len(diffs),
            }

    return {
        "control_head": CONTROL_HEAD,
        "tail_classes": list(TAIL_CLASSES),
        "folds": list(folds), "heads": list(heads), "seeds": list(seeds),
        "per_fold_per_head": per_fold_per_head,
        "cross_fold_video_level_macro_f1": cross_fold,
        "paired_diff_vs_control": paired_diff_vs_control,
    }


def _self_check(cross_fold: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    if CONTROL_HEAD not in cross_fold:
        return None
    measured_mean = cross_fold[CONTROL_HEAD]["mean"]
    measured_sd = cross_fold[CONTROL_HEAD]["sample_sd"]
    delta = abs(measured_mean - args.registered_gap_linear_mean)
    within_tolerance = delta <= SELF_CHECK_TOLERANCE
    check = {
        "registered_mean": args.registered_gap_linear_mean,
        "registered_sd": args.registered_gap_linear_sd,
        "measured_mean": measured_mean,
        "measured_sd": measured_sd,
        "abs_delta": delta,
        "tolerance": SELF_CHECK_TOLERANCE,
        "within_tolerance": within_tolerance,
    }
    if within_tolerance:
        print(
            f"SELF-CHECK OK: gap_linear on the patch cache {measured_mean:.4f} +/- {measured_sd:.4f} "
            f"vs registered {args.registered_gap_linear_mean:.4f} +/- {args.registered_gap_linear_sd:.4f} "
            f"(delta {delta:.4f} <= {SELF_CHECK_TOLERANCE})",
            flush=True,
        )
    else:
        print(
            f"SELF-CHECK FAILED: gap_linear on the patch cache {measured_mean:.4f} +/- {measured_sd:.4f} "
            f"does NOT reproduce the registered {args.registered_gap_linear_mean:.4f} +/- "
            f"{args.registered_gap_linear_sd:.4f} (delta {delta:.4f} > {SELF_CHECK_TOLERANCE}). "
            "Proceeding, but every other arm's comparison against this control is now suspect.",
            flush=True,
        )
    return check


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    cls_mmap, features_mmap, key_to_index = _load_patch_cache(args.patch_cache_root)
    print(f"patch cache: {features_mmap.shape[0]} rows, memmapped from {args.patch_cache_root}", flush=True)

    flow_cache: tuple[np.ndarray, dict[tuple[str, int], int]] | None = None
    if any(head in FLOW_FUSED_HEADS for head in args.heads):
        flow_features_mmap, flow_key_to_index = _load_flow_cache(args.flow_cache_root)
        flow_cache = (flow_features_mmap, flow_key_to_index)
        print(f"flow cache: {flow_features_mmap.shape[0]} rows, memmapped from {args.flow_cache_root}", flush=True)

    fold_data_by_fold: dict[int, dict[str, Any]] = {}
    runs: dict[tuple[int, str, int], dict[str, Any]] = {}
    class_names: Sequence[str] | None = None

    for fold in args.folds:
        fold_data = _load_fold(fold, args)
        fold_data_by_fold[fold] = fold_data
        class_names = fold_data["class_names"]
        print(
            f"fold {fold}: train {len(fold_data['train_rows'])} rows sampled "
            f"({args.frames}/video, seed={args.sampling_seed}), "
            f"validation {len(fold_data['validation_rows'])} rows (full, untouched), "
            f"cache {fold_data['cache_sha256'][:12]}",
            flush=True,
        )
        fold_dir = args.output / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for head in args.heads:
            for seed in args.seeds:
                payload = _run_one(
                    head, fold, seed, fold_data, (cls_mmap, features_mmap, key_to_index), args, flow_cache=flow_cache,
                )
                runs[(fold, head, seed)] = payload
                path = fold_dir / f"{head}-seed-{seed}.json"
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if class_names is None:
        raise RuntimeError("no fold was processed; --folds was empty")

    summary = _build_summary(runs, args.folds, args.heads, args.seeds, class_names)
    self_check = _self_check(summary["cross_fold_video_level_macro_f1"], args)
    if self_check is not None:
        summary["gap_linear_self_check"] = self_check

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
