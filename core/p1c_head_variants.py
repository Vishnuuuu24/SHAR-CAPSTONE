"""P3A: train a non-control head from `core.p1c_heads` on cached DINOv2 features, CPU.

Why this is a sibling module, not a change to `core.p1c_head_training`
------------------------------------------------------------------------
`core.p1c_head_training.train_cached_head` trains one specific architecture
(`nn.Linear` on an in-RAM `[N, 1536]` tensor) built INSIDE the function. Three
of P3A's four arms cannot fit that shape: `MultiScaleSeHead` needs the raw
`[N, 256, 768]` patch grid, which at fold scale is far too large to
materialise as one array (the patch cache is 32.5 GB; a fold's grid alone is
~42 GB as float32 -- see `scripts/run_p1c_head_comparison.py`), and
`StepHead` needs per-video frame *sequences*, not independent rows at all.
Contorting `train_cached_head` to also branch on head shape and batch source
would turn one already-load-bearing function into a much harder one to trust;
this module instead reuses every protocol constant and every scoring helper
`core.p1c_head_training` and `core.p1c_probes` already export (rule of two)
and adds exactly the two things that differ: accepting an arbitrary
`torch.nn.Module`, and three batch/predict strategies matched to each head's
input shape.

The three strategies
---------------------
- **vector** (`train_vector_head`, `GapLinearHead`/`MlpHead`): the pooled
  `[CLS ++ mean(patch)]` vector is small enough to hold as one `[N, 1536]`
  tensor per fold (~300 MB at 49,719 validation rows), so this path is
  almost byte-identical to `train_cached_head` -- same `TensorDataset` +
  seeded `DataLoader`, same `epoch_macro_f1` reused directly.
- **grid** (`train_grid_head`, `MultiScaleSeHead`): the caller passes the
  patch cache's `features.float16.npy`, opened with `mmap_mode="r"`, plus
  integer positions into it for this fold's train/validation rows. Every
  minibatch is gathered from the memmap on demand -- `features_mmap[batch_
  indices]` -- and never accumulated into a whole-fold array; validation is
  scored the same way, in fixed-order chunks.
- **sequence** (`train_sequence_head`, `StepHead`): pooled vectors are
  grouped into one ordered-by-frame-index sequence per video. Training
  sequences (<= the sampled `frames_per_video` budget) are zero-padded to the
  registered window with a boolean mask; a validation video longer than that
  window is split into consecutive non-overlapping chunks, each scored
  separately, and the video's final prediction is the argmax of those
  chunks' MEAN logit vector (STEP's own output is one CLS-position
  prediction per window; the P3A brief leaves multi-window combination
  unspecified, and mean-logit is the natural extension when the window
  policy is fixed but a video runs long -- this file's docstring records
  that choice explicitly rather than leaving it implicit).

All three return the SAME `core.p1c_head_training.HeadTrainingResult` shape,
so a comparison table can read every arm's JSON the same way. What differs
between arms is recorded IN that result's `recipe` dict (a free-form
mapping, unlike the dataclass's fixed fields): `recipe["video_level_
aggregation"]` is `"majority_vote"` for vector/grid heads (frame predictions
aggregated the existing way, `eval.video_level.aggregate_video_level_
predictions`) and `"step_direct_cls_prediction"` for the sequence head,
whose `ProbeResult.video_level_metrics` and `.frame_metrics` are IDENTICAL
by construction -- there is exactly one prediction per validation video, not
one per frame, so "frame-level" and "video-level" collapse to the same
number. That collapse is intentional (STEP predicts videos, not frames) and
is stated here rather than left to be discovered by a reader comparing
numbers across arms.
"""

from __future__ import annotations

import copy
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from core.p1c_head_training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_DELTA,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    HeadTrainingResult,
    epoch_macro_f1,
    set_local_seed,
    targets_from_rows,
)
from core.p1c_probes import score_predictions
from eval.metrics import classification_metrics

VIDEO_LEVEL_AGGREGATION_MAJORITY_VOTE = "majority_vote"
VIDEO_LEVEL_AGGREGATION_STEP_DIRECT = "step_direct_cls_prediction"


def pooled_vector_from_tokens(cls_batch: np.ndarray, patch_batch: np.ndarray) -> np.ndarray:
    """`[CLS ++ mean(patch, axis=1)]`, float32 -- the same pooling the registered cache stores.

    `cls_batch` is `[N, 768]`, `patch_batch` is `[N, 256, 768]`; both come
    from the patch cache's `cls.float16.npy` / `features.float16.npy`. Kept
    as one small function so every caller (all three vector-consuming heads,
    plus `load_pooled_features` below) computes it identically -- rule of
    two, and the one computation the P3A self-check (`gap_linear` must
    reproduce the registered 0.4068 +/- 0.0353 figure) depends on being
    exactly this.
    """

    if cls_batch.shape[0] != patch_batch.shape[0]:
        raise ValueError(f"cls/patch batch sizes disagree: {cls_batch.shape[0]} vs {patch_batch.shape[0]}")
    mean_patch = np.asarray(patch_batch, dtype=np.float32).mean(axis=1)
    return np.concatenate([np.asarray(cls_batch, dtype=np.float32), mean_patch], axis=1)


def load_pooled_features(
    cls_mmap: np.ndarray, features_mmap: np.ndarray, indices: np.ndarray, *, chunk_size: int = 2000
) -> np.ndarray:
    """`[len(indices), 1536]` float32 pooled vectors, streamed from the memmapped patch cache.

    The 32.5 GB `features.float16.npy` is never fancy-indexed with all of
    `indices` at once -- gathering, say, 49,719 rows in one call would
    materialise close to the whole-fold 42 GB the P3A brief warns against.
    Instead `indices` is walked in chunks of `chunk_size`, each chunk read
    from the memmap, pooled, cast to float32 and written into a
    pre-allocated `[len(indices), 1536]` output array -- the only thing held
    in full is the small pooled result, never the raw patch grid.
    """

    if len(indices) == 0:
        raise ValueError("indices must be non-empty")
    pooled = np.empty((len(indices), cls_mmap.shape[1] + features_mmap.shape[2]), dtype=np.float32)
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start : start + chunk_size]
        pooled[start : start + len(chunk)] = pooled_vector_from_tokens(
            np.asarray(cls_mmap[chunk]), np.asarray(features_mmap[chunk])
        )
    return pooled


def group_pooled_sequences_by_video(
    pooled: np.ndarray,
    rows: Sequence[dict[str, Any]],
    *,
    permute_frame_order: bool = False,
    permute_seed: int = 0,
) -> dict[tuple[str, str], tuple[np.ndarray, dict[str, Any]]]:
    """Group pooled rows into one ascending-frame-index sequence per video.

    Returns `{(source_dataset, source_video_id): (sequence[L, 1536],
    representative_row)}`. Raises if a video's rows disagree on `target` --
    the same defect `eval.video_level.aggregate_video_level_predictions`
    guards against, checked here too because `StepHead` never runs its
    frame-level predictions through that function.
    """

    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (row["source_dataset"], row["source_video_id"])
        groups[key].append((int(row["source_frame_index"]), index))

    result: dict[tuple[str, str], tuple[np.ndarray, dict[str, Any]]] = {}
    permute_rng = np.random.default_rng(permute_seed) if permute_frame_order else None
    for key, entries in groups.items():
        entries.sort(key=lambda item: item[0])
        order = [index for _frame_index, index in entries]
        if permute_rng is not None:
            # The STEP control arm (arXiv 2503.24298's own premise): destroy
            # temporal order while keeping the architecture, its capacity and
            # its video-level prediction identical. A gain that survives this
            # permutation is NOT evidence of temporal modelling -- it is
            # capacity or aggregation. Deterministic in `permute_seed` so the
            # control is reproducible, and applied per video so each video's
            # frames are shuffled independently.
            order = [order[i] for i in permute_rng.permutation(len(order))]
        target_indices = {int(rows[index]["target"]) for index in order}
        if len(target_indices) != 1:
            raise ValueError(f"video {key} frames disagree on target: {sorted(target_indices)}")
        result[key] = (pooled[order], dict(rows[order[0]]))
    return result


def _pad_sequence(sequence: np.ndarray, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """`(padded[frames, dim], mask[frames])`, mask True at padded (invalid) positions."""

    length = sequence.shape[0]
    if length > frames:
        raise ValueError(f"sequence has {length} frames, exceeds window {frames}")
    if length == frames:
        return sequence.astype(np.float32, copy=False), np.zeros(frames, dtype=bool)
    pad = np.zeros((frames - length, sequence.shape[1]), dtype=np.float32)
    padded = np.concatenate([sequence.astype(np.float32, copy=False), pad], axis=0)
    mask = np.concatenate([np.zeros(length, dtype=bool), np.ones(frames - length, dtype=bool)])
    return padded, mask


def _chunk_sequence(sequence: np.ndarray, frames: int) -> list[np.ndarray]:
    """Consecutive, non-overlapping `<= frames`-length chunks covering the whole sequence."""

    return [sequence[start : start + frames] for start in range(0, len(sequence), frames)]


def _macro_f1_from_predicted(predicted: np.ndarray, rows: Sequence[dict[str, Any]], class_names: Sequence[str]) -> float:
    truth_labels = [class_names[int(row["target"])] for row in rows]
    predicted_labels = [class_names[int(index)] for index in predicted]
    return classification_metrics(truth_labels, predicted_labels, class_names)["macro_f1"]


#: `_fit_head` checkpoint rules. The registered rule picks the best epoch on the monitored videos.
CHECKPOINT_BEST_VALIDATION = "best_validation"
CHECKPOINT_LAST_EPOCH = "last_epoch"


@dataclass
class _FitOutcome:
    best_metric: float
    best_epoch: int
    best_state_dict: dict[str, Any]
    best_predicted: np.ndarray
    epoch_records: tuple[dict[str, Any], ...]
    stopped_early: bool
    train_seconds: float


def _fit_head(
    *,
    run_train_epoch,
    run_validation,
    epochs: int,
    patience: int,
    min_delta: float,
    checkpoint_rule: str = "best_validation",
) -> _FitOutcome:
    """The one early-stopping/best-epoch loop shared by all three batch strategies.

    `checkpoint_rule=CHECKPOINT_LAST_EPOCH` trains exactly `epochs` epochs and keeps the
    final weights: no early stopping and no selection on the scored videos (the fixed-length
    rule of `core.inner_cv`). The per-epoch metric is still recorded, as a curve only.

    `run_train_epoch(epoch) -> dict` performs one epoch's optimizer steps and
    returns its bookkeeping fields (`train_examples`, `train_batches`,
    `train_mean_loss`); `run_validation() -> (macro_f1, predicted)` scores
    the current weights. Bookkeeping (best-epoch selection on validation
    macro-F1, patience/min_delta early stopping, wall-clock) is identical to
    `core.p1c_head_training.train_cached_head` -- lifted out here so the
    three `train_*_head` functions below differ only in how a batch is
    shaped, never in the stopping rule.
    """

    if checkpoint_rule not in (CHECKPOINT_BEST_VALIDATION, CHECKPOINT_LAST_EPOCH):
        raise ValueError(f"unknown checkpoint_rule {checkpoint_rule!r}")
    best_metric = float("-inf")
    stopping_anchor = float("-inf")
    stagnant_epochs = 0
    best_epoch = -1
    best_state_dict: dict[str, Any] | None = None
    best_predicted: np.ndarray | None = None
    epoch_records: list[dict[str, Any]] = []
    stopped_early = False

    started = time.perf_counter()
    for epoch in range(epochs):
        train_stats = run_train_epoch(epoch)
        metric, predicted = run_validation()
        epoch_records.append({"epoch": epoch, **train_stats, "validation_macro_f1": metric})
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state_dict = copy.deepcopy(train_stats["model_state_dict_source"].state_dict())
            best_predicted = predicted
        if metric > stopping_anchor + min_delta:
            stopping_anchor, stagnant_epochs = metric, 0
        else:
            stagnant_epochs += 1
        if checkpoint_rule == CHECKPOINT_LAST_EPOCH:
            best_metric, best_epoch, best_predicted = metric, epoch, predicted
            best_state_dict = None
            if epoch == epochs - 1:
                best_state_dict = copy.deepcopy(train_stats["model_state_dict_source"].state_dict())
            continue
        if stagnant_epochs >= patience:
            stopped_early = True
            break
    train_seconds = time.perf_counter() - started

    if best_state_dict is None or best_predicted is None:
        raise RuntimeError("no validation epoch was scored; cannot select a checkpoint")

    # `model_state_dict_source` was only a vehicle to reach `.state_dict()`
    # above; strip it before this becomes part of a JSON-serialisable record.
    cleaned_records = tuple(
        {key: value for key, value in record.items() if key != "model_state_dict_source"} for record in epoch_records
    )
    return _FitOutcome(best_metric, best_epoch, best_state_dict, best_predicted, cleaned_records, stopped_early, train_seconds)


def _recipe(
    *, learning_rate: float, weight_decay: float, batch_size: int, epochs: int, patience: int, min_delta: float,
    video_level_aggregation: str,
) -> dict[str, Any]:
    return {
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": "none",
        "loss": "cross_entropy_mean_unweighted",
        "batch_size": batch_size,
        "epochs_ceiling": epochs,
        "early_stopping_monitor": "macro_f1",
        "early_stopping_mode": "max",
        "early_stopping_patience": patience,
        "early_stopping_min_delta": min_delta,
        "selection_rule": "best_validation_macro_f1",
        "precision": "float32_cpu",
        "video_level_aggregation": video_level_aggregation,
    }


def train_vector_head(
    head: nn.Module,
    train_pooled: np.ndarray,
    train_rows: Sequence[dict[str, Any]],
    validation_pooled: np.ndarray,
    validation_rows: Sequence[dict[str, Any]],
    class_names: Sequence[str],
    *,
    seed: int,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    checkpoint_rule: str = CHECKPOINT_BEST_VALIDATION,
) -> HeadTrainingResult:
    """Train `GapLinearHead` or `MlpHead` on an in-RAM `[N, 1536]` pooled matrix.

    `train_pooled`/`validation_pooled` are produced by `load_pooled_features`
    (or, for the `gap_linear` self-check, must equal the registered cache's
    own pooled vectors bit-for-bit -- see `scripts/run_p1c_head_comparison.py`).
    """

    if len(train_pooled) != len(train_rows):
        raise ValueError(f"train pooled/rows disagree: {len(train_pooled)} vs {len(train_rows)}")
    if len(validation_pooled) != len(validation_rows):
        raise ValueError(f"validation pooled/rows disagree: {len(validation_pooled)} vs {len(validation_rows)}")

    set_local_seed(seed)

    train_x = torch.from_numpy(np.ascontiguousarray(train_pooled, dtype=np.float32))
    train_y = torch.from_numpy(targets_from_rows(train_rows))
    validation_x = torch.from_numpy(np.ascontiguousarray(validation_pooled, dtype=np.float32))

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    def run_train_epoch(_epoch: int) -> dict[str, Any]:
        head.train()
        total_loss, examples = 0.0, 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_x)
            loss_value = criterion(logits, batch_y)
            loss_value.backward()
            optimizer.step()
            batch_examples = int(batch_y.shape[0])
            total_loss += float(loss_value.detach()) * batch_examples
            examples += batch_examples
        return {
            "train_examples": examples,
            "train_batches": len(loader),
            "train_mean_loss": total_loss / examples if examples > 0 else float("nan"),
            "model_state_dict_source": head,
        }

    def run_validation() -> tuple[float, np.ndarray]:
        return epoch_macro_f1(head, validation_x, validation_rows, class_names)

    outcome = _fit_head(
        run_train_epoch=run_train_epoch, run_validation=run_validation, epochs=epochs, patience=patience, min_delta=min_delta,
        checkpoint_rule=checkpoint_rule,
    )
    head.load_state_dict(outcome.best_state_dict)

    probe_result = score_predictions(
        outcome.best_predicted, validation_rows, class_names,
        probe=f"{type(head).__name__}_variant_head", scaling="none",
        parameters={"optimizer": "AdamW", "learning_rate": learning_rate, "weight_decay": weight_decay, "batch_size": batch_size},
        seed=seed, train_row_count=len(train_rows), fit_seconds=outcome.train_seconds, predict_seconds=0.0,
        converged=outcome.stopped_early, bootstrap_iterations=bootstrap_iterations,
    )
    return HeadTrainingResult(
        seed=seed,
        recipe=_recipe(
            learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size, epochs=epochs,
            patience=patience, min_delta=min_delta, video_level_aggregation=VIDEO_LEVEL_AGGREGATION_MAJORITY_VOTE,
        ),
        epochs=outcome.epoch_records, best_epoch=outcome.best_epoch, best_validation_macro_f1=outcome.best_metric,
        stopped_early=outcome.stopped_early, train_seconds=outcome.train_seconds, probe_result=probe_result,
    )


def train_grid_head(
    head: nn.Module,
    features_mmap: np.ndarray,
    train_indices: np.ndarray,
    train_rows: Sequence[dict[str, Any]],
    validation_indices: np.ndarray,
    validation_rows: Sequence[dict[str, Any]],
    class_names: Sequence[str],
    *,
    seed: int,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> HeadTrainingResult:
    """Train `MultiScaleSeHead` from minibatches gathered live off the patch-cache memmap.

    `features_mmap` is `features.float16.npy` opened with `mmap_mode="r"`;
    `train_indices`/`validation_indices` are integer positions into it,
    aligned 1:1 with `train_rows`/`validation_rows` (built by the caller's
    join against `rows.csv`, per `scripts/run_p1c_roi_head.py`'s pattern).
    No `[N, 256, 768]` array for a whole split is ever materialised -- only
    one minibatch's worth at a time, for training and for validation alike.
    """

    if len(train_indices) != len(train_rows):
        raise ValueError(f"train indices/rows disagree: {len(train_indices)} vs {len(train_rows)}")
    if len(validation_indices) != len(validation_rows):
        raise ValueError(f"validation indices/rows disagree: {len(validation_indices)} vs {len(validation_rows)}")

    set_local_seed(seed)
    train_y = targets_from_rows(train_rows)
    rng = np.random.default_rng(seed)
    optimizer = AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    def _gather(indices: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(features_mmap[indices], dtype=np.float32))

    def run_train_epoch(_epoch: int) -> dict[str, Any]:
        head.train()
        order = rng.permutation(len(train_indices))
        total_loss, examples, batches = 0.0, 0, 0
        for start in range(0, len(order), batch_size):
            batch_positions = order[start : start + batch_size]
            batch_x = _gather(train_indices[batch_positions])
            batch_y = torch.from_numpy(train_y[batch_positions])
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_x)
            loss_value = criterion(logits, batch_y)
            loss_value.backward()
            optimizer.step()
            batch_examples = int(batch_y.shape[0])
            total_loss += float(loss_value.detach()) * batch_examples
            examples += batch_examples
            batches += 1
        return {
            "train_examples": examples, "train_batches": batches,
            "train_mean_loss": total_loss / examples if examples > 0 else float("nan"),
            "model_state_dict_source": head,
        }

    def run_validation() -> tuple[float, np.ndarray]:
        head.eval()
        predicted = np.empty(len(validation_indices), dtype=np.int64)
        with torch.no_grad():
            for start in range(0, len(validation_indices), batch_size):
                chunk = validation_indices[start : start + batch_size]
                logits = head(_gather(chunk))
                predicted[start : start + len(chunk)] = logits.argmax(dim=1).numpy()
        return _macro_f1_from_predicted(predicted, validation_rows, class_names), predicted

    outcome = _fit_head(run_train_epoch=run_train_epoch, run_validation=run_validation, epochs=epochs, patience=patience, min_delta=min_delta)
    head.load_state_dict(outcome.best_state_dict)

    probe_result = score_predictions(
        outcome.best_predicted, validation_rows, class_names,
        probe=f"{type(head).__name__}_variant_head", scaling="none",
        parameters={"optimizer": "AdamW", "learning_rate": learning_rate, "weight_decay": weight_decay, "batch_size": batch_size},
        seed=seed, train_row_count=len(train_rows), fit_seconds=outcome.train_seconds, predict_seconds=0.0,
        converged=outcome.stopped_early, bootstrap_iterations=bootstrap_iterations,
    )
    return HeadTrainingResult(
        seed=seed,
        recipe=_recipe(
            learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size, epochs=epochs,
            patience=patience, min_delta=min_delta, video_level_aggregation=VIDEO_LEVEL_AGGREGATION_MAJORITY_VOTE,
        ),
        epochs=outcome.epoch_records, best_epoch=outcome.best_epoch, best_validation_macro_f1=outcome.best_metric,
        stopped_early=outcome.stopped_early, train_seconds=outcome.train_seconds, probe_result=probe_result,
    )


def train_sequence_head(
    head: nn.Module,
    train_video_sequences: dict[tuple[str, str], tuple[np.ndarray, dict[str, Any]]],
    validation_video_sequences: dict[tuple[str, str], tuple[np.ndarray, dict[str, Any]]],
    class_names: Sequence[str],
    *,
    frames: int,
    seed: int,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    class_weights: Sequence[float] | None = None,
    epoch_resampler: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    label_smoothing: float = 0.0,
    optimizer_param_groups: Callable[[nn.Module], list[dict[str, Any]]] | None = None,
    checkpoint_rule: str = CHECKPOINT_BEST_VALIDATION,
) -> HeadTrainingResult:
    """Train `StepHead` on per-video pooled-frame sequences (`group_pooled_sequences_by_video`).

    `epoch_resampler(epoch) -> (sequences, masks, targets)` optionally replaces the fixed training tensors with
    freshly drawn ones every epoch (the video list and its order of targets must stay the same size); `None`
    keeps the registered fixed-draw behaviour exactly. `label_smoothing=0.0` is the registered plain loss.
    `optimizer_param_groups(head) -> [{"params": ..., "lr": ...}, ...]` optionally gives AdamW per-group learning
    rates (D-67: a fine-tuned backbone tail beside the head); groups without "lr" use `learning_rate`. `None` is the
    registered single-group optimiser over `head.parameters()`.

    Every training video's sequence is padded to `frames` (it was already
    sampled to at most that many frames, so this never truncates); every
    validation video's sequence -- full, untouched, per docs/PROJECT_RULES.md rule 1 --
    is split into consecutive `frames`-length chunks and scored as one
    prediction per video via mean-logit combination across its chunks (see
    the module docstring for why). The result's `probe_result.frame_metrics`
    and `.video_level_metrics` are therefore identical: there is one row per
    VIDEO here, never one per frame.
    """

    set_local_seed(seed)
    video_keys = sorted(train_video_sequences)
    padded = [_pad_sequence(train_video_sequences[key][0], frames) for key in video_keys]
    train_sequences = torch.from_numpy(np.stack([item[0] for item in padded]))
    train_masks = torch.from_numpy(np.stack([item[1] for item in padded]))
    train_targets = torch.tensor(
        [int(train_video_sequences[key][1]["target"]) for key in video_keys], dtype=torch.long
    )

    loader = DataLoader(
        TensorDataset(train_sequences, train_masks, train_targets),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    params = head.parameters() if optimizer_param_groups is None else optimizer_param_groups(head)
    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    # `class_weights=None` is the registered unweighted loss; the combined-dataset probe passes inverse-frequency weights.
    weight = None if class_weights is None else torch.tensor(list(class_weights), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight, reduction="mean", label_smoothing=label_smoothing)

    def run_train_epoch(_epoch: int) -> dict[str, Any]:
        head.train()
        total_loss, examples = 0.0, 0
        epoch_loader = loader
        if epoch_resampler is not None:
            drawn = epoch_resampler(_epoch)
            epoch_loader = DataLoader(
                TensorDataset(*(torch.from_numpy(np.ascontiguousarray(a)) for a in drawn)),
                batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed * 100003 + _epoch),
            )
        for batch_sequences, batch_masks, batch_targets in epoch_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_sequences, key_padding_mask=batch_masks)
            loss_value = criterion(logits, batch_targets)
            loss_value.backward()
            optimizer.step()
            batch_examples = int(batch_targets.shape[0])
            total_loss += float(loss_value.detach()) * batch_examples
            examples += batch_examples
        return {
            "train_examples": examples, "train_batches": len(loader),
            "train_mean_loss": total_loss / examples if examples > 0 else float("nan"),
            "model_state_dict_source": head,
        }

    validation_video_keys = sorted(validation_video_sequences)
    validation_video_rows = [validation_video_sequences[key][1] for key in validation_video_keys]

    def run_validation() -> tuple[float, np.ndarray]:
        head.eval()
        predicted = np.empty(len(validation_video_keys), dtype=np.int64)
        with torch.no_grad():
            for position, key in enumerate(validation_video_keys):
                sequence, _row = validation_video_sequences[key]
                chunks = [_pad_sequence(chunk, frames) for chunk in _chunk_sequence(sequence, frames)]
                chunk_sequences = torch.from_numpy(np.stack([item[0] for item in chunks]))
                chunk_masks = torch.from_numpy(np.stack([item[1] for item in chunks]))
                logits = head(chunk_sequences, key_padding_mask=chunk_masks)
                predicted[position] = int(logits.mean(dim=0).argmax().item())
        return _macro_f1_from_predicted(predicted, validation_video_rows, class_names), predicted

    outcome = _fit_head(
        run_train_epoch=run_train_epoch, run_validation=run_validation, epochs=epochs, patience=patience, min_delta=min_delta,
        checkpoint_rule=checkpoint_rule,
    )
    head.load_state_dict(outcome.best_state_dict)

    probe_result = score_predictions(
        outcome.best_predicted, validation_video_rows, class_names,
        probe="StepHead_variant_head", scaling="none",
        parameters={"optimizer": "AdamW", "learning_rate": learning_rate, "weight_decay": weight_decay, "batch_size": batch_size, "frames": frames},
        seed=seed, train_row_count=len(video_keys), fit_seconds=outcome.train_seconds, predict_seconds=0.0,
        converged=outcome.stopped_early, bootstrap_iterations=bootstrap_iterations,
    )
    return HeadTrainingResult(
        seed=seed,
        recipe=_recipe(
            learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size, epochs=epochs,
            patience=patience, min_delta=min_delta, video_level_aggregation=VIDEO_LEVEL_AGGREGATION_STEP_DIRECT,
        ),
        epochs=outcome.epoch_records, best_epoch=outcome.best_epoch, best_validation_macro_f1=outcome.best_metric,
        stopped_early=outcome.stopped_early, train_seconds=outcome.train_seconds, probe_result=probe_result,
    )
