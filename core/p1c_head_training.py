"""B-1x: train the registered linear head on cached features, on CPU.

Why this module exists
-----------------------
Every existing P1C head number comes from `scripts/run_p1c_frame_baseline.py`,
which reads pixels through a frozen or fine-tuned backbone on a GPU. Once the
backbone is frozen (P1C-VIDEO-FROZEN-CONFIRM-015 and everything after it), the
2048-D vector it emits for a given frame is bit-identical on every epoch --
`core/feature_cache.py`'s whole reason for existing -- so a head-only fit is a
small matmul over a fixed matrix and belongs on CPU, not on the one shared MIG
slice. This module is that CPU trainer.

It reproduces the CONFIRM-015 recipe exactly, as registered in
`configs/experiments/p1c/P1C-VIDEO-FROZEN-CONFIRM-015-AR-FOLD1.yaml`:
optimizer AdamW(lr=3e-4, weight_decay=1e-4), no scheduler, unweighted
`nn.CrossEntropyLoss(reduction="mean")`, batch_size 64 shuffled every epoch,
a 30-epoch ceiling, early stopping on validation macro_f1 (mode max, patience
5, min_delta 0.001), and `best_validation_macro_f1` selection -- the BEST
epoch's metrics are reported, not the last.

One declared deviation: this trains in float32 on CPU. CONFIRM-015 trained
under `amp_bfloat16` on CUDA (`core/precision.py`). That is a real numerical
difference (autocast dtype and CPU-vs-cuDNN kernel nondeterminism, K-24), not
an implementation shortcut, and `scripts/run_p1c_cached_head.py`'s validation
run against fold 1 is precisely the check for whether it matters.

Head architecture
------------------
`models.classifiers.GlobalAverageLinearHead` is NOT reused as-is: its
`forward` unconditionally does `AdaptiveAvgPool2d` then `Linear`, and requires
a `[B, C, H, W]` spatial tensor (`c5.ndim != 4` raises). The cached features
this module trains on are already the pooled `[N, D]` vectors -- that pooling
is exactly what the cache stores, per `core/feature_cache.py`'s
`FEATURE_CACHE_POOLING_BY_FAMILY`. Reshaping a pooled vector back into a fake
`[N, D, 1, 1]` map just to satisfy `forward()`'s pooling call would be the
same class of mistake docs/PROJECT_RULES.md's SEAMCNN entry warns against for a different
head (never reshape a pooled vector into spatial data). The correct reuse is
therefore the bare `nn.Linear` `GlobalAverageLinearHead` wraps as
`.classifier`: constructing the full module and using only `.classifier`
keeps the in/out dimensions and initialization identical to what the GPU run
used, without exercising the pooling path a pre-pooled vector must never see.

Scoring is delegated to `core.p1c_probes.score_predictions` (rule of two --
that function already builds `FramePrediction` objects with the NUL-separated
composite source-video key, frame metrics, clustered CI, and video-level
majority vote exactly as a training run needs). The per-epoch monitor metric
does *not* go through it: `core.p1c_engine.FrameClassifierEngine.evaluate`
only pays for the clustered bootstrap on the final selected checkpoint, never
per training epoch, and this module mirrors that -- 30 epochs x 1000 bootstrap
iterations would burn CPU time on a number no one reads.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from core.p1c_losses import LOSS_NAMES, build_loss, class_counts
from core.p1c_probes import ProbeResult, score_predictions
from eval.metrics import classification_metrics
from models.classifiers import GlobalAverageLinearHead

#: The registered CONFIRM-015 recipe. Every default below is read from
#: `configs/experiments/p1c/P1C-VIDEO-FROZEN-CONFIRM-015-AR-FOLD1.yaml`'s
#: `fixed_protocol.training` block, not re-derived.
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 30
DEFAULT_PATIENCE = 5
DEFAULT_MIN_DELTA = 0.001
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class HeadTrainingResult:
    """One seed's cached-feature head training run, scored like a real one."""

    seed: int
    recipe: dict[str, Any]
    epochs: tuple[dict[str, Any], ...]
    best_epoch: int
    best_validation_macro_f1: float
    stopped_early: bool
    train_seconds: float
    probe_result: ProbeResult
    #: Best-epoch per-frame validation class probabilities, ``(n_validation,
    #: n_classes)`` float, in `validation_rows` order. Populated only when
    #: `train_cached_head(..., capture_validation_probabilities=True)`; left
    #: None otherwise so the default path and its JSON are byte-unchanged.
    #: Consumed by `eval.video_probability_aggregation` (backlog B-6); it is
    #: bulk array data, so `to_json` drops it rather than inlining it -- a
    #: caller that wants it writes it to a sidecar ``.npy``.
    best_validation_probabilities: Any = None
    #: `{"sum": ..., "zero_weight_rows": ...}` when `train_cached_head(...,
    #: train_weights=...)` was given a weight array, so the effective
    #: training set size (rows that actually contributed gradient) is
    #: visible next to the row count. None -- and dropped from `to_json`,
    #: same as when the key never existed -- when weighting was not used.
    train_weight_summary: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload.pop("best_validation_probabilities", None)
        if payload.get("train_weight_summary") is None:
            payload.pop("train_weight_summary", None)
        payload["probe_result"] = self.probe_result.to_json()
        return payload


def set_local_seed(seed: int) -> None:
    """Seed python/numpy/torch exactly as `scripts/run_p1c_frame_baseline.py` does.

    `core.reproducibility.seed_everything` also exists and additionally flips
    `torch.use_deterministic_algorithms(True)` process-wide -- a global switch
    this module deliberately does not touch, both because the GPU recipe being
    reproduced never set it either, and because it would leak into every other
    test or script sharing this interpreter. Matching the runner's own local
    seeding is the conservative choice here.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#: Back-compat internal alias, see `targets_from_rows` below for the pattern;
#: `core.p1c_head_variants` imports the public name.
_set_seed = set_local_seed


def targets_from_rows(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """`int64` target-class-index array from cache-shaped rows. Public: `core.p1c_head_variants` reuses it (rule of two)."""

    return np.asarray([int(row["target"]) for row in rows], dtype=np.int64)


#: Back-compat internal alias -- every call site inside this module predates
#: the public name and is left untouched.
_targets = targets_from_rows


def epoch_macro_f1(model: nn.Module, features: torch.Tensor, rows: Sequence[dict[str, Any]], class_names: Sequence[str]) -> tuple[float, np.ndarray]:
    """The per-epoch monitor metric: frame macro-F1, no clustered bootstrap.

    Mirrors `FrameClassifierEngine.evaluate(..., bootstrap_iterations=None)`,
    which the training loop calls every epoch precisely to avoid paying for a
    bootstrap the early-stopping/selection decision never uses. Public:
    `core.p1c_head_variants` reuses it for heads whose validation input fits
    in memory as one tensor (rule of two); heads that must stream their
    validation predictions in minibatches score epochs separately.
    """

    model.eval()
    with torch.no_grad():
        predicted = model(features).argmax(dim=1).numpy()
    truth_labels = [class_names[int(row["target"])] for row in rows]
    predicted_labels = [class_names[index] for index in predicted]
    return classification_metrics(truth_labels, predicted_labels, class_names)["macro_f1"], predicted


#: Back-compat internal alias, see `targets_from_rows` above.
_epoch_macro_f1 = epoch_macro_f1


def train_cached_head(
    train_features: np.ndarray,
    train_rows: Sequence[dict[str, Any]],
    validation_features: np.ndarray,
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
    capture_validation_probabilities: bool = False,
    train_weights: np.ndarray | None = None,
    loss: str = "cross_entropy",
    focal_gamma: float = 2.0,
) -> HeadTrainingResult:
    """Train the registered linear head on cached features and score the best epoch.

    `train_features`/`validation_features` are the `[N, D]` arrays
    `core.feature_cache.load_feature_cache` returns; `train_rows`/
    `validation_rows` are its matching per-row provenance records.

    `train_weights`, when given, is one non-negative float per training row
    (`core.p1c_background_mask.background_weights` is the intended producer).
    A row's cross-entropy loss is scaled by its weight before averaging, so a
    weight of 0 removes that row's influence on every gradient step without
    removing it from the batch shuffle or the row count. When None (the
    default), behaviour is byte-identical to before this argument existed:
    the same `reduction="mean"` criterion object, the same arithmetic, no
    weight-related code on the hot path at all.

    `loss` (P3B, `core.p1c_losses`) selects the per-example criterion; it
    defaults to `"cross_entropy"`, which -- like `train_weights=None` above --
    keeps the hot path byte-identical to every earlier run: the exact same
    `nn.CrossEntropyLoss(reduction=...)` object is built, no other code from
    `core.p1c_losses` runs. Any other value in `core.p1c_losses.LOSS_NAMES`
    swaps in `weighted_cross_entropy`, `focal` (tuned by `focal_gamma`) or
    `balanced_softmax` instead; per-row `train_weights`, when also given,
    still multiply the (now non-CE) per-sample loss and divide by the batch
    weight sum exactly as before -- the two mechanisms compose.
    """

    if loss not in LOSS_NAMES:
        raise ValueError(f"loss must be one of {LOSS_NAMES}; got {loss!r}")
    if len(train_features) != len(train_rows):
        raise ValueError(f"train features/rows disagree: {len(train_features)} vs {len(train_rows)}")
    if len(validation_features) != len(validation_rows):
        raise ValueError(
            f"validation features/rows disagree: {len(validation_features)} vs {len(validation_rows)}"
        )
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if train_weights is not None:
        train_weights = np.asarray(train_weights, dtype=np.float64)
        if train_weights.shape != (len(train_rows),):
            raise ValueError(
                f"train_weights length {train_weights.shape} disagrees with {len(train_rows)} train rows"
            )
        if not np.all(np.isfinite(train_weights)):
            raise ValueError("train_weights must all be finite")
        if np.any(train_weights < 0):
            raise ValueError("train_weights must all be >= 0")
        if not np.any(train_weights > 0):
            raise ValueError("train_weights must have at least one value > 0")

    _set_seed(seed)

    in_channels = train_features.shape[1]
    num_classes = len(class_names)
    # The full module is built only to inherit its exact Linear shape/init;
    # see the module docstring for why `.forward` (which pools) is never
    # called on already-pooled cached features.
    model = GlobalAverageLinearHead(in_channels, num_classes).classifier

    train_x = torch.from_numpy(np.ascontiguousarray(train_features, dtype=np.float32))
    train_y = torch.from_numpy(_targets(train_rows))
    validation_x = torch.from_numpy(np.ascontiguousarray(validation_features, dtype=np.float32))

    loader_generator = torch.Generator().manual_seed(seed)
    if train_weights is None:
        dataset: TensorDataset = TensorDataset(train_x, train_y)
    else:
        # A weight tensor riding alongside features/labels in the same
        # TensorDataset is shuffled by the same DataLoader permutation each
        # epoch, so it always stays aligned to the row it was computed for.
        train_w = torch.from_numpy(np.ascontiguousarray(train_weights, dtype=np.float32))
        dataset = TensorDataset(train_x, train_y, train_w)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    reduction = "mean" if train_weights is None else "none"
    if loss == "cross_entropy":
        # Literally the same construction as before this argument existed --
        # no other line in core.p1c_losses runs on this path.
        criterion = nn.CrossEntropyLoss(reduction=reduction)
    else:
        criterion = build_loss(loss, counts=class_counts(train_rows, class_names), gamma=focal_gamma, reduction=reduction)

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
        model.train()
        total_loss = 0.0
        examples = 0
        for batch in loader:
            if train_weights is None:
                batch_x, batch_y = batch
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss_value = criterion(logits, batch_y)
                loss_value.backward()
                optimizer.step()
            else:
                batch_x, batch_y, batch_w = batch
                weight_sum = float(batch_w.sum())
                if weight_sum <= 0.0:
                    # A zero-sum batch (every row in it weighted to 0) carries
                    # no gradient signal; dividing by its weight sum would be
                    # a division by zero, and there is nothing to normalise
                    # against, so this batch's optimizer step is skipped
                    # entirely rather than stepping on a manufactured loss.
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                per_sample_loss = criterion(logits, batch_y)
                loss_value = (per_sample_loss * batch_w).sum() / weight_sum
                loss_value.backward()
                optimizer.step()
            batch_size_actual = int(batch_y.shape[0])
            total_loss += float(loss_value.detach()) * batch_size_actual
            examples += batch_size_actual

        metric, predicted = _epoch_macro_f1(model, validation_x, validation_rows, class_names)
        epoch_records.append(
            {
                "epoch": epoch,
                "train_examples": examples,
                "train_batches": len(loader),
                "train_mean_loss": total_loss / examples if examples > 0 else float("nan"),
                "validation_macro_f1": metric,
            }
        )
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_predicted = predicted
        if metric > stopping_anchor + min_delta:
            stopping_anchor, stagnant_epochs = metric, 0
        else:
            stagnant_epochs += 1
        if stagnant_epochs >= patience:
            stopped_early = True
            break
    train_seconds = time.perf_counter() - started

    if best_state_dict is None or best_predicted is None:
        raise RuntimeError("no validation epoch was scored; cannot select a checkpoint")

    model.load_state_dict(best_state_dict)

    # Inference-only, after training and RNG use are complete, so this cannot
    # perturb the default path: the softmax of the best epoch's logits over the
    # validation split, for `eval.video_probability_aggregation` (B-6).
    best_validation_probabilities = None
    if capture_validation_probabilities:
        model.eval()
        with torch.no_grad():
            best_validation_probabilities = (
                torch.softmax(model(validation_x), dim=1).detach().cpu().numpy()
            )

    probe_result = score_predictions(
        best_predicted,
        validation_rows,
        class_names,
        probe="cached_linear_head",
        scaling="none",
        parameters={
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs_ceiling": epochs,
            "early_stopping_patience": patience,
            "early_stopping_min_delta": min_delta,
            "precision": "float32_cpu",
        },
        seed=seed,
        train_row_count=len(train_rows),
        fit_seconds=train_seconds,
        predict_seconds=0.0,
        # Repurposed from sklearn's "did the solver converge": here, whether
        # training stopped from patience being exhausted (True) rather than
        # hitting the epoch ceiling with the anchor still improving (False).
        converged=stopped_early,
        bootstrap_iterations=bootstrap_iterations,
    )

    return HeadTrainingResult(
        seed=seed,
        recipe={
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler": "none",
            "loss": f"{loss}_mean_unweighted" if train_weights is None else f"{loss}_sample_weighted",
            "loss_name": loss,
            "focal_gamma": focal_gamma,
            "batch_size": batch_size,
            "epochs_ceiling": epochs,
            "early_stopping_monitor": "macro_f1",
            "early_stopping_mode": "max",
            "early_stopping_patience": patience,
            "early_stopping_min_delta": min_delta,
            "selection_rule": "best_validation_macro_f1",
            "precision": "float32_cpu",
        },
        epochs=tuple(epoch_records),
        best_epoch=best_epoch,
        best_validation_macro_f1=best_metric,
        stopped_early=stopped_early,
        train_seconds=train_seconds,
        probe_result=probe_result,
        best_validation_probabilities=best_validation_probabilities,
        train_weight_summary=(
            None
            if train_weights is None
            else {
                "sum": float(train_weights.sum()),
                "zero_weight_rows": int(np.sum(train_weights == 0.0)),
            }
        ),
    )
