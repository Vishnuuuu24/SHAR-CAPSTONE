"""Generic, provenance-preserving execution primitives for P1C frame controls.

This module deliberately contains no architecture, optimizer, loss, data-loader,
schedule, augmentation, checkpoint-selection, or device-policy default.  A
registered P1C runner must supply those choices explicitly.  Its narrow job is
to make one ordinary frame-classification epoch, deterministic evaluation
records, and exact resume state difficult to implement incorrectly.
"""

from __future__ import annotations

import copy
import os
import random
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from core.precision import PrecisionPolicy
from core.throughput import ThroughputPolicy

from core.device import accelerator_rng_state, restore_accelerator_rng_state
from eval.metrics import classification_metrics, video_clustered_macro_f1_ci


_INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
_CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class EpochTrainingResult:
    """Non-scientific accounting emitted by a completed training epoch."""

    examples: int
    batches: int
    mean_loss: float
    # Optimizer steps the gradient scaler discarded because the gradients
    # overflowed float16's range. Zero for float32 and bfloat16, which are not
    # scaled at all. Non-zero is not an error -- it is the scaler working --
    # but it means the epoch applied fewer updates than it ran batches, so it
    # is reported rather than swallowed (docs/PROJECT_RULES.md rule 4).
    skipped_optimizer_steps: int = 0


@dataclass(frozen=True)
class FramePrediction:
    """One evaluation prediction with the source-video key retained."""

    source_dataset: str
    source_video_id: str
    target_index: int
    predicted_index: int


@dataclass(frozen=True)
class EvaluationResult:
    """Deterministic metrics plus per-frame source-video traceability."""

    predictions: tuple[FramePrediction, ...]
    metrics: dict[str, Any]
    video_clustered_macro_f1_ci: dict[str, Any] | None

    @property
    def source_video_ids(self) -> tuple[str, ...]:
        """Globally unambiguous source-video keys for clustered resampling."""

        return tuple(
            f"{prediction.source_dataset}\0{prediction.source_video_id}" for prediction in self.predictions
        )


@dataclass(frozen=True)
class ResumeState:
    """Exact progression and RNG state required to resume a caller-owned run."""

    next_epoch: int
    global_step: int
    extra_state: dict[str, Any]
    rng_state: dict[str, Any]


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(class_names, (str, bytes)) or not isinstance(class_names, Sequence):
        raise TypeError("class_names must be an ordered sequence of strings")
    result = tuple(class_names)
    if len(result) < 2 or any(not isinstance(name, str) or not name for name in result):
        raise ValueError("class_names must contain at least two non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("class_names must not contain duplicates")
    return result


def _device(value: torch.device | str) -> torch.device:
    if not isinstance(value, (torch.device, str)):
        raise TypeError("device must be an explicit torch.device or device string")
    return torch.device(value)


def _source_video_ids(value: object, batch_size: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("batch source_video_id must be an ordered sequence")
    identifiers = tuple(value)
    if len(identifiers) != batch_size or any(not isinstance(item, str) or not item for item in identifiers):
        raise ValueError("batch source_video_id must contain one non-empty string per image")
    return identifiers


def _source_datasets(value: object, batch_size: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("batch source_dataset must be an ordered sequence")
    datasets = tuple(value)
    if len(datasets) != batch_size or any(not isinstance(item, str) or not item for item in datasets):
        raise ValueError("batch source_dataset must contain one non-empty string per image")
    return datasets


def _batch_from_mapping(batch: object, device: torch.device) -> tuple[Tensor, Tensor, tuple[str, ...], tuple[str, ...]]:
    if not isinstance(batch, Mapping):
        raise TypeError("P1C engine batches must be mappings")
    missing = {"image", "target", "source_dataset", "source_video_id"} - set(batch)
    if missing:
        raise ValueError(f"P1C engine batch lacks required fields: {sorted(missing)}")
    images = batch["image"]
    targets = batch["target"]
    if not isinstance(images, Tensor) or images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("batch image must be a non-empty [batch,channels,height,width] tensor")
    if not isinstance(targets, Tensor) or targets.ndim != 1 or targets.shape[0] != images.shape[0]:
        raise ValueError("batch target must be a [batch] tensor aligned with image")
    if targets.dtype not in _INTEGER_DTYPES:
        raise TypeError("batch target must use an integer dtype")
    source_video_ids = _source_video_ids(batch["source_video_id"], images.shape[0])
    source_datasets = _source_datasets(batch["source_dataset"], images.shape[0])
    return images.to(device), targets.to(device), source_datasets, source_video_ids


def _checked_logits(logits: object, targets: Tensor, class_names: tuple[str, ...]) -> Tensor:
    if not isinstance(logits, Tensor) or logits.ndim != 2:
        raise ValueError("frame classifier must return a [batch,num_classes] logits tensor")
    if logits.shape != (targets.shape[0], len(class_names)):
        raise ValueError(
            "frame classifier logits shape must match batch/class_names: "
            f"got {tuple(logits.shape)}, expected {(targets.shape[0], len(class_names))}"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("frame classifier produced non-finite logits")
    if torch.any(targets < 0) or torch.any(targets >= len(class_names)):
        raise ValueError("batch target index is outside class_names")
    return logits


class FrameClassifierEngine:
    """Execute caller-specified frame classification choices without defaults."""

    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer: Optimizer,
        loss: Callable[[Tensor, Tensor], Tensor],
        loss_reduction: str,
        device: torch.device | str,
        class_names: Sequence[str],
        precision: PrecisionPolicy,
        throughput: ThroughputPolicy,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be a caller-supplied torch Optimizer")
        if not callable(loss):
            raise TypeError("loss must be a caller-supplied callable")
        if loss_reduction != "mean":
            raise ValueError("P1C loss_reduction must be explicitly registered as mean")
        if not isinstance(precision, PrecisionPolicy):
            raise TypeError(
                "precision must be a core.precision.PrecisionPolicy resolved from a "
                "registered training.precision; the engine does not choose one"
            )
        if not isinstance(throughput, ThroughputPolicy):
            raise TypeError(
                "throughput must be a core.throughput.ThroughputPolicy resolved from a "
                "registered training.throughput; the engine does not choose one"
            )
        self.optimizer = optimizer
        self.loss = loss
        self.loss_reduction = loss_reduction
        self.device = _device(device)
        self.class_names = _class_names(class_names)
        self.precision = precision
        self.throughput = throughput
        # One scaler per engine: it carries mutable scale state, so sharing it
        # would couple two runs' training.
        self._scaler = precision.make_grad_scaler()
        model.to(self.device)
        # `torch.compile` (inside `prepare_model`, below) can wrap the module
        # in an `OptimizedModule` whose `state_dict()` keys are all prefixed
        # `_orig_mod.`. Captured here, before that wrapping, so checkpoints
        # always save/load against the uncompiled module and a run compiled
        # under one throughput policy stays loadable by a run under another.
        # See `checkpoint_module`.
        self._checkpoint_module = model
        self.model = throughput.prepare_model(model)

    @property
    def checkpoint_module(self) -> nn.Module:
        """The uncompiled module identity to pass to save/load checkpoint helpers.

        Never pass `self.model` to `save_training_checkpoint` /
        `load_training_checkpoint` when a throughput policy compiles it: its
        `state_dict()` keys would carry `torch.compile`'s `_orig_mod.` prefix,
        silently producing a checkpoint an uncompiled run cannot load.
        """
        return self._checkpoint_module

    def train_epoch(self, batches: Iterable[Mapping[str, object]]) -> EpochTrainingResult:
        """Run one explicit caller-owned epoch under the registered precision.

        No scheduler is implied. Precision is not implied either: it comes from
        the `PrecisionPolicy` the caller resolved from its config, and
        `float32` reproduces the original float32-only behaviour exactly
        (`nullcontext`, no scaler, no skipped steps).

        Only the forward pass and the loss run under autocast. Backward is
        deliberately outside it: it replays the dtypes the forward recorded,
        and wrapping it would be a no-op at best.

        The registered `ThroughputPolicy` also applies here: memory format
        and (if compiled) kernel selection change which CUDA kernels run, not
        the arithmetic they compute. Results are therefore deterministic
        *within* one throughput policy, but are not expected to be bitwise
        identical to a run under a different one -- the same caveat autocast
        already carries for precision.
        """

        self.model.train(True)
        total_loss = 0.0
        examples = 0
        batch_count = 0
        skipped_steps = 0
        for batch in batches:
            images, targets, _, _ = _batch_from_mapping(batch, self.device)
            images = self.throughput.prepare_input(images)
            self.optimizer.zero_grad(set_to_none=True)
            with self.precision.autocast():
                logits = _checked_logits(self.model(images), targets, self.class_names)
                loss_value = self.loss(logits, targets)
            if not isinstance(loss_value, Tensor) or loss_value.ndim != 0:
                raise ValueError("loss must return a scalar torch.Tensor")
            # A non-finite *loss* is a real fault in every policy: torch
            # autocasts cross entropy back to float32, so this is not the
            # float16 gradient underflow the scaler exists to absorb.
            if not torch.isfinite(loss_value):
                raise ValueError("loss is non-finite")
            if self._scaler is None:
                loss_value.backward()
                self.optimizer.step()
            else:
                self._scaler.scale(loss_value).backward()
                # A discarded step is inferred from the scale dropping:
                # update() lowers the scale by backoff_factor exactly when the
                # unscaled gradients held inf/NaN, and only raises it
                # otherwise. torch exposes no public "was that step applied"
                # flag, so this is the supported reading.
                scale_before = self._scaler.get_scale()
                self._scaler.step(self.optimizer)
                self._scaler.update()
                if self._scaler.get_scale() < scale_before:
                    skipped_steps += 1
            batch_size = int(targets.shape[0])
            total_loss += float(loss_value.detach().cpu()) * batch_size
            examples += batch_size
            batch_count += 1
        if not batch_count:
            raise ValueError("cannot train on an empty batch iterable")
        return EpochTrainingResult(
            examples=examples,
            batches=batch_count,
            mean_loss=total_loss / examples,
            skipped_optimizer_steps=skipped_steps,
        )

    def evaluate(
        self,
        batches: Iterable[Mapping[str, object]],
        *,
        bootstrap_iterations: int | None,
        bootstrap_seed: int | None,
    ) -> EvaluationResult:
        """Evaluate frames deterministically while retaining every source-video ID.

        Cluster bootstrap is optional but never implicit: callers provide both
        values or explicitly pass ``None`` for both values.
        """

        if (bootstrap_iterations is None) != (bootstrap_seed is None):
            raise ValueError("bootstrap_iterations and bootstrap_seed must both be supplied or both be None")
        if bootstrap_iterations is not None:
            _non_negative_int(bootstrap_iterations, "bootstrap_iterations")
            if bootstrap_iterations < 100:
                raise ValueError("bootstrap_iterations must be at least 100")
            _non_negative_int(bootstrap_seed, "bootstrap_seed")

        prior_mode = self.model.training
        self.model.train(False)
        predictions: list[FramePrediction] = []
        try:
            # Evaluate under the same precision the run trained in: scoring a
            # bfloat16-trained model in float32 would report numerics the run
            # never used. argmax is robust to the dtype either way.
            with torch.no_grad(), self.precision.autocast():
                for batch in batches:
                    images, targets, source_datasets, source_video_ids = _batch_from_mapping(batch, self.device)
                    images = self.throughput.prepare_input(images)
                    logits = _checked_logits(self.model(images), targets, self.class_names)
                    predicted = logits.argmax(dim=1)
                    for source_dataset, source_video_id, target, predicted_index in zip(
                        source_datasets,
                        source_video_ids,
                        targets.detach().cpu().tolist(),
                        predicted.detach().cpu().tolist(),
                        strict=True,
                    ):
                        predictions.append(
                            FramePrediction(
                                source_dataset=source_dataset,
                                source_video_id=source_video_id,
                                target_index=int(target),
                                predicted_index=int(predicted_index),
                            )
                        )
        finally:
            self.model.train(prior_mode)
        if not predictions:
            raise ValueError("cannot evaluate an empty batch iterable")

        truth = [self.class_names[prediction.target_index] for prediction in predictions]
        predicted = [self.class_names[prediction.predicted_index] for prediction in predictions]
        source_video_ids = [f"{prediction.source_dataset}\0{prediction.source_video_id}" for prediction in predictions]
        clustered = None
        if bootstrap_iterations is not None:
            clustered = video_clustered_macro_f1_ci(
                truth,
                predicted,
                source_video_ids,
                self.class_names,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            )
        return EvaluationResult(
            predictions=tuple(predictions),
            metrics=classification_metrics(truth, predicted, self.class_names),
            video_clustered_macro_f1_ci=clustered,
        )


def capture_resume_state(*, next_epoch: int, global_step: int, extra_state: Mapping[str, Any]) -> ResumeState:
    """Capture exact interpreter and torch RNG state with explicit progress data."""

    _non_negative_int(next_epoch, "next_epoch")
    _non_negative_int(global_step, "global_step")
    if not isinstance(extra_state, Mapping):
        raise TypeError("extra_state must be an explicit mapping")
    rng_state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    # Accelerator RNG is captured through the backend dispatcher so that MPS —
    # the backend these runs actually execute on — is preserved alongside CUDA.
    rng_state.update(accelerator_rng_state())
    return ResumeState(
        next_epoch=next_epoch,
        global_step=global_step,
        extra_state=copy.deepcopy(dict(extra_state)),
        rng_state=rng_state,
    )


def _validate_resume_state(state: object) -> ResumeState:
    if not isinstance(state, ResumeState):
        raise TypeError("state must be a ResumeState from capture_resume_state")
    _non_negative_int(state.next_epoch, "next_epoch")
    _non_negative_int(state.global_step, "global_step")
    if not isinstance(state.extra_state, dict) or not isinstance(state.rng_state, dict):
        raise TypeError("ResumeState state fields are invalid")
    required_rng = {"python", "numpy", "torch_cpu"}
    if required_rng - set(state.rng_state):
        raise ValueError(f"ResumeState lacks RNG fields: {sorted(required_rng - set(state.rng_state))}")
    return state


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    state: ResumeState,
    allow_overwrite: bool,
) -> None:
    """Atomically save caller-owned model, optimizer, progression, and RNG state."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not isinstance(model, nn.Module) or not isinstance(optimizer, Optimizer):
        raise TypeError("model and optimizer must be torch objects")
    if not isinstance(allow_overwrite, bool):
        raise TypeError("allow_overwrite must be explicit bool")
    _validate_resume_state(state)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "resume_state": state,
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _restore_rng_state(rng_state: Mapping[str, Any]) -> None:
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    restore_accelerator_rng_state(dict(rng_state))


def load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    device: torch.device | str,
    restore_rng: bool,
) -> ResumeState:
    """Restore a saved state exactly; RNG restoration is deliberately explicit."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    if not isinstance(model, nn.Module) or not isinstance(optimizer, Optimizer):
        raise TypeError("model and optimizer must be torch objects")
    if not isinstance(restore_rng, bool):
        raise TypeError("restore_rng must be explicit bool")
    map_location = _device(device)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported or malformed P1C checkpoint")
    for key in ("model_state", "optimizer_state", "resume_state"):
        if key not in payload:
            raise ValueError(f"malformed P1C checkpoint lacks {key}")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(map_location)
    optimizer.load_state_dict(payload["optimizer_state"])
    state = _validate_resume_state(payload["resume_state"])
    if restore_rng:
        _restore_rng_state(state.rng_state)
    return state
