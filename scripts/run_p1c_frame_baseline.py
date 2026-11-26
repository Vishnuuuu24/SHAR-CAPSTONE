#!/usr/bin/env python3
"""Run the registered P1C frame baseline without Test-driven selection.

This script intentionally has a narrow contract: it accepts only a fully
registered GAP-linear P1C experiment, writes one immutable lifecycle closure
per seed, and evaluates validation only.  Held-out Test evaluation belongs to
the subsequent frozen-checkpoint evaluator, not this training command.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import psutil
import torch
import yaml
from torch import nn
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.device import device_allocated_bytes, memory_report, reset_peak_memory, resolve_device_policy, synchronize
from core.precision import resolve_precision_policy
from core.throughput import resolve_throughput_policy
from core.p1c_engine import FrameClassifierEngine, capture_resume_state, load_training_checkpoint, save_training_checkpoint
from core.ioaccounting import IoSnapshot, io_delta
from core.p1c_readiness import (
    event_only_test_registration,
    sha256_file,
    validate_p1c_full_run_config,
    validate_p1c_screening_run_config,
)
from core.progress import log_phase
from core.run_lifecycle import RunLifecycle, artifact_digest
from data.augmentation import NONE_AUGMENTATION, resolve_augmentation_policy
from data.p1c_loader import (
    FrameLoaderOptions,
    FramePreprocessing,
    P1CFrameDataset,
    P1CPackedFrameDataset,
    build_p1c_frame_dataloader,
)
from data import pixel_source
from data.pixel_source import (
    extract_closure_path as _extract_closure_path,
    pack_closure_path as _pack_closure_path,
)
from data.video_dataset import P1CVideoFrameDataset
from eval.video_level import aggregate_video_level_predictions
from models.classifiers import (
    FrameClassifier,
    GlobalAverageLinearHead,
    ResNet50C5,
    apply_backbone_trainable_mode,
    count_parameters,
)


# Which readiness validator a registered status selects. A status this runner
# does not know is refused rather than defaulted to either one: a screening
# config silently validated as a full run would evaluate Test, and a full-run
# config silently validated as a screening run would accept one seed as a
# headline. Both are exactly the silent-degradation class this project refuses.
_VALIDATOR_BY_STATUS = {
    "REGISTERED_FULL_RUN_READY": ("full", validate_p1c_full_run_config),
    "REGISTERED_SCREENING_RUN_READY": ("screening", validate_p1c_screening_run_config),
}


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("P1C configuration must be a YAML mapping")
    experiment = config.get("experiment")
    status = experiment.get("status") if isinstance(experiment, dict) else None
    if status not in _VALIDATOR_BY_STATUS:
        raise RuntimeError(
            f"experiment.status must be one of {sorted(_VALIDATOR_BY_STATUS)}; got {status!r}"
        )
    run_class, validator = _VALIDATOR_BY_STATUS[status]
    errors = validator(config, base_dir=REPO_ROOT)
    if errors:
        raise RuntimeError(f"P1C {run_class}-run readiness is not closed:\n- " + "\n- ".join(errors))
    protocol = config.get("fixed_protocol")
    if not isinstance(protocol, dict):  # Defended by the readiness validator; keeps this runner self-contained.
        raise RuntimeError("registered P1C config lacks fixed_protocol")
    return config, protocol, run_class


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _utc_run_id(experiment_id: str, seed: int) -> str:
    safe_id = experiment_id.lower().replace("_", "-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_id}-s{seed}-{stamp}"


def _local_resnet50(weights: dict[str, Any]) -> ResNet50C5:
    path = REPO_ROOT / str(weights["path"])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("registered ResNet50 weight file is not a state dictionary")
    return ResNet50C5.from_local_weights(payload, weights_name=str(weights["enum"]))


def _packed_blob_sha256(protocol: dict[str, Any]) -> str | None:
    """The registered D-33 blob SHA-256 for the run manifest, or ``None`` when unpacked.

    Thin wrapper over ``data.pixel_source.packed_blob_sha256``; kept as a
    module-level name because this script calls it before a lifecycle exists.
    """

    return pixel_source.packed_blob_sha256(protocol)


def _video_blob_sha256(protocol: dict[str, Any]) -> str | None:
    """The registered D-34/D-35 blob SHA-256 for the run manifest, or ``None`` if absent."""

    return pixel_source.video_blob_sha256(protocol)


def _make_datasets(protocol: dict[str, Any], *, seed: int) -> tuple[P1CFrameDataset, P1CFrameDataset]:
    data = protocol["data"]
    roots = {name: REPO_ROOT / relative for name, relative in data["dataset_roots"].items()}
    preprocessing = FramePreprocessing.from_config(protocol["preprocessing"])
    classes = protocol["model"]["class_names"]
    verify = protocol["training"]["loader"]["verify_digests"]
    # Readiness already refused an unregistered/malformed block; resolving it
    # again here is what actually turns the registration into pixels. Only the
    # TRAIN dataset below is ever constructed with this -- validation always
    # gets no `augmentation=` argument at all, which is what makes it
    # structurally impossible to augment validation from this call site (see
    # data/augmentation.py's module docstring for the second layer, enforced
    # inside the dataset classes themselves).
    train_augmentation = resolve_augmentation_policy(protocol["augmentation"])
    pixel_source.refuse_if_both_registered(data)
    # K-21: a blob-derived manifest with neither pixel source registered used
    # to fall through to per-file loading and re-hash every source MP4 once
    # per manifest row instead of failing. Refuse before either branch below
    # can be reached.
    pixel_source.refuse_if_blob_manifest_lacks_pixel_source(
        REPO_ROOT / data["train"]["path"], data, label="train manifest"
    )
    pixel_source.refuse_if_blob_manifest_lacks_pixel_source(
        REPO_ROOT / data["validation"]["path"], data, label="validation manifest"
    )
    extract_closure_path = _extract_closure_path(data)
    pack_closure_path = _pack_closure_path(data)
    if extract_closure_path is not None:
        # D-34/D-35: as with the D-33 packed-blob path below, the blob's own
        # SHA-256 is verified once at construction rather than per row, so
        # verify_digests must still be registered true for the same reason --
        # a config may not claim a weaker posture than the run performs.
        pixel_source.require_verify_digests_true(protocol, field="video_frames")
        train = P1CVideoFrameDataset(
            REPO_ROOT / data["train"]["path"], closure_path=extract_closure_path,
            preprocessing=preprocessing, class_names=classes, split="train",
            augmentation=train_augmentation, augmentation_run_seed=seed,
        )
        validation = P1CVideoFrameDataset(
            REPO_ROOT / data["validation"]["path"], closure_path=extract_closure_path,
            preprocessing=preprocessing, class_names=classes, split="validation",
        )
    elif pack_closure_path is not None:
        # D-33: the packed blob's own SHA-256 is verified exactly once, at
        # dataset construction, in place of the per-file check. verify_digests
        # no longer gates a per-file digest walk under this branch, but it
        # must still be registered true -- a config that asked for weaker
        # verification (false) than the run actually performs would be a
        # silent, undocumented downgrade of what docs/PROJECT_RULES.md's reproducibility
        # rule requires a config to state about itself. Fail closed instead.
        pixel_source.require_verify_digests_true(protocol, field="packed_frames")
        train = P1CPackedFrameDataset(
            REPO_ROOT / data["train"]["path"], roots, split="train", class_names=classes,
            preprocessing=preprocessing, pack_closure_path=pack_closure_path,
            augmentation=train_augmentation, augmentation_run_seed=seed,
        )
        validation = P1CPackedFrameDataset(
            REPO_ROOT / data["validation"]["path"], roots, split="validation", class_names=classes,
            preprocessing=preprocessing, pack_closure_path=pack_closure_path,
        )
    else:
        train = P1CFrameDataset(
            REPO_ROOT / data["train"]["path"], roots, split="train", class_names=classes,
            preprocessing=preprocessing, verify_digests=verify,
            augmentation=train_augmentation, augmentation_run_seed=seed,
        )
        validation = P1CFrameDataset(
            REPO_ROOT / data["validation"]["path"], roots, split="validation", class_names=classes,
            preprocessing=preprocessing, verify_digests=verify,
        )
    pixel_source.assert_no_source_video_leakage(
        train, validation, message="registered Train/validation manifests have source-video leakage"
    )
    return train, validation


def _loader_options(protocol: dict[str, Any], *, seed: int, epoch: int, shuffle: bool) -> FrameLoaderOptions:
    loader = protocol["training"]["loader"]
    return FrameLoaderOptions(
        batch_size=protocol["training"]["batch_size"], shuffle=shuffle, num_workers=loader["num_workers"],
        seed=seed, epoch=epoch, pin_memory=loader["pin_memory"], drop_last=loader["drop_last"],
        persistent_workers=loader["persistent_workers"],
    )


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


class _PeakMemory:
    """Track the largest device allocation and host RSS actually observed.

    CUDA maintains its own peak, but MPS does not, so a truthful peak has to be
    sampled during the run.  Reporting a literal zero would satisfy the
    lifecycle schema while telling the reader nothing.
    """

    def __init__(self, device: torch.device, rss_before: int) -> None:
        self._device = device
        self._rss_before = rss_before
        self._peak_device: int | None = device_allocated_bytes(device)
        self._peak_rss = rss_before

    def sample(self, process: "psutil.Process") -> None:
        allocated = device_allocated_bytes(self._device)
        if allocated is not None:
            self._peak_device = max(self._peak_device or 0, allocated)
        self._peak_rss = max(self._peak_rss, process.memory_info().rss)

    def snapshot(self) -> dict[str, Any]:
        return {
            "peak_device_allocated_bytes": self._peak_device,
            "peak_process_rss_bytes": self._peak_rss,
            "process_rss_growth_bytes": max(0, self._peak_rss - self._rss_before),
            "device_memory_report": memory_report(self._device),
        }

    def lifecycle_peak_bytes(self) -> int:
        """The single non-negative integer the lifecycle verdict requires.

        Device allocation is preferred because it is what the accelerator
        actually held; host RSS growth is the fallback for CPU runs, where no
        device allocator exists.
        """

        if self._peak_device is not None:
            return int(self._peak_device)
        return int(max(0, self._peak_rss - self._rss_before))


def _video_level_report(evaluation: dict[str, Any], selected, class_names: list[str]) -> dict[str, Any]:
    """Build the additional, evaluation-only per-video report, clearly labelled.

    This is reported alongside `selected_checkpoint_validation_metrics`, never
    in place of it: `selection_rule: best_validation_macro_f1` above already
    picked the checkpoint from the frame-level metric before this runs, and
    nothing here feeds back into that choice.
    """

    setting = evaluation["video_level_aggregation"]
    if setting == "none":
        return {"aggregation": "none"}
    video_level = aggregate_video_level_predictions(selected.predictions, class_names)
    return {
        "aggregation": "majority_vote",
        "metrics": video_level.metrics,
        "video_count": video_level.video_count,
        "tie_broken_video_count": video_level.tie_broken_video_count,
        "predictions": [asdict(prediction) for prediction in video_level.predictions],
    }


def _verdict_reason(run_class: str) -> str:
    if run_class == "screening":
        return (
            "One screening seed completed. A single seed carries no error bar: the registered "
            "decision_threshold exists precisely so this number is compared against it rather "
            "than read as a result. Test was inaccessible by registration."
        )
    return "One registered seed completed; aggregate all three validation seeds before any Test evaluation."


def _verdict_next_action(run_class: str, experiment: dict[str, Any]) -> str:
    if run_class == "screening":
        screening = experiment.get("screening") or {}
        return (
            f"Compare validation macro_f1 against parent {screening.get('parent_id')} using the "
            f"registered decision_threshold {screening.get('decision_threshold')}. Below it, the change "
            "is inside seed noise and must not be promoted; above it, re-run as a three-seed "
            "REGISTERED_FULL_RUN_READY experiment before any number is quoted."
        )
    return "Run the remaining registered seeds, aggregate validation, then freeze one checkpoint-selection decision."


def _run_seed(
    config_path: Path,
    config: dict[str, Any],
    protocol: dict[str, Any],
    *,
    seed: int,
    run_class: str,
) -> Path:
    experiment = config["experiment"]
    run_id = _utc_run_id(str(experiment["id"]), seed)
    log_phase(f"seed {seed}: starting run {run_id}")
    training_settings = protocol["training"]
    # The registered policy is authoritative. Hardcoding strict MPS here would
    # let a CPU- or CUDA-registered configuration pass readiness and then run on
    # a device its own run manifest does not name.
    selected_device, device_report = resolve_device_policy(
        training_settings["device_policy"], cpu_threads=training_settings.get("cpu_threads")
    )
    log_phase(f"seed {seed}: device selected: {device_report.selected}")
    reset_peak_memory(selected_device)
    # Computed before the lifecycle exists (like dataset_manifest_digest, this
    # records the pack/extract closure's own declared SHA-256, not a value
    # recomputed here); a bad/missing closure, or a config that registers both
    # pixel sources, is deferred into the immutable failure path below rather
    # than crashing before any run directory exists.
    pixel_source_error: BaseException | None = None
    packed_blob_sha256: str | None = None
    video_blob_sha256: str | None = None
    event_only_test: dict[str, Any] | None = None
    try:
        pixel_source.refuse_if_both_registered(protocol["data"])
        packed_blob_sha256 = _packed_blob_sha256(protocol)
        video_blob_sha256 = _video_blob_sha256(protocol)
        # D-40: some registered folds have no constructible event-only test
        # view at all. Readiness already accepted either a real registration
        # or a validated `{unavailable: true, reason}` declaration; this just
        # records which one, and the reason verbatim, in the run manifest.
        if run_class == "screening" and "event_only_test" not in protocol["data"]:
            # A screening run may not touch Test at all, so an omitted
            # event-only registration is the correct state here rather than a
            # missing one. Record why, so the manifest never reads as though a
            # Test view existed and simply went unevaluated.
            event_only_test = {
                "available": False,
                "reason": "not applicable: screening run, Test is inaccessible by registration",
            }
        else:
            event_only_test = event_only_test_registration(protocol["data"])
    except Exception as exc:  # noqa: BLE001 - surfaced inside the immutable failure lifecycle below
        pixel_source_error = exc
    lifecycle = RunLifecycle.create(
        REPO_ROOT / "results/p1c",
        run_id,
        {
            "run_id": run_id,
            "run_kind": f"p1c_frame_baseline_{run_class}_validation_only",
                "run_class": run_class,
                "screening": config["experiment"].get("screening") if run_class == "screening" else None,
            "config_digest": sha256_file(config_path),
            "code_revision": "local working tree; commit is intentionally not assumed",
            "seed": seed,
            "package_versions": {"python": sys.version.split()[0], "torch": torch.__version__},
            "dataset_manifest_digest": protocol["data"]["train"]["sha256"],
            # D-33: the packed blob's own SHA-256, recorded alongside the
            # manifest digest so a run referencing the blob stays traceable;
            # None when this run reads per-file (no data.packed_frames).
            "packed_frames_blob_sha256": packed_blob_sha256,
            # D-34/D-35: same idea for the video-derived blob. Never folded
            # into packed_frames_blob_sha256 above -- they are different
            # pixel sources and mislabeling one as the other would misstate
            # provenance.
            "video_frames_blob_sha256": video_blob_sha256,
            # Resolved pixel source ("video_blob" / "packed_blob" / "per_file")
            # -- see the matching field in the epoch_metrics.json report below
            # and data.pixel_source.loader_mode for the shared vocabulary.
            "loader_mode": pixel_source.loader_mode(
                packed_blob_sha256=packed_blob_sha256, video_blob_sha256=video_blob_sha256
            ),
            # D-40: whether the headline event-only test view exists for this
            # config's data, and its declared reason verbatim if it does not
            # -- see event_only_test_registration for the shape.
            "event_only_test": event_only_test,
            "annotation_version": "P1B frozen grouped Train/validation manifests",
            "environment_digest": sha256_file(REPO_ROOT / "requirements/p0c-lock.txt"),
            "hardware": device_report.to_dict(),
            "metric_artifact_paths": ["epoch_metrics.json", "aggregate.json", "verdict.json"],
        },
    )
    started = datetime.now(timezone.utc)
    attempt_id = f"seed-{seed}-training"
    lifecycle.append_attempt(
        {
            "attempt_id": attempt_id, "seed": seed, "status": "RUNNING", "started_at": started.isoformat(),
            "finished_at": None, "reason": "registered validation-only P1C frame baseline",
            "hardware": device_report.to_dict(), "parent_checkpoint": None,
            "artifact_digest": None, "artifact_path": None,
        }
    )
    clock = perf_counter()
    # K-21: a run that reads two orders of magnitude more than its corpus is
    # pathological, and the 2026-08-27 burn was only caught because someone
    # read /proc/<pid>/io by hand mid-run. Snapshot here and at every exit so
    # the number lands in the artifact instead of needing to be caught live.
    io_start = IoSnapshot.capture()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    # COMPUTE_POLICY section 3 requires a real peak. MPS exposes no peak tracker,
    # so the maximum is sampled per epoch rather than read once at the end.
    peak_samples = _PeakMemory(selected_device, rss_before)
    try:
        if pixel_source_error is not None:
            raise pixel_source_error
        _set_seed(seed)
        resolved_loader_mode = pixel_source.loader_mode(
            packed_blob_sha256=packed_blob_sha256, video_blob_sha256=video_blob_sha256
        )
        log_phase(f"seed {seed}: pixel source resolved: loader_mode={resolved_loader_mode}")
        if video_blob_sha256 is not None:
            log_phase(f"seed {seed}: building datasets (video JPEG blob; blob SHA-256 verified once, no per-file digesting)")
        elif packed_blob_sha256 is not None:
            log_phase(f"seed {seed}: building datasets (packed-blob memmap; blob SHA-256 verified once, no per-file digesting)")
        else:
            log_phase(f"seed {seed}: building datasets (manifest load + digest verification)")
        train_dataset, validation_dataset = _make_datasets(protocol, seed=seed)
        log_phase(f"seed {seed}: datasets ready: {len(train_dataset)} train rows, {len(validation_dataset)} validation rows")
        # Train-only by construction (see _make_datasets): validation is never
        # even offered a real policy, so its half of this report is always the
        # NONE_AUGMENTATION sentinel, not merely "off by decision".
        augmentation_report = {"train": train_dataset.augmentation.report(), "validation": NONE_AUGMENTATION.report()}
        training = training_settings
        if training["scheduler"]["name"] != "none":
            raise ValueError("this baseline runner supports only the explicitly registered scheduler=name:none")
        log_phase(f"seed {seed}: loading model + pretrained weights ({protocol['backbone']['weights']['enum']})")
        backbone = _local_resnet50(protocol["backbone"]["weights"])
        # Absent resolves to "full" -- the exact current behaviour for every
        # existing config, the same contract an absent training.throughput
        # block already carries. core.p1c_readiness already refused any
        # value outside BACKBONE_TRAINABLE_MODES before this runner started.
        trainable_mode = protocol["backbone"].get("trainable", "full")
        backbone = apply_backbone_trainable_mode(backbone, trainable_mode)
        class_names = protocol["model"]["class_names"]
        # Moved to the device before the optimizer is constructed, not left to
        # the engine: a fused optimizer kernel is selected against the device its
        # parameters live on, so building it over CPU parameters that are about to
        # move is a footgun whose failure mode is torch-version dependent.
        model = FrameClassifier(
            backbone, GlobalAverageLinearHead(backbone.out_channels, len(class_names))
        ).to(selected_device)
        parameter_counts = count_parameters(model)
        log_phase(
            f"seed {seed}: backbone.trainable={trainable_mode}, "
            f"trainable_parameters={parameter_counts['trainable_parameters']}, "
            f"total_parameters={parameter_counts['total_parameters']}"
        )
        precision_policy = resolve_precision_policy(training["precision"], selected_device)
        throughput_policy = resolve_throughput_policy(training.get("throughput"), selected_device)
        log_phase(f"seed {seed}: throughput policy: channels_last={throughput_policy.channels_last}, "
                  f"compile={throughput_policy.compile_mode}, fused_optimizer={throughput_policy.fused_optimizer}")
        log_phase(
            f"seed {seed}: augmentation: train={augmentation_report['train']['name']}, "
            f"validation={augmentation_report['validation']['name']}"
        )
        # Only the trainable parameters: a frozen backbone's parameters carry
        # requires_grad=False (models.classifiers.FrozenBackbone), and handing
        # them to the optimizer anyway would build dead per-parameter state
        # for weights that never receive a gradient.
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = AdamW(
            trainable_parameters, lr=training["optimizer"]["lr"], weight_decay=training["optimizer"]["weight_decay"],
            **throughput_policy.optimizer_kwargs(),
        )
        # No weight= : the unweighted control is deliberate (owner decision
        # 2026-08-27), and core/p1c_readiness.py forces loss.class_balance to
        # the literal string "none" so a config cannot register a weighting
        # this line would then ignore. Implementing weighting requires
        # changing the validator AND this construction together -- editing
        # only the config is already rejected, and editing only the
        # validator would let a registered weighting silently do nothing.
        engine = FrameClassifierEngine(
            model, optimizer=optimizer, loss=nn.CrossEntropyLoss(reduction="mean"), loss_reduction="mean",
            device=selected_device, class_names=class_names,
            precision=precision_policy, throughput=throughput_policy,
        )
        best_metric = float("-inf")
        stopping_anchor = float("-inf")
        stagnant_epochs = 0
        best_checkpoint: str | None = None
        telemetry: list[dict[str, Any]] = []
        global_step = 0
        log_phase(f"seed {seed}: starting training ({training['epochs']} epoch ceiling, patience={training['early_stopping']['patience']})")
        for epoch in range(training["epochs"]):
            epoch_started = perf_counter()
            log_phase(f"seed {seed}: epoch {epoch} train starting")
            # Rebinds the augmentation seed's epoch component before the loader
            # (and its spawned workers, which pickle the dataset fresh on each
            # new DataLoader) is built for this epoch -- so epoch N's crops/flips/
            # jitter differ from epoch N-1's rather than repeating.
            train_dataset.set_epoch(epoch)
            train_result = engine.train_epoch(build_p1c_frame_dataloader(train_dataset, options=_loader_options(protocol, seed=seed, epoch=epoch, shuffle=True)))
            log_phase(f"seed {seed}: epoch {epoch} train done ({train_result.batches} batches); validation starting")
            validation = engine.evaluate(
                build_p1c_frame_dataloader(validation_dataset, options=_loader_options(protocol, seed=seed, epoch=epoch, shuffle=False)),
                bootstrap_iterations=None, bootstrap_seed=None,
            )
            metric = float(validation.metrics["macro_f1"])
            global_step += train_result.batches
            synchronize(selected_device)
            peak_samples.sample(process)
            log_phase(
                f"seed {seed}: epoch {epoch} done in {perf_counter() - epoch_started:.1f}s, "
                f"validation_macro_f1={metric:.4f}, best_so_far={max(best_metric, metric):.4f}, "
                f"stagnant_epochs={stagnant_epochs}"
            )
            epoch_record = {
                "epoch": epoch, "train_examples": train_result.examples, "train_batches": train_result.batches,
                "train_mean_loss": train_result.mean_loss, "validation": validation.metrics,
                "validation_macro_f1": metric, "scheduler_state": None,
                "memory": peak_samples.snapshot(),
            }
            telemetry.append(epoch_record)
            checkpoint_name = f"checkpoint_epoch_{epoch:03d}.pt"
            save_training_checkpoint(
                lifecycle.run_directory / checkpoint_name, model=engine.checkpoint_module, optimizer=optimizer,
                state=capture_resume_state(
                    next_epoch=epoch + 1, global_step=global_step,
                    extra_state={"best_validation_macro_f1": best_metric, "best_checkpoint": best_checkpoint, "scheduler_state": None},
                ), allow_overwrite=False,
            )
            if metric > best_metric:
                best_metric, best_checkpoint = metric, checkpoint_name
            if metric > stopping_anchor + training["early_stopping"]["min_delta"]:
                stopping_anchor, stagnant_epochs = metric, 0
            else:
                stagnant_epochs += 1
            if stagnant_epochs >= training["early_stopping"]["patience"]:
                break
        if best_checkpoint is None:
            raise RuntimeError("no validation checkpoint was selected")
        log_phase(f"seed {seed}: training loop ended; reloading best checkpoint ({best_checkpoint}) for final validation pass")
        load_training_checkpoint(
            lifecycle.run_directory / best_checkpoint,
            model=engine.checkpoint_module,
            optimizer=optimizer,
            device=selected_device,
            restore_rng=False,
        )
        selected = engine.evaluate(
            build_p1c_frame_dataloader(validation_dataset, options=_loader_options(protocol, seed=seed, epoch=0, shuffle=False)),
            bootstrap_iterations=protocol["evaluation"]["clustered_ci_iterations"],
            bootstrap_seed=protocol["evaluation"]["clustered_ci_seed"],
        )
        # Evaluation-only, additional to the frame-level metrics above; it is
        # never what checkpoint selection or early stopping used (those ran
        # per-epoch, above, from validation.metrics only). See
        # eval/video_level.py for why this is reported and its limitation.
        video_level_report = _video_level_report(protocol["evaluation"], selected, class_names)
        report = {
            "experiment_id": experiment["id"], "seed": seed, "selection_rule": training["selection_rule"],
            "selected_checkpoint": best_checkpoint, "best_validation_macro_f1": best_metric,
            "selected_checkpoint_validation_metrics": selected.metrics,
            "selected_checkpoint_validation_clustered_ci": selected.video_clustered_macro_f1_ci,
            "selected_checkpoint_validation_video_level_metrics": video_level_report,
            "epochs": telemetry, "test_evaluation_performed": False,
            "device": device_report.to_dict(), "memory": peak_samples.snapshot(),
            "io": io_delta(io_start, IoSnapshot.capture()),
            "throughput": throughput_policy.report(),
            "augmentation": augmentation_report,
            "backbone_trainable": trainable_mode,
            "parameter_counts": parameter_counts,
            # "video_blob" JPEG-decodes one row from the D-34/D-35 blob;
            # "packed_blob" reads pixels via numpy.memmap with the blob's
            # SHA-256 verified once at construction (D-33); "per_file" opens
            # and decodes one PNG per row.
            "loader_mode": resolved_loader_mode,
            "packed_frames_blob_sha256": packed_blob_sha256,
            "video_frames_blob_sha256": video_blob_sha256,
        }
        report_digest = _write_json(lifecycle.run_directory / "epoch_metrics.json", report)
        finished = datetime.now(timezone.utc)
        lifecycle.append_attempt(
            {
                "attempt_id": attempt_id, "seed": seed, "status": "COMPLETED", "started_at": started.isoformat(),
                "finished_at": finished.isoformat(), "reason": "completed validation-only seed; Test remained inaccessible",
                "hardware": device_report.to_dict(), "parent_checkpoint": None,
                "artifact_digest": report_digest, "artifact_path": "epoch_metrics.json",
            }
        )
        lifecycle.finalize(
            report,
            {
                "status": "COMPLETED", "verdict": "GOOD_ENOUGH", "claim_state": "INCONCLUSIVE",
                "reason": _verdict_reason(run_class),
                "next_action": _verdict_next_action(run_class, config["experiment"]),
                "runtime_seconds": perf_counter() - clock, "peak_memory_bytes": peak_samples.lifecycle_peak_bytes(),
                "storage_bytes": sum(item.stat().st_size for item in lifecycle.run_directory.iterdir() if item.is_file()),
                "stop_reason": "registered epoch ceiling or validation early stopping",
                "checkpoint_disposition": f"validation-selected checkpoint retained: {best_checkpoint}",
                "summary_artifact_digest": artifact_digest(report),
            },
        )
        log_phase(f"seed {seed}: done ({lifecycle.run_directory})")
    except BaseException as exc:
        finished = datetime.now(timezone.utc)
        peak_samples.sample(process)
        failure = {
            "exception_type": type(exc).__name__, "message": str(exc), "test_evaluation_performed": False,
            "device": device_report.to_dict(), "memory": peak_samples.snapshot(),
            # Recorded on the failure path too: a run killed for being
            # pathologically slow is exactly when these counters matter most.
            "io": io_delta(io_start, IoSnapshot.capture()),
        }
        failure_digest = _write_json(lifecycle.run_directory / "failure.json", failure)
        status = "ABORTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        lifecycle.append_attempt(
            {
                "attempt_id": attempt_id, "seed": seed, "status": status, "started_at": started.isoformat(),
                "finished_at": finished.isoformat(), "reason": "P1C seed stopped before a valid result",
                "hardware": device_report.to_dict(), "parent_checkpoint": None,
                "artifact_digest": failure_digest, "artifact_path": "failure.json",
            }
        )
        lifecycle.finalize(
            failure,
            {
                "status": status, "verdict": "INCONCLUSIVE", "claim_state": "NOT_SUPPORTED",
                "reason": "No research result is valid from a failed or interrupted seed.",
                "next_action": "Inspect the immutable failure artifact; resume only with the same registered configuration.",
                "runtime_seconds": perf_counter() - clock, "peak_memory_bytes": peak_samples.lifecycle_peak_bytes(),
                "storage_bytes": sum(item.stat().st_size for item in lifecycle.run_directory.iterdir() if item.is_file()),
                "stop_reason": type(exc).__name__, "checkpoint_disposition": "no selected checkpoint is eligible",
                "summary_artifact_digest": artifact_digest(failure),
            },
        )
        raise
    return lifecycle.run_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/experiments/p1c/P1C-FRAME-GAP-BASELINE-001.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    log_phase(f"loading config: {config_path}")
    config, protocol, run_class = _load_config(config_path)
    seeds = protocol["training"]["seeds"]
    log_phase(f"config loaded: {config['experiment']['id']} [{run_class} run], {len(seeds)} registered seed(s): {seeds}")
    for index, seed in enumerate(seeds, start=1):
        log_phase(f"seed {seed}: {index}/{len(seeds)}")
        print(_run_seed(config_path, config, protocol, seed=seed, run_class=run_class))
    log_phase("all registered seeds complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
