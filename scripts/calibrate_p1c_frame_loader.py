#!/usr/bin/env python3
"""Measure the registered P1C frame baseline on real Train/validation loading.

This is a calibration, not a research run: it never reads held-out Test rows,
does not save a model checkpoint, and cannot choose a scientific configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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

from core.device import DeviceReport, memory_report, resolve_device_policy, synchronize
from core.precision import resolve_precision_policy
from core.throughput import resolve_throughput_policy
from core.p1c_engine import FrameClassifierEngine
from core.p1c_readiness import sha256_file
from core.progress import BatchProgress, log_phase
from core.run_lifecycle import RunLifecycle, artifact_digest
from data.video_dataset import P1CVideoFrameDataset  # noqa: E402
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
from models.classifiers import FrameClassifier, GlobalAverageLinearHead, ResNet50C5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("experiment"), dict):
        raise ValueError("calibration config must contain an experiment mapping")
    experiment = config["experiment"]
    if experiment.get("status") != "REGISTERED_CALIBRATION_READY" or experiment.get("run_class") != "calibration":
        raise ValueError("calibration config must be REGISTERED_CALIBRATION_READY with run_class=calibration")
    protocol = config.get("fixed_protocol")
    calibration = protocol.get("calibration") if isinstance(protocol, dict) else None
    if not isinstance(protocol, dict) or not isinstance(calibration, dict):
        raise ValueError("calibration config must contain fixed_protocol.calibration")
    scope = calibration.get("scope")
    expected_scope = {
        "research_training": False,
        "checkpoint_saved": False,
        "held_out_test_access": False,
        "configuration_selection": False,
    }
    if scope != expected_scope:
        raise ValueError("calibration scope must explicitly prohibit research training, checkpoints, Test, and selection")
    return config, protocol


def _verified_file(registration: dict[str, Any], *, name: str) -> Path:
    if not isinstance(registration, dict) or not isinstance(registration.get("path"), str):
        raise ValueError(f"{name} must register a path")
    path = (REPO_ROOT / registration["path"]).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} path escapes project root") from exc
    if not path.is_file() or _sha256(path) != registration.get("sha256"):
        raise ValueError(f"{name} file is missing or its SHA-256 does not match")
    return path


def _packed_blob_sha256(protocol: dict[str, Any]) -> str | None:
    """The registered blob SHA-256 for the run manifest, or ``None`` when unpacked.

    Thin wrapper over ``data.pixel_source.packed_blob_sha256``; kept as a
    module-level name because this script calls it before a lifecycle exists.
    """

    return pixel_source.packed_blob_sha256(protocol)


def _video_blob_sha256(protocol: dict[str, Any]) -> str | None:
    """The registered D-34/D-35 blob SHA-256 for the run manifest, or ``None`` if absent."""

    return pixel_source.video_blob_sha256(protocol)


def _datasets(protocol: dict[str, Any]) -> tuple[P1CFrameDataset, P1CFrameDataset, float]:
    data = protocol["data"]
    _verified_file(data["train"], name="train manifest")
    _verified_file(data["validation"], name="validation manifest")
    _verified_file(protocol["backbone"]["weights"], name="pretrained weights")
    roots = {name: REPO_ROOT / relative for name, relative in data["dataset_roots"].items()}
    preprocessing = FramePreprocessing.from_config(protocol["preprocessing"])
    classes = protocol["model"]["class_names"]
    pack_closure_path = _pack_closure_path(data)
    extract_closure_path = _extract_closure_path(data)
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
    resolved_loader_mode = pixel_source.loader_mode(
        packed_blob_sha256=_packed_blob_sha256(protocol), video_blob_sha256=_video_blob_sha256(protocol)
    )
    log_phase(f"pixel source resolved: loader_mode={resolved_loader_mode}")
    start = time.perf_counter()
    if extract_closure_path is not None:
        # D-34/D-35: pixels come from the packed JPEG blob. As with the D-33
        # path, the blob's own SHA-256 is verified once at construction rather
        # than per row, so verify_digests must still be registered true for the
        # same reason -- a config may not claim a weaker posture than the run
        # performs.
        pixel_source.require_verify_digests_true(protocol, field="video_frames")
        log_phase("building train dataset (video JPEG blob; blob SHA-256 verified once at construction)")
        train = P1CVideoFrameDataset(
            REPO_ROOT / data["train"]["path"], closure_path=extract_closure_path,
            preprocessing=preprocessing, class_names=classes,
        )
        log_phase(f"train dataset ready: {len(train)} rows")
        log_phase("building validation dataset (video JPEG blob; blob SHA-256 verified once at construction)")
        validation = P1CVideoFrameDataset(
            REPO_ROOT / data["validation"]["path"], closure_path=extract_closure_path,
            preprocessing=preprocessing, class_names=classes,
        )
        log_phase(f"validation dataset ready: {len(validation)} rows")
    elif pack_closure_path is not None:
        # D-33: the packed blob's own SHA-256 is verified exactly once, at
        # dataset construction, in place of the per-file check. This field
        # therefore no longer gates a per-file digest walk under this branch,
        # but it must still be registered true -- a config that asked for
        # weaker verification (false) than the run actually performs would be
        # a silent, undocumented downgrade of what docs/PROJECT_RULES.md's reproducibility
        # rule requires a config to state about itself. Fail closed instead.
        pixel_source.require_verify_digests_true(protocol, field="packed_frames")
        log_phase("building train dataset (packed-blob memmap; blob SHA-256 verified once at construction, no per-file digesting)")
        train = P1CPackedFrameDataset(
            REPO_ROOT / data["train"]["path"], roots, split="train", class_names=classes,
            preprocessing=preprocessing, pack_closure_path=pack_closure_path,
        )
        log_phase(f"train dataset ready: {len(train)} rows")
        log_phase("building validation dataset (packed-blob memmap; blob SHA-256 verified once at construction, no per-file digesting)")
        validation = P1CPackedFrameDataset(
            REPO_ROOT / data["validation"]["path"], roots, split="validation", class_names=classes,
            preprocessing=preprocessing, pack_closure_path=pack_closure_path,
        )
        log_phase(f"validation dataset ready: {len(validation)} rows")
    else:
        log_phase("building train dataset (manifest load + full digest verification)")
        train = P1CFrameDataset(
            REPO_ROOT / data["train"]["path"], roots, split="train", class_names=classes,
            preprocessing=preprocessing, verify_digests=True,
        )
        log_phase(f"train dataset ready: {len(train)} rows")
        log_phase("building validation dataset (manifest load + full digest verification)")
        validation = P1CFrameDataset(
            REPO_ROOT / data["validation"]["path"], roots, split="validation", class_names=classes,
            preprocessing=preprocessing, verify_digests=True,
        )
        log_phase(f"validation dataset ready: {len(validation)} rows")
    pixel_source.assert_no_source_video_leakage(
        train, validation, message="calibration manifests have source-video leakage"
    )
    return train, validation, time.perf_counter() - start


def _loader(dataset: P1CFrameDataset, protocol: dict[str, Any], *, seed: int, shuffle: bool) -> Any:
    settings = protocol["training"]
    loader = settings["loader"]
    return build_p1c_frame_dataloader(
        dataset,
        options=FrameLoaderOptions(
            batch_size=settings["batch_size"], shuffle=shuffle, num_workers=loader["num_workers"], seed=seed,
            epoch=0, pin_memory=loader["pin_memory"], drop_last=loader["drop_last"],
            persistent_workers=loader["persistent_workers"],
        ),
    )


def _local_model(protocol: dict[str, Any]) -> FrameClassifier:
    weights = protocol["backbone"]["weights"]
    payload = torch.load(REPO_ROOT / weights["path"], map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("pretrained checkpoint is not a state dictionary")
    backbone = ResNet50C5.from_local_weights(payload, weights_name=weights["enum"])
    return FrameClassifier(backbone, GlobalAverageLinearHead(backbone.out_channels, len(protocol["model"]["class_names"])))


def _select_registered_device(policy: object, calibration: dict[str, Any]) -> tuple[torch.device, DeviceReport, dict[str, int]]:
    """Select exactly the registered device through the shared policy resolver.

    Device selection lives in ``core.device`` so that the calibration and the
    full run cannot drift apart about what a policy name means.
    """

    before = torch.get_num_threads()
    device, report = resolve_device_policy(policy, cpu_threads=calibration.get("cpu_threads"))
    return device, report, {"torch_threads_before": before, "torch_threads_used": torch.get_num_threads()}


@dataclass(frozen=True)
class _StepTiming:
    """Per-batch wall time separated into its data-loading and compute parts.

    The distinction matters because only wall time predicts a run.  Timing the
    compute step alone measures the accelerator, which is a throughput ceiling
    rather than a throughput: with ``num_workers=0`` the loader is serial and
    every millisecond it spends is added to the epoch, while with prefetching
    workers the same loading is hidden and wall collapses onto compute.
    """

    load_seconds: tuple[float, ...]
    compute_seconds: tuple[float, ...]

    @property
    def wall_seconds(self) -> tuple[float, ...]:
        return tuple(load + compute for load, compute in zip(self.load_seconds, self.compute_seconds, strict=True))

    def to_dict(self) -> dict[str, Any]:
        wall = self.wall_seconds
        return {
            "load_step_seconds": list(self.load_seconds),
            "compute_step_seconds": list(self.compute_seconds),
            "wall_step_seconds": list(wall),
            "median_load_seconds": statistics.median(self.load_seconds),
            "median_compute_seconds": statistics.median(self.compute_seconds),
            "median_wall_seconds": statistics.median(wall),
            "load_share_of_wall": statistics.median(self.load_seconds) / statistics.median(wall),
        }


def _timed_batches(
    step: Any, batches: Any, count: int, device: torch.device, *, label: str
) -> _StepTiming:
    """Time ``count`` batches, attributing loader wait and compute separately."""

    log_phase(f"{label}: starting {count} batch(es)")
    progress = BatchProgress(count, label=label, every=1, min_interval_seconds=15.0)
    loads: list[float] = []
    computes: list[float] = []
    for _ in range(count):
        synchronize(device)
        fetch_started = time.perf_counter()
        try:
            batch = next(batches)
        except StopIteration:
            break
        loads.append(time.perf_counter() - fetch_started)
        compute_started = time.perf_counter()
        step(batch)
        synchronize(device)
        computes.append(time.perf_counter() - compute_started)
        # Logged after this batch's timing is already recorded above, so
        # progress output cannot perturb the measured load/compute seconds.
        progress.step()
    if len(computes) != count:
        raise RuntimeError(f"calibration loader ended before registered {label} batch count")
    progress.done()
    return _StepTiming(load_seconds=tuple(loads), compute_seconds=tuple(computes))


def _warmup(step: Any, batches: Any, count: int, device: torch.device, *, label: str) -> None:
    _timed_batches(step, batches, count, device, label=label)


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/experiments/p1c/P1C-FRAME-GAP-CALIBRATION-001.yaml")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    log_phase(f"loading config: {config_path}")
    config, protocol = _load_config(config_path)
    calibration = protocol["calibration"]
    seed = calibration["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    log_phase(f"config loaded: {config['experiment']['id']} (seed={seed})")
    device: torch.device | None
    device_preflight_error: RuntimeError | None = None
    thread_report = {"torch_threads_before": torch.get_num_threads(), "torch_threads_used": torch.get_num_threads()}
    log_phase(f"selecting device (policy={protocol['training']['device_policy']})")
    try:
        device, device_report, thread_report = _select_registered_device(protocol["training"]["device_policy"], calibration)
        log_phase(f"device selected: {device_report.selected}")
    except RuntimeError as exc:
        # A strict-MPS refusal is material calibration evidence, so preserve it
        # in the same immutable lifecycle rather than losing it before a run
        # directory exists.
        device = None
        device_preflight_error = exc
        device_report = DeviceReport(
            selected="unavailable",
            mps_built=torch.backends.mps.is_built(),
            mps_available=torch.backends.mps.is_available(),
            cpu_fallback_allowed=False,
            mps_fallback_environment=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        )
    experiment = config["experiment"]
    run_id = f"{experiment['id'].lower().replace('_', '-')}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    # Computed before the lifecycle exists (like dataset_manifest_digest, this
    # records the pack/extract closure's own declared SHA-256, not a value
    # recomputed here); a bad/missing closure, or a config that registers both
    # pixel sources, is deferred into the immutable failure path below rather
    # than crashing before any run directory exists.
    pixel_source_error: BaseException | None = None
    packed_blob_sha256: str | None = None
    video_blob_sha256: str | None = None
    try:
        pixel_source.refuse_if_both_registered(protocol["data"])
        packed_blob_sha256 = _packed_blob_sha256(protocol)
        video_blob_sha256 = _video_blob_sha256(protocol)
    except Exception as exc:  # noqa: BLE001 - surfaced inside the immutable failure lifecycle below
        pixel_source_error = exc
    lifecycle = RunLifecycle.create(
        REPO_ROOT / "results/p1c",
        run_id,
        {
            "run_id": run_id, "run_kind": "p1c_real_loader_calibration", "config_digest": _sha256(config_path),
            "code_revision": "local working tree; commit is intentionally not assumed", "seed": seed,
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
            # -- see the matching field in the calibration.json report below
            # and data.pixel_source.loader_mode for the shared vocabulary.
            "loader_mode": pixel_source.loader_mode(
                packed_blob_sha256=packed_blob_sha256, video_blob_sha256=video_blob_sha256
            ),
            "annotation_version": "P1B frozen grouped Train/validation manifests; Test prohibited",
            "environment_digest": _sha256(REPO_ROOT / "requirements/p0c-lock.txt"), "hardware": device_report.to_dict(),
            "metric_artifact_paths": ["calibration.json", "aggregate.json", "verdict.json"],
        },
    )
    started = datetime.now(timezone.utc)
    attempt_id = "real-loader-throughput"
    lifecycle.append_attempt({
        "attempt_id": attempt_id, "seed": seed, "status": "RUNNING", "started_at": started.isoformat(),
        "finished_at": None, "reason": "registered Train/validation throughput and ETA calibration",
        "hardware": device_report.to_dict(), "parent_checkpoint": None, "artifact_digest": None, "artifact_path": None,
    })
    clock = time.perf_counter()
    try:
        if device_preflight_error is not None:
            raise device_preflight_error
        if device is None:
            raise RuntimeError("strict MPS calibration has no selected device")
        if pixel_source_error is not None:
            raise pixel_source_error
        process = psutil.Process()
        rss_before, memory_before = process.memory_info().rss, memory_report(device)
        train, validation, initialization_seconds = _datasets(protocol)
        log_phase(f"loading model + pretrained weights ({protocol['backbone']['weights']['enum']})")
        # Moved to the device before the optimizer is constructed, not left to
        # the engine: a fused optimizer kernel is selected against the device its
        # parameters live on, so building it over CPU parameters that are about to
        # move is a footgun whose failure mode is torch-version dependent.
        model = _local_model(protocol).to(device)
        precision_policy = resolve_precision_policy(protocol["training"]["precision"], device)
        log_phase(f"precision policy: {precision_policy.name} (autocast={precision_policy.is_mixed}, "
                  f"grad_scaler={precision_policy.uses_grad_scaler})")
        throughput_policy = resolve_throughput_policy(protocol["training"].get("throughput"), device)
        log_phase(f"throughput policy: channels_last={throughput_policy.channels_last}, "
                  f"compile={throughput_policy.compile_mode}, fused_optimizer={throughput_policy.fused_optimizer}")
        optimizer = AdamW(
            model.parameters(), lr=protocol["training"]["optimizer"]["lr"],
            weight_decay=protocol["training"]["optimizer"]["weight_decay"],
            **throughput_policy.optimizer_kwargs(),
        )
        engine = FrameClassifierEngine(
            model, optimizer=optimizer, loss=nn.CrossEntropyLoss(reduction="mean"), loss_reduction="mean",
            device=device, class_names=protocol["model"]["class_names"],
            precision=precision_policy, throughput=throughput_policy,
        )
        log_phase("model ready")

        # Accumulated across warmup and measured batches: under float16 a
        # scaler-discarded step is normal, but a run that discards many has
        # trained less than its batch count claims and must say so.
        skipped_steps_seen = [0]

        def train_step(batch: Any) -> None:
            result = engine.train_epoch((batch,))
            skipped_steps_seen[0] += result.skipped_optimizer_steps

        def validation_step(batch: Any) -> None:
            engine.evaluate((batch,), bootstrap_iterations=None, bootstrap_seed=None)

        train_batches = iter(_loader(train, protocol, seed=seed, shuffle=True))
        _warmup(train_step, train_batches, calibration["warmup_train_batches"], device, label="train warmup")
        train_timing = _timed_batches(
            train_step, train_batches, calibration["measured_train_batches"], device, label="train"
        )
        validation_timing = _timed_batches(
            validation_step, iter(_loader(validation, protocol, seed=seed, shuffle=False)),
            calibration["measured_validation_batches"], device, label="validation",
        )
        log_phase("measured batches complete; computing ETA and writing artifact")
        # Wall time, not compute time, is what an epoch actually costs.
        median_train = statistics.median(train_timing.wall_seconds)
        median_validation = statistics.median(validation_timing.wall_seconds)
        batch_size = protocol["training"]["batch_size"]
        train_batches_per_epoch = math.ceil(len(train) / batch_size)
        validation_batches_per_epoch = math.ceil(len(validation) / batch_size)
        epoch_seconds = train_batches_per_epoch * median_train + validation_batches_per_epoch * median_validation
        full_epochs = calibration["full_epochs_for_estimate"]
        per_seed_seconds = initialization_seconds + full_epochs * epoch_seconds
        overhead_low, overhead_high = calibration["overhead_factor_range"]
        total_low, total_high = (per_seed_seconds * 3 * overhead_low, per_seed_seconds * 3 * overhead_high)
        report = {
            "kind": f"P1C real-loader {device.type.upper()} calibration; not training, model selection, or Test evaluation",
            "experiment_id": experiment["id"], "parent_id": experiment["parent_id"], "config_sha256": _sha256(config_path),
            "source_scope": {"train_rows": len(train), "validation_rows": len(validation), "held_out_test_access": False},
            "weight_sha256": protocol["backbone"]["weights"]["sha256"], "device": device_report.to_dict(),
            # Recorded from the *resolved* policy, not copied from the config:
            # the artifact must state the numerics the run actually executed.
            "precision": precision_policy.report(),
            "throughput": throughput_policy.report(),
            "thread_configuration": thread_report,
            # "video_blob" JPEG-decodes one row from the D-34/D-35 blob;
            # "packed_blob" reads pixels via numpy.memmap with the blob's
            # SHA-256 verified once at construction (D-33); "per_file" opens
            # and decodes one PNG per row with verify_digests=True. The
            # initialization-seconds field name below predates the packed
            # path and still says "full digest verification" for all three,
            # so this field is what actually disambiguates them.
            "loader_mode": pixel_source.loader_mode(
                packed_blob_sha256=packed_blob_sha256, video_blob_sha256=video_blob_sha256
            ),
            "packed_frames_blob_sha256": packed_blob_sha256,
            "video_frames_blob_sha256": video_blob_sha256,
            "loader_initialization_seconds_with_full_digest_verification": initialization_seconds,
            "measurements": {
                "warmup_train_batches": calibration["warmup_train_batches"],
                "measured_train_batches": len(train_timing.compute_seconds),
                "measured_validation_batches": len(validation_timing.compute_seconds),
                "loader_num_workers": protocol["training"]["loader"]["num_workers"],
                "skipped_optimizer_steps": skipped_steps_seen[0],
                "train": train_timing.to_dict(), "validation": validation_timing.to_dict(),
                # Wall-clock rates predict the run; compute-only rates are the
                # accelerator ceiling and must never be quoted as throughput.
                "train_median_images_per_second": batch_size / median_train,
                "validation_median_images_per_second": batch_size / median_validation,
                "train_compute_only_images_per_second": batch_size / statistics.median(train_timing.compute_seconds),
                "validation_compute_only_images_per_second": batch_size / statistics.median(validation_timing.compute_seconds),
                "process_rss_before_bytes": rss_before, "process_rss_after_bytes": process.memory_info().rss,
                "device_memory_before": memory_before, "device_memory_after": memory_report(device),
            },
            "eta": {
                "train_batches_per_epoch": train_batches_per_epoch, "validation_batches_per_epoch": validation_batches_per_epoch,
                "median_epoch_seconds": epoch_seconds, "max_epochs_per_seed": full_epochs, "sequential_seeds": 3,
                "full_digest_initialization_seconds_per_seed": initialization_seconds,
                "overhead_factor_range": [overhead_low, overhead_high], "three_seed_wall_seconds_range": [total_low, total_high],
                "formula": "3 * (digest_initialization + full_epochs * (train_batches * train_median_wall + validation_batches * validation_median_wall)) * overhead_factor",
                "median_basis": "wall_seconds_per_batch_including_data_loading",
                "reestimate_rule": "re-estimate after first full epoch if actual pace differs by more than 20 percent",
            },
            "scope_assertions": calibration["scope"],
        }
        report_digest = _write_json(lifecycle.run_directory / "calibration.json", report)
        finished = datetime.now(timezone.utc)
        lifecycle.append_attempt({
            "attempt_id": attempt_id, "seed": seed, "status": "COMPLETED", "started_at": started.isoformat(),
            "finished_at": finished.isoformat(), "reason": "real Train/validation loader calibration completed; Test remained inaccessible",
            "hardware": device_report.to_dict(), "parent_checkpoint": None,
            "artifact_digest": report_digest, "artifact_path": "calibration.json",
        })
        lifecycle.finalize(report, {
            "status": "COMPLETED", "verdict": "GOOD_ENOUGH", "claim_state": "NOT_APPLICABLE",
            "reason": "Real-loader calibration provides a measured device-specific ETA only; it has no scientific result.",
            "next_action": "Review ETA and resource headroom, then create the immutable full-run registration before training.",
            "runtime_seconds": time.perf_counter() - clock, "peak_memory_bytes": max(0, process.memory_info().rss - rss_before),
            "storage_bytes": sum(item.stat().st_size for item in lifecycle.run_directory.iterdir() if item.is_file()),
            "stop_reason": "registered calibration batch counts completed", "checkpoint_disposition": "none; calibration checkpoints are prohibited",
            "summary_artifact_digest": artifact_digest(report),
        })
    except BaseException as exc:
        failure = {"exception_type": type(exc).__name__, "message": str(exc), "scope_assertions": calibration["scope"]}
        failure_digest = _write_json(lifecycle.run_directory / "failure.json", failure)
        # The immutable manifest always declares calibration.json, including a
        # preflight failure. This preserves the failure without weakening the
        # repository closure contract.
        _write_json(
            lifecycle.run_directory / "calibration.json",
            {"kind": "P1C calibration preflight failure; not training", "failure": failure, "device": device_report.to_dict()},
        )
        status = "ABORTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        lifecycle.append_attempt({
            "attempt_id": attempt_id, "seed": seed, "status": status, "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(), "reason": "real-loader calibration stopped before valid ETA evidence",
            "hardware": device_report.to_dict(), "parent_checkpoint": None,
            "artifact_digest": failure_digest, "artifact_path": "failure.json",
        })
        lifecycle.finalize(failure, {
            "status": status, "verdict": "INCONCLUSIVE", "claim_state": "NOT_APPLICABLE",
            "reason": "Calibration did not produce a valid ETA.", "next_action": "Inspect the immutable failure artifact before any full-run registration.",
            "runtime_seconds": time.perf_counter() - clock, "peak_memory_bytes": 0,
            "storage_bytes": sum(item.stat().st_size for item in lifecycle.run_directory.iterdir() if item.is_file()),
            "stop_reason": type(exc).__name__, "checkpoint_disposition": "none; calibration checkpoints are prohibited",
            "summary_artifact_digest": artifact_digest(failure),
        })
        raise
    log_phase(f"done: {lifecycle.run_directory}")
    print(lifecycle.run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
