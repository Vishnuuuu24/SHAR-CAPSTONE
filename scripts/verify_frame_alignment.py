#!/usr/bin/env python3
"""Fail-closed pixel evidence for the Kaggle-frame-index assumption.

The P0B interval mapper treats ``<video>_<frame>.png`` as an original-video
frame index.  This tool samples every registered label deterministically,
compares each PNG with the claimed video frame and competing index hypotheses,
and records an authenticated immutable run.  It is verification tooling: it
never writes a manifest, trains a model, or computes a headline metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.progress import BatchProgress, log_phase
from core.run_lifecycle import RunLifecycle, artifact_digest, artifact_payload
from data.frame_alignment import INTERPOLATIONS, best_policy_result
from data.ucf_intervals import (
    CANONICAL_UCF_LABELS,
    FrameReference,
    QuarantinedFrame,
    iter_kaggle_frame_references,
)
from core.reproducibility import stable_sample_seed


REQUIRED_CONFIG = {
    "schema_version": "1.0.0",
    "status": "VERIFICATION_TOOLING_NO_MANIFEST_OR_RESEARCH",
}
CONFIG_KEYS = frozenset({
    *REQUIRED_CONFIG,
    "frame_root",
    "video_root",
    "results_root",
    "split",
    "required_labels",
    "sample",
    "interpolation",
    "ssim_pass_threshold",
    "index_hypotheses",
})
SAMPLE_KEYS = frozenset({"videos_per_label", "frames_per_video", "global_seed"})
HYPOTHESIS_KEYS = frozenset({"offsets", "scale_factors", "minimum_declared_margin"})


def _configured_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"config {field} must be a non-empty relative path")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"config {field} escapes repository root") from exc
    return candidate


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"config {field} must be a positive integer")
    return value


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"config {field} must be a non-negative integer")
    return value


def _unit_interval(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"config {field} must be a finite number in [0,1]")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"config {field} must be in [0,1]")
    return result


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError(f"frame-alignment config must contain exactly {sorted(CONFIG_KEYS)}")
    for key, expected in REQUIRED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"frame-alignment config {key} must equal {expected!r}")
    for field in ("frame_root", "video_root", "results_root"):
        if not isinstance(config[field], str) or not config[field]:
            raise ValueError(f"config {field} must be a non-empty relative path")
    if config["split"] not in ("Train", "Test"):
        raise ValueError("config split must be 'Train' or 'Test'")
    labels = config["required_labels"]
    if not isinstance(labels, list) or tuple(labels) != tuple(sorted(CANONICAL_UCF_LABELS)):
        raise ValueError("config required_labels must list every canonical UCF label in registered order")
    sample = config["sample"]
    if not isinstance(sample, dict) or set(sample) != SAMPLE_KEYS:
        raise ValueError(f"config sample must contain exactly {sorted(SAMPLE_KEYS)}")
    _positive_int(sample["videos_per_label"], "sample.videos_per_label")
    frames_per_video = _positive_int(sample["frames_per_video"], "sample.frames_per_video")
    if frames_per_video > 3:
        raise ValueError("config sample.frames_per_video cannot exceed the registered first/middle/last positions")
    _non_negative_int(sample["global_seed"], "sample.global_seed")
    if config["interpolation"] not in INTERPOLATIONS:
        raise ValueError(f"config interpolation must be one of {sorted(INTERPOLATIONS)}")
    _unit_interval(config["ssim_pass_threshold"], "ssim_pass_threshold")
    hypotheses = config["index_hypotheses"]
    if not isinstance(hypotheses, dict) or set(hypotheses) != HYPOTHESIS_KEYS:
        raise ValueError(f"config index_hypotheses must contain exactly {sorted(HYPOTHESIS_KEYS)}")
    offsets = hypotheses["offsets"]
    if not isinstance(offsets, list) or not offsets or any(isinstance(item, bool) or not isinstance(item, int) or item == 0 for item in offsets):
        raise ValueError("config index_hypotheses.offsets must be non-zero integers")
    scales = hypotheses["scale_factors"]
    if not isinstance(scales, list) or not scales:
        raise ValueError("config index_hypotheses.scale_factors must be a non-empty list")
    for scale in scales:
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("config index_hypotheses.scale_factors must contain finite positive numbers")
    _unit_interval(hypotheses["minimum_declared_margin"], "index_hypotheses.minimum_declared_margin")
    return config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_video_file(video_root: Path, source_video_id: str) -> Path | None:
    matches = sorted(video_root.rglob(f"{source_video_id}.mp4"))
    return matches[0] if len(matches) == 1 else None


def _sample_seed(label: str, global_seed: int) -> int:
    label_digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return stable_sample_seed(f"frame-alignment/{label}", label_digest, global_seed, "video-sample")


def _select_videos_and_frames(
    references: list[FrameReference], *, videos_per_label: int, frames_per_video: int, global_seed: int
) -> list[tuple[str, str, int]]:
    """Pick the registered number of videos and first/middle/last frames per label."""

    by_label: dict[str, dict[str, list[FrameReference]]] = {label: {} for label in CANONICAL_UCF_LABELS}
    for reference in references:
        by_label[reference.label].setdefault(reference.source_video_id, []).append(reference)
    selected: list[tuple[str, str, int]] = []
    for label in CANONICAL_UCF_LABELS:
        videos = by_label[label]
        if len(videos) < videos_per_label:
            raise ValueError(f"label {label} has only {len(videos)} source videos; need {videos_per_label}")
        generator = np.random.Generator(np.random.PCG64(_sample_seed(label, global_seed)))
        chosen = generator.choice(np.array(sorted(videos), dtype=object), size=videos_per_label, replace=False)
        for source_video_id in sorted(str(item) for item in chosen):
            frame_indices = sorted(reference.source_frame_index for reference in videos[source_video_id])
            positions = [frame_indices[0], frame_indices[len(frame_indices) // 2], frame_indices[-1]][:frames_per_video]
            if len(set(positions)) != frames_per_video:
                raise ValueError(f"source video {source_video_id} lacks {frames_per_video} distinct registered positions")
            selected.extend((label, source_video_id, frame_index) for frame_index in positions)
    return selected


def _reference_lookup(references: Iterable[FrameReference]) -> dict[tuple[str, int], FrameReference]:
    return {(reference.source_video_id, reference.source_frame_index): reference for reference in references}


def _candidate_indices(frame_index: int, *, offsets: list[int], scale_factors: list[float]) -> list[int]:
    candidates = {frame_index + offset for offset in offsets if frame_index + offset >= 0}
    candidates.update(round(frame_index * factor) for factor in scale_factors if round(frame_index * factor) >= 0)
    candidates.discard(frame_index)
    return sorted(candidates)


def _result_record(result: Any) -> dict[str, Any]:
    return {
        "source_frame_index": result.source_frame_index,
        "resize_policy": result.resize_policy,
        "interpolation": result.interpolation,
        "source_resolution": list(result.source_resolution),
        "kaggle_resolution": list(result.kaggle_resolution),
        "ssim": result.ssim,
        "psnr": result.psnr,
    }


def _code_revision() -> str:
    files = (Path(__file__), REPO_ROOT / "data/frame_alignment.py")
    return "; ".join(f"{path.name}:{_sha256_file(path)}" for path in files)


def _run_id(value: str | None, config_digest: str) -> str:
    if value is not None:
        return value
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"p0b-frame-alignment-{stamp}-{config_digest[:8]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/frame_alignment_check.yaml")
    parser.add_argument("--run-id", help="unique immutable result-directory name; defaults to a UTC timestamped ID")
    args = parser.parse_args()

    started_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc)
    config_path = args.config.resolve()
    config = load_config(config_path)
    frame_root = _configured_path(REPO_ROOT, config["frame_root"], "frame_root")
    video_root = _configured_path(REPO_ROOT, config["video_root"], "video_root")
    results_root = _configured_path(REPO_ROOT, config["results_root"], "results_root")
    if not frame_root.is_dir() or not video_root.is_dir():
        missing = frame_root if not frame_root.is_dir() else video_root
        raise FileNotFoundError(f"configured alignment input is unavailable: {missing}")

    sample = config["sample"]
    videos_per_label = sample["videos_per_label"]
    frames_per_video = sample["frames_per_video"]
    global_seed = sample["global_seed"]
    interpolation = config["interpolation"]
    threshold = float(config["ssim_pass_threshold"])
    hypotheses = config["index_hypotheses"]
    margin = float(hypotheses["minimum_declared_margin"])

    log_phase(f"discovering frame references under {frame_root}")
    discovery_progress = BatchProgress(None, label="frame references scanned", every=50_000, min_interval_seconds=15.0)
    references: list[FrameReference] = []
    quarantined: list[QuarantinedFrame] = []
    for entry in iter_kaggle_frame_references(frame_root):
        discovery_progress.step()
        if isinstance(entry, QuarantinedFrame):
            quarantined.append(entry)
        elif entry.split == config["split"].lower():
            references.append(entry)
    discovery_progress.done()
    log_phase(f"discovery complete: {len(references)} references, {len(quarantined)} quarantined")
    lookup = _reference_lookup(references)
    if len(lookup) != len(references):
        raise ValueError("configured split contains duplicate source-video/frame-index references")
    selection = _select_videos_and_frames(
        references,
        videos_per_label=videos_per_label,
        frames_per_video=frames_per_video,
        global_seed=global_seed,
    )

    log_phase(f"starting {len(selection)} sampled pixel comparisons (each hashes its source video on first use)")
    comparison_progress = BatchProgress(len(selection), label="comparisons", every=5, min_interval_seconds=15.0)
    comparisons: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    video_digests: dict[Path, str] = {}
    for label, source_video_id, frame_index in selection:
        comparison_progress.step()
        reference = lookup[(source_video_id, frame_index)]
        png_path = frame_root / reference.filepath
        video_path = _find_video_file(video_root, source_video_id)
        if video_path is None:
            unresolved.append({"reason": "source_video_not_found_or_ambiguous", "source_video_id": source_video_id, "source_frame_index": frame_index})
            continue
        video_digest = video_digests.get(video_path)
        if video_digest is None:
            video_digest = _sha256_file(video_path)
            video_digests[video_path] = video_digest
        input_rows.append({
            "kaggle_relative_path": reference.filepath,
            "kaggle_png_sha256": _sha256_file(png_path),
            "source_video_id": source_video_id,
            "source_video_path": video_path.relative_to(REPO_ROOT).as_posix(),
            "source_video_sha256": video_digest,
            "declared_source_frame_index": frame_index,
        })
        try:
            declared = best_policy_result(png_path, video_path, source_video_id=source_video_id, source_frame_index=frame_index, interpolation=interpolation)
            alternatives = []
            for candidate in _candidate_indices(frame_index, offsets=hypotheses["offsets"], scale_factors=hypotheses["scale_factors"]):
                try:
                    alternative = best_policy_result(png_path, video_path, source_video_id=source_video_id, source_frame_index=candidate, interpolation=interpolation)
                except (RuntimeError, ValueError):
                    continue
                alternatives.append(_result_record(alternative))
            if not alternatives:
                unresolved.append({"reason": "no_valid_competing_index_hypothesis", "source_video_id": source_video_id, "source_frame_index": frame_index})
                continue
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            unresolved.append({"reason": f"comparison_failed:{exc}", "source_video_id": source_video_id, "source_frame_index": frame_index})
            continue
        best_alternative = max(alternatives, key=lambda item: item["ssim"])
        declared_record = _result_record(declared)
        declared_record.update({
            "label": label,
            "kaggle_relative_path": reference.filepath,
            "declared_index_passes_ssim_threshold": declared.ssim >= threshold,
            "best_competing_index": best_alternative,
            "declared_ssim_margin_over_competitor": declared.ssim - best_alternative["ssim"],
            "declared_index_wins_margin": declared.ssim - best_alternative["ssim"] >= margin,
            "competing_indices": alternatives,
        })
        comparisons.append(declared_record)
    comparison_progress.done()
    log_phase(f"comparisons complete: {len(comparisons)} resolved, {len(unresolved)} unresolved")

    failing = [item for item in comparisons if not item["declared_index_passes_ssim_threshold"] or not item["declared_index_wins_margin"]]
    expected_count = len(CANONICAL_UCF_LABELS) * videos_per_label * frames_per_video
    input_manifest = sorted(input_rows, key=lambda item: (item["source_video_id"], item["declared_source_frame_index"]))
    input_digest = hashlib.sha256(json.dumps(input_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    report = {
        "schema_version": "2.0.0",
        "kind": "Kaggle-frame-index-vs-original-video fail-closed pixel-alignment evidence",
        "scope_assertions": {"manifest_written": False, "model_training": False, "headline_evaluation": False, "sampled_content_only": True},
        "config_sha256": _sha256_file(config_path),
        "code_revision": _code_revision(),
        "sampled_input_manifest_sha256": input_digest,
        "split": config["split"],
        "required_labels": list(CANONICAL_UCF_LABELS),
        "sample": {"videos_per_label": videos_per_label, "frames_per_video": frames_per_video, "global_seed": global_seed},
        "interpolation": interpolation,
        "ssim_pass_threshold": threshold,
        "minimum_declared_margin": margin,
        "index_hypotheses": hypotheses,
        "discovered_frame_count": len(references),
        "discovery_quarantine_count": len(quarantined),
        "discovery_quarantine_examples": [item.__dict__ for item in quarantined[:100]],
        "expected_comparison_count": expected_count,
        "attempted_comparison_count": len(selection),
        "resolved_comparison_count": len(comparisons),
        "unresolved_count": len(unresolved),
        "unresolved": unresolved[:100],
        "passing_count": len(comparisons) - len(failing),
        "failing_count": len(failing),
        "all_selected_comparisons_pass": len(comparisons) == expected_count and not quarantined and not unresolved and not failing,
        "worst_comparisons": sorted(comparisons, key=lambda item: item["declared_ssim_margin_over_competitor"])[:20],
        "sampled_inputs": input_manifest,
        "comparisons": comparisons,
    }

    run_id = _run_id(args.run_id, report["config_sha256"])
    results_root.mkdir(parents=True, exist_ok=True)
    lifecycle = RunLifecycle.create(results_root, run_id, {
        "run_id": run_id,
        "run_kind": "frame_alignment_verification",
        "config_digest": report["config_sha256"],
        "code_revision": report["code_revision"],
        "seed": global_seed,
        "package_versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__, "PyYAML": yaml.__version__},
        "dataset_manifest_digest": input_digest,
        "annotation_version": "not_applicable_pixel_alignment_no_temporal_annotation_lookup",
        "environment_digest": _sha256_file(REPO_ROOT / "requirements/p0c-lock.txt"),
        "hardware": {"device": "CPU OpenCV decode and SSIM/PSNR comparison"},
        "metric_artifact_paths": ["evidence.json", "aggregate.json", "verdict.json"],
    })
    evidence_path = lifecycle.run_directory / "evidence.json"
    evidence_path.write_bytes(artifact_payload(report))
    finished = datetime.now(timezone.utc)
    runtime_seconds = time.monotonic() - started_monotonic
    success = report["all_selected_comparisons_pass"]
    lifecycle.append_attempt({
        "attempt_id": "frame-alignment-001",
        "seed": global_seed,
        "status": "COMPLETED" if success else "INVALID",
        "started_at": started_at.isoformat(),
        "finished_at": finished.isoformat(),
        "reason": "all registered sampled comparisons passed" if success else "one or more required alignment checks were unresolved or failed",
        "hardware": {"device": "CPU OpenCV decode and SSIM/PSNR comparison"},
        "parent_checkpoint": None,
        "artifact_digest": _sha256_file(evidence_path),
        "artifact_path": "evidence.json",
    })
    verdict = {
        "status": "COMPLETED" if success else "INVALID",
        "verdict": "GOOD" if success else "INVALID",
        "claim_state": "SUPPORTED" if success else "INCONCLUSIVE",
        "reason": "Every registered sampled PNG matched its declared index and beat competing hypotheses." if success else "The registered frame-alignment evidence gate did not pass.",
        "next_action": "Do not rewrite prior mapping artifacts; inspect this immutable evidence before changing their status." if success else "Inspect the evidence report and correct input, configuration, or index assumptions before rerunning with a new run ID.",
        "runtime_seconds": runtime_seconds,
        "peak_memory_bytes": 0,
        "storage_bytes": evidence_path.stat().st_size,
        "stop_reason": "completed fail-closed evidence scan" if success else "fail-closed evidence gate rejected the scan",
        "checkpoint_disposition": "not applicable; no model checkpoint exists",
        "summary_artifact_digest": artifact_digest(report),
    }
    lifecycle.finalize(report, verdict)
    log_phase(f"done: {lifecycle.run_directory}")
    print(lifecycle.run_directory)
    if not success:
        raise RuntimeError("frame-alignment evidence gate failed; see immutable run directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
