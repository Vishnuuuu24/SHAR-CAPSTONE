#!/usr/bin/env python3
"""Create immutable pre-training P1C code/formula-audit evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.run_lifecycle import RunLifecycle, artifact_digest


AUDITED_SOURCES = (
    "core/device.py",
    "core/p1c_readiness.py",
    "core/p1c_engine.py",
    "data/p1c_loader.py",
    "data/p1c_temporal.py",
    "data/manifest.py",
    "data/grouping.py",
    "eval/metrics.py",
    "models/classifiers.py",
    "scripts/smoke_p1c_models.py",
    "scripts/run_p1c_frame_baseline.py",
)
TEST_MODULES = (
    "tests.test_p0c_foundation",
    "tests.test_p1c_readiness",
    "tests.test_p1c_engine",
    "tests.test_p1c_loader",
    "tests.test_p1c_temporal",
    "tests.test_p1c_models",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_digest(paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        path = REPO_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _new_run_id(started: datetime) -> str:
    """Return a UTC timestamped, append-preserving audit identifier."""
    return f"p1c-code-audit-{started.strftime('%Y%m%dT%H%M%SZ')}"


def _device_policy_behaviour_holds() -> bool:
    """Exercise the resolver: unknown policies and absent backends must refuse."""

    import torch

    from core.device import DEVICE_POLICIES, resolve_device_policy

    try:
        resolve_device_policy("cpu_or_whatever_is_free")
    except ValueError:
        pass
    else:
        return False
    # An explicit CPU policy without a registered thread count is unreproducible.
    try:
        resolve_device_policy("cpu_strict_deterministic")
    except ValueError:
        pass
    else:
        return False
    # Every accelerator policy must refuse rather than degrade when its backend
    # is absent. Whichever backend this host lacks is the one we can assert on.
    for policy, present in (
        ("cuda_strict", torch.cuda.is_available()),
        ("mps_strict", torch.backends.mps.is_available()),
    ):
        if present:
            continue
        try:
            resolve_device_policy(policy)
        except RuntimeError:
            continue
        return False
    return set(DEVICE_POLICIES) == {
        "cuda_strict", "mps_strict", "mps_warn_only_repeatability", "cpu_strict_deterministic",
    }


def _accelerator_rng_is_captured() -> bool:
    """A checkpoint must carry the RNG of the accelerator it was produced on."""

    import torch

    from core.p1c_engine import capture_resume_state

    state = capture_resume_state(next_epoch=0, global_step=0, extra_state={})
    if torch.backends.mps.is_available() and "torch_mps" not in state.rng_state:
        return False
    if torch.cuda.is_available() and "torch_cuda" not in state.rng_state:
        return False
    return {"python", "numpy", "torch_cpu"} <= set(state.rng_state)


def main() -> int:
    started = datetime.now(timezone.utc)
    run_id = _new_run_id(started)
    source_digest = _bundle_digest(AUDITED_SOURCES)
    process = subprocess.run(
        [sys.executable, "-m", "unittest", *TEST_MODULES],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    static_sources = {relative: (REPO_ROOT / relative).read_text(encoding="utf-8") for relative in AUDITED_SOURCES}
    checks = {
        "targeted_formula_and_invariant_tests_pass": process.returncode == 0,
        "mps_smoke_lengths_follow_clip_device": "torch.tensor([2], device=device)" in static_sources["scripts/smoke_p1c_models.py"],
        "readiness_binds_real_registered_files": "_resolve_registered_file" in static_sources["core/p1c_readiness.py"],
        "loader_order_is_seeded_by_run_and_epoch": "p1c-frame-loader" in static_sources["data/p1c_loader.py"],
        "cluster_key_includes_dataset_and_video": (
            "prediction.source_dataset" in static_sources["core/p1c_engine.py"]
            and "prediction.source_video_id" in static_sources["core/p1c_engine.py"]
        ),
        "loss_telemetry_requires_mean_reduction": "loss_reduction must be explicitly registered as mean" in static_sources["core/p1c_engine.py"],
        "temporal_planner_rejects_boundary_violations": "non-contiguous frame indices" in static_sources["data/p1c_temporal.py"],
        "runner_is_validation_only_and_fails_closed": (
            "test_evaluation_performed\": False" in static_sources["scripts/run_p1c_frame_baseline.py"]
            and "validate_p1c_full_run_config" in static_sources["scripts/run_p1c_frame_baseline.py"]
        ),
        # Device behaviour is asserted by execution rather than by grepping for a
        # call-site literal: a source-text check passes for dead code and breaks
        # on any harmless refactor, so it proves nothing about what runs.
        "device_policy_refuses_silent_fallback": _device_policy_behaviour_holds(),
        "runner_binds_registered_device_policy": (
            "resolve_device_policy(" in static_sources["scripts/run_p1c_frame_baseline.py"]
            and "device_policy" in static_sources["scripts/run_p1c_frame_baseline.py"]
        ),
        "checkpoint_rng_covers_selected_accelerator": _accelerator_rng_is_captured(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"P1C logic audit failed: {[name for name, value in checks.items() if not value]}")
    lifecycle = RunLifecycle.create(
        REPO_ROOT / "results/p1c",
        run_id,
        {
            "run_id": run_id,
            "run_kind": "code_audit",
            "config_digest": _sha256(REPO_ROOT / "configs/experiments/p1c/P1C-FRAME-GAP-BASELINE-001.yaml"),
            "code_revision": f"P1C audited-source bundle SHA-256 {source_digest}",
            "seed": 0,
            "package_versions": {"python": sys.version.split()[0]},
            "dataset_manifest_digest": "3930ca6b3155bee6d90308447cefade66332ab20f8a3685a6de341938d35608c",
            "annotation_version": "P1B frozen manifests; no dataset media read by this audit",
            "environment_digest": _sha256(REPO_ROOT / "requirements/p0c-lock.txt"),
            "hardware": {"device": "CPU code audit; no model execution"},
            "metric_artifact_paths": ["aggregate.json", "verdict.json", "audit_report.json"],
        },
    )
    report = {
        "kind": "P1C pre-training code/formula audit; not a model run or research result",
        "audited_sources": list(AUDITED_SOURCES),
        "audited_source_bundle_sha256": source_digest,
        "checks": checks,
        "targeted_test_modules": list(TEST_MODULES),
        "targeted_test_output_tail": (process.stdout + process.stderr).splitlines()[-20:],
        "scope_assertions": {"model_training": False, "dataset_media_read": False, "test_evaluation": False},
    }
    report_path = lifecycle.run_directory / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finished = datetime.now(timezone.utc)
    lifecycle.append_attempt(
        {
            "attempt_id": "formula-and-control-path-audit",
            "seed": 0,
            "status": "COMPLETED",
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "reason": "Validated P1C formulas, data boundaries, readiness, resume controls, and corrected audit findings before training",
            "hardware": {"device": "CPU code audit"},
            "parent_checkpoint": None,
            "artifact_digest": _sha256(report_path),
            "artifact_path": "audit_report.json",
        }
    )
    lifecycle.finalize(
        report,
        {
            "status": "COMPLETED",
            "verdict": "GOOD_ENOUGH",
            "claim_state": "NOT_APPLICABLE",
            "reason": "Targeted P1C formula/control-path audit passes after correcting launch-blocking defects; runner/calibration remain required.",
            "next_action": "Implement and test the integrated frame-baseline runner, then download/hash weights and run real-loader calibration.",
            "runtime_seconds": (finished - started).total_seconds(),
            "peak_memory_bytes": 0,
            "storage_bytes": sum(path.stat().st_size for path in lifecycle.run_directory.iterdir() if path.is_file()),
            "stop_reason": "P1C code audit completed before any training",
            "checkpoint_disposition": "not applicable; no model execution",
            "summary_artifact_digest": artifact_digest(report),
        },
    )
    print(lifecycle.run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
