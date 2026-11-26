#!/usr/bin/env python3
"""Close the registered three-seed P1C validation aggregate without Test access."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_aggregation import aggregate_validation_seeds
from core.p1c_readiness import sha256_file, validate_p1c_full_run_config
from core.run_lifecycle import RunLifecycle, artifact_digest


def _read_seed_run(path: Path, *, config_digest: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    try:
        resolved.relative_to((REPO_ROOT / "results/p1c").resolve())
    except ValueError as exc:
        raise ValueError("seed runs must be under results/p1c") from exc
    if not (resolved / ".complete").is_file():
        raise ValueError(f"seed run is not an immutable closure: {resolved}")
    manifest = json.loads((resolved / "run_manifest.json").read_text(encoding="utf-8"))
    verdict = json.loads((resolved / "verdict.json").read_text(encoding="utf-8"))
    report = json.loads((resolved / "epoch_metrics.json").read_text(encoding="utf-8"))
    if manifest.get("config_digest") != config_digest or verdict.get("status") != "COMPLETED":
        raise ValueError(f"seed run does not match the completed registered configuration: {resolved}")
    return report, {"path": str(resolved.relative_to(REPO_ROOT)), "epoch_metrics_sha256": sha256_file(resolved / "epoch_metrics.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed-run", type=Path, action="append", required=True, help="repeat exactly three times")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    errors = validate_p1c_full_run_config(config, base_dir=REPO_ROOT)
    if errors:
        raise RuntimeError("registered full-run configuration is not ready:\n- " + "\n- ".join(errors))
    experiment = config["experiment"]
    protocol = config["fixed_protocol"]
    config_digest = sha256_file(config_path)
    reads = [_read_seed_run(path, config_digest=config_digest) for path in args.seed_run]
    aggregate = aggregate_validation_seeds([report for report, _ in reads], expected_experiment_id=experiment["id"])
    aggregate["seed_run_artifacts"] = [reference for _, reference in reads]
    run_id = f"{experiment['id'].lower().replace('_', '-')}-aggregate-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    lifecycle = RunLifecycle.create(
        REPO_ROOT / "results/p1c", run_id,
        {
            "run_id": run_id, "run_kind": "p1c_frame_three_seed_validation_aggregate", "config_digest": config_digest,
            "code_revision": "local working tree; commit is intentionally not assumed", "seed": -1,
            "package_versions": {"python": sys.version.split()[0], "torch": torch.__version__},
            "dataset_manifest_digest": protocol["data"]["train"]["sha256"],
            "annotation_version": "P1B frozen grouped Train/validation manifests; Test prohibited",
            "environment_digest": sha256_file(REPO_ROOT / "requirements/p0c-lock.txt"),
            "hardware": {"aggregation": "validation-only", "seed_hardware": [json.loads((REPO_ROOT / item["path"] / "run_manifest.json").read_text(encoding="utf-8"))["hardware"] for item in aggregate["seed_run_artifacts"]]},
            "metric_artifact_paths": ["cross_seed_aggregate.json", "aggregate.json", "verdict.json"],
        },
    )
    started = datetime.now(timezone.utc).isoformat()
    lifecycle.append_attempt({"attempt_id": "three-seed-validation-aggregate", "seed": -1, "status": "RUNNING", "started_at": started, "finished_at": None, "reason": "registered validation-only aggregation", "hardware": {"aggregation": "validation-only"}, "parent_checkpoint": None, "artifact_digest": None, "artifact_path": None})
    report_path = lifecycle.run_directory / "cross_seed_aggregate.json"
    report_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_digest = sha256_file(report_path)
    finished = datetime.now(timezone.utc).isoformat()
    lifecycle.append_attempt({"attempt_id": "three-seed-validation-aggregate", "seed": -1, "status": "COMPLETED", "started_at": started, "finished_at": finished, "reason": "all three registered validation seeds aggregated; Test remained inaccessible", "hardware": {"aggregation": "validation-only"}, "parent_checkpoint": None, "artifact_digest": report_digest, "artifact_path": "cross_seed_aggregate.json"})
    lifecycle.finalize(aggregate, {"status": "COMPLETED", "verdict": "GOOD_ENOUGH", "claim_state": "INCONCLUSIVE", "reason": "Three validation-only seeds are aggregated; no held-out Test result exists.", "next_action": "Run frozen-checkpoint Test evaluation only through the separate post-selection path.", "runtime_seconds": 0.0, "peak_memory_bytes": 0, "storage_bytes": report_path.stat().st_size, "stop_reason": "three registered completed seed closures", "checkpoint_disposition": "three validation-selected checkpoints retained in referenced seed runs", "summary_artifact_digest": artifact_digest(aggregate)})
    print(lifecycle.run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
