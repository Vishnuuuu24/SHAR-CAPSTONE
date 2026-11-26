#!/usr/bin/env python3
"""Close one interrupted P1C calibration lifecycle without rewriting history."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_readiness import sha256_file
from core.run_lifecycle import RunLifecycle, artifact_digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    lifecycle = RunLifecycle(directory)
    if lifecycle.complete_marker.exists():
        raise RuntimeError("completed runs are immutable and cannot be recovered")
    events = [json.loads(line) for line in lifecycle.attempts_path.read_text(encoding="utf-8").splitlines() if line]
    if len(events) != 1 or events[0].get("status") != "RUNNING":
        raise RuntimeError("recovery requires exactly one unfinished RUNNING attempt")
    started = events[0]
    recovery = {
        "kind": "P1C calibration infrastructure-abort recovery; not a calibration result",
        "reason": args.reason,
        "scope_assertions": {
            "research_training": False,
            "checkpoint_saved": False,
            "held_out_test_access": False,
            "configuration_selection": False,
        },
    }
    failure_path = directory / "recovery_failure.json"
    failure_path.write_text(json.dumps(recovery, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    calibration_path = directory / "calibration.json"
    calibration_path.write_text(json.dumps(recovery, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finished = datetime.now(timezone.utc).isoformat()
    lifecycle.append_attempt(
        {
            "attempt_id": started["attempt_id"], "seed": started["seed"], "status": "ABORTED",
            "started_at": started["started_at"], "finished_at": finished, "reason": args.reason,
            "hardware": started["hardware"], "parent_checkpoint": started["parent_checkpoint"],
            "artifact_digest": sha256_file(failure_path), "artifact_path": "recovery_failure.json",
        }
    )
    lifecycle.finalize(
        recovery,
        {
            "status": "ABORTED", "verdict": "INCONCLUSIVE", "claim_state": "NOT_APPLICABLE",
            "reason": "The host terminated calibration before its first recorded batch; no ETA or research result exists.",
            "next_action": "Fix the loader memory path and repeat calibration under a new immutable attempt.",
            "runtime_seconds": 0.0, "peak_memory_bytes": 0, "storage_bytes": sum(item.stat().st_size for item in directory.iterdir() if item.is_file()),
            "stop_reason": "external process termination", "checkpoint_disposition": "none; calibration checkpoints are prohibited",
            "summary_artifact_digest": artifact_digest(recovery),
        },
    )
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
