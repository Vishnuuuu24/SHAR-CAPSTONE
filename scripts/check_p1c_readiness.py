#!/usr/bin/env python3
"""Report whether the P1C full-run configuration is fully preregistered."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_readiness import (
    sha256_file,
    validate_p1c_full_run_config,
    validate_p1c_screening_run_config,
)

# Mirrors run_p1c_frame_baseline._VALIDATOR_BY_STATUS: the checker must apply the
# same validator the runner will, or a config could pass here and be refused at
# launch (or worse, pass here under the wrong rules).
_VALIDATOR_BY_STATUS = {
    "REGISTERED_FULL_RUN_READY": ("full", validate_p1c_full_run_config),
    "REGISTERED_SCREENING_RUN_READY": ("screening", validate_p1c_screening_run_config),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/experiments/p1c/P1C-FRAME-GAP-BASELINE-001.yaml",
    )
    parser.add_argument("--require-ready", action="store_true", help="return nonzero unless every P1C launch gate is closed")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("P1C configuration must be a YAML mapping")
    experiment = config.get("experiment")
    status = experiment.get("status") if isinstance(experiment, dict) else None
    if status not in _VALIDATOR_BY_STATUS:
        run_class = "unknown"
        errors = [f"experiment.status must be one of {sorted(_VALIDATOR_BY_STATUS)}; got {status!r}"]
    else:
        run_class, validator = _VALIDATOR_BY_STATUS[status]
        errors = validator(config, base_dir=REPO_ROOT)
    print(
        json.dumps(
            {
                "kind": f"P1C {run_class}-run readiness registration check; not training evidence",
                "run_class": run_class,
                "config": str(config_path),
                "config_sha256": sha256_file(config_path),
                "ready": not errors,
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if args.require_ready and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
