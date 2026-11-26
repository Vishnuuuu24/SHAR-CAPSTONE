# P1C immutable experiment configurations

One YAML file is one logical P1C configuration. Create it before its first attempt; never edit it after a `RUNNING` event. A material change creates a new child file and experiment ID. An unchanged resume appends an attempt to the same artifact instead.

Every configuration records its parent, one material change, fixed protocol, immutable input digests, validation-only selection rule, Test-access gate, and code/formula audit. The runner copies the resolved configuration and SHA-256 into its immutable result directory.

AI may recommend the next child only from closed **validation-only** evidence in this form:

```text
Given <parent validation aggregate>, change only <factor old -> new> because <observation>.
Keep <fixed factors>. Expected decision criterion: <registered gate>.
```

AI must not use Test results to choose a child, silently alter a configuration, combine multiple experimental changes, or convert a pilot into a headline result.

## Current files

| File | Status | Notes |
|---|---|---|
| `P1C-FRAME-GAP-BASELINE-001.yaml` | `PLANNED` | Immutable planned root under D-27; do not edit. |
| `P1C-FRAME-GAP-CALIBRATION-001.yaml` | superseded | Its artifacts are `INVALID` as throughput evidence under D-30 (compute-only timing). |
| `P1C-FRAME-GAP-CPU-CALIBRATION-002.yaml` | superseded | Same D-30 timing defect; CPU remains rejected on magnitude. |
| `P1C-FRAME-GAP-CALIBRATION-003.yaml` | `REGISTERED_CALIBRATION_READY` | Corrected wall-clock calibration of the exact `FULL-001` settings. |
| `P1C-FRAME-GAP-FULL-001.yaml` | `REGISTERED_FULL_RUN_READY` | The launchable three-seed frame baseline. |
| `P1C-FRAME-GAP-FULL-002-WORKERS.yaml` | `PLANNED` | AI-recommended `num_workers` 0 → 8 child. Not launchable: readiness refuses a non-`REGISTERED_FULL_RUN_READY` status. Promotion is an owner decision and needs its own calibration first. |
