# Coding Conventions

## 1) Naming Rules

| Item | Rule | Example |
|---|---|---|
| Files | `snake_case.py` | `data/ucf_intervals.py` |
| Functions | `snake_case`; private helpers begin `_` | `_select_videos_and_frames` |
| Types | `PascalCase` dataclasses/modules | `FrameReference`, `RunLifecycle` |
| Constants | `UPPER_CASE` | `CANONICAL_UCF_LABELS` |

## 2) Formatting and Linting

- No formatter or linter configuration was detected. Tests and `python scripts/check_repository.py` are the enforced quality signals.

## 3) Import and Module Conventions

- Standard library imports precede third-party imports, then project imports.
- Scripts add the repository root explicitly, then import `core`, `data`, `eval`, or `models` packages.

## 4) Error and Logging Conventions

- Validation generally raises `ValueError`, `TypeError`, or `RuntimeError` with a precise reason. Invalid rows are quarantined where the mapping contract permits it.
- Scripts print artifact paths; there is no central logging framework.
- Raw licensed media paths are not intended for committed artifacts.

## 5) Testing Conventions

- Tests live in `tests/test_*.py`, use `unittest`, temporary directories, and `unittest.mock.patch` for command-line arguments or unavailable backends.
- Synthetic data proves code paths; it is never research evidence.
- Coverage threshold: [TODO] no coverage configuration was found.

## 6) Evidence

- `core/device.py`
- `data/ucf_intervals.py`
- `tests/test_verify_frame_alignment.py`
- `scripts/check_repository.py`
