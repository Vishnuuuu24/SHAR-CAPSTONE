# Testing Patterns

## 1) Test Stack and Commands

- Framework: Python `unittest`.
- Assertions/mocks: `unittest.TestCase`, `unittest.mock`, synthetic NumPy/OpenCV fixtures.

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/check_repository.py
```

## 2) Test Layout

- Files are `tests/test_*.py`.
- Tests commonly create temporary fixture data and delete it on cleanup.
- There is no separate integration-test or coverage configuration.

## 3) Test Scope Matrix

| Scope | Covered? | Typical target | Notes |
|---|---|---|---|
| Unit | Yes | Metrics, mapping, models, device policy | Primary test type |
| Synthetic end-to-end | Yes | CLI lifecycle, alignment, P1B views | No licensed data required |
| Real-data integration | Not in checkout | P0B/P1B alignment/materialization | Must run only on the authorized data machine |

## 4) Mocking and Isolation

- Backend availability is conditionally tested or mocked; CLI arguments are patched through `sys.argv`.
- Fixture images/videos are generated locally so tests do not read licensed datasets.

## 5) Coverage and Quality Signals

- Coverage threshold/current coverage: [TODO] not configured.
- The current current-shell unit suite passed 186 tests after the latest mapping/alignment changes; the pinned Python 3.12 repository gate remains the closure requirement.

## 6) Evidence

- `tests/test_p0b_mapping.py`
- `tests/test_p1c_aggregation.py`
- `tests/test_verify_frame_alignment.py`
- `scripts/check_repository.py`
