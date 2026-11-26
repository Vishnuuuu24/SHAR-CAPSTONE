# Codebase Structure

## 1) Top-Level Map

| Path | Purpose |
|---|---|
| `core/` | Device policy, reproducibility, provenance, run lifecycle, training engine and aggregate validation |
| `data/` | Frame parsing, manifests, splits, views, temporal clips, degradations, alignment checks |
| `models/` | ResNet50 frame/temporal classifiers and classical restoration contract |
| `eval/` | Classification metrics and image-quality metrics |
| `scripts/` | Explicit CLI entry points for verification, materialization, calibration, training, and aggregation |
| `configs/` | Registered, versioned experiment and verification parameters |
| `docs/` | Governing research protocol and live state |
| `tests/` | Dataset-free unit and synthetic end-to-end tests |
| `results/` | Immutable evidence artifacts; raw licensed-data artifacts are Git-ignored |

## 2) Entry Points

- There is no single application server. Work is performed through explicit scripts such as `scripts/run_p1c_frame_baseline.py`, `scripts/materialize_p1b_views.py`, and `scripts/verify_frame_alignment.py`.
- `scripts/check_repository.py` is the portable repository gate.

## 3) Module Boundaries

| Boundary | Owns | Must not own |
|---|---|---|
| `data/` | Provenance-preserving preparation and views | Model selection or headline claims |
| `models/` | Neural/classical transformations | Dataset split decisions |
| `eval/` | Task-local metric computation | Data relabelling |
| `core/` | Cross-cutting reproducibility and immutable artifacts | Experiment-specific scientific defaults |
| `scripts/` | Composition of registered components | Silent configuration changes |

## 4) Naming and Organization Rules

- Python modules use `snake_case`; classes use `PascalCase`; internal helpers use a leading underscore.
- The project is organized by responsibility rather than a web-service request stack.
- Scripts add the repository root to `sys.path`; imports use top-level package paths such as `data.ucf_intervals`.

## 5) Evidence

- `docs/00_README.md`
- `docs/PHASE_PLAN.md`
- `core/run_lifecycle.py`
- `scripts/check_repository.py`
