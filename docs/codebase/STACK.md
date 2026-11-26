# Technology Stack

## 1) Runtime Summary

| Area | Value | Evidence |
|---|---|---|
| Primary language | Python | `pyproject.toml` |
| Runtime + version | Exactly Python 3.12.x | `pyproject.toml` |
| Package manager | pip / setuptools build backend | `pyproject.toml`, `requirements/p0c-lock.txt` |
| Module/build system | Setuptools package `shar-research` | `pyproject.toml` |

## 2) Production Dependencies

| Dependency | Version | Role |
|---|---:|---|
| PyTorch / torchvision | 2.13.0 / 0.28.0 | Model definition, training, MPS execution |
| NumPy | 2.5.1 | Arrays, deterministic sampling |
| OpenCV | 4.13.0.92 | Image/video decoding and classical restoration |
| scikit-image | 0.26.0 | PSNR and SSIM quality metrics |
| Pillow | 12.3.0 | Image I/O/rendering support |
| PyYAML | 6.0.3 | Registered experiment/config files |
| psutil | 7.2.2 | Resource telemetry |

## 3) Development Toolchain

- Tests use the standard-library `unittest` runner; GitHub Actions runs the repository verifier on Python 3.12.
- No formatter, linter, container configuration, database, or web framework is configured.

## 4) Key Commands

```bash
python -m pip install .
python -m unittest discover -s tests -p 'test_*.py'
python scripts/check_repository.py
python scripts/verify_frame_alignment.py --run-id <new-id>
```

## 5) Environment and Config

- Config sources: `configs/*.yaml`, `configs/experiments/p1c/*.yaml`.
- Required environment variables: none declared. `PYTORCH_ENABLE_MPS_FALLBACK=1` is explicitly refused for strict accelerator policies.
- Constraint: current project evidence requires the pinned Python 3.12 environment; a Python 3.13 shell is not a valid closure environment.

## 6) Evidence

- `pyproject.toml`
- `requirements/p0c-lock.txt`
- `.github/workflows/ci.yml`
- `docs/TECH_STACK.md`
