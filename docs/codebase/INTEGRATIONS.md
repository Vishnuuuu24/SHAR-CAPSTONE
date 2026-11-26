# External Integrations

## 1) Integration Inventory

| System | Type | Purpose | Auth model | Criticality |
|---|---|---|---|---|
| Local Kaggle UCF-Crime mirror | Licensed local dataset | Weak-label frame source | Owner-authorized local access | High |
| Local official UCF annotations/videos | Licensed local data | Intervals and optional original-video checks | Owner-authorized local access | High |
| PyTorch MPS/CUDA/CPU | Compute backend | Training and calibration | Local runtime | High |
| GitHub Actions | CI | Dataset-free Python 3.12 portability gate | GitHub workflow permissions | Medium |

## 2) Data Stores

- There is no database or message queue. Data resides in configured local directories; results are filesystem artifacts under `results/`.

## 3) Secrets and Credentials Handling

- No environment-variable template or secret manager is present.
- Dataset access is documented as owner-authorized; raw media, annotations, and crops are not to be committed or redistributed.

## 4) Reliability and Failure Behavior

- Strict device policies refuse silent accelerator fallback.
- Mapping and alignment tools quarantine/invalidates unresolved input rather than manufacture a result.
- No HTTP retry, backoff, or circuit-breaker code exists because there is no runtime web integration.

## 5) Observability

- Run manifests, attempt streams, digests, and verdict files provide experiment observability.
- [TODO] There is no centralized telemetry/monitoring service.

## 6) Evidence

- `data/registry/dataset_access.json`
- `core/device.py`
- `core/run_lifecycle.py`
- `.github/workflows/ci.yml`
