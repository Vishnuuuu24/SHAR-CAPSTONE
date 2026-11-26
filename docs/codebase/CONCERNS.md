# Codebase Concerns

## 1) Top Risks

| Severity | Concern | Impact | Suggested action |
|---|---|---|---|
| High | Historical P0B/P1B outputs predate official-Normal-source validation and real pixel alignment evidence | Training/evaluation manifests could rest on stale provenance | Run the immutable alignment gate, then regenerate P0B/P1B on the authorized machine. |
| High | Current shell is Python 3.13 while the project requires Python 3.12 | Local green unit tests are not release/research closure | Recreate/use the pinned Python 3.12 environment before repository or research gates. |
| Medium | P1C training has not begun | No model performance claim exists yet | Complete revalidation, then launch only the preregistered three-seed controls. |
| Medium | P2A restoration conventions remain owner-blocked | Real denoising experiments could be underspecified | Freeze color/range, degradation, filter, and metric settings before real P2A work. |

## 2) Technical Debt

| Debt item | Where | Risk if ignored | Suggested fix |
|---|---|---|---|
| No formatter/linter/coverage threshold | Root configuration | Style and coverage regressions rely on review | [ASK USER] decide whether to adopt one after the active revalidation work. |
| Script-driven entry points | `scripts/` | Discoverability is lower than a single CLI | Keep this onboarding map current; do not consolidate during active research gates without need. |

## 3) Security Concerns

| Risk | Evidence | Current mitigation | Gap |
|---|---|---|---|
| Licensed-media disclosure | `docs/DATA_SPEC.md`, `.gitignore` | Local-only/raw-media restrictions and Git-ignored artifacts | Team process must continue to prevent accidental publication. |
| Dataset-path/config misuse | `scripts/verify_frame_alignment.py` | Repository-root confinement and validation | No external access-control system is in this codebase. |

## 4) Performance and Scaling Concerns

| Concern | Evidence | Scaling risk | Suggested improvement |
|---|---|---|---|
| 1.38M local frames with serial loading | `docs/TRAINING_LOG.md` | Full P1C run estimate is the `P1C-FRAME-GAP-CALIBRATION-003` three-seed range | Use only a registered loader-worker change and remeasure wall time. |
| Full-video/frame digesting for alignment evidence | `scripts/verify_frame_alignment.py` | Real alignment scans may be I/O-heavy | Treat its measured time/storage as evidence; do not reduce coverage silently. |

## 5) Fragile/High-Churn Areas

| Area | Why fragile | Churn signal | Safe change strategy |
|---|---|---|---|
| `docs/PROJECT_STATUS.md` | Sole live resume pointer | 9 commits in scan history | Update it only with evidence-backed state changes. |
| Mapping/verification scripts | They define data provenance | Recent changes to P0B alignment | Add boundary/failure tests and retain prior artifacts rather than overwriting them. |
| Device policy | Backend behavior differs across MPS/CUDA/CPU | `core/device.py` recently changed | Test real torch state, not only report metadata. |

## 6) [ASK USER] Questions

1. [ASK USER] After the P0B/P1B revalidation is complete, do you want a small team-facing tutorial/notebook that walks through one real manifest row and one model batch?
2. [ASK USER] Do the owners want a formatter/linter and coverage threshold now, or should changes remain focused on the registered research gates?

## 7) Evidence

- `docs/PROJECT_STATUS.md`
- `docs/PROJECT_CONTROL.md`
- `docs/COMPUTE_POLICY.md`
- `docs/codebase/.codebase-scan.txt`
