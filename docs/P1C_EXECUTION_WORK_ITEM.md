# P1C execution framework work item

> Started: 2026-07-24 · State: `IN_PROGRESS` — P1C full-run registration and MPS calibration are complete; no real training has started.

## Control packet

- **Sprint/work item:** `P1C` / `P1C-CODE-W2`
- **Governs:** FR-1, FR-2, FR-10, FR-13, FR-16, NFR-1, NFR-3, NFR-8, V-G3–V-G4, D-21, D-25, and B-006.
- **Objective:** complete the fail-closed data-loading, training, evaluation, checkpoint/resume, aggregation, and run-lifecycle framework required by the already implemented P1C architectures.
- **In scope:** explicit RGB/preprocessing contract; Train/validation-only frame loading; source-video-preserving temporal clips; deterministic loader configuration; train/validation loop; validation-only checkpoint selection; resume state; frozen Test evaluators; three-seed aggregation; immutable lifecycle artifacts; readiness/estimate checks; eligible faculty-render trigger.
- **Non-goals:** selecting architecture dimensions, pretrained weights, loss/augmentation/optimizer defaults, execution device policy, or starting a real-model run. Those remain owner-registered inputs.
- **Dependencies/blockers:** P1B manifests are complete. `P1C-FRAME-GAP-FULL-001` binds local weights and the strict-MPS calibration. B-006 remains temporal-only; the remaining frame-run blocker is cross-seed aggregate closure.

## Definition of done

- [x] `P1C-CODE-AUDIT-001` reviews formulas and control paths; it corrected artifact binding, MPS length placement, epoch-seeded loader order, composite video clusters, loss accounting, and file-integrity gates before training.
- [x] `P1C-CODE-AUDIT-002` closes the D-30 corrections: wall-clock calibration timing, registered-policy device resolution, accelerator RNG in checkpoints, and real peak-memory reporting. Its device and RNG checks assert behaviour by execution rather than by matching source text, so a refactor can no longer break the gate without breaking the contract.
- [x] The readiness validator rejects placeholders, incomplete/incorrect real artifact identities, non-three-seed policies, Test-based selection, and incomplete protocol fields.
- [x] Frame loading accepts only configured Train/validation manifest rows and requires explicit resize/interpolation/RGB/normalization values.
- [x] Temporal clip construction preserves source-video ordering and never crosses a source or split boundary.
- [x] Generic frame-engine primitives preserve source-video IDs through evaluation, require caller-supplied loss/optimizer/device, and atomically save/restore model/optimizer/progress/RNG state.
- [x] The integrated runner records scheduler state, validation-selected best checkpoints, per-epoch telemetry, and exact resumable attempts. `aggregate_p1c_frame_baseline.py` fail-closes unless it receives exactly the three registered completed validation-only seed closures and writes their immutable aggregate.
- [ ] Evaluation accepts a frozen selected checkpoint and writes macro/per-class metrics, confusion matrix, and source-video-clustered CI; Test is never used for selection.
- [ ] Aggregate/lifecycle outputs preserve every seed/attempt and reconcile compute estimates with actuals.
- [ ] All code-only tests and repository verification pass; no licensed model training/evaluation claim occurs.

## Owner inputs required before launch

1. Recheck strict MPS availability at launch; do not enable CPU fallback. The runner now resolves whatever policy the config names, so the config is what decides the device.
2. Read the `P1C-FRAME-GAP-CALIBRATION-003` wall-clock ETA in [`TRAINING_LOG.md`](TRAINING_LOG.md) before committing the machine; the superseded `P1C-FRAME-GAP-CALIBRATION-001` figure excluded data loading.
3. Decide whether to register `P1C-FRAME-GAP-FULL-002-WORKERS` (`num_workers` 0 → 8). It is an AI recommendation held at `PLANNED` and cannot launch; the argued case for scientific neutrality and the measured throughput gap are recorded in the file itself.
4. Run the three registered frame seeds, then pass their immutable run directories to `scripts/aggregate_p1c_frame_baseline.py`.
5. MLP/CNN-LSTM dimensions and temporal clip policy remain later temporal-row inputs; B-006 does not block this frame-only baseline.

## Faculty rule

The execution framework itself is not renderable. A new faculty pack is created automatically only after a registered, provenance-complete P1C checkpoint creates a named learned stage for the fixed three Test examples. Until then, the latest faculty pack remains `ucf-temporal-preview-v2-p1b`.

## Current code evidence

- `core/p1c_readiness.py` and `scripts/check_p1c_readiness.py` provide the fail-closed registration report.
- `data/p1c_loader.py` enforces the explicit frame-loading contract; `data/p1c_temporal.py` creates source-bound fixed/padded clips.
- `core/p1c_engine.py` provides caller-owned frame train/evaluation and exact checkpoint-resume primitives.
- `core/device.py` is the single device authority: `resolve_device_policy` resolves every registered `device_policy` (`cuda_strict`, `mps_strict`, `mps_warn_only_repeatability`, `cpu_strict_deterministic`), refuses rather than degrades when a backend is absent, and owns backend dispatch for synchronize, memory telemetry, and accelerator RNG. No runner branches on `torch.mps` versus `torch.cuda` by hand.
- `scripts/run_p1c_frame_baseline.py` trains validation-only, retains every epoch checkpoint, selects only by validation macro-F1, and refuses held-out Test access. Under D-30 it resolves the device from the registered policy instead of hardcoding strict MPS, and records a sampled peak device/host memory instead of a literal zero.
- `scripts/calibrate_p1c_frame_loader.py` measures the real Train/validation loader without checkpoints, Test access, or model selection. Under D-30 it attributes each batch to loader wait versus compute and estimates from wall time; the rates recorded by the superseded `P1C-FRAME-GAP-CALIBRATION-001` and `P1C-FRAME-GAP-CPU-CALIBRATION-002` rows were compute-only.
- Dataset-free fixtures cover registration, image decoding/normalization, deterministic loader planning, clip boundaries, metrics traceability, and checkpoint RNG restoration. The full repository suite currently passes 133 tests.
- Immutable code/formula evidence: `results/p1c/p1c-code-audit-20260724/`. It is not a model, media, or Test result.
