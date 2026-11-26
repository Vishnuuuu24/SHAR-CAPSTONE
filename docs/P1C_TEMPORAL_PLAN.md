# P1C — video-level eval, per-video sampling, temporal probe, temporal model

> Created 2026-09-10. Owner-directed: implement Fix #1–#4 as a staged ML experiment —
> reproduce the baseline, change one thing at a time, measure each change on the SAME
> folds and metric, keep negatives, do not fabricate.
> Live numbers: [`TRAINING_LOG.md`](TRAINING_LOG.md). Queue rationale: [`EXPERIMENT_BACKLOG.md`](EXPERIMENT_BACKLOG.md).
> This file tracks **implemented vs experimentally-validated** for the temporal work only.

## The question this whole plan exists to answer

Is the current low macro-F1 primarily caused by (a) lack of temporal information and
(b) poor frame sampling — or by something else (label noise, tiny effective sample,
domain mismatch)? Then: build the **simplest** system that demonstrably raises
**cross-fold, video-level** macro-F1.

What we already know before starting (all in `TRAINING_LOG.md` / `EXPERIMENT_BACKLOG.md`):

- The **head is not the bottleneck** — a logistic probe recovers 96% of the trained head
  (B-9). So no head-architecture / dropout / focal-loss / LR sweeps (Fix #12).
- **Fold 1 is an easy split** (D-45): ResNet probe 0.3336 on fold 1 vs 0.1579–0.2001 on
  folds 2–4. Every headline before 2026-09-09 rests on fold 1. Cross-fold is mandatory.
- **Motion classes are stuck regardless of backbone**: Fighting F1 has been 0.003–0.117
  across every run; DINOv2 made it *worse* (0.059). This is the primary evidence that
  the ceiling is temporal — the temporal probe (Phase 4) is the direct test.

## Status legend

- **IMPLEMENTED** — code + unit tests landed, not yet run on real data.
- **VALIDATED(fold N…)** — run on real data, numbers in `TRAINING_LOG.md`.
- **DESIGNED** — specified below, no code yet.

---

## Phase 1 — baseline reproduction — VALIDATED (fold 1)

The cached-feature CPU trainer (`core/p1c_head_training.py`, `scripts/run_p1c_cached_head.py`)
reproduces D-42's GPU/bf16 trained head:

| | fold 1 val frame macro-F1 |
|---|---|
| D-42 `CONFIRM-015` (GPU, amp_bfloat16, 3 seeds) | 0.3481 ± 0.0049 |
| cached head (CPU, float32, 3 seeds) `fold1-frozen-VALIDATION` | **0.3482 ± 0.0008** |

The CPU path is a faithful instrument. Cross-fold ResNet folds 2–4 + DINOv2 folds 1–4
were being run by a concurrent session on 2026-09-10 (`p1c-cached-head-crossfold/`).

---

## Phase 2 — Fix #1 — video-level probability aggregation — IMPLEMENTED

**What.** A third evaluation score alongside frame-level and video-majority-vote:

    video_probability = mean_over_frames( softmax(frame_logits) )      # per source video
    video_prediction  = argmax(video_probability)

**Code.**

| file | role |
|---|---|
| `eval/video_probability_aggregation.py` (new) | `aggregate_video_level_probabilities(frame_records, frame_probabilities, class_names)` → per-video mean-prob prediction + `classification_metrics` (macro-F1, per-class P/R/F1/support, confusion matrix). Composite `(source_dataset, source_video_id)` key. Refuses on target-index disagreement within a video. Reports `low_margin_video_count` (videos decided by < 1/K margin). |
| `core/p1c_head_training.py` (edited, additive) | `train_cached_head(..., capture_validation_probabilities=False)`. Default path byte-unchanged and JSON-unchanged (verified by test). When True, attaches the best epoch's validation softmax as `best_validation_probabilities`; inference-only, after all RNG use, so it cannot perturb the selected result (verified by test). |
| `scripts/eval_p1c_video_aggregation.py` (new) | Thin CLI. Trains the registered head per seed, writes per-seed JSON with all three metric blocks, `seed-N-validation-probabilities.npy`, and readable confusion matrices (`seed-N-confusion-frame.txt`, `-video-meanprob.txt`, `-video-majorityvote.txt`), plus `aggregate.json` (mean ± SD across seeds, per-class). |
| `tests/test_video_probability_aggregation.py` (new) | 11 tests: grouping, mean≠majority disagreement case, hand-computed metrics, low-margin counting, input validation. |
| `tests/test_p1c_head_training.py` (edited) | 3 new tests: default path unchanged, captured-probability shape/normalisation, capture does not change the selected result. |

**Not yet done:** fold the same three-metric reporting into `scripts/run_p1c_cached_head.py`
so the cross-fold sweep emits it in one pass (held back only because a concurrent session's
sweep was mid-flight and edits to that script would have reached its later folds).

**Result:** pending — `results/p1c/p1c-video-aggregation-b6/` (fold 1 ResNet running 2026-09-10).

---

## Phase 3 — Fix #2 — per-video frame sampling — DESIGNED

**What.** Sample a fixed `N` frames per source video per epoch instead of consuming every
frame. Attacks two problems at once: the 4.48× frame-count imbalance (purely from video
length) and near-duplicate frames within a video.

**Design.**

- New `data/video_frame_sampler.py`: `select_frame_indices(frame_indices_per_video, n, *, seed, deterministic)`.
  - Split each video's frame sequence into `N` contiguous temporal segments; pick one frame per segment.
  - `deterministic=True` (validation/test) → the segment **midpoint**.
  - `deterministic=False` (train) → a seeded random frame **within** each segment (temporal jitter, never cross-segment).
  - Videos with `< N` frames → take all frames, no oversampling; record the shortfall.
  - Class balance is already exact at the video level (34/4/12 per class — verified,
    `tests/test_split_leakage.py`), so sampling `N`/video keeps it exact at the frame level too.
- Operates on the **cached features** for the probe ablation (the cache rows carry
  `source_frame_index`), and on the loader for any training run.
- `N` is a config field (`data.frames_per_video`), not hard-coded.

**Ablation (probe-level, CPU, no new backbone compute):** logistic probe on DINOv2 features,
folds 1–4, at `current (all frames) / N=8 / N=16 / N=32`. Same C, same eval protocol.

**Cost:** CPU-minutes per point (probe fits shrink with fewer rows). No GPU.

---

## Phase 4 — Fix #3 — cheap temporal probe — DESIGNED

**What.** Before any temporal network: does combining nearby frames' features add signal?

**Design.**

- New `core/temporal_features.py`: given cached per-frame features + `source_video_id` +
  `source_frame_index`, build a per-(video, centre-frame) temporal vector from a window
  `t-w … t+w` (`w` config, default 2 → 5 frames):
  - **A. concat** `[f_{t-2}, …, f_{t+2}]` → `5D`
  - **B. mean** `mean_k f` → `D`
  - **C. mean+std** `[mean_k f, std_k f]` → `2D`
  - Windows never cross a video boundary; a centre frame closer to the edge than `w`
    uses a shorter, edge-clamped window (recorded).
- Fit the **same logistic probe** (standardize, per-backbone C from D-45) on the temporal
  vector; compare to the single-frame probe on **identical folds and eval protocol**
  (frame-level + video mean-prob).
- Cache the temporal vectors so C-sweeps are cheap.

**Decision rule:** if temporal features raise cross-fold video mean-prob macro-F1 outside
the fold-to-fold noise (SD ≈ 0.04 for DINOv2), **and** move the motion classes
(Fighting/Assault/Shooting/Vandalism) specifically → proceed to Phase 5. If not →
write up why (candidate reasons: window too short; frozen frame features already
near-static; label noise dominates) before building anything.

**Cost:** CPU only, reuses existing caches. This is the pivotal cheap experiment.

---

## Phase 5 — Fix #4 — lightweight temporal model — DESIGNED (gated on Phase 4)

Only if Phase 4 shows signal. Architecture, cheapest first:

    video → sample N frames (Phase 3) → DINOv2 (frozen, cached) → [N, D] sequence → temporal model → MLP → 14 classes

| option | temporal model | when |
|---|---|---|
| A | temporal mean-pool + MLP | first — it is Phase 4-B with a learned head |
| B | 1-layer GRU (hidden ≈ 128) + MLP | if A ≈ mean-prob baseline |
| C | 2-layer tiny Transformer encoder | only if B clearly wins and is still cheap |

- Frozen DINOv2, features cached → the temporal model trains on `[N, D]` sequences on
  **one MIG slice or CPU**, not a full-frame forward per step.
- New config family `P1C-VIDEO-TEMPORAL-<opt>-AR-FOLD<n>`; every field in Fix #13 recorded.
- 3 seeds, all 4 folds, validation-only selection; Test untouched until a single final config.

**Cost:** GPU. A slice booking > 30 min gets an ETA heads-up to the owner first.

---

## Phase 6 — cross-fold validation — DESIGNED

Every candidate that passes its phase gate runs folds 1–4, 3 seeds, reported as
`mean ± SD` with per-fold values shown. The results table is built only from completed
runs; missing cells are marked `incomplete`, never filled with a guess.

| Method | F1 fold 1 | fold 2 | fold 3 | fold 4 | Mean ± SD |
|---|---|---|---|---|---|
| ResNet50 frame, trained head | 0.3482 | — | — | — | incomplete (cross-fold sweep running) |
| DINOv2 frame, trained head (fixed recipe) | ~0.336 | — | — | — | incomplete |
| DINOv2 frame + video mean-prob | — | — | — | — | incomplete (Phase 2) |
| DINOv2 + per-video sampling | — | — | — | — | incomplete (Phase 3) |
| DINOv2 + temporal probe | — | — | — | — | incomplete (Phase 4) |
| DINOv2 + temporal model | — | — | — | — | incomplete (Phase 5) |

Probe cross-fold reference (D-45, not a trained head): ResNet 0.2199 ± 0.0778,
DINOv2 0.2973 ± 0.0438.

---

## DINOv2 recipe caveat (Fix #7)

The fixed AdamW recipe (`weight_decay = 1e-4`) was tuned around ResNet features; the B-2
probe found DINOv2's optimum ~1000× more regularised (ResNet C=1e-2 vs DINOv2 C=3e-5).
Early cross-fold numbers show the DINOv2 trained head at ~0.336 — **below** its own probe
(0.3625) and below the ResNet head. **Do not read this as "DINOv2 is worse."** Before any
such conclusion, run a small controlled `weight_decay ∈ {1e-4, 1e-3, 1e-2, 1e-1}` sweep
for DINOv2, everything else fixed, folds 1–2 first. This is the one permitted regularisation
sweep — it is disambiguating a backbone comparison, not tuning the head for its own sake.

## Leakage guards (Fix #8 / #15) — VALIDATED

`tests/test_split_leakage.py` (new) asserts, against the real artifacts:

- train/validation/test **video** sets pairwise disjoint — **verified ∅ on all 4 folds**.
- one label per video across every split it appears in — verified.
- video-level class balance 34/4/12 per class, 14 classes — verified all folds.
- every feature-cache row keeps `source_video_id` and stays inside its manifest split —
  verified for all present ResNet + DINOv2 caches.
- cache `manifest.json` records the matching split and fold.

Frame-count imbalance (357k–400k train frames from 476 videos) is real but is a
sampling problem (Phase 3), not leakage.

## Label noise (Fix #16)

Inherited video-level labels stay the protocol. Every number is named an
**inherited-label** score. Fold 1 has **no** constructible event-only view (D-40: 48/168
test videos carry an interval; Abuse/Robbery/Stealing have none) — no event-only claim is
made for it. An event-only sensitivity analysis is possible only on folds/videos that have
intervals and would be reported separately.

## Compute discipline (Fix #14)

Reuse the existing DINOv2/ResNet caches (never rebuild). Prefer CPU probes. No large
sweeps. Every expensive step configurable and resumable. Do not delete caches. Do not
launch uncontrolled parallel CPU jobs — pin to 4 cores (D-39 budget), `nice`, one job
at a time when a peer session is already running one. GPU only via the existing pod
infra (`scripts/dgx/`), never a separate execution path.
