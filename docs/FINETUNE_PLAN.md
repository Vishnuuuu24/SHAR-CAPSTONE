# Partial fine-tuning of DINOv2-B — plan (D-67) and result (D-68: not established)

> **Result (2026-11-03, D-68): neither arm passes** — B_ft2 −0.002 [−0.039, +0.033], B_ft4 +0.012 [−0.019, +0.044].
> Status: **approved by the owner 2026-11-03 with a 4-block arm added** ("alright lets work on 4 blocks"). Nothing here
> has been measured yet.
> Protocol: D-65 (inner-monitor epoch selection, paired claim rule, validation only).

## 1. The question

Every DINOv2 result so far used a **frozen** backbone: the image model's weights never change, only the small
head on top learns. This experiment lets the **last 2, or the last 4, of DINOv2-B's 12 transformer blocks** (plus its
final LayerNorm) adapt to UCF-Crime frames, and asks:

> Does adapting DINOv2-B's last two (or four) blocks raise 14-class macro-F1 over the same head on frozen features?

Each candidate changes exactly one factor against the control: how many final blocks (+ final norm) are trainable. Frames, head, seeds, optimiser,
budget, epoch selection and scoring are identical to the control.

## 2. Prior evidence (why this is worth one run, and why the odds are modest)

| Source | What it showed | Bearing |
|---|---|---|
| A5 (`docs/TRAINING_LOG.md`, 2026-09-11) | DINOv2 blocks 10–11 unfrozen: Shooting-vs-not ROC 0.600 → 0.800 | **Lead only**: one split, 18 held-out videos (SE ≈ 0.12), training loss 0.000 (memorised), binary task |
| N29 (`docs/REMEDIES_TRIED.md`) | VideoMAE-B, top k blocks unfrozen: k2 −0.026 (0/4 folds), k4 +0.007 (1/4) | **Negative prior**: partial fine-tuning has already failed once on this data, though on a weaker backbone (0.35 vs 0.46) with no epoch selection (fixed 8 epochs) |
| D-63 learning curve | +0.089 per doubling of training videos | The model is data-starved; fine-tuning adds parameters, not data — overfitting is the main risk |

Correction to my earlier message: partial fine-tuning is **not** untried on this project — N29 tried it for VideoMAE.
It is untried for DINOv2 on the 14-class task.

## 3. Arms

| Arm | Backbone | Trainable | Role |
|---|---|---|---|
| `B_frozen` | DINOv2-B, cached features | D-53 `step_permuted` head (7.1 M) | **Control**; must reproduce D-66's `B_ref` (0.3835) |
| `B_ft2` | DINOv2-B, blocks 0–9 frozen, 10–11 + norm trainable | head + 14.2 M backbone | **Candidate** |
| `B_ft4` | DINOv2-B, blocks 0–7 frozen, 8–11 + norm trainable | head + 28.4 M backbone | **Candidate** (added by the owner) |
| `B_ft2_lr0` | same pipeline as `B_ft2`, backbone LR = 0 | head only | Pipeline check, fold 1 seed 0 only; not a contrast |

Registered contrasts: **`B_ft2` vs `B_frozen`** and **`B_ft4` vs `B_frozen`**, Holm-adjusted over the two
(the stronger one must clear one-sided p ≤ 0.0125, i.e. a 97.5% interval above zero; the other then p ≤ 0.025).
`B_ft2` vs `B_ft4` is reported descriptively, not claimed.

## 4. Fixed settings (all copied from the control unless stated)

- **Folds/seeds:** AR folds 1–4 × seeds 0, 1, 2.
- **Training frames:** the same 8-frame seed-0 draw per training video as `load_fold_features`. **No augmentation**
  (it would be a second factor).
- **Split:** 30 training videos per class; inner monitor = 4 per class, fixed per fold (`inner_monitor_videos(seed=fold)`,
  as D-66); 108 validation videos scored once at the selected epoch; **test not read**.
- **Head:** D-53 `step_permuted`, seeded init (identical across arms), LR 3e-4, weight decay 1e-4, batch 64 videos,
  ≤30 epochs, patience 5, best monitor macro-F1 selects the epoch.
- **Backbone LR (the one new hyperparameter):** grid {1e-5, 3e-5, 1e-4}, AdamW, same weight decay.
  Chosen **per fold and per arm** by mean inner-monitor macro-F1 over the 3 seeds (ties → lower LR) — validation never sees the choice. The grid is
  centred on A5's 3e-5 (10× below the head) and brackets it both sides; if the winner sits on an edge in ≥2 folds,
  that is reported as "grid edge", not silently extended.
- **Precision:** float32 throughout, as the cache was built. Trainable blocks use activation checkpointing
  (recompute in the backward pass) to fit 18 GB; gradients are exact (unit-tested against the plain path).
- **Head on CPU, tail on GPU:** the head runs where the control's does; only the backbone blocks use the GPU.

## 5. How it runs cheaply (design)

Blocks 0–7 never change, so their output is computed **once per fold**:
1. Decode the training + monitor frames (3,808 per fold) through the registered preprocessing and blocks 0–7; keep
   the 261×768 hidden state per frame on the GPU (~3 GB, float32).
2. `B_ft4` trains blocks 8–11 from that cache. For `B_ft2`, frozen blocks 8–9 are applied once to make a second
   cache, and the first is freed; each epoch then runs only blocks 10–11 + norm + pooling + head.
3. Per fold: 2 arms × 3 LRs × 3 seeds = 18 trainings; only the 6 chosen runs are scored on validation.
4. Validation (37,600–49,719 frames per fold): one decode + blocks 0–7 pass per fold, feeding all six chosen tails.

**Estimated slice time: about 3 h** (worst case 4–5 h). A smoke run (fold 1, seed 0, 2 epochs, 2,000 validation
frames) measures the real rate first; the full ETA is posted before launch.

## 6. Checks before any number counts

1. **Pixel path = cache:** 128 frames through this pipeline vs the B fold cache, min cosine ≥ 0.9999
   (the D-66 check gave 0.9999997).
2. **Split = whole:** blocks 0–9 (cached) + 10–11 at initialisation reproduce the full backbone's pooled features
   (max |Δ| reported, cosine ≥ 0.9999).
3. **`B_ft2_lr0` reproduces `B_frozen`** on fold 1 seed 0 within ±0.01 macro-F1.
4. **Freeze audit:** after training, blocks 0–9, patch embedding, position embedding and register tokens are
   bit-identical to the registered checkpoint (sha256 `c24ecfb4…`); the trainable set is exactly blocks 10–11 + norm.
5. Unit tests for 2 and 4 with a tiny model; `unittest discover` passes before launch.

## 7. Claim rule and what gets reported

- **Claim:** paired source-video cluster bootstrap (`paired_cluster_bootstrap`, 2,000 draws) 95% CI lower bound > 0
  **and** point gain ≥ +0.02 macro-F1 (D-65).
- **Always reported:** per-fold and per-class F1, violence-class mean (Abuse, Assault, Fighting, Shooting), selected
  backbone LR per fold, selected epoch, training loss at that epoch (memorisation check), and the historical
  +0.04-on-4/4 gate.
- **Label honesty:** inherited-label score, never event-only (D-40).
- **Reproducibility:** seed, config, git revision, fold-cache sha256 and weight sha256 in `summary.json`.

## 8. What happens after

| Outcome | Next step |
|---|---|
| A claim met | That arm (the larger validated gain if both pass) becomes the candidate for D-65's single flagged test look |
| Claim not met | Recorded as N47 with all numbers; the classifier-improvement track closes; Phase 5 |

No rescue arms after seeing validation (no other block count, no augmentation, no other LR). Anything else would need its own
prospective registration.

## 9. Files (planned)

- `core/vit_split.py` — trunk/tail split of the timm ViT, `CachedTailHead`, freeze-audit digests
- `core/finetune_experiment.py` — per-fold trunk cache, LR grid, selection, validation pass, checks
- `core/d65_claim.py` — the shared D-65 contrast report and Holm rule (also used by the D-66 runner now)
- `core/frame_batches.py` — shared blob → registered-preprocessing batches (also used by the DINOv2-L extractor)
- `core/p1c_head_variants.train_sequence_head` — new optional `optimizer_param_groups` (default unchanged)
- `scripts/run_finetune_experiment.py` — thin CLI (`--smoke` for the calibration run)
- `tests/test_finetune_experiment.py` — split = whole, trainable set, source network untouched, checkpointed
  gradients = plain, index lookup = direct features, permutation preserved, LR selection, Holm
- Output: `results/p1c/finetune-b-last2-last4/{runs.json, probs.csv, checks.json, summary.json}`, log in
  `results/p1c/pod-logs/`
