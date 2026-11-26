# Experiment backlog — P1C classification

> Created: 2026-08-28 · Owner-directed: "let this run finish and then we'll try ALL the
> other options." This is the queue and the reasoning behind its order.
> Live status: [`PROJECT_STATUS.md`](PROJECT_STATUS.md). Measured numbers:
> [`TRAINING_LOG.md`](TRAINING_LOG.md). This document holds *plans*, not results —
> when something here runs, its numbers go to `TRAINING_LOG.md` and its row here is
> marked done with a pointer, never with a copy of the figures.

## Where we are

Four runs on official Action-Recognition **fold 1**, all validation-only, Test untouched:

| run | one change | val frame macro-F1 | s/epoch |
|---|---|---|---|
| BASELINE-011 | — (3 seeds) | 0.2939 ± 0.0196 | 634 |
| **SCREEN-012** | backbone frozen | **0.3537** (1 seed) | **216** |
| SCREEN-013 | lr 3e-4 → 3e-5 | 0.3364 (1 seed) | 635 |
| SCREEN-014 | frozen + augmentation | 0.3421 (1 seed) — **negative** | 308 |

Chance is 0.0714. CONFIRM-015 is re-measuring the frozen configuration over three seeds.

**The diagnosis everything below responds to:** 391,014 training frames come from only
**476 source videos**, so frames within a video are near-duplicates and a 23,536,718-parameter
fine-tune memorises per-video appearance instead of per-class activity. Freezing the backbone
(28,686 trainable parameters) is the only screened change that helped.

## The measurement problem — read this before trusting any single result below

The validation split is **56 videos**. SCREEN-012's clustered 95% CI was **0.228–0.421**, a
span of 0.19 around a headline of 0.354. Most items below would move the number by less than
that. Two consequences that are not optional:

- A single-seed screen decides *direction*, never magnitude. The registered 0.04 threshold
  exists for exactly this and is carried forward unchanged run to run.
- **Running folds 2–4 sharpens every future comparison more than any single item below**,
  because it averages over 224 validation videos instead of 56. It is cheap at the frozen
  cost (~1–1.5 h per fold). Treat it as infrastructure for the rest of the backlog, not as
  one more experiment competing with them.

  **DONE 2026-09-09, and it was worth more than predicted** (`P1C-PROBES-FOLDS-CONFIRM-AR`,
  D-45). This entry expected folds to *sharpen* comparisons. They did something stronger:
  they showed fold 1 is an easy split whose numbers are not representative. The ResNet probe
  scores 0.3336 on fold 1 against 0.1579–0.2001 on folds 2–4 — a spread of 0.1757, i.e. 2.3×
  the DINOv2 effect. Every headline produced before this date rests on that fold, D-42's
  0.3481 included. Folds are now mandatory for any effect claim, not optional sharpening.

---

## Tier 0 — do this first, it changes the cost of everything else

### B-1. Cache the frozen backbone features

**What.** With the backbone frozen it computes the *same* 2048-D vector for a given frame on
every epoch. We currently recompute 391,014 identical forward passes 30 times. Run the
backbone once, write the features to disk, and every later experiment trains only the head.

**Effect.** ~216 s/epoch → **seconds per epoch**. The entire head-design space (B-4, B-5, B-8,
B-9, B-10) becomes a same-afternoon sweep instead of a queue of 3.5 h slice bookings.

**Size.** 391,014 × 2048 × 4 B = **3.2 GB** fp32, 1.6 GB fp16. Validation adds 37,600 × 2048.
Fits in RAM; goes to the pod's emptyDir like the frame blob (never read from NFS at random —
D-33).

**Why it is legal now.** Cached features are only valid if the input is deterministic.
Augmentation would change the features every epoch and make caching impossible — **SCREEN-014's
negative result is what unlocks this.** Note the dependency explicitly: if augmentation is ever
revisited, it cannot use the cache.

**Correctness requirement.** The cache must record the backbone weight digest, the
preprocessing block, and the frame-blob SHA-256, and refuse to load against a mismatch. A
stale feature cache silently answering for a different backbone is exactly the class of
undetected-wrong-result this project does not accept.

**Cost as built (2026-08-29).** ~13 min of forward pass, not the ~4 min estimated: stripping away the
backward pass leaves the run loader-bound, the same effect that made SCREEN-014's augmented epochs 43%
slower. Paid once.

### B-1 — DONE, 2026-08-29

Built and verified. `results/p1c/p1c-frozen-feature-cache-ar-fold1/`:

| split | rows | size | verification |
|---|---|---|---|
| train | 391,014 | 1.6 GB | 482/512 sampled rows bit-identical, max abs delta 3.906e-3 |
| validation | 37,600 | 150 MB | 478/512, max abs delta 3.906e-3 |

float16, 2048-D, `model_control: gap_linear`, `pooling: global_average`. Evidence artifact
`train/verification.json`; cache SHA-256 in each `manifest.json`.

**Consequence for everything below: head-only experiments no longer need the GPU.** A linear-head
epoch on cached features is a 391,014 x 2048 x 14 matmul, ~11 GFLOP -- under a second on 4 CPU cores,
against 216 s before. B-8, B-9 and any head sweep therefore stop competing for the MIG slice entirely,
leaving it for the arms that genuinely need it (B-2, B-3, B-12).

**Verification lesson, recorded so it is not repeated.** The check was rewritten three times across
three pod runs (~50 min of slice) before it was right, and every failure was the check, not the cache
-- the train cache has been byte-identical since its first build. The decisive fact, which should have
been established first: `scripts/run_p1c_frame_baseline.py` seeds RNGs but never calls
`torch.use_deterministic_algorithms`, so every training run in this project already accepts cuDNN's
last-bit nondeterminism. Bit-exactness was a stricter standard than the consuming pipeline provides.
The gate is now `|a - b| <= atol + rtol * |b|` (both 1e-3), the only form that behaves at both ends of
this data -- activations peak at 3-9 where a relative bound applies, and ReLU leaves many channels at
or near zero where only an absolute floor is meaningful. Integer ULP distance is reported as a
diagnostic, never the gate. Provenance keeps zero tolerance. **Before writing any future numerical
equality check, establish what determinism the producing code actually guarantees.**

---

## Tier 1 — could move the number, not just nudge it

### B-2. A better frozen backbone: DINOv2 or CLIP

**What.** We are linear-probing 2015-era ImageNet-supervised ResNet50 features on grainy
320×240 surveillance footage — a real domain mismatch. Self-supervised (DINOv2) and
vision-language (CLIP) features are substantially stronger under linear probing on
out-of-domain data.

**Why I rank it first for expected gain.** Our current ceiling may be the *features*, not the
head. Everything in Tier 2 tunes the head against features we have never questioned.

**Cost.** Frozen, so cheap — one-off weight download plus one cached forward pass per backbone
(B-1). Needs a registered weight digest and source URL like every other backbone here.
**Caveat:** different input resolution and normalisation, so it is not a one-line swap and it
is *not* single-factor against SCREEN-012 — it is a new baseline in its own right.

### B-2 — DONE (DINOv2 half), 2026-09-02. Video-backbone half CLOSED 2026-09-16 (D-50): VideoMAE-B (Kinetics) scores Fighting 0.000 / Shooting 0.000 on CCTV; fusion +0.0069, bar not met.

Run as `P1C-PROBES-B2-DINOV2-AR-FOLD1`; numbers live in [`TRAINING_LOG.md`](TRAINING_LOG.md),
the decision in [`DECISIONS.md`](DECISIONS.md) D-44. Headline: best probe **0.3625** against
ResNet's **0.3336** and the trained head's **0.3481 ± 0.0049** — a direction, *not* a promotion
(it does not clear the registered 0.04 threshold; see D-44 for the full set of reasons).

**The caveat above was wrong, and it is worth saying why.** This entry predicted a different
input resolution and normalisation would make the swap non-single-factor. Measured: DINOv2's
registered normalisation mean/std is *identical* to the project's existing preprocessing, and
the model runs at our registered 224×224 (14 divides 224 evenly; the position grid interpolates
per forward pass). So it *was* single-factor after all — only `backbone.family`, `.feature` and
`.weights` moved, and the result is attributable to the backbone alone. A planning assumption
overturned by measurement rather than carried forward.

**The operational lesson, which cost nothing here but would have inverted the conclusion:**
**a hyperparameter grid is a property of the backbone, not of the task.** The two optima sit
more than two orders of magnitude apart — ResNet peaks at C=1e-2, DINOv2 at C=3e-5. Scoring
DINOv2 on ResNet's grid gives 0.3226, i.e. a *loss*. Any future backbone swap must re-sweep,
and must bracket its maximum on both sides before the number is quoted.

**What it did not do.** It did not help the motion classes. Fighting fell to 0.059 from the
head's 0.115, and the macro gain is a small net of large offsetting per-class swings
(RoadAccidents +0.318 … Shooting −0.458). B-12 keeps its full justification.

**Still open under B-2:** CLIP, as a separately-labelled secondary backbone and zero-shot
reference. Zero-shot uses no training data, so per rule 3 it is a different task setup and must
never be ranked against these supervised numbers; prompt wording becomes a hidden tunable, and
CLIP's training corpus is unauditable.

### B-3. Partial unfreezing (`layer4` trainable, everything below frozen)

**What.** The middle ground we skipped entirely. Full fine-tune failed, full freeze won, and
we never measured between them. `layer4` carries the most ImageNet-specific semantics and is
the layer most worth adapting to surveillance imagery.

**Cost.** ~400 s/epoch (gradients flow through the last block only). Incompatible with B-1 —
the cache would have to start below `layer4`, which is a variant worth building if this wins.

### B-4. Use the 7×7 spatial map instead of GAP

**What.** Global average pooling collapses the pre-GAP 7×7×2048 grid into one vector,
discarding *where* in the frame anything is. The project's own architecture plan already calls
for this — the **SEMSCNN-inspired 2-D multiscale-SE head** in [`ARCHITECTURE.md`](ARCHITECTURE.md)
— and it has never been built. It consumes the pre-GAP spatial map; it must never be given a
reshaped 2048-D GAP vector (see the proposal-status table in `docs/PROJECT_RULES.md`).

**Cost.** Cheap on a cached feature map, but the cache is then 7×7 larger: 391,014 × 7 × 7 ×
2048 × 2 B ≈ **157 GB** in fp16, which does **not** fit the emptyDir comfortably alongside the
84.3 GiB frame blob. Either cache a reduced map, or recompute for this arm only. Size this
before building it.

### B-5. Per-video frame sampling

**What.** Sample a fixed N frames per source video per epoch instead of consuming every frame.

**Why it may be the best cheap idea here.** It fixes two things at once: the 4.48× class
imbalance (RoadAccidents 10,656 → Arrest 47,704 frames, caused purely by videos differing in
length while videos themselves are exactly 34/class), *and* the near-duplicate problem, by
never showing the model 800 near-identical frames of one scene in one epoch. That is arguably a
more direct attack on the diagnosis than augmentation was — and augmentation lost.

**Cost.** Free at runtime; a sampler change. Changes what an "epoch" means, so the epoch
ceiling and patience need restating rather than inheriting.

---

## Tier 2 — cheap, worth sweeping once B-1 exists

### B-6. Mean-probability video aggregation

Our video-level score uses majority vote, which discards confidence — `eval/video_level.py`
documents this limitation itself. Averaging per-frame probabilities usually beats it.
**Evaluation-only, changes no training**, so it can be recomputed against checkpoints we
already hold. Free.

### B-7. Discriminative learning rates (backbone ~1e-5, head 3e-4)

The one point on the freeze/lr axis never measured. SCREEN-013 dropped the rate on the backbone
*and* the head together, so the head learned 10× slower too — that may be exactly why it lost
to freezing, which keeps the head at full rate. Needs per-parameter-group LR support in the
runner. ~635 s/epoch, incompatible with B-1.

### B-8. Head regularisation sweep — weight decay, dropout, label smoothing

Never swept. Trivial on cached features, and with 28,686 trainable parameters the head's
regularisation is one of the few remaining knobs that is genuinely free to explore.

### B-9. Classical linear probes as a reference point — logistic regression, SVM, kNN

Not a proposal to ship one, a **sanity check**: these are the standard reference for "how good
are these frozen features, independent of our training loop." If scikit-learn on cached
features matches or beats our head, the head/optimizer is not where the loss is. Cheap and
diagnostic.

### B-9 — DONE, 2026-09-01. **It answered the question, and the answer redirects the queue.**

Full figures in the `P1C-PROBES-B9-AR-FOLD1` row of [`TRAINING_LOG.md`](TRAINING_LOG.md).
The short version:

| probe | best configuration | frame macro-F1 |
|---|---|---|
| logistic regression | standardize, C=0.01 | **0.3336** |
| linear SVM | standardize, C=1e-4 *(edge of swept range — a lower bound, see K-25)* | 0.2880 |
| k-NN (k=20, distance) | standardize | 0.2713 |
| k-NN (k=20, distance) | l2 / cosine | 0.2667 |
| *trained head, CONFIRM-015, 3 seeds* | *—* | *0.3481 ± 0.0049* |

**A plain logistic regression recovers 96% of the trained head**, with none of our optimizer,
schedule, early stopping or checkpoint selection. The 0.0145 shortfall is 3.0x the head's seed
SD, so it is not nothing — and the probe's C was tuned on the same validation split it is scored
on, so the honest gap is if anything slightly larger. But it is far inside the registered 0.04
promotion threshold and inside the fold's clustered CI several times over.

**Consequence for this backlog: Tier 2 head work is deprioritised.** B-4 (head design) and B-8
(head regularisation) are sweeping a knob worth at most ~0.015 on this fold, and the 56-video
validation split cannot resolve that — its 95% CI is near ±0.19. Running them would produce
numbers we could not distinguish from noise. They stay on the list; they stop being next.

**What this does not say.** It does not say the head is optimal — it says the *ceiling set by
these features* is close, so the next real move is a better representation (B-2), or more
information in the input (B-12), not a better classifier on top of ResNet50 C5 features.
B-5 (per-video sampling) and B-10 (class weighting) are unaffected: they change what the
classifier is *fitted on*, not its capacity, and the probe inherited exactly the same imbalance.

Two by-products worth keeping. Feature scaling is a real factor, not a formality — best
`standardize` 0.3336 against best `none` 0.3042, a 0.029 spread that would have been silently
attributed to the classifier had only one been reported. And `LinearSVC` costs ~20x a logistic
fit and does not parallelise, which is now **K-25**.

### B-10. Class-weighted or focal loss (K-22)

Targets imbalance, not overfitting — a different disease from the one that has dominated every
run. Registered as K-22 and needs the validator and runner changed together. Expect it to move
per-class recall more than the macro headline. Note it partly overlaps B-5, which addresses the
same imbalance at its source; run one before assuming the other is still needed.

### B-11. Test-time augmentation

Average predictions over a few deterministic crops/flips at inference. Evaluation-only, no
retraining. Small, reliable, and it is the *evaluation* side of the augmentation idea that
failed on the training side — worth knowing whether that asymmetry holds.

---

## Tier 3 — structural, and the only thing that can fix Fighting

### B-12 — CLOSED, 2026-09-11 (D-46, D-48, D-49)

Not built, on evidence. Temporal windows at 0.7 s and 2.8 s (+0.0037), feature temporal
differences (−0.0146, 0/4), dense 2.67 s clips (+0.008, p=0.84) and patch-level motion
(+0.071 within-video, p=0.09, no transfer) all failed to move video-level classification.
A GRU or Transformer over the same frozen features would inherit the same ceiling. The
Fighting/Assault/Vandalism problem is real and unsolved; what the evidence rules out is that
sequence modelling *over frozen appearance features* is the fix. See the Shooting section of
`CROSS_FOLD_REPORT.md`.

### B-12. Multi-frame / temporal model (original entry, retained)

**Fighting has scored 0.003, 0.024, 0.031, 0.111 and 0.117 across every run so far.**
Assault sits at 0.061–0.103. These are *motion* classes: a single still frame of a fight and a
still frame of two people standing close together are near-identical pixels. No head design,
learning-rate schedule, loss weighting or backbone swap in Tier 1–2 can fix this, because the
information is not in the input.

This is the largest remaining item and the one most likely to change the project's conclusion
rather than its third decimal place. It is gated on B-006 (MPS determinism) only for the
historical MacBook path; on CUDA that gate does not apply. Scope it properly before starting —
clip sampling, memory on one MIG slice, and a fresh epoch-cost measurement.

---

## Sequencing

1. ~~CONFIRM-015~~ **DONE** → 0.3481 +/- 0.0049, D-42.
2. ~~B-1 feature cache~~ **DONE** → built and verified 2026-08-29.
3. ~~B-9 linear probes~~ **DONE** → best probe 0.3336 against the head's 0.3481. Verdict: invest in
   **features**, not heads.
4. **B-2 better backbone** ← **NEXT**, and **B-5 per-video sampling** — the two highest-expected-value
   arms, and B-9 is the reason B-2 now leads.
5. **Folds 2–4** once a candidate is worth measuring properly, to escape the 56-video CI. B-9 sharpened
   this: a 0.0145 effect is not measurable on one fold, so any Tier 2 head result would need the folds
   before it could be believed at all.
6. **B-12 temporal** as the structural project, planned separately rather than queued behind
   the sweep. B-9 added evidence: the probe scores Fighting at **0.026** against the head's 0.115 —
   neither is usable, and no classifier on single-frame features fixes it.

## Rules that still apply to every item above

- One material factor per comparison when the comparison is meant to attribute a cause. B-2 and
  B-5 both break single-factor comparability against SCREEN-012 and must be declared as new
  baselines, not as one-change children.
- Every fold-1 number is an **inherited-label** score. Fold 1 has no constructible event-only
  view — 48 of 168 test videos carry an official interval and Abuse, Robbery and Stealing have
  none (D-40). Never report one as event-only.
- Negative results stay visible (rule 4). SCREEN-014 is in `TRAINING_LOG.md` as a negative, and
  anything here that fails gets the same treatment.
- Test is spent once, on a final configuration. The runner is validation-only for every run
  class and nothing above changes that.
- Do not edit or commit `core/`, `data/`, `models/` or `scripts/` while a pod is running
  (K-23). Creating a *new* module is safe; editing one the run imports is not.

### ROI study -- CLOSED, 2026-09-22 (D-51)

Person-crop input (YOLO11m-pose union box, +10% margin, full-frame fallback at 40.3% of rows) against the full frame on the validated pipeline: 0.3975 vs 0.4068, paired -0.0092 +/- 0.0462, 1/4 folds positive. Fighting 0.193 -> 0.185, Assault 0.215 -> 0.132. Weapon detection: zero hits in 1,017 Shooting frames. Artifacts: `results/p1c/p1c-roi-study/`, `results/p1c/p1c-roi-cache/`, `results/p1c/p1c-roi-detector-check/`. Keypoints retained in `detections.jsonl`; a skeleton experiment is the one motion representation not tried and is not scheduled.
