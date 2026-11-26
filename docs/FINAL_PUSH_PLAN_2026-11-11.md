# Final push plan — SHAR

> Written 2026-11-11 after the full audit. Status: **executed on 2026-11-11; all steps run-complete; results in D-76 and the `P5-*` rows of `TRAINING_LOG.md`.**
> Goal: take the project as far as the data allows, with numbers that survive review,
> then stop and write the paper.
> Next free decision ID is D-71. The five rules in `docs/PROJECT_RULES.md` apply to every step.

## Where we start from (audit findings, all measured 2026-11-11)

| Fact | Number | Source |
|---|---|---|
| Headline 14-class test | macro-F1 0.4138, accuracy 41.8% | `results/p1c/headline-14class-test/test.json` |
| Plain baseline: average each video's DINOv2 features, logistic regression | test 0.394 / 39.9%; validation 0.399 | `scripts/review_2026_11_11/03_*.py`, `04_*.py` |
| Same baseline, cross-validated inside each fold's 476 training videos | 0.394 / 0.390 / 0.409 / 0.397 | same scripts |
| Human captions (UCA), bag-of-words classifier, same folds | 0.80 macro-F1 (0.79 with class words removed) | `scripts/review_2026_11_11/05_*.py` |
| Stored test predictions regrouped into 7 meaning-based families | 55.7% accuracy, 0.52 macro-F1 | computed from `test.json` |
| Near-duplicate videos across the 700 | 4 pairs; `Arrest009` = `Shooting015` | `scripts/review_2026_11_11/02_*.py` |

The five review scripts are drafts written for the audit. They read a temporary folder
through `SHAR_SCRATCH_DIR`; Step 1 turns them into proper scripts.

Three claims in the current docs are not supported as written and are repaired below:
the 0.46 headline (Step 0), "attention pooling +0.057" (Step 1), and "restoring clean
frames always hurts" (Step 3).

## Rules for this plan

- **One evaluation method for every classifier comparison** (defined in Step 1). No more
  scoring on the 56 validation videos per fold.
- **Write the pass rule before the run.** A candidate passes if the paired 95% interval of
  its gain excludes zero and the gain is at least +0.02 macro-F1 (this is D-65's rule,
  kept). With several candidates in one round, apply the Holm correction.
- **Test folds are not read in Steps 1, 2 and 6.** They are read once more in Step 7,
  declared in advance.
- **No rescue arms.** A candidate that fails is recorded and left.
- Pods: check `kubectl get pods` first, never touch a `Running` pod, save logs before
  deleting, give an ETA before anything over about 30 minutes, and do not edit
  importable code in the mounted checkout while a pod runs (K-23).

---

## Step 0 — Set the record straight (no compute)

1. Add an `EVIDENCE_REGISTER.md` entry: the honest validation level is about 0.40, and
   the 0.4626 figure carries about 0.06 of epoch-selection optimism (D-66 already says
   this; the status page and faculty page still lead with 0.46).
2. Record that the audit scored a logistic baseline on the test folds on 2026-11-11.
   It selected nothing, but it is a second look at test and must be on record.
3. Fix K-01's stale count: the suite is 1,007 tests, 6 skipped, all passing.
4. Replace the long status paragraph in `PROJECT_STATUS.md` with a pointer to this plan.

Done when: the four edits are made and reviewed.

## Step 1 — Stronger evaluation and the missing baseline rows (CPU, cached features)

**Why:** 56 validation videos (4 per class) cannot detect a gain smaller than about 0.05.
Most past "neutral" results are really "could not tell".

**The evaluation method.** For each official fold:

- Pool = that fold's 532 official training videos (the current train and validation
  parts together, 38 per class). The fold's 168 test videos are not touched.
- Split the pool into 5 parts, class-balanced, whole videos only. Train on 4, predict
  the 5th, rotate. Repeat with 3 different splits.
- Every arm uses a training length fixed in advance. No early stopping on scored videos.
- Score = macro-F1 over the held-out predictions of all 532 videos. Compare two arms
  with the paired, video-level bootstrap already in `core/failure_analysis.py`
  (2,000 draws), resampling by source video because videos repeat across folds.

**Arms (all on existing caches):**

| Arm | What it is | Claim it tests |
|---|---|---|
| R0 | Average DINOv2-B features per video, logistic regression | the reference row |
| R1 | Linear head on 8 frames, majority vote (D-46 pipeline) | sampling and voting gains |
| R2 | Attention pooling head (D-53) | "+0.057 from attention" |
| R3 | R0 on ResNet50 features | "DINOv2 beats ResNet" |

**Code:** one module `core/inner_cv.py` (splits, fixed-length training, scoring) and one
thin script `scripts/run_inner_cv.py`. Reuse `train_sequence_head` and the cache loaders.

**Cost:** CPU only inside a pod; my estimate is under an hour for all four arms.

Done when: a table of R0–R3 with paired intervals is in `results/p1c/inner-cv-baselines/`
and in `TRAINING_LOG.md`, and D-53's claim is either confirmed or corrected in
`EVIDENCE_REGISTER.md`.

## Step 2 — Re-test the near misses and stack what survives (CPU, cached features)

**Why:** this is the honest route to a higher number. Several past candidates pointed
upward but could not clear the old gate. With about ten times more scoring videos, real
small gains become visible and can be combined.

| Candidate | Old result | Cache |
|---|---|---|
| DINOv2-L instead of B | +0.023 [-0.022, +0.075] | `results/p1c/p1c-dinov2l-cache` |
| DINOv2 + SigLIP fusion | +0.005 | `results/p1c/p1c-siglip-patch-cache` |
| DINOv2 + V-JEPA 2 fusion | +0.029, fold 4 negative | `results/p1c/p1c-exp-vjepa2-probe` |
| Training frames from GlanceVAD clicks | +0.016 | `results/p1c/glance-annotations` |
| 300 extra UCF videos with class weights | +0.029, 2/4 folds | `results/p1c/p1c-extra-ucf-cache` |

Procedure: test each against the Step 1 winner with the pass rule and Holm correction
over the five. Then add survivors one at a time, largest gain first, keeping an addition
only if the stack still passes against the previous stack.

My expectation, which is an inference and not a measurement: one or two survive and the
honest score lands between 0.42 and 0.46. If none survive, that is the answer and the
Step 1 winner is final.

Done when: a results table and the chosen stack are recorded under
`results/p1c/inner-cv-candidates/`.

## Step 3 — Repair the denoising reference (CPU, cached features)

**Problem:** `core/p2_downstream.py` picks the classifier's epoch on the clean validation
videos, so the clean reference (0.4513) is inflated and every "restoring clean frames"
loss is measured against it. Restormer changes almost nothing (51.4 dB) yet shows -0.037.

**Fix:** retrain the downstream head with a fixed training length (the one chosen in
Step 1), then re-score all 80 rows from the cached condition features in
`results/p2/p2-classical`, `p2b-learned`, `p2b-nafnet` and `p2b-extra`.

Keep the old table. Write the new one beside it as `results/p2/result1_denoise_v2.csv`
and record which statements changed. The salt-and-pepper gains should hold; the small
clean-frame losses may shrink or vanish.

Done when: v2 table exists, D-59 to D-62's clean-frame statements are confirmed or
corrected in `EVIDENCE_REGISTER.md`.

## Step 4 — Live detector on the official anomaly split (one GPU pass, then CPU)

**What it is:** a separate task and a separate table (rule 3). The model scores each
moment of a video for "is something abnormal happening now". This is the task the
dataset was built for, and it has real time labels on test, so it satisfies rule 1
directly.

**Data:** `Anomaly_Train.txt` (1,610 videos, video-level labels only) and
`Anomaly_Test.txt` (290 videos with official intervals). All 2,783,603 frames are
already in the blob at `results/p1c/p1c-video-frames/`.

**Before any run — D-71:** K-12 says the anomaly split refuses to build manifests because
the two Normal directories carry different names. Register the mapping: both Normal
directories are "normal", all 13 crime folders are "anomaly".

**4a. Features (GPU).** Existing caches cover the 700 Action-Recognition videos only.
Extract frozen features for every frame in the blob. Run a 5-minute throughput smoke
first, then state the ETA. My estimate for the full pass is 1–2 hours for one backbone;
this needs the heads-up.

**4b. Models (CPU), in this order:**

| Row | Model | Purpose |
|---|---|---|
| D0 | Video-level classifier score spread to every frame | floor: no localisation at all |
| D1 | Sultani-style ranking loss on per-frame features | the published baseline recipe |
| D2 | Top-k multiple-instance loss with a small MLP | the standard modern recipe |
| D3 | D2 plus a causal temporal layer that sees only the past | the live version |
| D4 | D3 with a two-sided temporal layer | measures what "live" costs |

Model choices are made on held-out training videos using video-level AUC only. The 290
test videos are read once, for these five declared rows.

**4c. Metrics:** frame-level AUC on all test frames (the standard number), AUC on anomaly
videos only, false alarms per hour on the 150 normal test videos at a fixed threshold,
and seconds from event start to first alarm. The last two are rarely reported and are
what "live" actually means.

**Reference points from the literature:** Sultani 2018 reports 75.4% with C3D features;
with frozen CLIP features published methods reach 84–88% (VadCLIP 88.02%). My
expectation for D2–D3 is the low-to-mid 80s. That is an inference.

**Not in scope:** feeding detector output back into the 14-class classifier. The
oracle-interval test already showed frames from inside the true event did not help
(0.341 against 0.390 accuracy on 41 videos).

**Optional row:** training with the GlanceVAD click annotations. If run, it is named
"glance-supervised" and never mixed with the weakly supervised rows.

Done when: the five-row table with all four metrics is in `results/detect/` and
`TRAINING_LOG.md`.

## Step 5 — Meaning-based family grouping (no compute)

Declare the mapping first, as D-72, before computing anything further:

| Family | Classes |
|---|---|
| violence | Abuse, Assault, Fighting, Shooting |
| theft | Burglary, Robbery, Shoplifting, Stealing |
| fire/blast | Arson, Explosion |
| vandalism, arrest, road accident, normal | one class each |

Report, as its own table: 7-family accuracy and macro-F1 from the final 14-class
predictions, plus top-3 accuracy on the 14 classes. This replaces the score-picked
7-class subset (D-56) as the reframing shown in the paper; D-56 stays on record.

## Step 6 — One shot at caption supervision (CPU; GPU only if 6a passes and needs it)

**Why:** the 0.80 caption result says the classes are learnable from a description of
what happens, with the same 38 videos per class. The visual features are the limit.

**Caution:** published work that added UCA-trained captions to detectors gained about
one AUC point. I put a clear win here at roughly one in four.

**6a. Gate (cheap).** Turn each UCA sentence into a 128-number vector (TF-IDF then
truncated SVD, no new model weights). Fit a linear map from the visual features of the
sentence's time span to that vector, using training videos only. On held-out videos,
check whether the right sentence is retrieved more often than chance. If it is not,
stop: the visual features do not hold the information.

**6b. The experiment (only if 6a passes).** Train the classifier with a second target:
besides the class, predict the caption vector. Training captions come from the fold's
training videos plus the 1,200 videos outside the 700-video protocol. Captions of
held-out videos are never used, and captions are never an input at prediction time.

Pass rule and evaluation exactly as Step 1. No second attempt if it fails.

## Step 7 — Final configuration, final test, stop

1. Final classifier = Step 1 winner plus Step 2 and Step 6 survivors, retrained on all
   532 training videos per fold with the fixed training length.
2. **One declared final test read (D-73)**, scored on the four official test folds. Both
   test numbers go in the paper: 0.4138 (first read, D-57) and the final one.
3. Detector table, denoising v2 table, family table, XD table: each stays separate.
4. **Stop.** After Step 7 there are no new backbones, heads, fusions or data mixes. The
   project is finished when Steps 0–7 are recorded; the next activity is writing.

## Amendments made while executing (2026-11-11)

- **Decisions recorded before results:** D-71 (detector protocol and label mapping), D-72 (family
  grouping), D-73 (final test read), D-74 (inner cross-validation protocol and pass rules), D-75
  (caption attempt). See `DECISIONS.md`.
- **Step 2 is run on the reference classifier.** Each candidate is the Step 1 reference arm (one
  vector per video, logistic regression) given a different feature matrix or training set, so each
  contrast changes one factor. Backbone contrasts use the 82,679 "study frames" that all three
  backbone caches share.
- **Step 4 sampling:** one frame per 15 source frames (0.5 s), close to the one-in-16 sampling of
  published CLIP-feature detectors. The causal row adds each frame's deviation from the running
  mean of past frames, following this project's own N13 finding.
- **Step 6 design:** a two-stage projection (visual span feature to caption vector, then
  classify) instead of a second training target. Same idea, no new model weights, no GPU.
- **Stacking rule, declared after Step 1 and before any stack was computed.** Step 1's winner
  and base is R2 (attention pooling, 8 epochs). A stack is the equal-weight average (weights fixed,
  never tuned) of the held-out class probabilities of R2 and one or more logistic arms. Order:
  first R2 + R0 (the plain mean-feature arm); then each Step 2 survivor, largest gain first, added
  to the previous stack. An addition is kept only if it passes the D-74 rule against the previous
  stack. Tool: `scripts/fuse_inner_cv.py`.
- **Stacking, second declaration (after the Step 2 results and the first two stack tests, before
  anything below was computed).** Step 2 passed three candidates: `L_study` (+0.034), `B+VJ`
  (+0.034), `B+S_study` (+0.031). Stacks S1 (R2 + R0: -0.002) and S2 (R2 + B+VJ: +0.013
  [+0.004, +0.023]) do not pass the +0.02 rule. Remaining tests, all against R2, all equal-weight
  probability averages: S3 = R2 + L_study; S4 = R2 + B+S_study; E3 = the three surviving logistic
  arms without R2; E4 = R2 + the three surviving arms. Final pipeline: E4 if it passes against R2
  and also passes against E3; otherwise E3 if it passes against R2; otherwise a single passing
  S-stack with the largest gain; otherwise R2 alone. No further combinations.
- **Stacking, third and last declaration (after Step 6 passed, before computing).** The caption arm
  `cap_proj` passed its own rule (+0.023 over its control). Two tests against the current final
  stack E3: F4 = E3 + `cap_proj` (four-member equal-weight average); C1 = `cap_proj` alone. Final
  pipeline: F4 if it passes against E3; otherwise C1 if it passes against E3; otherwise E3.
- **Pods run from a frozen copy of committed code** (`scripts/dgx/make_code_snapshot.sh`), so
  editing the branch cannot disturb a running pod (K-23).

## Decisions needed from the owner

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 1 | Local commits for this work | approve / keep uncommitted | Approve local commits on branch `review-plan-2026-11-11`. Pods mount the main checkout, so code has to reach it before a run. |
| 2 | Second test read in Step 7 | allow once, declared / report cross-validated score only | Allow once. Without it a changed final model has no test number. |
| 3 | Detector features | DINOv2-B only / DINOv2-B and SigLIP in one pass | Both in one pass. Decoding is paid once, and the published 84–88% results use CLIP-family features. Roughly doubles the GPU time. |
| 4 | GlanceVAD row in Step 4 | run / skip | Run only if Steps 0–6 are done. |

## What the paper can claim

Claims the numbers support today:

- 41.8% on the official 4-fold recognition split, against the dataset authors' 23.0% and
  28.4%, using frozen features and no extra annotation. Maqsood et al.'s 45% must be
  cited with its conditions (one split, added frame-level labels).
- Five different pretrained backbones land within a few points of each other, and a
  text classifier on human descriptions reaches 0.80. The limit is what current visual
  features encode about small, low-resolution actions, not the labels or the video count.
- Choosing the training epoch on 56 scored videos inflates the score by about 0.06.
- Image-quality scores do not predict recognition: the denoiser with the best PSNR is
  not the one that helps, and a switching median filter recovers 84–91% of the loss
  under impulse noise.

Claims that depend on this plan: the detector's AUC and live metrics (Step 4), any
classifier gain (Steps 2 and 6), the repaired clean-frame statement (Step 3).

Claims that must not appear: 0.46 as the headline; the score-picked 7 classes as the
main result; "beats all existing results"; XD-Violence "mAP 0.813" placed beside
published frame-level AP figures (ours is video-level); any table with failed arms
removed (rule 4).

## Suggested order for one working day

Step 0, then Step 1. Launch the Step 4a feature pass as soon as Step 1's pod is done,
since it is the only long GPU job. Step 5 is arithmetic on stored predictions and can be
done while it runs. Steps 3, 2 and 6a are CPU work but still need a pod (the head node is
off limits and only one pod runs at a time), so they queue behind the feature pass. Then
Step 4b–4c, then 6b if the gate passed, then Step 7.
