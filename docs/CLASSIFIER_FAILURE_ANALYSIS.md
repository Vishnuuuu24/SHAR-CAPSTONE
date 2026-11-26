# Classifier failure analysis (validation only)

> Written 2026-11-02. Subject: the D-53 headline classifier (frozen DINOv2 ViT-B/14-reg, 8 frames per video,
> attention-pooling head). Evidence: its stored validation probabilities (4 folds x 3 seeds x 56 videos,
> `results/p1c/p1c-late-fusion/dinov2_probs.csv`, sha256 in `results/p1c/classifier-failure-analysis/analysis.json`),
> the contact sheets and the 42 recorded remedies. **The test split (spent, D-57) was not used to choose anything here.**
> Reproduce: `./.venv/bin/python scripts/analyze_classifier_failures.py`.

## 1. Bottom line

The classifier tells normal from crime well (AUC 0.938) and tells **which** crime badly. It does well on crimes that
*look* different (fire, theft in shops, vandalism) and fails on crimes that are *actions* (the four violence classes).
The failures are consistent per video, not random, and most trace to the data: very few training videos per class,
long videos where the crime is a small part, compilation and title-card videos, and a crime that is simply not
visible at 320x240.

## 2. What it scores

| Measure (validation, 4 folds x 3 seeds) | Value |
|---|---|
| Macro-F1 / accuracy | **0.4626** / 47.5% |
| 95% interval (global source-video bootstrap within class, 2000 draws; corrected 2026-11-02, see EVIDENCE_REGISTER) | **0.367 - 0.511** (SD 0.037). The first version resampled inside each fold and gave 0.398-0.502, too narrow because 74 validation videos recur across folds |
| Per fold | 0.519 / 0.436 / 0.470 / 0.425 |
| Seed spread inside a fold | SD 0.009: seeds agree, so training noise is not the problem |
| Top-1 / top-2 / top-3 / top-5 accuracy | 47.5% / 61.3% / 68.6% / 81.7% (median rank of the true class: 2) |
| Normal vs crime (ranking, AUC) | 0.938 |
| Test, spent once (D-57), for reference | 0.4138 macro-F1 / 41.8% |

Only **108 unique validation videos** exist (4 per class per fold, overlapping across folds), so any single
number carries about +-0.07. Paired comparisons on the same videos are much tighter (difference SD 0.010-0.013 for
training-procedure changes, 0.024-0.031 for data changes). 107 of the 108 are test videos of another fold. The test figure sits 0.049 below validation; this is likely partly because the best epoch is
chosen on the validation set.

## 3. Where it fails

Per class, seed-averaged (about 8 unique validation videos each):

| Class | Family | Precision | Recall | F1 |
|---|---|---|---|---|
| Normal | normal | 0.81 | 0.83 | **0.81** |
| Explosion | fire | 0.83 | 0.54 | 0.64 |
| Arson | fire | 0.59 | 0.71 | 0.63 |
| Shoplifting | theft | 0.54 | 0.67 | 0.59 |
| Vandalism | damage | 0.56 | 0.65 | 0.59 |
| Stealing | theft | 0.76 | 0.44 | 0.53 |
| RoadAccidents | traffic | 0.51 | 0.54 | 0.51 |
| Arrest | police | 0.64 | 0.48 | 0.51 |
| Robbery | theft | 0.52 | 0.52 | 0.50 |
| Burglary | theft | 0.28 | 0.42 | 0.33 |
| Abuse | violence | 0.34 | 0.29 | 0.30 |
| Fighting | violence | 0.28 | 0.31 | 0.29 |
| Assault | violence | 0.21 | 0.23 | 0.21 |
| **Shooting** | violence | 0.03 | 0.02 | **0.02** |

The four violence classes are the four worst. Shooting is at zero.

Error structure (118 errors per seed-average of 224 video evaluations):

- **94.9% of errors are one crime called another crime.** Only 2.8% send a crime to Normal and 2.3% send Normal to a crime.
- Errors stay inside the same family only 22% of the time (17% if errors were spread evenly): the confusions are
  wide, not neat pairs. Most frequent pairs: RoadAccidents-Shooting, Assault-Burglary, Abuse-Assault, Abuse-Fighting.
- **Errors are consistent per video.** Of 108 videos, 35 are right in every evaluation, 41 are wrong in every
  evaluation, and only 32 flip. The 41 always-wrong videos produce **73% of all errors**. Shooting: 5 of 6 are
  always wrong; Stealing 5 of 8; Abuse 4 of 8; Assault 4 of 7.
- **It is not overconfident.** Calibration error is 0.086; predictions above 0.9 confidence are 94% right and hold
  under 1% of the errors. Keeping only the most confident 20% of videos gives 73% accuracy. (This is a property of the
  model, not a way to raise the headline: picking confident predictions after the fact is circular.)
- **Longer videos fail more**: error rate 47%, 50%, 52%, 61% across length quartiles (weak trend, Spearman 0.06, p = 0.11).

## 4. Why it fails (evidence)

1. **It reads the scene, not the action.** Classes with a distinctive look (fire, shop interior, a damaged object) score
   0.59-0.64; classes defined by what people do (hit, shoot, abuse) score 0.02-0.30. Removing the per-video offset
   improves localisation and destroys classification (N13 + N14), i.e. the scene signature *is* the signal. Adversarial
   removal of the source made it worse (N14, D-49).
2. **Too little data per class.** 34 training videos per class (476 per fold). The extra data that exists for the violence
   classes is from another domain and did not help (XD-Violence, N28; domain separable at ROC-AUC 0.98-1.00).
3. **The training procedure uses about 1% of the cached frames.** The head trains on one fixed 8-frame draw per video
   (3,808 of 391,014 cached frames); validation scores all frames. See section 6.
4. **The label is weak and the crime is often not visible.** The contact sheets show a Shooting video that is a
   compilation of unrelated scenes whose "event" is a police car on a street, and a Fighting video that opens and
   closes with title cards ("Teen girl using real karate...") with the fight in a short middle segment. At 320x240 no
   weapon is detectable (N16). Only 48 of 168 test videos in fold 1 have an official interval (D-40).
5. **Where the frames come from is not the bottleneck.** Giving the head frames from inside the true event did not help
   (oracle-interval probe, covered-video accuracy 0.390 uniform vs 0.341 event-only), and training on in-event frames did not help either (train-oracle macro-F1 0.4614 vs 0.4636; human-click frames N36: +0.016, 1/4 folds).

## 5. What has already been ruled out

Seeds and ensembling (N37), aggregation rules (N1, N2), temporal windows, deltas and dense clips (N3-N8), higher
resolution (N8), video backbones (N9, N10, N29, N30), optical flow (N26, N27), SigLIP and fusion (N31, N32), person
crops and weapon detection (N15, N16), loss functions (N21-N23), augmentation at pixel level (N25), extra data (N28,
N33), human click labels (N36), nearest-neighbour memory (N38), VLM zero-shot (N35), and all denoising (N39-N42).

## 6. The one lever not yet pulled

The registered head sees one fixed 8-frame bag per training video for the whole of training; the seeded sampler in
`core/p1c_video_sampling.py` was designed as a training-time augmentation ("different seeds see different frames") but
was only ever used once with seed 0. Redrawing the bag every epoch from the cached frames costs no new extraction.
The experiment is pre-registered in `core/head_experiment.py` (see D-63 in `docs/DECISIONS.md`).

### Result (D-63, validation, 4 folds x 3 seeds, +0.04-on-every-fold gate fixed in advance)

| Arm (one change from the control) | Macro-F1 | Per-fold change vs control | Gate |
|---|---|---|---|
| Control (D-53 head, reproduced) | 0.4636 | - | - |
| Fresh 8-frame bag every epoch | 0.4477 | -0.041 / +0.002 / -0.025 / 0.000 | fail |
| Same, 100-epoch ceiling, patience 10 | 0.4486 | -0.038 / +0.002 / -0.024 / 0.000 | fail |
| Label smoothing 0.1 | 0.4587 | -0.017 / +0.004 / -0.012 / +0.006 | fail |

The control matches the stored probabilities exactly in 10 of 12 fold-seed cells; the other two differ by +0.0125 and
-0.0006 (multithreaded floating point changing which epoch wins), and its mean equals the 0.4636 recorded for the same
runs in D-57. **Resampling did not help and slightly hurt.** Frames from one video are near-duplicates (cosine 0.948
between event and non-event frames), so redrawing them adds almost no information: the head still converges in about
7 epochs and the effective training-set size is the number of videos, not frames.

### The learning curve (the main finding)

Same head, trained on a fraction of each class's training videos, scored on the same validation videos:

| Training videos per class | 9 | 17 | 26 | 34 (all) |
|---|---|---|---|---|
| Macro-F1 | 0.296 | 0.357 | 0.431 | 0.464 |
| Per fold | .32/.32/.30/.25 | .42/.34/.36/.30 | .47/.43/.46/.36 | .52/.44/.47/.43 |

Every fold rises at every step: **+0.089 macro-F1 per doubling of training videos, with no sign of flattening** (a log-linear
extrapolation, to be treated as indicative only, gives about +0.05 at 51 and +0.09 at 68 videos per class). The classifier
is limited by the number of distinct labelled videos. More frames of the same videos do not help; more videos do.

### What this points to

Ranked by what the evidence supports:

1. **More distinct training videos.** UCF-Crime has 300 videos outside our 700-video protocol for four classes (Burglary
   +50, RoadAccidents +100, Robbery +100, Stealing +50) and none for the four violence classes. Adding them is a named protocol
   deviation and must be reported as such, not as the benchmark result.
2. **A representation that learns more from few examples** (a larger frozen backbone). Untested; the DINOv2 ViT-L/14-reg
   weights are downloadable.
3. For the violence classes, more UCF videos do not exist and the XD-Violence videos did not transfer (N28). These classes
   are probably capped by the 320x240 footage and by weak labels unless a different kind of evidence is brought in.
4. Do not expect gains from further changes to how the head is trained on the existing 476 videos per fold.

### Round 2 (D-64): the unused UCF videos

UCF-Crime has 300 videos outside the 700-video protocol, for four classes only (Burglary +50, RoadAccidents +100,
Robbery +100, Stealing +50). Added as training data (a named protocol deviation), validation only:

| Arm | Macro-F1 | Per-fold change vs control | Gate |
|---|---|---|---|
| Control | 0.4636 | - | - |
| +34 videos per class (those four) | 0.4617 | -0.000 / +0.028 / -0.030 / -0.006 | fail |
| All 300 | 0.4484 | -0.001 / +0.050 / -0.075 / -0.034 | fail |
| All 300, inverse-frequency class weights | 0.4927 | -0.003 / +0.080 / -0.006 / +0.046 | fail (2 of 4 folds) |

The one positive arm is not what it looks like. The four classes that received data gained only +0.022 F1; the
other ten gained +0.032 (Shooting 0.02 to 0.18, Arrest +0.12, Explosion +0.11, but Vandalism -0.19), which is a
class-prior effect of the weighting, and it comes from two folds. **It is not a validated improvement.**

### What limits further progress: the measurement

The validation set has 108 unique videos and a 95% interval of about +-0.07 on macro-F1 (corrected). Paired
intervals for the two rounds: resample -0.016 [-0.035, +0.003], label smoothing -0.005 [-0.029, +0.019], extra videos
with class weights +0.029 [-0.022, +0.080]; none is established in either direction. The learning curve
(+0.089 per doubling) shows more data would help, but the only safe way to find and certify smaller improvements is
a larger evaluation set.

### Round 3 (D-65 / D-66): a larger frozen backbone, under the repaired protocol

Epoch selection now uses 4 training videos per class (not the scored videos); the claim rule is the paired
global source-video bootstrap (95% interval above 0 and gain >= +0.02).

| Arm | Macro-F1 (108 validation videos) |
|---|---|
| D-53 head on ViT-B, epoch chosen on the inner monitor | 0.3835 |
| ViT-B + matched projection head (control) | 0.3900 |
| **ViT-L + matched projection head** | **0.4134** |

ViT-L minus ViT-B: **+0.023, 95% interval [-0.022, +0.075]**, per fold -0.015 / +0.022 / -0.013 / +0.099. Not
established. The violence classes moved from 0.175 to 0.210 mean F1 (Assault +0.10, Abuse +0.05, Shooting -0.04).

The same D-53 head scores 0.4636 when its epoch is picked on the validation videos and 0.3835 when picked on the
inner monitor with four fewer training videos per class (about -0.015 by the learning curve). Roughly 0.06 of the
earlier validation figures was therefore optimism from choosing the epoch on the scored videos. The test result
(0.414, D-57) is not affected: its test videos never chose an epoch.

