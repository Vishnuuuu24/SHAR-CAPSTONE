# Remedies tried against the motion-class failure

> Compiled 2026-09-25 from `TRAINING_LOG.md` rows and `DECISIONS.md` D-42 → D-51.
> This file is a **register, not a new source of numbers** — every figure here is
> traceable to the ledger row named in its Artifact column. Where a remedy was
> tried more than once (different radius, different rule), the row states the best
> variant and the spread.

## The failure being remedied

Best validated system: **DINOv2 ViT-B/14-reg + 8 frames/video + linear head + majority
vote**, video-level macro-F1 **0.4068 ± 0.0353** (4 AR folds × 3 seeds), +0.1056 over the
ResNet baseline on 4/4 folds. Per class it splits cleanly:

| Scene-defined (works) | | Motion-defined (fails) | |
|---|---|---|---|
| Normal | 0.739 | Vandalism | 0.397 |
| Shoplifting | 0.633 | Assault | 0.215 |
| Stealing | 0.554 | Fighting | 0.193 |
| Explosion | 0.538 | Shooting | 0.033 |
| Arson | 0.512 | | |
| Robbery | 0.473 | | |

Two measured mechanisms, not assumptions:

1. **No motion in the input.** A frozen image model maps a punch mid-swing and a person
   standing to nearly the same vector.
2. **Extreme event sparsity.** The event occupies a median 9.9 % of a Shooting video
   (1.0 % worst case); event and non-event frames of the *same* video sit at cosine
   **0.9477**, pooled cross-video separability ROC **0.381**.

Class imbalance is **not** a mechanism: under the selected 8-frames/video pipeline the
training set is exactly balanced at 272 rows per class (the 4.5× row imbalance exists only
in the all-frames configuration, which scores 0.3433 and is not the selected pipeline).

## What worked

| # | Remedy | Effect | Artifact |
|---|---|---|---|
| W1 | **Video-level aggregation** instead of frame-level scoring | **+0.0616 ± 0.0259, 8/8 cells** | `P1C-VIDEO-AGGREGATION-B6` |
| W2 | **Per-video frame sampling**, 8 frames/video (vs all frames) | **+0.0635, 4/4 folds**; monotone in budget (8 > 16 > 32 > all) | `P1C-SAMPLING-ABLATION` |
| W3 | **Backbone swap** ResNet50 → DINOv2 ViT-B/14-reg | **+0.0774 ± 0.0346 probe, 4/4 folds**; +0.0421 in the full pipeline | `P1C-PROBES-FOLDS-CONFIRM-AR` |
| W4 | **Freezing the backbone** instead of fine-tuning it | +0.0575 on fold 1, and it removed the epoch-0 overfitting collapse | `P1C-VIDEO-SCREEN-012` |

Note what these have in common: none of them added motion. They fixed *what the classifier
is scored on*, *which frames it sees*, and *how good the frame features are*. The motion
classes moved very little under all four — W2 in particular **regressed Shooting 0.279 → 0.033**
while raising macro-F1, because 8 evenly-spaced frames usually miss a 9.9 % event.

## What worked, added 2026-09-26

| # | Remedy | Effect | Artifact |
|---|---|---|---|
| W5 | **Learned attention pooling over a video's frames** instead of majority vote (STEP's architecture with frame order destroyed) | **+0.0570 +/- 0.0303 over a matched-capacity no-attention control, 4/4 folds**; 0.4599 +/- 0.0468 video-level macro-F1. Motion classes move for the first time: Vandalism +0.137, Fighting +0.129, Assault +0.085. Shooting goes to 0.000 | `P1C-P3A-HEAD-COMPARISON` (D-53) |

W5 is the same axis as W1 and W2 -- it changes how a video's frames are combined, not what sits on top of the features.

## What did not work -- 16 controlled negatives

### A. Aggregation and decision rules

| # | Remedy | Mechanism it tested | Result | Artifact |
|---|---|---|---|---|
| N1 | Majority vote vs mean probability | which pooling elects the video label | **tie**, paired +0.0035 ± 0.0083, 4/8 cells each | `P1C-VIDEO-AGGREGATION-B6` |
| N2 | Six aggregation rules (majority, mean, max, top-1/5/10 mean) | can a minority-frame rule elect a sparse event | **Shooting F1 0.033 under all six rules, identical to three decimals.** Top-10 nominally +0.0138 but 3/4 folds, variance up (0.0462 vs 0.0353), Vandalism 0.397 → 0.200 | `P1C-AGGREGATION-EXPERIMENT` |

### B. Temporal structure over frozen features

| # | Remedy | Mechanism it tested | Result | Artifact |
|---|---|---|---|---|
| N3 | Temporal windows (concat / mean / mean+std, radius 2 and 8) | does local temporal context help | best **+0.0037** (mean, r8, 2/4 folds); mean+std negative. Frame-level gain +0.0117 (4/4) does **not** survive aggregation | `P1C-TEMPORAL-PROBE-V2` |
| N4 | Feature delta (f_t − f_{t−1}) | explicit temporal difference in latent space | **−0.0146 ± 0.0031, 0/4 folds**; Shooting 0.118 → 0.075 | `P1C-EXP-A-TEMPDIFF` |
| N5 | Dense 16-frame clips, mean-pooled | denser temporal coverage of the event | per-video ROC +0.008 (p=0.84); pooled 0.419 → 0.413 | `P1C-EXP-A2-CLIPS` |
| N6 | Dense clips + delta | both together | **−0.084 (p=0.05)** | `P1C-EXP-A2-CLIPS` |
| N7 | Patch-token motion, mean \|P_t − P_{t−1}\| before pooling | motion measured before spatial pooling destroys it | per-video ROC **+0.071 (p=0.09)** — the only representation that moves within-video ranking — but **pooled unchanged** (0.433) | `P1C-EXP-A3-TOKENS` |
| N8 | Resolution 224 → 448 | is the action too small to resolve | **worst row of the matrix**: pooled 0.350 vs 0.433 | `P1C-EXP-A3-TOKENS` |

### C. Motion-pretrained backbone

| # | Remedy | Mechanism it tested | Result | Artifact |
|---|---|---|---|---|
| N9 | VideoMAE-B Kinetics-400 clip features, alone | does a real video model transfer to CCTV | **Fighting 0.000, Shooting 0.000**; overall 0.3459 ± 0.0674, −0.0476 vs DINOv2 | `P1C-EXP-P2-FUSION` |
| N10 | DINOv2 ++ VideoMAE fusion | appearance + motion concatenated | **+0.0069, 3/4 folds** (fold 4 −0.0517); motion-class mean 0.180 → 0.200 | `P1C-EXP-P2-FUSION` |

### D. Event sparsity and supervision

| # | Remedy | Mechanism it tested | Result | Artifact |
|---|---|---|---|---|
| N11 | Shooting exposure raised 1.50 → 6.71 event frames per annotated video | give the head more of the actual event | **+0.0004 ± 0.0274**; Shooting F1 0.118 → 0.152 but precision halved (0.250 → 0.103, predicts Shooting 5.8× more often) and **Arrest collapsed 0.457 → 0.239** | `P1C-SHOOTING-EXPOSURE` |
| N12 | Background-masked loss (zero-weight known-background rows of the target class) | remove the wrong-label signal | **+0.0009, 2/4 folds**; event/background P(Shooting) ratio 1.09× → 1.18× but pooled event ROC 0.574 → 0.526. Only 46 % annotation coverage (21 of 181 rows cleaned) | `P1C-EXP-B-BGMASK` |
| N13 | Per-video score normalisation (z-score, rank) | remove the per-video offset that hides the event | localisation improves a lot (patch motion pooled ROC 0.433 → 0.645 → **0.726**) but **classification is destroyed**: video-level Shooting-vs-not max-z **0.478** against max-raw **0.729** | `P1C-EXP-A4-NORMALIZATION` |

### E. Spurious correlation / domain

| # | Remedy | Mechanism it tested | Result | Artifact |
|---|---|---|---|---|
| N14 | Domain-adversarial fine-tuning (DANN, GRL, source-video as domain, blocks 10–11 unfrozen) | unlearn the per-video identity the model leans on | **classification 0.385 — below chance** (frozen 0.600, plain fine-tune 0.800), while localisation rose on all 5 held-out videos (0.668 pooled / 0.857 per-video) | `P1C-EXP-A5-DANN` |

### F. Input view

| # | Remedy | Mechanism it tested | Result | Artifact |
|---|---|---|---|---|
| N15 | Person-ROI crop (YOLO11m-pose union box, +10 % margin, full-frame fallback) | is the background diluting the person | **0.3975 vs 0.4068**, paired −0.0092 ± 0.0462, 1/4 folds positive. **Fighting flat 0.193 → 0.185; Assault worse 0.215 → 0.132.** Context losses: Robbery −0.156, Shoplifting −0.111, Normal −0.099. Fallback 40.3 % of rows. **Paired by video (672 points, 108 videos): ROI-only correct 44, full-frame-only 54, McNemar exact p=0.363** -- the crop changes the answer on 98 videos with no significant direction | `P1C-ROI-STUDY` |
| N16 | Open-vocabulary weapon detection (YOLO-World-v2-s, {gun, pistol, rifle}) | can the object be localised directly | **0 detections in 233 in-interval and 784 out-of-interval Shooting frames, even at conf 0.10**; 0 false positives on 500 controls | `P1C-ROI-STUDY` |

### G. Heads and capacity (P3A, 2026-09-26)

| # | Remedy | Mechanism it tested | Result | Artifact |
|---|---|---|---|---|
| N17 | MLP head instead of linear | head capacity | **-0.0018, 2/4 folds** -- extends D-43 | `P1C-P3A-HEAD-COMPARISON` |
| N18 | SEMSCNN-inspired 2-D multiscale-SE head on the 16x16 patch grid | spatial structure the pooled vector discards | **0.3252 +/- 0.0422, -0.0864, 0/4 folds** -- worst arm measured, and 10-20x slower per fit | `P1C-P3A-HEAD-COMPARISON` |
| N19 | Temporal order via STEP's positional encoding + attention | frame order as signal | **-0.0197 against the same architecture with order destroyed, 1/4 folds.** Order-sensitivity contributes nothing; the gain in W5 is attention alone | `P1C-P3A-HEAD-COMPARISON` |
| N20 | 7.1M-parameter video-level MLP (masked mean over frames) | capacity + video-level prediction without attention | **-0.0087, 2/4 folds** -- the control that proves W5 is attention, not parameters | `P1C-P3A-HEAD-COMPARISON` |

### H. Loss functions (P3B, 2026-09-25)

| # | Remedy | Mechanism it tested | Result | Artifact |
|---|---|---|---|---|
| N21 | Inverse-frequency weighted CE | class reweighting | **exactly identical to CE** (paired 0.0000, 4/4 folds) -- weights are all 1.0 because the selected pipeline's TRAIN is exactly balanced at 272 rows/class | `P1C-P3B-LOSS-ABLATION` |
| N22 | Balanced Softmax (log-prior logit adjustment) | class-prior correction | **exactly identical to CE** -- equal priors shift every logit by the same constant, and softmax is shift-invariant | `P1C-P3B-LOSS-ABLATION` |
| N23 | Focal loss (gamma=2) | per-example difficulty reweighting | **0.3860 +/- 0.0334 vs 0.4068, paired -0.0208 +/- 0.0176, 0/4 folds.** Tail mean 0.2095 -> 0.1903. Gains Shooting +0.018, Stealing +0.041; loses Shoplifting -0.069, Normal -0.059, Assault -0.050 | `P1C-P3B-LOSS-ABLATION` |

### I. Explicit motion input (2026-09-27)

| # | Remedy | Mechanism it tested | Result | Artifact |
|---|---|---|---|---|
| N26 | **RAFT-small optical flow**, 162-dim pooled descriptor fused with DINOv2, under majority vote | explicit pixel-displacement motion, immune to appearance | **0.3987, paired -0.0129 +/- 0.0241, 1/4 folds** | `P1C-FLOW-FUSION` |
| N27 | The same flow descriptor fused under D-53's attention pooling | flow where the aggregator can weight motion spikes | **0.4638, paired +0.0038 +/- 0.0408, 2/4 folds** -- inside noise. Tail mean 0.2696 -> 0.2874; Fighting +0.048 and Shooting +0.052 paid for by Explosion -0.111, Stealing -0.060 | `P1C-FLOW-FUSION` |

**Why, measured:** median global flow magnitude is 0.141 px over 0.167 s and **Fighting (0.123) is below Normal (0.140)** -- the motion defining the class is smaller than the background's own flow. This closes pooled flow descriptors *at this spacing*; frame-adjacent flow and a learned flow branch remain untried (D-54).

### J. Optimisation and regularisation (early screens, fold 1 only)

| # | Remedy | Result | Artifact |
|---|---|---|---|
| N24 | Learning rate 3e-4 → 3e-5 | frame macro-F1 0.3364 vs 0.3537; still peaks at epoch 1 — `INCONCLUSIVE` | `P1C-VIDEO-SCREEN-013` |
| N25 | Augmentation policy added | frame 0.3421, video 0.4033 — within fold-1 noise, `INCONCLUSIVE` | `P1C-VIDEO-SCREEN-014` |

### K. Extra training data from another dataset (2026-10-07)

| # | Remedy | Mechanism it tested | Result | Artifact |
|---|---|---|---|---|
| N28 | **XD-Violence training videos added for Shooting / Fighting / Abuse** (train only; k=34 or 102 extra per class, all 31 Abuse; validation stays UCF), `step_permuted`, seeded head init, 4 folds x 3 seeds, CPU | does more labelled data for the failing classes lift them | Macro-F1 0.4626 baseline vs **0.4599 (k34, 0/4 folds, deltas -0.0017/-0.0003/-0.0050/-0.0037)** vs **0.4551 (k102, 0/4 folds)**; gate +0.04 on 4/4 not met. **Shooting F1 0.024 -> 0.021 (k34) -> 0.000 (k102); Fighting 0.289 -> 0.295 -> 0.219; Abuse 0.298 -> 0.255 -> 0.243.** Non-target mean F1 flat (0.533 / 0.533 / 0.537). Near-duplicate check: no XD video above cosine 0.95 to a UCF validation video (max 0.537, centred features). 4 of 232 Shooting videos checked for decode only; features are CPU-extracted (max abs diff to GPU cache 0.004) | `p1c-xd-extra-probe`, `p1c-xd-extra-cache` |

**Diagnosed 2026-10-08** (`p1c-xd-domain-gap`, 4f7c8f8): XD vs UCF of the same class is separable at ROC-AUC 1.000 (Shooting), 1.000 (Abuse), 0.978 (Fighting), against 0.802 for UCF Shooting vs UCF Fighting. XD-Shooting centroid ranks 9 of 14 UCF classes (cosine 0.543 to UCF Shooting); XD Fighting and Abuse rank 1. 3-way Shooting/Fighting/Abuse accuracy (chance 0.333): XD->UCF 0.367 (Shooting recall 0.17, Abuse 0.04), UCF->UCF leave-one-fold-out 0.407, XD->XD 0.688. UCF test has 150 videos (SE about 0.04). So: a large domain gap, and these frozen features barely separate the three classes even within UCF.

**Reading:** 232 movie/web Shooting videos did not teach the head to recognise CCTV Shooting. Fits the earlier finding that the bottleneck is the frozen representation at 320x240, not the amount of labelled data; also a domain gap (movies and web clips vs CCTV). Limits: plain CE with train no longer balanced (k102), no domain-adaptation, XD frames were not downscaled to 320x240 first.

### L. Partial fine-tune of a motion-pretrained backbone (2026-10-13)

| # | Remedy | Hypothesis | Result | Artifacts |
|---|---|---|---|---|
| N29 | **VideoMAE-B, top k blocks unfrozen** (k=0 head-only control, k=2, k=4), hidden state cached after block 8, 16-frame clips x 8 per video, 8 fixed epochs, final-epoch metric (no epoch selection), 4 folds x 3 seeds, 56 validation videos per fold, test untouched | does adapting the representation, not the data or the head, lift the starved classes | Macro-F1 **k0 0.3546 / k2 0.3291 / k4 0.3615**. Paired vs k0: **k2 -0.0255, 0/4 folds; k4 +0.0070 +/- 0.0258, 1/4 folds (fold 3 only, +0.0419)**; gate +0.04 on 4/4 not met. k4 training loss falls to 0.03 on 34 videos/class (memorises). Per class (k0 -> k4 F1): **Shooting 0.094 -> 0.095, Fighting 0.000 -> 0.082, Assault 0.061 -> 0.019**. All three arms sit 0.098-0.131 below the registered 0.4599 (cross-protocol: DINOv2 STEP head vs VideoMAE clips; compared, not ranked, and the evaluation split of 0.4599 was not re-verified here). Oracle-epoch F1 (0.39) is reported in the summary and is not a headline | `p1c-videomae-finetune`, `p1c-videomae-block8-cache` (digest 21a9a855a74f7945) |
| N30 | **Late fusion: DINOv2 `step_permuted` head + fine-tuned VideoMAE-B (k4, k0 control)**, equal-weight probability average (fixed in advance, no tuning), 4 folds x 3 seeds, 56 validation videos per fold, CPU | do the motion and appearance models make different mistakes | Macro-F1 A 0.4626 (reproduces the registered probe exactly) vs **A+k4 0.4324 (-0.0302, 1/4 folds)** vs A+k0 0.4306 (-0.0320, 0/4 folds); gate +0.04 on 4/4 not met. F1 A -> A+k4: **Shooting 0.024 -> 0.021, Fighting 0.289 -> 0.165, Assault 0.213 -> 0.072**. The weaker model dilutes the stronger one | `p1c-late-fusion` |
| N31 | **Frozen SigLIP-base (google/siglip-base-patch16-224, vision tower) replacing DINOv2** through the identical pipeline: 8 frames/video, `step_permuted` head, seeded init, video-level aggregation, 4 folds x 3 seeds, 56 validation videos per fold, CPU head | does a language-aligned backbone separate the starved classes | Macro-F1 **0.4524** vs DINOv2 0.4626 (paired **-0.0102, 1/4 folds**: -0.006 / +0.023 / -0.023 / -0.036); gate +0.04 on 4/4 not met. F1 DINOv2 -> SigLIP: **Shooting 0.024 -> 0.013, Fighting 0.289 -> 0.150, Assault 0.213 -> 0.128**, but **Abuse 0.298 -> 0.538, Robbery 0.503 -> 0.641, Stealing 0.531 -> 0.610**; non-target mean 0.533 -> 0.512. Not apples-to-apples: mean/std 0.5 normalisation, 14x14 patch grid, bilinear resize, learned attention-pool in place of a CLS token. The two backbones fail on different classes | `p1c-siglip-probe`, `p1c-siglip-patch-cache` (features sha256 6261a231...) |
| N32 | **Late fusion: DINOv2 head (A) + frozen SigLIP-base head (S)**, equal-weight probability average (0.5, fixed in advance, no tuning), 4 folds x 3 seeds, 56 validation videos/fold. Macro-F1: A 0.4626, S 0.4524, A+S **0.4680**. Paired A+S minus A: +0.0054 mean, per-fold +0.004/+0.039/+0.011/-0.033 (3/4 folds positive, +0.04 gate on 4/4 NOT met). A+S minus S: +0.0156, 4/4 positive, gate not met. Per-class A+S vs A: Abuse +0.19, Explosion +0.08, Robbery +0.06; Assault -0.08, Burglary -0.05, Arrest -0.05, Fighting -0.04, Shooting 0.024 -> 0.000. Complementarity is weaker than hoped: both right 35.9%, only A 11.6%, only S 11.3%, neither 41.2% (phi 0.54). Sanity: S re-run reproduces the N31 probe exactly (0.4524, per-fold diff 0.0). **Negative/neutral: gain is inside fold noise and the anomaly-vs-anomaly classes (Shooting/Fighting/Assault) do not improve.** Validation numbers only; test untouched. Artifacts: `results/p1c/p1c-late-fusion-siglip/`. |
| N33 | **SHAR-Combined: `ucf_only` vs `all_sources` (UCF+XD+CUVA pooled), frozen DINOv2 + step_permuted head**, 14-class, class-weighted CE, 4 folds x 3 seeds, scored on the UCF validation part. UCF-part macro-F1: ucf_only **0.4112**, all_sources **0.4007** (-0.0105). Paired per-fold delta -0.021/+0.031/-0.036/-0.016: 0/4 folds pass the +0.04 gate. Per-source for all_sources: XD 0.528, CUVA 0.603 (ucf_only on those: 0.064, 0.270 - in-domain data trivially helps its own domain, not evidence of transfer). New benchmark (official AD splits), NOT comparable with the 0.4626 headline. **Negative: adding XD and CUVA does not improve UCF; it slightly hurts.** Validation only; sealed test untouched. Artifacts: `results/p1c/combined-probe/`, `results/p1c/combined-patch-cache/`. |
| N34 | **Confusion-guided divide-and-conquer, E0-E2 only (oracle ceilings on the stored DINOv2 probabilities).** 94.9% of errors are anomaly->anomaly; recurring pairs are fold-inconsistent; leave-one-fold-out groups differ on every fold. Perfect-specialist hard-route ceilings: violent_trio +0.026 (0/4 folds), fight_assault +0.005, property +0.037 (1/4), LOFO size 2/3 +0.013/+0.014 - all rejected before training; violent_quad +0.069 and LOFO size 4 +0.056 (4/4) survive, but only with a near-perfect specialist. **E3-E5 (training specialists) is code-complete but not run** (cluster workers NotReady). Not a result about specialists. Report: `docs/DIVIDE_AND_CONQUER.md`. |
| N35 | **Zero-shot video-language model: Qwen2.5-VL-7B-Instruct (8-bit, rev cc594898, SHA-256 verified)**, 8 native-resolution frames/video, 14-option multiple choice, class probabilities from the answer-letter logits averaged over forward/reversed option order; alone and fused with DINOv2 at a fixed 0.5 weight. 108 unique validation videos, 4 folds. **Alone 0.2638 macro-F1 (-0.199 vs 0.4626, 0/4 folds); fused 0.4101 (-0.0525; -0.045/-0.066/-0.041/-0.059, 0/4).** The model answers with a letter (letter mass 0.998) but defaults to 'Robbery' for 47/108 videos; Abuse 0/8, Assault 0/7, Fighting 0/8, Shooting 0/6, Vandalism 0/8 correct. Outside knowledge does not resolve the crime-vs-crime confusion on this CCTV. **Negative.** Validation only. `results/vlm/qwen25vl7b-zeroshot/` |
| N36 | **Training frames from GlanceVAD human clicks (+/-2 s windows, per-video fps) plus official intervals, covering 442/442 anomaly training videos per fold** (vs ~17% in the earlier in-event-frame test); 8 frames drawn inside the windows; matched evenly-spaced control on the same fold-cache features; validation unchanged. **0.4795 vs control 0.4636 (+0.0159); per fold +0.005/+0.070/+0.001/-0.012: 1/4 folds pass +0.04.** Control reproduces the earlier run exactly. Choosing in-event training frames for every video does not reliably help; the fold-2 jump is not repeated on any other fold. **Negative (inside the noise).** Closes the weak-label/frame-selection question. `results/p1c/p1c-oracle-interval-probe/train-glance/`, labels `results/p1c/glance-annotations/` |
| N37 | **Seed ensembling: average the 3 seeds' probabilities per fold** (stored D-53 probabilities, no retraining). Per fold vs single-seed mean: -0.017/+0.003/+0.004/-0.001, 0/4 folds. **Negative**: the seeds already agree; averaging does not lift the final number. `results/p1c/p1c-late-fusion/dinov2_probs.csv` |
| N38 | **Tip-Adapter-style memory classifier**: one embedding per video (mean of L2-normalised fold-cache DINOv2 frame features), kernel vote exp(-beta(1-cos)) over the fold's 476 training videos, fused with the stored D-53 head probabilities. Pre-registered beta=5, w=0.5: **0.4501 (-0.0125; -0.001/-0.025/-0.006/-0.018, 0/4)**. Memory vote alone 0.2374 (-0.225; Normal F1 collapses). Leave-one-fold-out tuned (beta, w): 0.4603 (-0.0023, 0/4; it chose the smallest weights, i.e. 'use the head'). **Negative**: nearest-video similarity in the frozen feature space carries less class information than the trained attention head. `results/p1c/cache-classifier/` |
| N39 | **Classical denoising of clean frames before the classifier** (RESULT 1 identity-degradation rows, D-59): median 3x3, Gaussian blur 5x5, bilateral 9/75/75, NLM h10 on native 320x240 frames, D-53 head retrained per fold x seed, 108 validation videos x 32 frames. **Median -0.055 (0/4 folds), blur -0.021 (1/4), bilateral -0.139 (0/4), NLM -0.160 (0/4)** vs unfiltered 0.4513. Smoothing removes texture the frozen features use; UCF's native compression noise is not what limits classification. **Negative.** `results/p2/p2-classical/` |
| N40 | **Learned denoising of clean frames before the classifier** (D-60 identity rows): conv denoising autoencoder and VAE trained on UCF frames, 3 seeds each. **Autoencoder -0.073 (0/4 folds) at 38.3 dB / SSIM 0.969; VAE -0.119 (0/4) at 32.7 dB.** Near-invisible pixel changes still cost recognition, more than a 5x5 Gaussian blur (-0.021). **Negative.** `results/p2/p2b-learned/` |
| N41 | **Pretrained modern restoration of clean frames before the classifier** (D-61): NAFNet-SIDD-width64 as published. **-0.116 (0/4 folds) at 29.7 dB.** **Negative.** `results/p2/p2b-nafnet/` |
| N42 | **D-62 restorers on clean frames before the classifier**: adaptive median **-0.014 (2/4)**, Restormer blind Gaussian **-0.037 (0/4)**, SCUNet real-PSNR **-0.079 (0/4)**. The switching median is the least harmful restoration tried (it leaves most pixels untouched), but none helps clean UCF frames. **Negative.** `results/p2/p2b-extra/` |
| N43 | **Per-epoch frame resampling for the attention head** (a fresh seeded 8-frame bag per training video every epoch from all cached frames, instead of one fixed draw; D-63). **0.4477 vs control 0.4636 (-0.016; -0.041/+0.002/-0.025/0.000, 0/4 pass); with a 100-epoch ceiling 0.4486.** Frames within a video are near-duplicates (cosine 0.948), so extra frames add no information. **Negative.** `results/p1c/head-improvement/` |
| N44 | **Label smoothing 0.1 on the attention head** (D-63). **0.4587 (-0.005; -0.017/+0.004/-0.012/+0.006, 0/4 pass).** **Negative (inside the noise).** `results/p1c/head-improvement/` |
| N45 | **Extra UCF training videos for Burglary / RoadAccidents / Robbery / Stealing** (300 videos outside the 700-video protocol; D-64). +34 per class **-0.002**; all 300 **-0.015**; all 300 + inverse-frequency weights **+0.029 (-0.003/+0.080/-0.006/+0.046, 2/4 folds)**, with the gain mostly in the ten classes that got no data (class-prior effect). **Negative / inconclusive.** `results/p1c/head-extra-videos/` |
| N46 | **Frozen DINOv2 ViT-L/14-reg4 instead of ViT-B** (D-65/D-66; matched projection head, inner-monitor epoch selection). **+0.023 macro-F1, paired 95% CI [-0.022, +0.075], P(gain) 0.85; per fold -0.015/+0.022/-0.013/+0.099.** Violence mean F1 0.175 -> 0.210. **Not established (positive direction, inside the noise).** `results/p1c/backbone-l-vs-b/` |
| N47 | **Partial fine-tuning of DINOv2-B: last 2 or last 4 blocks + final norm** (D-67, D-65 protocol; backbone LR chosen on the inner monitor). **B_ft2 -0.002 [-0.039, +0.033]; B_ft4 +0.012 [-0.019, +0.044]; Holm over 2: neither significant.** Per fold ft4 +0.003/-0.020/-0.014/+0.079. Violence mean F1 0.184 -> 0.174 / 0.164. LR-0 pipeline check reproduced the frozen control exactly. **Not established (negative); classifier-improvement track closed.** `results/p1c/finetune-b-last2-last4/` |
| N48 | **Denoising or a larger detector input before pose** (pose-clarity check, 2026-11-06; 360 frames, 534 cached people, YOLO11m-pose settings of the cache; Restormer-blind and SCUNet-real at 640, original frames at imgsz 960 and 1280). Share of people with shoulder+hip+knee visible at confidence >= 0.5 vs the re-run original 0.644: **orig 960 -0.075 [-0.103, -0.050]; orig 1280 -0.064 [-0.095, -0.034]; Restormer -0.114 [-0.147, -0.083]; SCUNet -0.073 [-0.103, -0.046]** (paired, videos resampled). Worst on people under 40 px: 0.471 -> Restormer 0.115, SCUNet 0.310. Larger input found 9 / 18 extra people on the 60 empty frames (6 / 12 with a usable body), not inspected. Detector-confidence measure, NOT pose accuracy; reference people are those the original frame detected, which biases against every alternative (direction known, size not). Human check partial (one sheet viewed). | **NEGATIVE: nothing helped, all four lower.** `results/p1c/pose-clarity-20261105/`, `results/p1c/pod-logs/posenb.log` |
| N49 | **Frozen V-JEPA 2 ViT-L clip features** (Meta original checkpoint sha256 5346856e, `encoder` branch, 16 frames at 224, mean over tokens; same 16-row clips, folds, 3 seeds and head as N9; 8 clips/video for training). Cross-fold video macro-F1: **DINOv2 (standardised clip mean) 0.3794 +/- 0.0323; VideoMAE 0.3459 +/- 0.0674; V-JEPA 2 0.3763 +/- 0.0099; DINOv2++V-JEPA 2 0.4086 +/- 0.0558.** Paired: V-JEPA 2 - DINOv2 **-0.0031** (folds -0.036/+0.039/+0.024/-0.039); V-JEPA 2 - VideoMAE +0.030 (folds -0.070/+0.060/+0.058/+0.074); fusion - DINOv2 +0.029 (folds +0.025/+0.074/+0.092/-0.074). Per class (mean over folds and seeds, about 4 validation videos per class per fold, so noisy): V-JEPA 2 vs DINOv2 Fighting 0.196 vs 0.118, RoadAccidents 0.501 vs 0.336, Explosion 0.543 vs 0.481, but Assault 0.028 vs 0.175, Shooting 0.019 vs 0.123, Arrest 0.214 vs 0.501. Inherited-label validation, Test untouched. The VideoMAE and DINOv2 arms reproduce the earlier N9 and P2-FUSION figures exactly (0.3459, 0.3794). | **NEUTRAL: no arm met the pre-declared bar (+0.04 on 4/4 folds).** V-JEPA 2 alone is level with DINOv2 and far steadier across folds; fusion's gain is inside the fold spread and negative on fold 4. Shuffled-frame arm not run. `results/p1c/p1c-exp-vjepa2-probe/vjepa2-probe-results.json`, `results/p1c/pod-logs/vjepa2-natural.log` |

Read: negative. Unfreezing blocks of a motion-pretrained model does not move Shooting or Assault, and the one fold that clears +0.04 is a single fold of an arm whose mean gain is inside the fold-to-fold spread. Together with N28, extra data and a changed representation have both failed to lift the starved classes at 34 videos per class.

## Diagnostics (not remedies, but they constrain what can work)

| Diagnostic | Finding | Artifact |
|---|---|---|
| Learning curve (D-63), validation | Macro-F1 0.2957 / 0.3565 / 0.4308 / 0.4636 at 9 / 17 / 26 / 34 training videos per class; every fold rises at every step; +0.089 per doubling, not flattening. **The head is limited by distinct training videos, not frames, loss or seeds.** | `results/p1c/head-improvement/` |
| Failure analysis (D-63), validation | Normal-vs-crime AUC 0.938; 94.9% of errors crime->other crime; violence family F1 0.02-0.30 vs 0.59-0.64 for look-distinct classes; 41/108 videos always wrong = 73% of errors; ECE 0.086; macro-F1 95% interval [0.398, 0.502]. See `docs/CLASSIFIER_FAILURE_ANALYSIS.md`. | `results/p1c/classifier-failure-analysis/` |
| Event separability | per-video ROC 0.575 ± 0.205 (vs shuffled control +0.127, p=0.006); **pooled cross-video ROC 0.381** — within a video the event is weakly rankable, across videos it is not | `P1C-EVENT-SEPARABILITY` |
| Cosine similarity, event vs non-event, same video | **0.9477** | `P1C-EVENT-SEPARABILITY` |
| Camera drift as class signal | N13 + N14 together: removing the per-video offset improves localisation and destroys classification, so the offset **is** the class signal the model uses | A4, A5 |
| Fold difficulty | fold 1 is an easy split (ResNet 0.3336 there vs 0.1579–0.2001 elsewhere); fold spread 0.1757 is 2.3× the backbone effect | `P1C-PROBES-FOLDS-CONFIRM-AR` |
| Detector viability | people **are** detectable at 320×240 (Fighting 100 %, Assault 93 % of sampled frames; median box height 55–135 px) — N15 is not a detector failure | `p1c-roi-detector-check` |
| Error breakdown, headline head (`step_permuted`, 4 folds x 3 seeds, 16 val videos/class) | Only **3 of 114.7** anomaly-class errors (3%) go to Normal; the rest land on other anomaly classes, spread widely. Errors that stay inside the aggression group (Abuse/Assault/Fighting) are 27% (chance 15%), inside the theft group (Burglary/Robbery/Shoplifting/Stealing) 32% (chance 23%). Per-class recall: Shooting 0.00, Assault 0.25, Fighting 0.27, Abuse 0.31. **Consequence:** the failure is *which anomaly*, not *whether an anomaly*; a binary anomaly-localisation (MIL) stage would not fix it directly, and a group-specialist head can address at most about a third of group errors. Caveat: 16 videos/class, so per-class recall is wide (n=16). | `results/p1c/p1c-error-breakdown/breakdown.json`, `scripts/export_p1c_error_breakdown.py` |
| UCA (UCF-Crime Annotation, Apache-2.0) as a source of anomaly intervals | UCA matches 696 of our 700 videos, but its captions describe the whole video: they cover a median **0.80** of each video's duration, 7 segments per video, of which a median 2 overlap the official event. Against the 121 videos that have both an official interval and a UCA entry, class-keyword segment selection recovers the official event with median IoU **0.22** (22/121 videos with no keyword hit); even taking every overlapping segment, IoU is 0.44 (union) to 0.53 (hull). **UCA does not give anomaly intervals** for the ~580 videos without official ones. | `results/p1c/p1c-uca-interval-recovery/`, `scripts/diagnose_uca_interval_recovery.py` |
| Oracle-interval upper bound, inference-time (head trained on uniform frames, scored on the 41 interval-covered validation videos, 4 folds x 3 seeds) | Accuracy: uniform8 0.390, **oracle_event 0.341**, anti_event 0.317 (uniform with all cached frames 0.398). McNemar uniform vs oracle_event 6 vs 4 discordant, p = 0.75. Giving the head frames from inside the true event did **not** help. Shooting 0/9 under every arm. Low power: 41 videos, dominated by Shooting/Burglary/Explosion/Shoplifting. | `results/p1c/p1c-oracle-interval-probe/` |
| Oracle-interval, train-time (the 74-94 covered train videos per fold draw their 8 frames from inside the event) | Overall 14-class macro-F1 0.4636 +/- 0.0409 (control, fold-cache features) vs **0.4614 +/- 0.0468** (train_oracle). Covered-subset accuracy under the oracle_event arm: control 0.341 vs train_oracle 0.333; McNemar 1 vs 0, p = 1.0. Training on in-event frames neither helped nor hurt, and did not make event frames more useful at test time. Only ~17% of train videos have intervals, so the intervention is partial. | `results/p1c/p1c-oracle-interval-probe/train-oracle/`, `scripts/run_p1c_oracle_interval_probe.py` |

## What this rules out, and what is left

**Ruled out by measurement:** better aggregation rules (N1, N2); temporal treatments over
frozen appearance features (N3–N8); Kinetics motion transfer (N9, N10); exposure and loss
masking (N11, N12); per-video normalisation as a classification fix (N13); adversarial
de-biasing (N14); person cropping (N15); direct weapon localisation (N16).

**The pattern:** every remedy that changed *what sits on top of* frozen appearance features
failed. W1, W2 and W5 all changed how a video's frames are selected, weighted or combined;
W3 changed the features themselves -- but to other *appearance* features. W5 (2026-09-26)
is the first evidence that part of the motion-class failure is an **aggregation** failure
rather than purely a representation failure: an unweighted majority vote cannot elect a
minority event, and a learned weighting recovers Vandalism, Fighting and Assault.

**Not yet tried** (stated so the gap is on record, not because it is scheduled):

- ~~**Explicit optical flow** fused with appearance features.~~ **Closed negative 2026-09-27**
  (N26/N27, D-54): -0.0129 under majority vote, +0.0038 under attention pooling. What remains
  untested is *frame-adjacent* flow (ours is 0.167 s-spaced) and a *learned* flow branch rather
  than a pooled descriptor -- but the magnitude diagnostic argues resolution is the binding
  constraint, not descriptor design.
- **Multiple-instance learning** (RTFM/MGFN-style top-K magnitude) instead of majority vote.
  **Partly answered 2026-09-26:** W5 shows learned attention pooling over frames beats majority
  vote by +0.0570 on 4/4 folds, which is the same family. A magnitude-ranked top-K MIL head is the
  natural follow-on and is still untried.
- **Latent temporal structuring** (LiFT, SDM) over the frozen features.
- **Plain supervised fine-tuning of a small video network** (X3D/MViT). N14 fine-tuned
  adversarially, which is not the same experiment; "fine-tuning" was closed on that basis
  and that closure is weaker than it should have been.
- **Skeleton / ST-GCN.** Per-detection keypoints for all 82,679 study frames are already
  saved in `results/p1c/p1c-roi-cache/detections.jsonl`; published evidence at this
  resolution argues against it.
