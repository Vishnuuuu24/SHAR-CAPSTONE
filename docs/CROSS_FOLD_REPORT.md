# Cross-fold report — P1C classification

> Live results table. Every cell is a measured number or explicitly `pending`/`N/A`.
> **Never fill a missing experiment with an estimate.** Numbers are quoted from
> `TRAINING_LOG.md`'s experiment rows and the artifact directories named at the bottom.
> Updated 2026-09-15.

## Metric discipline

Frame-level and video-level macro-F1 are **different metrics and are never compared across
rows**. Each table below reports one metric throughout. The project's prediction target is
the video, so video-level is the metric a final system should be judged on; frame-level is
retained because every earlier result is stated in it.

Scope for every number: official Action-Recognition folds, **validation splits only**,
inherited video-level labels (D-40, never event-only), **Test never touched**, 3 seeds
(0/1/2) per cell unless stated.

## Frame-level macro-F1

| Method | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± Std |
|---|---:|---:|---:|---:|---:|
| ResNet frame (trained head) | 0.3482 | 0.2379 | 0.1933 | 0.1798 | **0.2398 ± 0.0764** |
| DINOv2 frame (trained head) | 0.3352 | 0.2907 | 0.2396 | 0.2600 | **0.2814 ± 0.0416** |
| Best backbone + majority vote | pending | pending | pending | pending | pending |
| Best backbone + mean probability | pending | pending | pending | pending | pending |
| Best backbone + 8 frames/video | pending | pending | pending | pending | pending |
| Best backbone + 16 frames/video | pending | pending | pending | pending | pending |
| Best backbone + 32 frames/video | pending | pending | pending | pending | pending |
| Best backbone + temporal mean | pending | pending | pending | pending | pending |
| Best backbone + temporal concat | pending | pending | pending | pending | pending |
| Best backbone + temporal mean+std | pending | pending | pending | pending | pending |
| Best temporal neural model | pending | pending | pending | pending | pending |

## Video-level macro-F1 (majority vote) — the decision metric

| Method | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± Std |
|---|---:|---:|---:|---:|---:|
| ResNet, all frames (baseline) | 0.3788 | 0.3297 | 0.2328 | 0.2635 | 0.3012 ± 0.0657 |
| DINOv2, all frames | 0.3608 | 0.3709 | 0.3147 | 0.3269 | 0.3433 ± 0.0268 |
| DINOv2 + mean probability | 0.3624 | 0.3796 | 0.3136 | 0.3155 | 0.3428 ± 0.0333 |
| **DINOv2 + 8 frames/video** | **0.4584** | **0.3991** | **0.3901** | **0.3795** | **0.4068 ± 0.0353** |
| DINOv2 + 16 frames/video | 0.4373 | 0.3594 | 0.3924 | 0.3920 | 0.3953 ± 0.0320 |
| DINOv2 + 32 frames/video | 0.4282 | 0.3166 | 0.3797 | 0.4056 | 0.3825 ± 0.0482 |
| DINOv2 + 16/video + temporal mean r2 | 0.4470 | 0.3559 | 0.3829 | 0.4189 | 0.4012 ± 0.0400 |
| DINOv2 + 16/video + temporal mean r8 | 0.4569 | 0.3442 | 0.3988 | 0.4131 | 0.4032 ± 0.0465 |
| DINOv2 + 16/video + temporal mean_std r2 | 0.4197 | 0.3514 | 0.3939 | 0.4015 | 0.3916 ± 0.0289 |
| DINOv2 + 16/video + temporal mean_std r8 | 0.4274 | 0.3547 | 0.3664 | 0.4141 | 0.3907 ± 0.0355 |
| Temporal neural model (GRU/Transformer) | N/A | N/A | N/A | N/A | **not built — probe negative** |


## Aggregation rules — controlled experiment (2026-09-11)

Same trained model, same predictions, same folds and seeds; only the video-level
aggregation rule changes. `P1C-AGGREGATION-EXPERIMENT`.

| rule | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± Std | vs MV | folds won |
|---|---:|---:|---:|---:|---:|---:|---:|
| **majority vote (kept)** | 0.4584 | 0.3991 | 0.3901 | 0.3795 | **0.4068 ± 0.0353** | — | — |
| max prob (k=1) | 0.4715 | 0.4198 | 0.3661 | 0.4075 | 0.4162 ± 0.0434 | +0.0095 | 3/4 |
| top-3 mean | 0.4866 | 0.4187 | 0.3659 | 0.4006 | 0.4179 ± 0.0507 | +0.0112 | 3/4 |
| top-5 mean | 0.4702 | 0.4311 | 0.3705 | 0.3945 | 0.4166 ± 0.0436 | +0.0098 | 3/4 |
| top-10 mean | 0.4754 | 0.4423 | 0.3807 | 0.3840 | 0.4206 ± 0.0462 | +0.0138 | 3/4 |
| mean prob (all frames) | 0.4374 | 0.4214 | 0.3929 | 0.3893 | 0.4103 ± 0.0231 | +0.0035 | 3/4 |

**Shooting F1 is 0.033 under every rule, identical to three decimals.** Still 1 of 48
correct; the model predicts Shooting only 11-16 times whichever rule is used.

**Majority vote is kept.** Top-10 mean is nominally highest but does not earn the change:
+0.0138 is 3x below the registered 0.04 threshold, it loses fold 3, its cross-fold variance
is *higher* (0.0462 vs 0.0353), and it costs Vandalism 0.397 -> 0.200 (correct videos
18/48 -> 9/48, leaking to Arson and Stealing — Vandalism's evidence is spread across many
frames, which top-k discards).

Per-class F1 by rule and the top-10 confusion matrix:
`results/p1c/p1c-aggregation-experiment/`.

## The four effects, separated

Each measured against its own controlled baseline; never mixed.

| | effect | cross-fold change | consistency | clears 0.04? |
|---|---|---|---|---|
| **A. Per-video sampling (8/video)** | **+0.0635** | 0.3433 → 0.4068 | **4/4 folds** | **YES** |
| **B. Video-level aggregation** | **+0.0616** | frame → majority vote | **8/8 cells** | **YES** |
| **D. Backbone (DINOv2 vs ResNet)** | +0.0421 | 0.3012 → 0.3433 | 3/4 folds | marginal |
| **C. Temporal representation** | **+0.0037** | 0.3996 → 0.4032 | 2/4 folds | **NO** |

A and B each exceed the backbone effect, and neither needed a new model or a GPU.
C is an order of magnitude smaller than any of them.

## Current best validated pipeline

```
DINOv2 ViT-B/14-reg (frozen)  ->  8 frames/video, temporally spread
  ->  linear head (CONFIRM-015 recipe)  ->  per-video majority vote
```

**0.4068 ± 0.0353** video-level macro-F1, cross-fold, 3 seeds.
Against the ResNet all-frames baseline's **0.3012 ± 0.0657**: **+0.1056, positive on 4/4
folds** (+0.0796 / +0.0695 / +0.1573 / +0.1161). Cross-fold sd nearly halves.
Chance is 0.0714.

It also trains on **3,808 rows instead of 391,014 — 1.0% of the data, 103x less.**

## Per-class F1, video level, cross-fold mean (baseline -> best)

| class | baseline | best | diff | precision | recall |
|---|---:|---:|---:|---:|---:|
| Shoplifting | 0.337 | 0.633 | **+0.296** | 0.540 | 0.792 |
| Vandalism ★ | 0.110 | 0.397 | **+0.287** | 0.522 | 0.375 |
| RoadAccidents | 0.199 | 0.400 | +0.201 | 0.405 | 0.417 |
| Stealing | 0.393 | 0.554 | +0.161 | 0.611 | 0.562 |
| Fighting ★ | 0.033 | 0.193 | +0.159 | 0.181 | 0.208 |
| Assault ★ | 0.061 | 0.215 | +0.154 | 0.235 | 0.208 |
| Arson | 0.360 | 0.512 | +0.152 | 0.443 | 0.625 |
| Arrest | 0.261 | 0.398 | +0.137 | 0.398 | 0.417 |
| Normal_Videos_event | 0.605 | 0.739 | +0.134 | 0.756 | 0.750 |
| Robbery | 0.362 | 0.473 | +0.111 | 0.421 | 0.562 |
| Explosion | 0.442 | 0.538 | +0.096 | 0.639 | 0.479 |
| Burglary | 0.295 | 0.283 | −0.012 | 0.263 | 0.312 |
| Abuse | 0.480 | 0.328 | −0.152 | 0.357 | 0.333 |
| Shooting ★ | 0.279 | **0.033** | **−0.245** | 0.083 | 0.021 |

★ = the four motion classes. 11 of 14 classes improve.

**Shooting collapsed and it is not hidden.** From the summed confusion matrix (4 folds
x 3 seeds, 48 true Shooting videos): only **1 of 48** is predicted correctly. They scatter
to Shoplifting (11), RoadAccidents (9), Abuse (7), Normal (6). The model also almost never
*predicts* Shooting — 16 predictions against 48 true videos — so the class has been
effectively squeezed out. This is a genuine regression of the best system and it caps what
can be claimed.

Most-confused pairs overall: Vandalism→Arson (13), Assault→Arson (12), Arrest→Robbery (11),
Assault→Burglary (11), Fighting→Robbery (11), RoadAccidents→Stealing (11).

## Confusion matrices

Saved under the existing results structure:
- `results/p1c/p1c-sampling-ablation/confusion-best-system-video-majorityvote.txt` (best system)
- `results/p1c/p1c-video-aggregation/fold*-*/seed-*-confusion-{frame,video-majorityvote,video-meanprob}.txt`
  (baseline and both aggregations, every fold and seed)

## Backbone comparison — the empirical question, not an assumption

Paired difference (DINOv2 − ResNet), per fold:

| metric | fold 1 | fold 2 | fold 3 | fold 4 | mean ± sd | folds won |
|---|---:|---:|---:|---:|---:|---:|
| frame | −0.0129 | +0.0529 | +0.0464 | +0.0801 | **+0.0416 ± 0.0392** | 3/4 |
| video | −0.0180 | +0.0412 | +0.0819 | +0.0634 | **+0.0421 ± 0.0434** | 3/4 |

**Working conclusion: DINOv2, on consistency more than peak.** It wins 3 of 4 folds on both
metrics and roughly halves the cross-fold variance (frame sd 0.0416 vs 0.0764; video 0.0268
vs 0.0657). It **loses fold 1** on both — the backbone choice is genuinely empirical and
fold-1-only evidence would have picked the other way.

**Two limits on that conclusion, both open:**

1. The trained head gives **+0.0416**, where the classical probe gave **+0.0774**
   (`P1C-PROBES-FOLDS-CONFIRM-AR`). The probe standardised features and swept C per
   backbone; the head does neither. So the head comparison runs DINOv2 on a recipe tuned
   for ResNet, and +0.0416 is a **floor** on DINOv2's advantage rather than a settled value.
2. `implemented` is not `experimentally validated`. This row is validated; every `pending`
   row below it is neither.

## The epoch-0 finding

Best epoch by fold (3 seeds each):

| backbone | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|
| ResNet | 14, 13, 10 | 0, 0, 0 | 0, 0, 0 | 2, 1, 1 |
| DINOv2 | 4, 0, 0 | 3, 0, 7 | 0, 1, 0 | 0, 2, 0 |

**On 3 of 4 folds the head peaks at or within two epochs of initialisation.** It is not
learning and then overfitting; it is being selected at essentially its starting point. This
happens to **both** backbones, so it is not a DINOv2 feature-scale problem — fold difficulty
dominates. It is the strongest form yet of B-9's finding: on most folds the trained head
adds almost nothing over its own initialisation. Open as a recipe question (learning rate,
weight decay, feature standardisation), deliberately ranked **below** the temporal work.

## Per-class F1, cross-fold mean over 4 folds × 3 seeds

| class | ResNet | DINOv2 | diff |
|---|---:|---:|---:|
| Shoplifting | 0.236 | 0.621 | +0.385 |
| Stealing | 0.333 | 0.575 | +0.242 |
| Vandalism ★ | 0.072 | 0.183 | +0.111 |
| Normal_Videos_event | 0.468 | 0.567 | +0.099 |
| Arrest | 0.097 | 0.190 | +0.093 |
| RoadAccidents | 0.120 | 0.209 | +0.089 |
| Assault ★ | 0.050 | 0.121 | +0.071 |
| Arson | 0.257 | 0.298 | +0.041 |
| Fighting ★ | 0.038 | 0.020 | **−0.018** |
| Explosion | 0.415 | 0.391 | −0.024 |
| Robbery | 0.271 | 0.184 | −0.087 |
| Abuse | 0.366 | 0.254 | −0.111 |
| Shooting ★ | 0.261 | 0.144 | **−0.117** |
| Burglary | 0.375 | 0.184 | −0.191 |

★ = the four motion classes under scrutiny.

**Fighting is 0.020–0.038 cross-fold**, against the 0.115 that fold 1 alone reported — the
class is effectively non-functional and the better backbone made it *worse*. Shooting also
fell sharply. DINOv2's gains sit almost entirely in appearance-driven classes (Shoplifting,
Stealing). This is the clearest evidence so far that the missing ingredient is **motion, not
representation quality**, and it is what the temporal probe exists to test.


## The Shooting track — closed 2026-09-11 (D-49)

Shooting is the one class the best pipeline cannot recognise (F1 0.033, 1 of 48 videos). Ten
experiments, each with a pre-registered stopping rule, each closed negative. Kept here because
the *shape* of the failure is the finding.

| experiment | what it tested | result | verdict |
|---|---|---|---|
| Aggregation rules | max / top-k / mean vs majority vote | Shooting **0.033 under all six** | not aggregation |
| Exposure (32/video for Shooting) | more event frames in training | +0.034 F1, precision halved, Arrest −0.218 | prior shift, not learning |
| Separability (LOVO frame probe) | is event vs background linearly separable? | per-video 0.575, **pooled 0.381** | not across videos |
| A: feature temporal difference | motion hidden in inputs? | **−0.0146, 0/4 folds** | no |
| B: background-masked loss | label noise? | +0.0009, 2/4 | no |
| A2: dense 2.67 s clips | local temporal context? | per-video 0.743 vs 0.735, pooled 0.413 | no |
| A3: patch-token motion | motion below the pooling? | per-video **0.806** (p=0.09), pooled 0.433 | strongest within-video hint; no transfer |
| A3: 448 resolution | spatial detail? | pooled **0.350** — worst row | no |
| A4: per-video normalisation | score drift? | pooled 0.433 → **0.726** rank; but Shooting-vs-not **0.729 → 0.478** | drift is real and **is the class signal** |
| A5: domain-adversarial fine-tuning | remove source identity in the weights | localisation **0.857**, classification **0.385** | same trade, in the representation |

**The shape, present in every row:** frozen DINOv2 ranks event clips above background *within* a
held-out video (0.65–0.86) and cannot say *whether* a video contains an event (0.35–0.44 pooled).
Removing the between-video difference — by normalisation or by an adversarial objective — sharpens
the first and destroys the second, because whether a video is a shooting scene *is* a between-source
distinction.

**The defensible statement:** frozen DINOv2 contains useful within-video event structure, but the
representation does not transfer sufficiently across source videos for reliable Shooting-vs-not
classification under the current dataset.

**Recorded as a lead, not a result:** plain 2-block fine-tuning without the domain loss raised
held-out Shooting-vs-not ROC 0.600 → 0.800 on a single 5-vs-13 split (SE ≈ 0.12) while memorising
its 95 event clips to loss 0.000. If the track is ever reopened, that is the experiment — under a
multi-split protocol.

## Artifacts

- `results/p1c/p1c-cached-head-crossfold/` — 8 configs × 3 seeds, per-seed JSON + aggregates
  + `crossfold-summary.json`
- `results/p1c/p1c-probes-folds-confirm/` — the classical-probe four-fold comparison (D-45)
- Instrument validation: `fold1-frozen-VALIDATION` reproduces CONFIRM-015's GPU number
  (0.3482 ± 0.0008 against 0.3481 ± 0.0049) — see `TRAINING_LOG.md`.
