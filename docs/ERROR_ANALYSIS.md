# Error analysis — best validated system

> System: frozen DINOv2 + 8 frames/video + linear head + per-video majority vote.
> **0.4068 ± 0.0353** video-level macro-F1 cross-fold. Analysis date 2026-09-10.
> **No model change was made.** Measured findings and hypotheses are separated throughout.

## What could be measured, and what could not

41 of 224 validation videos carry an official temporal interval (D-40: the AR folds have
poor annotation coverage). **9 of them are Shooting** — the best-covered class, and the one
that collapsed. So the Shooting question is answerable by measurement rather than inference.
Classes with no interval coverage in validation (Abuse, Normal, Robbery, Stealing) are
analysed from predictions and imagery only, and that limit is stated where it applies.

## MEASURED — event duration relative to video length

Median over annotated videos, both splits:

| class | n | median event as % of video |
|---|---:|---:|
| Abuse | 2 | 5.4% |
| RoadAccidents | 15 | 8.7% |
| Explosion | 24 | 9.7% |
| **Shooting** | **25** | **9.9%** |
| Shoplifting | 24 | 11.5% |
| Stealing | 3 | 13.2% |
| Arrest | 6 | 19.5% |
| Vandalism | 5 | 22.9% |
| Assault | 3 | 25.6% |
| Burglary | 11 | 29.5% |
| Fighting | 6 | 30.6% |
| Arson | 10 | 39.4% |
| Robbery | 2 | 44.0% |

**Shooting events occupy ~10% of their video; individual cases go to 1.0%.**

## MEASURED — does uniform 8-frame TRAINING sampling contain the event?

| class | videos | ≥1 sampled frame in event | mean event frames of 8 |
|---|---:|---:|---:|
| Shoplifting | 15 | 9/15 | 0.80 |
| Explosion | 14 | 10/14 | 1.21 |
| **Shooting** | **14** | **11/14** | **1.50** |
| RoadAccidents | 5 | 5/5 | 1.60 |
| Vandalism | 4 | 4/4 | 1.75 |
| Fighting | 3 | 3/3 | 2.33 |
| Arson | 5 | 5/5 | 2.60 |

Per Shooting video: **3 of 14 get zero event frames**; 6 more get exactly 1 of 8. So a
Shooting training video typically contributes **7 frames of ordinary street and 1 frame of
event** — while its label says "Shooting" for all 8.

## MEASURED — the evaluation side, and this is the decisive constraint

Validation is **not** sampled: majority vote runs over every frame. For all 9 annotated
validation videos, the event is a **minority of frames**:

| video | frames | in event | % of the vote | can majority vote elect it? |
|---|---:|---:|---:|---|
| Burglary035 | 810 | 348 | 43.0% | no |
| Fighting018 | 278 | 69 | 24.8% | no |
| Burglary017 | 423 | 91 | 21.5% | no |
| Shoplifting028 | 272 | 55 | 20.2% | no |
| Shooting048 | 549 | 65 | 11.8% | no |
| Shooting013 | 215 | 18 | 8.4% | no |
| RoadAccidents020 | 355 | 25 | 7.0% | no |
| Shooting034 | 282 | 19 | 6.7% | no |
| Explosion036 | 1066 | 25 | 2.3% | no |

**In 9 of 9, event frames cannot win a majority vote.** The system can only be right when the
*non-event* frames also look like the class. That is a structural property of the evaluation
protocol, not a model weakness.

## MEASURED — vote margins on the errors

Shooting's errors are **unanimous, not uncertain**: `Shooting034 -> RoadAccidents` at margin
**1.00** across all 282 frames; `Shooting022 -> Abuse` at 1.00 over 911 frames;
`Shooting054 -> Shoplifting` at 1.00. Mean margin is 0.809 on correct videos and 0.719 on
wrong ones — the model is nearly as confident when wrong as when right.

## Visual evidence

`results/p1c/error-analysis-contact-sheets/` — the 8 sampled frames per video, red box
where a frame falls inside the annotated event.

- **`Shooting032`** — 21,680 frames, event at 7,995–8,205 (**1.0%**). The 8 samples land at
  0 / 3,095 / 6,195 / 9,290 / 12,390 / 15,485 / 18,585 / 21,680. The event falls *between*
  two samples. **Zero event frames.** Every frame shows an ordinary street.
- **`Shooting034`** — overhead street camera, event 960–1,050 of 1,405 (**6.4%**). Seven of
  eight frames are parked cars on a road, visually indistinguishable from RoadAccidents.
  Predicted RoadAccidents, margin 1.00.

## Attribution per confusion (measured basis where it exists)

| confusion | primary cause | evidence |
|---|---|---|
| **Shooting → Shoplifting / RoadAccidents / Abuse / Normal** | **sampling + event-minority evaluation** | event 9.9% median (1.0% worst); 3/14 videos get 0 event frames; event never reaches majority in 9/9 validation videos; errors unanimous at margin 1.00 |
| **Vandalism → Arson** | **visual similarity** | both are property-damage scenes, often night/outdoor; Vandalism→Arson 13 occurrences summed |
| **Assault → Arson / Burglary** | **visual similarity + few examples** | 34 train videos/class; Assault event 25.6% so sampling is not the constraint |
| **Fighting → Robbery** | **visual similarity** | `Fighting041` is 6,329 frames predicted Robbery at margin 1.00; Fighting event 30.6%, so frames DO contain the fight |
| **Arrest → Robbery** | **visual similarity + label semantics** | an arrest often follows a robbery; the two co-occur in one scene |

## HYPOTHESES — explicitly not yet measured

1. **Shooting fails primarily at evaluation, not training.** Even a perfect frame classifier
   would lose the majority vote at 6.7–11.8% event coverage. *Test:* score Shooting videos by
   *max* or *top-k* frame probability instead of majority vote and see whether it recovers.
   This is an evaluation change, needs no new model.
2. **Shoplifting succeeds despite an 11.5% event because context carries the label** — the
   whole video is a shop interior. Shooting's context is a street, which is also
   RoadAccidents' context. *Test:* compare per-class accuracy against the visual
   distinctiveness of the non-event background.
3. **Fighting/Assault/Vandalism→Arson is representational, not temporal.** Their events
   occupy 23–31% of frames and sampling reaches them, yet they still fail — consistent with
   D-46's finding that temporal windows did not help.
4. **Class count is a floor.** 34 training videos per class; Assault and Fighting may simply
   lack examples. Untestable without more data.

## What this suggests, without committing to it

The single highest-value next intervention appears to be **changing how a video's frames are
aggregated into a video decision** — majority vote structurally cannot represent a brief
event. That is an evaluation-side change costing no new architecture, and it is testable on
the caches already built. A GRU or Transformer over frozen frame features would not address
it, since D-46 measured temporal windows at +0.0037.

**No model change has been made. This document records the analysis only.**
