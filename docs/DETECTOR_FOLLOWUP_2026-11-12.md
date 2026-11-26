# Detector follow-up — prospective execution record

Owner request: continue the AI assistant's unfinished work on better MIL, language-aligned detectors,
and RWF-2000 after supplying the two missing files. Recovered 2026-11-12 from the AI assistant's
local session `981411ac-17e0-4e33-a9fb-a7932a146101`. No follow-up implementation had been saved.
Work stays on `review-plan-2026-11-11`, uncommitted under the current user's Git boundary.
The final 14-class recognition experiment remains closed. Paper work stays parked.

## UCF detector: fixed before training

Same D-71 official anomaly split and cached DINOv2-B + SigLIP features (one per 15 source frames).
Same 10% whole-training-video holdout, stratified normal/anomaly, seed 0. Fit mean/std on
**fitting videos only**; use no holdout/test frames to fit preprocessing. Temporal context is
the current frame's deviation from its causal running mean; five-sample causal score smoothing.
Training labels are video-level only. Test intervals are evaluation truth, never training targets.

All trained rows: MLP 512/128, ReLU, dropout 0.5, AdamW lr 1e-4, weight decay .01, up to
60 epochs, patience 10 on held-out training-video top-k AUC, improvement tolerance 1e-4.
Seeds 0/1/2 seeded before model construction; strict deterministic PyTorch algorithms and the pod's registered cuBLAS workspace. Balanced batches contain 32 bags of each class;
cycle the smaller class to cover the larger class each epoch. This fixes the old runner's
discarded majority tail; hence the new control, rather than D-71's saved score, is the primary comparison.
Random stratified temporal sample of at most 256 frames per training bag, ceil(T/16) aggregation.

| Row | Fixed definition |
|---|---|
| control | BCE on mean of top-k frame sigmoid scores |
| magnitude | Same architecture; rank latent vectors by L2 norm, BCE on their top-k scores. Add .001 times [mean relu(10 - abnormal top-k norm)^2 + mean normal top-k norm^2]. Margin/weight are project choices, not copied paper constants. |
| selftrain | Control + .5 times frame pseudo-label BCE after five warm-up epochs. EMA teacher (decay .99), detached; positive target when teacher p >= .9; negative when p <= .1; ignore other abnormal-bag frames. All normal-bag frames are negative. |
| uncertainty | Same pseudo-label branch using four teacher dropout passes; confidence = clamp(1 - variance(p)/.25, .1, 1), detached, weighting frame BCE. This is a project uncertainty adaptation, **not UR-DMU's dual-memory method**. |
| language | Control with 14 frozen SigLIP prompt-cosine features appended before causal context. Uses raw first-768 pooler dimensions, never standardized image vectors for text cosine. |
| zero_shot_language | No training: sigmoid[10*(max anomaly prompt cosine - normal prompt cosine)], same smoothing and training-normal alarm threshold. |

Text: two templates ("a surveillance video of {}.", "a security camera recording of {}.")
for normal activity and the 13 named anomaly concepts. Each template embedding is normalized;
average per concept and normalize again. Prompts and checksums are saved before scoring.
Text tower is strictly loaded; full SigLIP SHA256 must equal the already registered
`2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8`.
Its 208 vision tensors were measured equal to the cached-feature tower before these runs.
This is a SigLIP prompt-feature adaptation, **not a reproduction of VadCLIP**.

Develop/check all rows using training videos only. Run all declared rows on test together
in **one further declared detector test read**, after every seed checkpoint exists. Never
choose a row, epoch, prompt, loss setting, alarm threshold, or rerun from test outcomes.
Keep all negative rows. Report exact D-71 metrics, seeds and variability; no improvement claim
without paired source-video-clustered uncertainty. New-control contrasts are the reference;
the five contrasts against control comprise one comparison family. Use 2,000 paired whole-source-video bootstrap draws (seed 0), exact run-length-weighted frame AUC, mean per-seed AUC per draw, percentile intervals and Holm-adjusted two-sided bootstrap sign probabilities. These describe uncertainty; no automatic winner adoption. No automatic adoption or further search.
Threshold: each seed's 99th percentile of scores on held-out normal **training** frames.

## RWF-2000: separate binary clip study

Use all 2,000 supplied clips, native train 1,600 and val 400; val is the terminal test,
never a tuning set. Preserve Fight/NonFight labels. No UCF label merging or localization claim.
Internal noncommercial research only; mirror metadata is not proof of underlying footage rights.
Supplied mirror: `DanJoshua/RWF-2000`; original paper [Cheng et al.](https://arxiv.org/abs/1911.05913).

Intake hashes archive and every clip, checks split counts, and preserves original member names.
Long Unicode filenames exceed the filesystem limit: extracted filenames are SHA256(original
member name) with an AVI extension; the immutable intake manifest maps them back.
**Intake measured six shared video SHA256 values across train and val.** Preserve native split;
report native metrics as contaminated, with a separately named overlap-excluded sensitivity.
Detect original-source groups by stripping a final `_integer` from the original stem, and
join groups sharing byte content. This is a conservative name/content audit, not proof that all
remaining videos are independent. Any overlapping terminal groups are excluded from sensitivity.

Frozen DINOv2-B and SigLIP at 224 pixels. Decode each clip to EOF, select 16 rounded uniformly
spaced source-frame indices; OpenCV INTER_LINEAR resize, RGB conversion, registered backbone
normalizations. Mean the 1,536-D frame features per video. Rows: DINOv2-B, SigLIP, concatenation.
Logistic regression with StandardScaler fitted only on fitting data. C in {.01,.1,1}, choose
by macro-F1 on the first split of stratified 5-fold **source-grouped** training CV, seeds 0/1/2;
tie -> lower C. Retrain on all native train, report every seed on native val and overlap-excluded
sensitivity. Metrics: accuracy, macro-F1, ROC-AUC, confusion matrix and source-cluster bootstrap
95% intervals (2,000 draws, seed 0). No terminal-set selection or additional candidates.

## Source boundaries

The mechanisms are motivated by [RTFM official code](https://github.com/tianyu0207/RTFM),
[MIST official code](https://github.com/fjchange/MIST_VAD),
[UR-DMU official code](https://github.com/henrryzh1/UR-DMU), and
[VadCLIP official code](https://github.com/nwpu-zxr/VadCLIP). Their full networks/recipes are
not implemented here; avoid attaching their published numbers to these adaptations.

## Execution and closure

Heavy work runs only in the account's serial MIG pod (8 CPUs, 16 GiB). Source snapshot is
made from the actual working files and hashed, not falsely identified as committed HEAD.
Keep best and latest resumable per-epoch MIL states, RNG/optimizer/teacher states, configs,
package versions, data/checkpoint digests, attempt streams and logs. Preserve failed intake
attempts. Calibration is not a result. Give throughput-based estimates before long runs.
Close with input manifest, unit tests, language cache, all training checkpoints and terminal
metric tables, and updates to status/training/evidence. Clearly separate code/run/research completion.


Calibration before full execution: main `results/detect/followup-20261112/ucf-001/calibration.json` measured 84.22 s/full epoch, 0.697 GiB GPU; RWF `rwf-001/calibrate.json` measured 6.92 clips/s, 0.862 GiB GPU. Full research settings retained. UCF estimate 4–30 h with early stopping/teacher overhead; RWF 5–15 min. First full epoch is logged and ETA must be revised if pace changes >20%.
