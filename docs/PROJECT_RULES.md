# Project rules

The working rules the SHAR project followed. Code comments and documents in this repository cite them as
"rule 1" to "rule 5".

## The five science rules

1. **Label honesty.** UCF-Crime frames inherit a video-level label. That is weak, noisy supervision, never
   frame-level ground truth. Where an event-only test view built from the official intervals exists, it is the
   headline and the full-video inherited-label score is a separately named sensitivity analysis. Where it does not
   exist, say so and name the reported metric for what it is. The Action-Recognition folds have no such view (only
   48 of fold 1's 168 test videos carry an official interval), so those runs report an honestly named
   inherited-label score.
2. **No circular evaluation.** Model-generated boxes or masks may propose regions of interest or act as an ablated
   auxiliary signal. They are never evaluation ground truth for that model or anything descended from it.
3. **Task separation.** UCF 14-class accuracy and F1, temporal anomaly-detection AUC and AP, COCO mAP, AVA
   Frame-mAP and UCF-Crime2Local localisation are different tasks. Never rank them against each other.
4. **Keep negative results.** Failed, neutral and negative outcomes stay visible. If a gain is inside the noise,
   the claim fails. Do not quietly drop an experiment that disappointed.
5. **Reproducibility minimum.** Any run that produces a number someone might cite records its seed,
   configuration, code revision and data digest.

Two supporting habits: prefer direct measurement and primary sources; and change one material factor per
comparison when the comparison is meant to attribute a cause.

## Code practices

- **Rule of two:** the second time a helper is written, extract it to a shared module.
- **No monoliths:** about 400 lines is a soft ceiling per module; crossing it is a prompt to split by
  responsibility.
- **Thin command-line scripts:** scripts parse arguments and orchestrate; the real work lives in importable
  functions that can be tested without a subprocess.
- Preserve source-video grouping and label provenance end to end.
- Never invent equations, thresholds, licences, class mappings or results.

## Recording work

A measured figure that a decision rests on goes in `docs/TRAINING_LOG.md`; a changed decision in
`docs/DECISIONS.md`; changed evidence in `docs/EVIDENCE_REGISTER.md`; something that broke in
`docs/KNOWN_PROBLEMS.md`. Dated decision rows are frozen at their date: a later correction goes in the evidence
register rather than rewriting history.

## Compute

GPU work ran on one slice of a shared university DGX server, so it was strictly serial. The full protocol was
always run on the real data, resolution and seeds; where it was infeasible, the alternatives were presented rather
than the science silently shrunk.
