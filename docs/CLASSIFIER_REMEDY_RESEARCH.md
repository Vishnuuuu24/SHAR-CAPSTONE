# Research options for the classifier failure

> 2026-11-02. Literature and repository review, not an experiment or an adopted protocol.
> Read with `CLASSIFIER_FAILURE_ANALYSIS.md`, D-63/D-64, and `REMEDIES_TRIED.md` (now through N45).
> Sources inspected: primary-paper abstracts, official model/repository documentation, and the
> authors' AnomalyCLIP project page. This is not a claim that every full paper was read.
> No training, downloads, test evaluation, or checkpoint selection performed.

The best next modelling experiment is a **frozen DINOv2 ViT-L/14-reg backbone**, with the existing
attention-pooling recipe held fixed. The most relevant alternative research method is
**supervised text-guided anomaly recognition**, following AnomalyCLIP rather than another
zero-shot VLM. Both require an evaluation repair before a convincing improvement claim.
Neither is a demonstrated solution on SHAR.

## 1. What the diagnosis establishes, and what it does not

The local evidence strongly supports a crime-type discrimination problem: Normal-versus-crime
ranking is good, while almost all errors exchange crime labels. Better binary anomaly detection
alone is therefore poorly matched to the objective. The failing violence classes need features
that distinguish interactions, not merely a score that something unusual happened.

Several conclusions in the analysis need narrower wording:

- The learning curve supports **sensitivity to the number of distinct training videos over the
  measured range**. It does not isolate the sole bottleneck, establish a future learning rate,
  or guarantee gains from arbitrary additional videos. D-64 and the XD negatives already
  demonstrate the importance of class coverage and domain.
- Cosine similarity near 0.948 is a diagnostic of this feature space, not proof that event and
  background contain no discriminative information. Small directions can still carry class signal.
- Oracle frame selection failed for the tested representation and partially annotated subset.
  This weakens the case for selection-only remedies; it does not establish an information-theoretic
  ceiling for every representation.
- Small changes that fail the historical gate are **unvalidated**, not necessarily zero effects.
  A single model's confidence interval does not determine the interval of a paired difference.
- The validation–test gap alone cannot identify epoch-selection bias as its cause; split difficulty,
  sampling, and accumulated method selection are competing explanations.

These are interpretations of existing evidence, not new measurements.

## 2. Repair the comparison first

**Use paired uncertainty.** For each candidate and control, retain predictions on exactly the
same videos. Resample a source video once, carrying all its fold/seed appearances and both
models together, stratified by class; recompute the registered mean-of-cell macro-F1 and its
candidate-minus-control difference. Preserve the intended fold weighting. Also report per-fold
deltas and per-class precision/recall/F1. This estimates conditional uncertainty in the stored
predictions; it does not capture all uncertainty from retraining or prior method selection.

Direct inspection of `core/failure_analysis.py::cluster_bootstrap` shows that videos are currently
sampled separately inside each fold. Seeds share the draw, but the same source appearing in
multiple folds need not. The published interval therefore is not a global source-cluster interval
across overlapping folds. Its numeric correction requires a separate analysis run.

**Separate epoch selection from scoring.** Use an inner, source-grouped development split to
select epoch and any hyperparameters; score once on an outer split. More seeds on the same
videos do not supply more independent evaluation examples. Nested evaluation addresses selection
bias, but cannot undo the fact that previous research choices already used these data.
[Cawley & Talbot, JMLR 2010](https://www.jmlr.org/beta/papers/v11/cawley10a.html).

**Investigate additional eligible evaluation coverage, without recycling Test.** Audit canonical
video identities across *all* historical folds and test uses before proposing any new split.
Using more available training-pool videos for grouped outer evaluation is a separately named
development protocol, requires head retraining, and changes the training budget. New truly
untouched videos are stronger confirmation. There may be too few eligible violence videos to
create a substantially larger independent evaluation set; do not promise one before the audit.
Overlapping folds are dependent; naïve fold-standard-error calculations are inappropriate.
[Bengio & Grandvalet, JMLR 2004](https://www.jmlr.org/papers/v5/grandvalet04a.html).

Keep the historical +0.04-on-every-fold gate for already registered comparisons. Any replacement
must be a dated prospective decision specifying practical effect size, paired uncertainty,
multiple comparisons, and class-level reporting before results. Do not retroactively promote D-64.

## 3. Ranked modelling options

| Priority | Method | Why it fits this failure | Principal limitation |
|---|---|---|---|
| 1 | Frozen DINOv2 ViT-L/14-reg | Tests a stronger representation with minimal pipeline change | Still an image encoder; may improve scene cues without resolving actions |
| 2 | AnomalyCLIP-style supervised text-guided recognition | Learns class-specific visual/text directions with video-level labels | New encoder and objective; small-data overfitting and shortcut risks |
| 3 | Frozen V-JEPA video features | Tests a different pretrained source of interaction/motion information | VideoMAE already failed; higher extraction and memory cost |
| Conditional | Additional distinct, compatible surveillance videos | Directly addresses training coverage | Access, labels, domain and rare-class coverage are the hard parts |
| Conditional | Audio plus vision | Adds evidence potentially absent from pixels | Usable, event-aligned audio in UCF is unverified; changes input protocol |

### 3.1 Frozen DINOv2-L: the cleanest first experiment

The official release supplies register-equipped B and L models (86M and 300M parameters).
This establishes availability, not SHAR benefit.
[Official DINOv2 code and models](https://github.com/facebookresearch/dinov2).

Compare B-reg and L-reg using the same source frames, resolution, sampling, folds, seeds,
attention architecture and optimization. Fix the attention hidden width so the larger input
dimension does not silently enlarge the whole head. Its input projection necessarily changes;
report trainable parameter counts and treat the result as a backbone-pipeline comparison.
Inspect the four violence classes separately: a macro gain confined to fire/theft does not
resolve the diagnosed problem.

DINOv3 is another frozen-feature candidate, with official B/L releases, but changes more than
backbone size and uses a separate model licence. Keep it as a subsequent experiment, not a
simultaneous replacement of the size comparison.
[Official DINOv3 release](https://github.com/facebookresearch/dinov3).

### 3.2 Supervised language-guided recognition: the closest method match

AnomalyCLIP combines CLIP, video-level supervision, multiple-instance learning, learned
text-driven abnormal-class directions and temporal modelling. Its authors explicitly address
recognition as well as detection. Their recognition metrics are class-averaged AUC/AP, not
SHAR's exclusive 14-label video macro-F1.
[Paper](https://arxiv.org/abs/2310.02835),
[official implementation](https://github.com/lucazanella/AnomalyCLIP),
[authors' metric definitions](https://lucazanella.github.io/AnomalyCLIP/).

This differs materially from N31 (SigLIP vision features) and N35 (Qwen zero-shot answering):
the new ingredient is **learning visual–text class alignment from training bags**. Neither
negative settles this experiment. Conversely, prior appearance-feature negatives make a
large benefit uncertain.

Start with a matched frozen CLIP visual-only attention control, then a text-guided arm using
the same visual encoder and training videos. Write prompts from class definitions before
scoring; never feed a video's true class into its input prompt. Learn normal reference features
from training videos only. Decompose added temporal processing in a later arm if attributing
the effect to language. A full reproduction can instead be labelled a system comparison.

VadCLIP is related primary evidence for dual visual/alignment branches, but its reported VAD
AUC/AP cannot be imported as a recognition benchmark.
[VadCLIP, AAAI 2024](https://arxiv.org/abs/2308.11681).

Do not import a public UCF-trained checkpoint until its training-video overlap is audited:
the official anomaly-detection and action-recognition protocols need not have the same split.
Extract or remap features with canonical IDs and documented preprocessing.

### 3.3 V-JEPA: a bounded alternative to another VideoMAE attempt

V-JEPA 2 learns from video with a latent prediction objective and supplies frozen-feature
attentive-probe evaluation. The current official repository also lists V-JEPA 2.1 checkpoints.
Pin one exact version/model rather than treating the names interchangeably.
[V-JEPA 2 paper](https://arxiv.org/abs/2506.09985),
[official release and probing code](https://github.com/facebookresearch/vjepa2).

The hypothesis is that a different video pretraining objective yields useful interactions
that image features and the tested VideoMAE recipe missed. General action-benchmark results
are not surveillance evidence. Use the published clip/preprocessing requirements, cache
frozen features, and train a small head before considering fine-tuning. Pre-register temporal
coverage and compare a shuffled-frame control to test whether any gain uses frame order.
Fusion should follow standalone evidence of useful, complementary errors; N30 already shows
that averaging a weaker branch can hurt.

### 3.4 Data and audio: potentially valuable, conditional paths

For additional videos, prioritize new sources for the failing classes, preserve existing
published label semantics, audit duplicate events and scenes, and verify access/terms.
Naïvely pooling XD has failed. Domain alignment or sample weighting is a new hypothesis,
not an established rescue. Never discard validation errors to create an easier dataset.

The XD-Violence primary paper supports investigating audio–visual input in its own domain.
Audio could supply gunshots or impacts when pixels do not, but first inventory UCF audio
presence, silence, synchronization and soundtrack contamination. If coverage is inadequate,
retain this as a separate XD experiment. Any UCF audio arm needs visual-only, audio-only and
combined controls on the same eligible videos; missing audio must not silently exclude errors.
[Wu et al., ECCV 2020](https://arxiv.org/abs/2007.04687).

## 4. Methods to deprioritize

- More frame resampling, label smoothing, generic head/loss sweeps or denoising: directly
  weakened by N21–N25 and N39–N45. Do not repeat without a distinct mechanism.
- Binary anomaly cascades: most errors are already crime-to-crime; binary AUC is ranking,
  not proof of a reliable hard routing threshold.
- Family specialist heads: only a minority of errors remain within families, and incorrect
  hard routing prevents recovery. A soft hierarchy is possible but low priority.
- Pose-only pipelines: person detection is feasible, but detection does not establish accurate
  joints or action discrimination; local ROI results offer little support.
- Super-resolution, synthetic examples and pseudo-label expansion: cannot manufacture verified
  event evidence or independent videos; self-training risks amplifying systematic mistakes.
- Dropping Shooting, merging violence classes or reporting only confident cases: potentially
  useful separately named tasks or product outputs, not fixes to 14-label macro-F1.

## 5. Concrete next work

1. Audit source grouping, historical split exposure and paired uncertainty using existing
   prediction artifacts. No feature extraction is needed for this diagnostic.
2. Specify the prospective evaluation, epoch-selection rule, candidate count and claim gate.
3. Calibrate frozen DINOv2-L extraction on the actual DGX allocation; report measured runtime,
   device/host memory and storage before a long run. Do not invent an ETA from parameter counts.
4. Run the controlled B-versus-L comparison with the full registered seeds and budgets.
5. If it fails, prioritize the supervised CLIP/text comparison over another head sweep;
   reserve V-JEPA for an explicitly budgeted interaction-representation hypothesis.

Research review complete; proposed experiments remain unrun. The spent test score and
historical decisions remain unchanged.
