# Literature review: reasoning-based vision for violence, fighting, shooting and vandalism

> Written 2026-11-03. **Evidence level: abstract-only.** Every claim and number below comes from
> the abstract returned by an arXiv/PubMed paper-search index. No paper here was full-read, and none of
> the numbers were reproduced. Treat this as a reading list with a map, not as verified evidence.
> Before any of it enters `docs/EVIDENCE_REGISTER.md`, read the paper's evaluation protocol.
> IDs are arXiv unless marked PMC/PMID.

## 1. Bottom line

1. **"Reasoning" in this field means three different things.** (a) Prompting a frozen VLM/LLM to describe
   and judge a clip (training-free). (b) Fine-tuning a multimodal LLM on anomaly explanations
   (instruction tuning). (c) Reinforcement learning (GRPO-style) that rewards a step-by-step
   chain of thought. They are not interchangeable, and papers rarely compare across them.
2. **Almost all of it targets generic "video anomaly detection" (UCF-Crime, XD-Violence), not violence
   subtypes.** Fighting and shooting are two classes among 13. Vandalism-specific work is nearly absent.
3. **The best-documented weakness is bias toward "normal".** Large VLMs under-call crimes (2512.20417).
   Spatial and role reasoning ("who hit whom") is only slightly above chance on synthetic probes (2601.15780).
4. **Benchmark numbers may partly measure source artifacts, not events.** 2608.27879 finds that
   video-level AUC on UCF-Crime/XD-Violence keeps 39-91% of its above-chance separation when scoring
   only frames *before* the annotated event; it identifies title cards and platform watermarks in the
   normal class. This bears directly on SHAR rule 1 (label honesty).
5. **Shooting is mostly solved as object detection, not as reasoning.** Gun detectors are small-object
   problems with high false-positive cost. The reasoning-style fix is to confirm with a second, temporal
   stage (see 4.3).

## 2. Scope and how this was searched

Five parallel searches (violence + VLM reasoning; anomaly + MLLM explainable; gun/weapon/shooting;
vandalism; fight + skeleton/GCN), then three more (chain-of-thought/GRPO; training-free LLM; violence
benchmarks). Then the key abstracts were read in full. **Not done:** citation chasing, full-text reading,
any Google Scholar/venue-quality check, any check of whether a 2026 preprint was later peer-reviewed.

## 3. Taxonomy of approaches

| Family | Idea | Reasoning? | Needs training? | Typical output |
|---|---|---|---|---|
| Classic clip classifiers (3D-CNN, flow, ViT, Mamba) | Classify short clips violent/non-violent | None | Yes | label |
| Skeleton / pose methods | Keypoints, joint speed, proximity, GCN | Implicit (geometry) | Light | label |
| Weakly-supervised VAD (MIL, VadCLIP, multimodal fusion) | Video-level label to snippet scores | None | Yes | score curve |
| Training-free VLM+LLM | Caption frames, LLM scores them | Explicit, text | No | score + text |
| Instruction-tuned MLLM | Fine-tune on anomaly Q&A | Explicit | Yes | text + score |
| RL / chain-of-thought MLLM | Reward step-by-step reasoning | Explicit, trained | Yes | CoT + label + grounding |
| Agentic / tool-using | Model crops, re-inspects, self-corrects | Explicit, iterative | Yes | CoT + temporal span |
| Detector + temporal confirmation | Gun detector then 3D-CNN/multi-frame vote | Rule-like | Yes | alarm |

## 4. Findings by target behaviour

### 4.1 Violence and fighting (the best-studied slice)

**Classic baselines and datasets.** RWF-2000 (1911.05913): 2,000 surveillance clips, Flow Gated Network 87.25% accuracy.
XD-Violence (2007.04687): 4,754 untrimmed videos, 217 h, audio plus weak labels. DVD (2506.05372): 500 videos,
2.7M frames, *frame-level* labels, motivated by the coarse labels of older sets. A literature review of
violence detection is at PMC11207446.

**Reported accuracy is saturated and not comparable.** Papers report 97-99% on Hockey/Movies/Crowd Violence
and 87-89% on RWF-2000 (e.g. PMC12567496, IDG-ViolenceNet: 97.5 / 99.5 / 89.4). The saturation is on small, easy,
often staged datasets; RWF-2000 is the harder one. Cross-dataset results drop (2606.01225 reports "persistent
dataset shift").

**Skeleton / pose.** DIFEM (2412.05386) uses joint velocity and intersection with classical classifiers and
claims far fewer parameters than deep SOTA. 2605.02659 (YOLO11-Pose, Random Forest) reports precision 0.98 in
controlled frontal views but 0.72 on steep high-altitude real CCTV, a clean illustration of domain gap.
2608.27879 compared bounding-box geometry, handcrafted pose and learned pose under a fixed pipeline: **no pose
representation beat coarse box geometry** (with only 15 anomalous UCF-Crime videos, so small effects cannot be ruled out).
Whole-frame context matched or beat person-crop appearance.

**VLM / reasoning on violence.**
- 2511.07171 and 2510.17651: zero-shot and LoRA-tuned VLMs (LLaVA-NeXT-Video-7B) exceed 90% binary accuracy
  on RWF-2000/RLVS, but a personalised 65.8M-parameter 3D CNN had better calibration (ROC AUC 92.59%) at
  about half the energy (240 Wh vs 570 Wh). Hierarchical class grouping lifted VLM 14-class UCF-Crime accuracy from
  65.31% to 81%. Authors' conclusion: use the CNN by default, call the VLM only for contextual cases.
- 2511.10866: LLM auto-captioned 1-2 s clips, 95.25% on RWF-2000, 83.25% on UCF-Crime.
- 2504.18866 (PiercingEye, TPAMI): LLM-written "logic-guided ambiguous event descriptions" supervise a
  hyperbolic vision-language loss, helps on a curated ambiguous-event subset.
- 2501.06224: violence detection with knowledge graphs plus GAT for retrieval and explanation.
- 2609.40014: short-horizon *pre-incident* risk from face, audio and keypoint motion, 96.38% ROC-AUC on
  443 XD-Violence samples. Small sample and, given 2608.27879, a source-cue risk.

### 4.2 Weapons and shooting

- **Datasets:** 51K-image gun set (2105.01058); YouTube-GDD with 16,064 gun instances (2203.04129);
  CCTV-Gun, with a cross-dataset protocol (2303.10703); Monash/IMFDB sets; thermal concealed-weapon TICW (2510.13326).
- **Core difficulty:** handguns are tiny, occluded and look like phones or hands. CCTV-Gun's authors argue
  for cross-dataset evaluation because intra-dataset scores overstate real performance.
- **Pose + appearance** (2010.13753, 2503.12215): combining hand-region pose with weapon appearance gave
  +4.23 to +18.9 AP over prior best in the 2020 paper.
- **Human-gun association** (2005.09329, 2309.09236): pair each person with each gun and classify who holds it.
  AP 77.8-78.5 reported. This is the closest thing to "reasoning" in the detection line.
- **Temporal / two-stage confirmation to cut false alarms:** TYolov5 (2111.08867), a two-stage video gun
  detector (2503.06317), and an IoT system (2410.09731) where YOLOv5s (mAP 0.87, 4.43 FPS) triggers a cloud
  3D-CNN robbery/normal classifier (0.88 accuracy) to suppress false positives.
- **Shooter-as-a-whole** detection plus tracking trained with Unreal Engine synthetic data (2309.03381).
- **Audio** is a separate, strong channel for gunshots (CNN on spectrograms, 2506.20609, PMC12387842).
- **Open gap:** I found no reasoning/VLM paper focused on shooting events specifically. Shooting appears only
  as one UCF-Crime class inside general VAD work, and DeepSVU (2602.18019) names shootings and robberies as the
  threats it covers.

### 4.3 Vandalism

Sparse. Found:
- A 614-clip purpose-built vandalism set, SlowFast-R50 at 94.33% accuracy, inside a multi-task surveillance
  system with multi-frame confirmation (2607.03131). Dataset appears private to that paper.
- A forensic event ontology applied to the 2011 London riots (1903.09012), older, symbolic reasoning.
- Camera tampering (not the same thing) (2602.05706).
UCF-Crime has a Vandalism class, but I found no work that analyses it separately. **This is the clearest gap.**

### 4.4 Reasoning-centred video-anomaly work (applies to all three above)

**Training-free (frozen models, no weights changed)**
- LAVAD (2404.01014): VLM captions each frame, LLM aggregates and scores; beats unsupervised and one-class methods on UCF-Crime and XD-Violence.
- AnyAnomaly (2503.04504): user supplies the event as text; context-aware VQA with an LVLM; claims SOTA on UBnormal and UCF-Crime.
- EventVAD (2504.13092), VADTree (2510.22693), ASK-Hint (2510.02155, prompts grouped into violence / property crime / public safety), CoReVAD (2605.23116), QVAD (2604.03040), Cog-VADU (2610.01754).
- ReCoVAD (2511.17094): "sparse reasoning is enough"; matches training-free SOTA using 28.55% (UCF) and 16.04% (XD) of frames.
- 2610.16324: feedback from a generated description raised AUC 84.36% to 86.25% (UCF-Crime), AP 68.07% to 72.58% (XD-Violence).

**Instruction-tuned**
- Holmes-VAD (2406.12235): VAD-Instruct50k from single-frame annotations plus captioner plus LLM.
- Holmes-VAU (2412.06171): HIVAU-70k, 70,000+ annotations at clip / event / video level.
- UCVL (2502.09325): 1,829 videos, six question types, 8 MLLMs benchmarked from 0.5B to 40B, GPT-4o as judge. LLaVA-OneVision fine-tuning improves it.
- VERA (2412.01095): learns the guiding *questions* as parameters, no weight changes.

**RL / chain-of-thought**
- Vad-R1 (2505.19877): perception-to-cognition CoT, AVA-GRPO with self-verification, Vad-Reasoning dataset.
- VANGUARD (2605.02912): three-stage curriculum, Qwen3-VL-4B teacher, GroundingDINO boxes; 94% ROC-AUC, 84% F1 on UCF-Crime.
- Glance-Scrutinize-Think (2608.11260): tool-using agent that crops and re-inspects; VAGU-T with 7,567 videos, 21 categories, CoT tool traces.
- LRPO (2607.00654), DeepSVU (2602.18019, security-oriented, explains *causes* of threats).

**Corrective findings (read these before trusting any headline)**
- *Normality bias*: CoT alone pushes models toward "normal"; CoAT (2512.20417) adds an anomaly-focused final layer: +11.8 pp F1 on low-resolution footage, +3.78 pp classification on high-resolution.
- *Spatial reasoning is weak*: near-chance on minimal-pair violence/benign and assailant-role tasks (2601.15780). 2510.23190 finds small VLMs work on "simple, spatially salient events" and fail under noisy spatial cues; GAN privacy filters degrade accuracy.
- *Cost*: federated LoRA VLM used 570 Wh vs 240 Wh for a CNN (2511.07171). Several 2026 papers (ReCoVAD, SAAF 2610.32486, Flashback 2505.15205) exist mainly to cut that cost.
- *Benchmark leakage*: 2608.27879 (see point 4 of section 1).

## 5. What the reasoning approaches do and do not buy

| Claim | Evidence in what I read | Confidence |
|---|---|---|
| Gives human-readable explanations | Many papers; but explanations are mostly scored by an LLM judge (GPT-4o, GPT-rating) | Medium. Judge-model bias is not addressed in the abstracts |
| Improves detection AUC over non-reasoning | Reported by the training-free line on UCF-Crime/XD-Violence | Medium. Task is weakly supervised and score is video-level |
| Better than a small CNN for plain violence | **Not supported** by 2511.07171; CNN matched or beat on calibration at lower cost | Medium |
| Handles "who did what to whom" | **Contradicted** by 2601.15780 (near chance) | Medium |
| Generalises to unseen crime types | Zero-shot transfer claimed (VANGUARD, AnyAnomaly) | Low. Few cross-domain protocols stated |

## 6. Implications for SHAR (for the team to decide, not decided here)

1. **Rule 1 (label honesty).** UCF-Crime frames inherit video-level labels. 2608.27879 gives a concrete,
   cheap diagnostic: score anomalous videos using only pre-onset frames, with a control for sequence length. If
   our classifier scores well on pre-onset frames, part of the headline is a source cue. This needs only the
   official interval annotations and is worth reading in full.
2. **Rule 3 (task separation).** Almost every paper above reports video-level AUC/AP for detection or
   binary accuracy. Do not rank those against our 14-label macro-F1. The closest same-task comparisons are
   multiclass UCF-Crime ones, e.g. the 3D-ConvNet work (2101.01073, class set and protocol not stated in its
   abstract), and 2511.07171's 65.31% to 81% VLM number (zero-shot VLM, different protocol). Check the label
   set and split before comparing either to our 14-label result.
3. **Rule 2 (no circular evaluation).** VANGUARD uses GroundingDINO boxes as training supervision and
   Qwen3-VL as a reasoning teacher. If we use a VLM to generate reasoning or labels, that VLM's output cannot
   also be the evaluation reference.
4. **Possible low-cost ablation:** use a frozen VLM only as a second-stage verifier on the classifier's
   confusable pairs (fighting vs assault/abuse; shooting vs robbery), mirroring the detector-then-confirm
   pattern in 4.2. This is a hypothesis from the literature, not a finding.
5. **Vandalism gap:** if the team wants a contribution, per-class vandalism analysis on UCF-Crime is under-served.
   Note docs/PROJECT_RULES.md: "claim the protocol, not the novelty".

## 7. Limits of this review

- Abstract-only; no paper was full-read; no number reproduced.
- The search index is semantic and returned many 2026 preprints I could not cross-check; some may be
  revised, withdrawn or unreviewed. Peer-reviewed items I saw: PiercingEye (TPAMI), several Scientific
  Reports and Sensors papers (quality varies by venue, Sensors/MDPI especially).
- Reported accuracies on Hockey Fights, Movies, Crowd Violence are near ceiling and not discriminative.
- Audio-only and thermal work is included only in passing.
- I did not search XD-Violence-specific reasoning papers, AVA-style action localisation, or crowd/riot
  literature in depth.

## 8. Suggested reading order (highest value for SHAR first)

1. 2608.27879: what the benchmark AUC measures (provenance artifacts).
2. 2512.20417: normality bias in VLM crime detection.
3. 2601.15780 and 2510.23190: where VLM spatial reasoning fails.
4. 2511.07171: CNN vs VLM, accuracy/calibration/energy.
5. 2404.01014 (LAVAD) then 2510.02155 (ASK-Hint): the training-free baseline and its violence-aware prompt design.
6. 2505.19877 (Vad-R1) and 2605.02912 (VANGUARD): trained reasoning, and its label-generation pipeline.
7. 2303.10703 (CCTV-Gun): cross-dataset evaluation done properly.
8. 2502.09325 (UCVL) and 2412.06171 (HIVAU-70k): benchmarks for explanation quality.
