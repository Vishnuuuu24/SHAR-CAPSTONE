# Posture detection: literature review and data check

Date: 2026-11-05. Literature search plus a read-only summary of existing cached detections.
No training, no new pose extraction, no Test use, no protocol or decision change.
Status: review-complete. Posture recognition itself is still **unrun**; research-complete: no.

**Read depth is marked on every paper.** F = I read passages of the full text. S = summary of the
publisher or repository page. A = abstract or search listing only. Numbers from A/S papers are the
authors' claims, unchecked. Skim the abstract, method and results table yourself before relying on
any paper your argument rests on: this write-up can misread a method, and only that check catches it.

**Metrics are not comparable across rows.** AUC, frame AUC, accuracy and macro-F1 here come from
different tasks and datasets. Nothing below ranks them against SHAR's 14-class scores (rule 3).

## 1. Main findings

1. **UCF-Crime has no posture labels, so the labels must come from elsewhere.** AVA has 14
   human-labelled pose classes (§3). It is movie footage, so it is a separate-domain source.
2. **Our cached joints support per-frame posture, not motion over time.** 592 of the 700 cached
   videos have only 8-16 sampled frames. Only 96 (13.7%) have 16 consecutive frames with a person
   (§2). Skeleton-sequence models such as ST-GCN need new extraction or HR-Crime's published tracks.
3. **Whether a person was detected at all differs strongly by class, and that alone can look like
   a result.** Per-video person coverage runs from 22.5% (RoadAccidents) to 81.2% (Shoplifting).
   Kim & Lim 2026 show this kind of signal matching a motion model on UCF-Crime (§4A), so any
   posture gain must beat a "coverage only" control.
4. **Direct evidence on UCF is cautious.** Ganesh 2026 found no pose descriptor beating simple box
   geometry, but on only 15 anomalous videos and for binary detection. Pose may help; it is unproven.
5. **Treat high accuracies in this literature sceptically.** The 96-98% figures come from staged,
   synthesized or custom datasets, or from papers that do not describe their split (§6).

## 2. What our cached pose data allows (measured today)

Source: `results/p1c/p1c-roi-cache/` (YOLO11m-pose, `imgsz=640, conf=0.25`). 700 videos, 50 per
class, 82,679 frames. Video labels are inherited labels (rule 1). Keypoint confidence is the model's
own score, rounded to one decimal; it is not accuracy, because UCF has no pose ground truth.

| Measure | Value | Why it matters |
|---|---|---|
| Frames per video | 326 videos have 8; 266 have 12-16; 108 have 100+ | `plan_roi_frames` keeps every validation frame but thins training to 8 per fold |
| Videos with 16+ consecutive stride-5 frames containing a person | 96 of 700 (13.7%); median longest run is 1 frame | Sequence models cannot train on this cache |
| Frames with no person / with 2+ people | 40.3% / 33.9% | Many frames carry no posture signal at all |
| Person box height, 10th / 50th / 90th percentile | 46 / 82 / 135 px (6.3% under 40 px, 47.5% under 80 px) | Small bodies, so joints are coarse |
| Share of detected joints with confidence >= 0.5 | shoulders 0.97, hips 0.92, elbows 0.80, knees 0.71, wrists 0.62, ankles 0.58, ears 0.53, nose 0.47, eyes 0.35 | Torso is usable; wrists and ankles are weak; face is poor |
| Track IDs | None saved | Person identity across frames does not exist yet |

Per-video mean, so long videos do not dominate. Most videos here have only 8-16 frames, so treat
single values as noisy.

| Class | Frames with a person | Frames with 2+ people |
|---|---|---|
| Shoplifting | 0.812 | 0.537 |
| Fighting | 0.770 | 0.606 |
| Assault | 0.765 | 0.542 |
| Robbery | 0.764 | 0.441 |
| Abuse | 0.652 | 0.364 |
| Normal | 0.598 | 0.437 |
| Arrest | 0.584 | 0.369 |
| Shooting | 0.529 | 0.292 |
| Burglary | 0.502 | 0.105 |
| Stealing | 0.442 | 0.184 |
| Vandalism | 0.435 | 0.151 |
| Explosion | 0.335 | 0.178 |
| Arson | 0.310 | 0.085 |
| RoadAccidents | 0.225 | 0.103 |

Rough scale for new extraction: about 575,000 stride-5 frames for all 700 videos (scaled from fold
1's 391,014 rows over 476 videos). Measure throughput on a smoke run before quoting an ETA; this
would likely exceed 30 minutes of slice time, so it needs the usual heads-up.

### Visual check: are the videos clear enough?

Added after the owner asked. Informal, one reviewer, 42 frames; artifacts and method in
`results/p1c/posture-visual-check-20261105/`. It judges whether skeletons follow the body, not
posture-model accuracy, and it is not ground truth.

- Of 30 sampled frames with people, the skeleton was good on 20 (95% interval roughly 49-81%), partial
  on 6 and failed on 4. Good by person height: under 40 px 3/6, then 4/6, 4/6, 5/6 and 4/6.
- Failures came more from camera angle, hidden bodies and crowds than from pixel clarity. Overhead views
  are a specific risk because AVA is mostly eye-level movie footage.
- Of 12 frames with no detected person, 9 looked empty, 2 had people the model missed, 1 was unclear.
- Enough body to judge coarse posture (shoulder, hip and knee each at confidence >= 0.5): 73.1% of detected
  people (51.1% under 40 px, 83.5% over 120 px). As a share of frames, per-video mean: Assault 0.710,
  Fighting 0.697, Shoplifting 0.625, Robbery 0.560, Arrest 0.530, Abuse 0.494, Normal 0.486, Shooting 0.456,
  Vandalism 0.380, Burglary 0.361, Stealing 0.353, Explosion 0.266, Arson 0.241, RoadAccidents 0.190.

Reading: coarse posture looks feasible on about half of frames, unevenly by class. Fine detail such as hands
looks unreliable. This raises the value of the small hand-labelled UCF check set in §8, question 5.

## 3. Posture labels: the datasets that exist

| Source | What it offers | Limits | Read |
|---|---|---|---|
| [AVA](https://arxiv.org/abs/1705.08421) | **14 pose classes:** stand, sit, walk, bend/bow, lie/sleep, dance, run/jog, crouch/kneel, martial art, get up, fall down, jump/leap, crawl, swim. Also interaction labels: fight/hit, grab, push, kick. Trainval counts range from stand 208,332 to fall down 378 | Movies, close-range people; the register warns of poor CCTV transfer. 313 video files are local but licence and completeness are unverified (`DATA_SPEC.md`) | F |
| COCO keypoints | Independent human-labelled joints, the right way to score the pose estimator itself | No posture classes. Local copy equally unverified | known from repo docs |
| [HR-Crime](https://arxiv.org/abs/2108.00246) ([data](https://doi.org/10.34894/IRRDJE), 3.5 GB, CC BY-NC-ND 4.0) | Skeleton tracks **from UCF-Crime itself**: 789 anomaly + 782 normal videos, YOLOv3-spp + AlphaPose + PoseFlow. Could fill our sparse-training gap and act as a second pose pipeline | Machine-generated, so never ground truth (rule 2). Drops videos without a clear view of people, which is a selection effect. Zip not opened; frame rate and video naming unchecked. `data.4tu.nl` did not resolve from the head node | F (paper), S (data page) |
| [PoseLift](https://arxiv.org/abs/2501.06591) | Real retail CCTV pose sequences: 155 videos, 43 shoplifting, 6 cameras, 15 fps | Anomaly task, one store, no posture labels | F |
| [CHAD](https://arxiv.org/abs/2212.09258), [HuVAD](https://arxiv.org/abs/2408.14329) | Surveillance pose/box/ID annotations, multi-camera | Binary anomaly, not posture classes | A |

Small posture studies, mostly controlled settings, useful for method rather than evidence:
[joint-angle standing/sitting/lying from ordinary cameras](https://journal.uob.edu.bh/handle/123456789/4207) (A),
[YOLOv11 pose-angle fall detection, PIFR](https://pmc.ncbi.nlm.nih.gov/articles/PMC12173392/) (A),
[two-stage fall recognition from posture features](https://pmc.ncbi.nlm.nih.gov/articles/PMC7729773/) (A),
[standing/sitting/lying/"dangerous sitting" with skeletons](https://pmc.ncbi.nlm.nih.gov/articles/PMC7240018/) (A),
[fall detection from body geometry and pose](https://doi.org/10.1016/j.jvcir.2021.103407) (A),
[overlapping-camera posture classes](https://doi.org/10.1177/14604582211030954) (A),
[atomic poses then Markov chains for suspicious behaviour](https://core.ac.uk/download/161508195.pdf) (A),
[projection-histogram posture in indoor surveillance, 2004](https://aimagelab.ing.unimore.it/imagelab/pubblicazioni/ia2004.pdf) (A).

## 4. Reading list by question

### A. Does pose carry signal on UCF-Crime itself? (start here)

| Paper | Setup | Finding | What to take | Read |
|---|---|---|---|---|
| [Kim & Lim 2026](https://arxiv.org/abs/2610.16471) | YOLOv8-pose, greedy tracker; Abuse/Assault/Fighting/Shooting vs Normal; 564 videos; video-level AUC | Detection/pose/tracking quality alone: AUC 0.827-0.862, against motion 0.821-0.844. After balancing quality, motion gains 0.046-0.141. By class: Fighting strong, Assault and Shooting weak | **Use their 11 quality variables as a control** | F |
| [Ganesh 2026](https://arxiv.org/abs/2608.27879) | YOLOv8m + BoT-SORT + RTMPose-m; 165 videos, 15 anomalous; video-level AUC | Box geometry 0.724 (probe) / 0.684 (MIL); pose variants 0.580-0.643 / 0.597-0.632. 39-91% of above-chance separation exists *before* the event | Box-only control and pre-event test. Author says this is a limit of the benchmark, not proof pose is useless | F |
| [Matei 2022](https://arxiv.org/abs/2207.01687) | HR-Crime skeleton trajectories, MPED-RNN encoder, 13 or 14 classes | Best 13-crime weighted accuracy 0.382; Normal accuracy about 0.18 in the 14-class model | Closest task to ours, **with problems**: split by trajectory, not video; trajectory labels come from a model's anomaly score (see §6) | F |
| [Boekhoudt 2021](https://arxiv.org/abs/2108.00246) | Builds HR-Crime | Their pipeline scored 0.534 against 0.735 AUROC on the original authors' HR-ShanghaiTech trajectories; poor video and lighting break detection | Pose pipeline choice alone can move results a lot | F |
| [Action Hints 2025](https://arxiv.org/abs/2509.11058) | AlphaPose on UCF test set (290 videos), trained on Kinetics skeletons; frame AUC | 62.7; STG-NF 51.6-51.9; language-vision model LAVAD 80.3 in the same table | Skeleton-only sits well below appearance methods here | F |
| [π-VAD 2025](https://arxiv.org/abs/2505.13123) | RGB plus modalities used only during training; frame AUC | RGB 86.97, +pose 87.65. Abnormal-only AUC 70.81 to 73.24. No intervals. Pose called key for human-based anomalies | Small gain; train-time-only pose avoids missing-pose problems at inference | F |
| [Noghre 2023](https://arxiv.org/abs/2303.05463) | ShanghaiTech and UBnormal analysis | Pose separates better than trajectory, but neither is enough alone on hard datasets | Plain statement of failure modes | F |
| [Kilic & Tuceryan 2023](https://doi.org/10.1109/eit57321.2023.10187325) | Shoplifting pre-crime clips, 3D CNN + pose | Claims pose improves accuracy | **Locked (§5).** Pre-crime results need Ganesh's caution | S |

### B. Pose for crime and violence on other datasets

| Paper | Idea | Fit for SHAR | Read |
|---|---|---|---|
| [DIFEM](https://arxiv.org/abs/2412.05386) | Only 5 features: joint velocity and joints overlapping another person's box, then classical classifiers | Very cheap baseline, but velocity needs consecutive frames | F |
| [Garcia-Cobo 2023](https://doi.org/10.1016/j.cviu.2023.103739) | Skeleton + frame-change inputs, added rather than multiplied so one empty input does not kill the signal | Fusion idea handles missing poses | S |
| [SPIL](https://arxiv.org/abs/2308.13866) | Joint points as a point cloud, with interaction learning | Fighting/Assault | A |
| [WeDePE](https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/weaponsDetection/HumanPose.pdf) (free copy of a closed article) | Pose locates hand regions, then a weapon detector runs there | Shooting; D-51 found no weapon detectable at 320x240, so low expectation | S |
| [Handgun + pose](https://arxiv.org/abs/2010.13753), [gun + pose 2025](https://arxiv.org/abs/2503.12215) | Combines hand regions, pose and appearance | As above | A |
| [Snatch-and-run robbery](https://arxiv.org/abs/2604.14329), [pushing detection](https://arxiv.org/abs/2605.02659) | YOLO-pose + kinematic features + Random Forest | Same detector family as ours; staged or small data | A |
| [ViTPose violence at airports](https://arxiv.org/abs/2308.16325), [multi-scale GCN](https://pmc.ncbi.nlm.nih.gov/articles/PMC9797844/) | Keypoint sequences + CNN-BiLSTM or GCN | Needs consecutive frames | A |
| [Dangerous student behaviour](https://arxiv.org/abs/2202.09550), [looking-around detection](https://journal.ugm.ac.id/v3/JISE/article/view/31998) | Pose features for behaviour classes from CCTV | Close to the "suspicious posture" wording | A |
| [YOLO-pose + CLIP VAD 2026](https://arxiv.org/abs/2608.31074) | Person crops scored against text descriptions | Bridges to the unrun text-guided idea | A |
| [Skeletal VAD survey](https://arxiv.org/abs/2301.00114) | Survey of skeleton anomaly detection | Orientation; one-class task, not ours | A |

One-class skeleton anomaly detectors ([STG-NF](https://arxiv.org/abs/2211.10946),
[MPED-RNN](https://arxiv.org/abs/1903.03295), [SeeKer](https://arxiv.org/abs/2506.18368),
[STEP](https://arxiv.org/abs/2608.19987), [SPARTA](https://arxiv.org/abs/2408.15185)) are a different
task (normal vs not normal). All A. Useful for ideas such as confidence weighting, not as baselines.

### C. Skeleton models, and surviving bad joints

| Paper | Finding | Fit for SHAR | Read |
|---|---|---|---|
| [PoseC3D](https://arxiv.org/abs/2104.13586) | Joints drawn as heatmaps, then a 3D CNN. Dropping one limb joint per frame: 92.4 to 91.5, against GCN 92.0 to 77.7 (FineGYM accuracy). Kinetics skeleton-only just 47.7% because people are missing or small. Samples frames uniformly | **Best fit for noisy joints**, and its Kinetics failure is a warning that applies to us | F |
| [SkeleTR](https://arxiv.org/abs/2309.11445) | Short skeleton pieces plus person-interaction transformer, built for imperfect association | Suits no reliable tracks | A |
| [PYSKL / ST-GCN++](https://arxiv.org/abs/2205.09443) | Baseline toolbox with pretrained models | Our 2026-11-04 review notes it is now unmaintained | A |
| [2P-GCN](https://arxiv.org/abs/2208.06174) | One graph across two people | Fighting/Assault; 54-61% of their frames have 2+ people | A |
| [MotionBERT](https://arxiv.org/abs/2210.06551), [ViA](https://arxiv.org/abs/2209.00065) | Pretrained motion encoders from 2D skeletons | Few videos per class suits a frozen encoder | A |
| [STARS](https://arxiv.org/abs/2407.10935) | Self-supervised skeleton training, stronger in few-shot | Same reason | A |
| [PartialVisGraph](https://arxiv.org/abs/2607.00716), [PeGCN](https://arxiv.org/abs/2003.07514), [RA-GCN](https://arxiv.org/abs/2008.03791) | Handling missing or noisy joints | Wrists and ankles are weak in our data | A |

### D. Combining pose with RGB

| Paper | Idea | Fit for SHAR | Read |
|---|---|---|---|
| [IntegralAction](https://arxiv.org/abs/2007.06317) | Pose stream decides how much appearance to use, based on pose reliability | **Direct fit:** gives a principled fallback for the 40% of frames with no person | F |
| [π-ViT](https://arxiv.org/abs/2311.18840), [PERF-Net](https://arxiv.org/abs/2009.13087) | Pose used only in training; not needed at inference | Avoids the missing-pose problem | A |
| [EPAM-Net](https://arxiv.org/abs/2408.05421), [Skeleton + one RGB frame](https://arxiv.org/abs/2202.11374) | Skeleton guides attention on frames | Could guide D-53's attention pooling | A |
| [Patch as Node](https://arxiv.org/abs/2512.21916) | Uses RGB patch tokens at joint locations | Interesting: we already cache DINOv2 patch tokens | A |
| [DRAF-Net](https://pmc.ncbi.nlm.nih.gov/articles/PMC13709200/) | Quality-aware RGB-skeleton fusion | Same principle as IntegralAction | A |

Note: D-53 found temporal order did not help on our data, and the frozen backbone may already encode
body shape. Fusion gains seen on NTU-style datasets do not transfer automatically.

### E. Pose estimators and low resolution

| Paper | Finding | Fit for SHAR | Read |
|---|---|---|---|
| [YOLO-Pose](https://arxiv.org/abs/2204.06806) | Boxes and joints in one pass | Our current family | A |
| [RTMPose](https://arxiv.org/abs/2303.07399), [RTMO](https://arxiv.org/abs/2312.07526), [ViTPose](https://arxiv.org/abs/2204.12484) | Alternatives. Ganesh uses RTMPose-m on UCF | Second estimator to compare if the audit shows a problem | A |
| [Low-resolution pose (CAL)](https://doi.org/10.1016/j.patcog.2022.108579) | Standard heatmaps degrade at low resolution; reports 72.3% AP at 128x96 on COCO | COCO persons, not CCTV | S |
| [Super-resolution for pose](https://arxiv.org/abs/2107.02108) | Upscaling helps small people (AP +160% in smallest group) and hurts above about 3,500 px of body area | Links to RESULT 1 restoration work. Our box sizes are not segmentation areas, so the threshold is not directly comparable | F |
| [STAR-Pose](https://arxiv.org/abs/2506.16061) | Task-driven upscaling for video pose | Later option | A |
| [TPAMI 2026 comparison](https://doi.org/10.1109/tpami.2026.3672463) | Reviews 118 papers on video pose accuracy | Marked open access by metadata, but the page could not be opened automatically; please check | A |
| [Cormier 2021](https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/html/Cormier_Interactive_Labeling_for_Human_Pose_Estimation_in_Surveillance_Videos_ICCVW_2021_paper.html) | Labelling surveillance pose takes 40-60 s per person; tool cuts time 55% in simpler scenes | Cost estimate if we ever hand-label a check set | S |

## 5. Locked papers: please try to get these

Every paper below was closed in Crossref licence data, Semantic Scholar and OpenAlex, and not in DOAJ.
High and Medium papers also had a title search for a free copy. Low papers had the database routes only. arXiv's API and Europe PMC were rate-limited or down, so
arXiv was covered through Semantic Scholar's arXiv IDs, the Firecrawl paper index and title search.
I only used legal routes.

**Tip:** where marked RG, ResearchGate lists a PDF or full text. I did not open it, so try that first
(free, may need an account) or use "Request full-text".

| # | Priority | Paper | Why it is worth having | Link | RG? |
|---|---|---|---|---|---|
| 1 | High | Kilic & Tuceryan 2023, *Crime Detection from Pre-crime Video Analysis with Augmented Pose Information* | Pose + shoplifting pre-crime clips; closest match to SHAR | https://doi.org/10.1109/eit57321.2023.10187325 | [yes](https://www.researchgate.net/publication/372638040) |
| 2 | High | Ansari & Singh 2022, *Identifying human activities in megastores through postural data to monitor shoplifting events* | Most explicit "postural" method; handling of occlusion | https://doi.org/10.1007/s00521-022-08028-0 | [listed](https://www.researchgate.net/publication/365670889) |
| 3 | High | Dileep et al. 2022, *Suspicious Human Activity Recognition using 2D Pose Estimation and CNN* | Same project wording; check dataset and split | https://doi.org/10.1109/wispnet54241.2022.9767152 | request only |
| 4 | High | Nale et al. 2021, *Suspicious Human Activity Detection Using Pose Estimation and LSTM* | Pose sequences for suspicious activity | https://doi.org/10.1109/iria53009.2021.9588719 | not checked |
| 5 | High | 2023, *Semantic segmentation-based system for fall detection and post-fall posture classification* (EAAI) | Actual posture classifier | https://doi.org/10.1016/j.engappai.2022.105616 | [listed](https://www.researchgate.net/publication/365758810) |
| 6 | High | Li et al. 2026, *Spatial-Temporal Self-Compensating GCN ... Under Data Constraints* (TIP) | Occlusion, missing frames, noise: our exact problem | https://doi.org/10.1109/tip.2026.3695673 | not checked |
| 7 | Medium | Luo et al. 2021, *Normal Graph* (Neurocomputing) | Widely cited skeleton GCN baseline | https://doi.org/10.1016/j.neucom.2019.12.148 | [listed](https://www.researchgate.net/publication/347139479) |
| 8 | Medium | 2023, *MP-Abr: abnormal behaviour in multi-person scenarios* (MTAP) | Multi-person handling | https://doi.org/10.1007/s11042-023-17667-x | [yes](https://www.researchgate.net/publication/375892687) |
| 9 | Medium | Pang et al. 2022, *Predicting skeleton trajectories using a Skeleton-Transformer* | Trajectory modelling | https://doi.org/10.1007/s00530-022-00915-9 | [listed](https://www.researchgate.net/publication/359608536) |
| 10 | Medium | Yue et al. 2022, *Action recognition based on RGB and skeleton data sets: A survey* | Fusion overview | https://doi.org/10.1016/j.neucom.2022.09.071 | [request](https://www.researchgate.net/publication/363727202) |
| 11 | Medium | Li et al. 2026, *Star-Shaped Multi-Person Interaction Graph* (TIP) | Group skeleton interaction | https://doi.org/10.1109/tip.2026.3684390 | [request](https://www.researchgate.net/publication/404057081) |
| 12 | Low | 2025, *Abnormal behavior detection in controlled areas based on body-part trajectory tracking* (JINST) | Body-part tracking | https://doi.org/10.1088/1748-0221/20/05/p05032 | not checked |
| 13 | Low | 2025, *A Review of Abnormal Behaviour Detection in Crowd* (Expert Systems) | Recent survey, optional | https://doi.org/10.1111/exsy.70013 | not checked |
| 14 | Low | Ding & Zhang 2024, *Key point trajectory prediction ... posture falls* (Network) | Falls only; skip unless easy | https://doi.org/10.1080/0954898x.2024.2412673 | not checked |

Checked and **not** locked: WeDePE (free copy above), Cormier 2021 (CVF), COSKAD (arXiv 2301.09489),
Garcia-Cobo 2023 and CAL 2022 (open at publisher). Dropped: *Skeleton-Based Detection of Abnormalities*
(TransAI 2020), which is about rehabilitation exercises, not surveillance.

## 6. Results not to reuse as baselines

| Source | Claim | Problem |
|---|---|---|
| `old_paper` (untracked file, *Optimized ST-GCN ... Edge Computing*) | 98.1% on UCF-Crime with "ST-GCN-OGRU" | **Never mentions skeletons, joints or a pose estimator**, though ST-GCN is defined on skeleton graphs. The graph nodes are not specified. Test confusion counts look small (13-38 samples per class) and only six classes are named. Provenance unknown |
| Matei 2022 | 0.382 on 13 crimes | Split by trajectory, so one video may sit on both sides (the paper does not establish grouping). Labels come from an anomaly model, which rule 2 would not allow as evaluation truth for descendants |
| Ansari & Singh 2022 | 96.87% | Measured on a synthesized dataset, per its abstract |
| [JISE 2026](https://journal.ugm.ac.id/v3/JISE/article/view/31998) | 97.35% looking-around | Custom dataset |
| [Naeen 2025](https://arxiv.org/abs/2510.22056) | 92.41% on UCF-Crime | Five-class subset, not the 14-class task |
| Any video-level AUC on UCF | various | Ganesh: much of it is available before the event. Kim & Lim: much of it comes from detection quality |

## 7. Proposed path (waiting on §8, question 1)

Nothing here has run.

1. **Audit:** render overlays, add track IDs on the 108 dense videos, compute Kim & Lim's quality
   variables, and open HR-Crime to check its frame rate and whether its videos map to our folds.
2. **Posture module:** a small classifier from normalized joints to a reduced label set, trained and
   scored on AVA with independent human labels. Check AVA's licence and completeness first. Score the
   estimator itself on COCO. On UCF, report only descriptive distributions and overlays, explicitly
   as unverified predictions.
3. **Value for crime classification** (only if D-67's closure is reopened): feed per-frame posture
   probabilities to the D-53 head, which works with 8 sparse frames and needs no tracks. Compare
   under D-65's paired protocol: coverage-only, box-only, posture-only, RGB, RGB+box, RGB+posture,
   with Fighting/Assault reported separately and the pre-event check on videos with official intervals.
   All results carry the inherited-label name (D-40).
4. **Sequence models** (PoseC3D, SkeleTR) only if step 3 or the audit justifies new extraction.

## 8. Questions for the owner

**1. What should "posture detection" produce?** This sets all later work.

| Option | Output | How it is checked | Cost and risk |
|---|---|---|---|
| A. Posture labels per person | Standing, sitting, lying, etc. on each person, plus overlays | AVA labels; no direct UCF check | Modest; AVA's movie footage may transfer poorly |
| B. Posture as a crime-classifier input | Maybe a higher 14-class score | D-65 paired comparison | Reopens D-67; Ganesh suggests gain may be small |
| C. A, then B **(recommended)** | Both, in order | Both | A gives a usable result even if B is negative, and B's negative would still be reportable |

2. **Who asked for this and by when?** Faculty requirement, paper section or demo? Is there a date?
3. **Which labels matter?** I suggest stand, sit, lie, bend/crouch, walk/run, fall. Static body shape
   only, or movement too? Do you also want two-person contact (grab, push, hit)?
4. **May we reopen the closed classifier track (D-67) for option B?**
5. **Is AVA acceptable as the label source?** Or would your team hand-label a small UCF check set
   (say 500 people, roughly 6-8 person-hours at Cormier's 40-60 s each)? That is the only direct evidence
   on UCF, but D-04 removed manual annotation from the core plan, so it is your call. Recommendation: AVA
   for training, plus the small check set if you can spare the time.
6. **Will code or the demo be released publicly?** The manifest records YOLO11's licence as AGPL-3.0.
   This is fine internally, but public release could make it matter.
7. **Is `old_paper` yours, and is its 98.1% meant to be a target?** I recommend not using it (§6).
8. **Which publishers does your library cover?** IEEE, Elsevier, Springer? I can reorder §5 to match.
9. **The message ended with `/mod`.** That is not a command available here, so I ignored it. What did
   you intend?

## 9. How this search was done

Searched a paper index (arXiv and biomedical sources) and the general web, then Crossref for
publisher-hosted work. Thirty likely-closed papers were checked one by one (§5). This is broad but not
exhaustive: IEEE Xplore and Scopus were not searched directly, and open-copy results depend on what
those databases record. Most entries are read at abstract level, as marked.
