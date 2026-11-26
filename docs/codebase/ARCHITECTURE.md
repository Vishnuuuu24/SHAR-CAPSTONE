# Architecture

## 1) Architectural Style

- Primary style: a research pipeline with fail-closed data/provenance gates and script-driven experiments.
- Constraints: weak inherited frame labels must not be treated as clean ground truth; Test data cannot guide model selection; every material run needs immutable provenance.

## 2) System Flow

```text
UCF mirror + official intervals
  -> P0B frame/source mapping and quarantine
  -> P1B grouped train/validation manifests + two Test views
  -> P1C registered classifiers and validation-only selection
  -> per-class/macro metrics + video-clustered CIs
  -> immutable result artifacts
```

The headline Test view contains Normal-video frames plus anomalous frames inside official intervals. The full inherited-folder Test view is retained only as a noisy-proxy sensitivity analysis.

## 3) Model Roadmap

| Component | Status | Purpose / why |
|---|---|---|
| ResNet50 C5 -> global-average-pooling -> linear head | Implemented P1C control | Establishes a conventional, auditable frame-classification baseline. |
| Same C5 -> GAP -> MLP | Implemented P1C control | Tests whether extra classifier capacity helps while holding the backbone feature map fixed. |
| ResNet50 C5 -> GAP -> projection -> LSTM | Implemented P1C temporal control | Tests whether short, source-video-local temporal context helps over a single frame. |
| SEMSCNN-inspired 2-D multiscale-SE head | Planned P3A | Tests multiscale spatial filtering and channel attention on the real C5 spatial map; it is an explicit visual adaptation, not copied EEG semantics. |
| CE, weighted CE, focal, Balanced Softmax | Planned P3B | Compares ways to avoid majority classes dominating a 14-label loss. |
| Stock YOLO11 ROI proposals | Planned P3C | Tests whether cropping likely relevant regions improves the same classifier; boxes are proposals, never UCF ground truth. |
| Median, Gaussian, bilateral, NLM | Implemented P2A interface | Cheap interpretable controls for denoising; each gets the same noisy input. |
| Denoising VAE + pinned pretrained restoration model | Planned P2B | Learns restoration/noise structure and provides a modern comparator; the historical ESVDAE name is blocked until its equation is specified. |

## 4) Reused Patterns

| Pattern | Where | Why |
|---|---|---|
| Fail-closed validation | `data/ucf_intervals.py`, `core/p1c_aggregation.py` | Bad provenance/metrics should stop a run rather than be silently tolerated. |
| Immutable run lifecycle | `core/run_lifecycle.py` | Keeps completed, failed, and invalid attempts reviewable. |
| Registered configuration | `configs/` and script loaders | Makes parameters inspectable instead of hidden defaults. |

## 5) Known Architectural Risks

- The 64x64 mirror is information-limited; upscaling cannot create surveillance detail.
- Original-video frame-index alignment and prior P0B/P1B artifacts require real-data revalidation before training.

## 6) Evidence

- `docs/ARCHITECTURE.md`
- `docs/PHASE_PLAN.md`
- `models/classifiers.py`
- `models/restoration.py`
- `data/views.py`
