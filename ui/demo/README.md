# Testing the SHAR classifier on your own video

Give the evaluated D-53 classifier a video and get its answer: predicted class, top-5 probabilities, a per-window
timeline, sample frames, and a comparison with the true label if you give one. There are three ways in, all using the
same model code.

| Way in | Best for | GPU slice |
|---|---|---|
| `scripts/dgx/test_video.sh` | Trying videos quickly; folders with labels | Freed as soon as it finishes |
| `notebooks/test_video.ipynb` | Working in a notebook | Held while open; Jupyter stops itself after 30 idle minutes |
| Upload page (below) | Showing someone in a browser | Held until you stop it |

## Quick start: command line

Copy videos anywhere under the home directory (`my_videos/` at the repo root is git-ignored), then on the head node,
from the repo root:

```bash
scripts/dgx/test_video.sh my_videos/clip.mp4                      # one video, no label
scripts/dgx/test_video.sh my_videos/clip.mp4 --label Fighting     # compare with the true class
scripts/dgx/test_video.sh my_videos/ --labels my_videos/labels.csv    # a folder; CSV columns: video,label
scripts/dgx/test_video.sh by_class/ --labels-from-folders         # by_class/Fighting/a.mp4 is labelled Fighting
scripts/dgx/test_video.sh --self-check                            # check the tool against recorded scores (about 5 min)
```

Classes: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Normal, RoadAccidents, Robbery, Shooting,
Shoplifting, Stealing, Vandalism.

Each run writes `report.html`, `results.json` and `results.csv` to `results/video-tests/<UTC time>/`. Reports embed video
frames, so that folder is git-ignored; do not publish them. If the slice is in use, the command says what is running and
starts nothing.

## Quick start: notebook

```bash
scripts/dgx/notebook.sh    # prints the login, port-forward and stop steps
```

## What runs

- **Model (default): E3, the final 14-class model (D-76).** It averages 36 logistic members: 3 arms (DINOv2-L;
  DINOv2-B + SigLIP; DINOv2-B + V-JEPA 2) x 4 folds x 3 seeds, over whole-video feature averages. Members load from
  `results/demo/e3-comparison-20261116/models/`, digest-checked against its `export.json`. At startup the encoder
  source files are checked to be byte-identical to the snapshot E3 ran with. Code: `core/e3_inference.py`.
  Test scores: macro-F1 0.455, accuracy 46.2%, top-3 72.2% (D-73).
- **Frames:** every 5th frame, JPEG q95 round trip as in the evaluated frame blob (K-33), the registered 224
  preprocessing, no frame cap (the E3 deployment rule). With it, uploads reproduce the saved E3 replay exactly.
- **Prototype boxes (owner choice 2026-11-25):** the D-78 control detector (seed 0, registered threshold 0.701; official
  test frame AUC 0.813, event recall 0.718, 19.1 false alarms/h) decides when a moment is flagged; a box around the
  area of strongest motion marks where; E3's whole-video class labels it. Option B (owner, 2026-11-25): the box shows only for the first 2 s of
  each flagged stretch, frozen at the stretch's first box, because later boxes drift onto unrelated motion; the
  rule was chosen after viewing the three example clips and the page says so. Outside that window the video is
  untouched. The detector trained on its 1,610 Anomaly-Detection training videos, so the page warns when an upload
  is one of them. Code: `core/demo_detector.py`; GPU check: `results/demo/e3-live-demo-20261124/detector_smoke.json`.
- **Honesty rule:** a UCF-Crime video is recognised by its *content* (SHA-256), not its name. Only members whose fold
  never trained on it are used, which is the test condition. Renaming the file does not change this. Other videos use all 36.
- **Not a claim:** the timeline scores consecutive ~5-second windows with the same members. It shows what each
  stretch resembles; the model never saw temporal labels, and it usually reads the setting rather than the moment.
- **Older model:** `--model d53` runs the previous D-53 demo (DINOv2-B + attention heads, test macro-F1 0.414,
  accuracy 41.8%, D-57), with weights in `.cache/demo/d53_heads.pt` from `scripts/export_demo_heads.py`.

### Older model (D-53, `--model d53`; also used by `scripts/test_video.py` and the notebook)

- **Model:** frozen DINOv2-B (registered weights) and the 12 D-53 heads (4 folds x 3 seeds), recreated by
  `scripts/export_demo_heads.py` with the D-57 recipe and checked against the recorded validation scores
  (`results/demo/export.json`). Weights: `.cache/demo/d53_heads.pt` (git-ignored).
- **Frames:** every 5th frame, JPEG quality 95 as in the frame blob every cached feature came from (K-33), the registered
  224 preprocessing, thinned to 2,400 for very long videos.
- **Honesty rule:** if a file's contents or name identify a UCF protocol video, only heads whose fold never trained or
  validated on it are used, which matches the test condition, and the dataset label overrides any label you give. A
  renamed copy is still recognised. A re-encoded or trimmed copy under a new name is not, so it would look like unseen
  footage; do not test on those. On unseen footage all 12 heads are averaged, an ensemble never scored separately.
- **What counts in the score:** one entry per unique file, only with a label, and never a video every fold saw. Under 30
  videos the report says it is an example, not a measurement.
- **Not a claim:** the timeline is per 8-frame window and only shows what each stretch resembles; the model never
  saw temporal labels. Scores: test macro-F1 0.414, accuracy 41.8% (D-57), inherited video-level labels, not event-only.
  The model always picks one of 14 classes and was trained on 320x240 CCTV; other footage is outside what was evaluated.
- **Self-check:** real MP4s through the whole path, compared with recorded validation scores. Last result on fold 1:
  all three heads reproduced their recorded macro-F1 with identical predictions on 56 videos, minimum frame cosine
  0.99997 (`results/demo/self_check_fold1.json`). It confirms the tool matches the evaluated model, not that the model is good.

## Upload page (GPU slice; occupies it while the page is up)

On the head node, from the repo root:

```bash
scripts/dgx/start_pod_job.sh demo-server './.venv/bin/python ui/demo/server.py --port 8765'
kubectl logs -f <pod-name>                     # wait for {"ready": true, ...}, then Ctrl-C
kubectl port-forward pod/<pod-name> 8765:8765  # leave running
```

On your laptop:

```bash
ssh -N -L 8765:localhost:8765 <user>@<internal-ip>
```

Open <http://localhost:8765>. When finished, stop the port-forward and run
`scripts/dgx/stop_pod_job.sh <pod-name>` so the GPU slice is free again.

## Files

- `ui/demo/index.html`: the page (no build step)
- `ui/demo/server.py`: standard-library HTTP server (`GET /`, `GET /health`, `POST /predict`)
- `core/e3_inference.py`: E3 encoders, members, held-out rule, timeline (default model)
- `core/demo_inference.py`: the older D-53 path (`--model d53`)
- `scripts/smoke_demo_e3.py`: GPU check of the E3 path against the saved E3 replay and the HTTP server
- `core/demo_inference.py`: decoding, features, head ensemble, held-out rule, timeline
- `core/video_report.py`, `core/video_selfcheck.py`: labels, fair score, HTML report; the end-to-end check
- `scripts/test_video.py`, `scripts/dgx/test_video.sh`: the command line and its pod wrapper
- `notebooks/test_video.ipynb`, `scripts/dgx/notebook.sh`: the notebook and its launcher
- `scripts/build_demo_digests.py`: builds `results/demo/protocol_digests.json`, which recognises renamed protocol videos
- `scripts/export_demo_heads.py`: recreates and verifies the 12 heads
