# Presentation showcase (2026-11-24)

`index.html`: a local page that plays three UCF clips the final E3 model (D-76) classifies correctly. It shows
the model's per-second reading, a latched "Possible <class> detected" alert, tagged frames and the whole-video
verdict. Open the file in a browser. It needs `media/`, which holds licensed UCF pixels, is git-ignored and must
not be published (UCF raw-media redistribution is prohibited).

| Clip | Label | Whole-video E3 | Members (none trained on clip) | Alert fires | Official event |
|---|---|---|---|---|---|
| Explosion021_x264 (26 s) | Explosion | Explosion 0.864 | 9 (fold 2) | 9.2 s | 4.5–9.0 s |
| RoadAccidents086_x264 (30 s) | RoadAccidents | RoadAccidents 0.937 | 36 (all folds) | 7.2 s | none (not an AD test video) |
| Robbery044_x264 (16 s) | Robbery | Robbery 0.927 | 9 (fold 4) | 7.2 s | none |

**Selection (rule 4):** these clips are chosen examples, not a sample. Candidates were held-out, correct, non-conflicting
E3 replay answers from `e3-comparison-20261116/answers/ucf`, ranked by confidence and limited to 15–150 s. Timelines
were built for 14 of them (`timelines/`). The final three were picked by hand for short length, class variety and
an event that is visible on screen. The headline remains test accuracy 46.2%, macro-F1 0.455 (D-73).

**What a window score is:** E3 re-applied to a trailing 32-selected-frame (~5.3 s) window every 6 selected frames (~1 s).
Same members, same pooling. This is a display aid, not an evaluated quantity. On 13 of the 14 timelines, every window's
top class equals the label from the first window onward (Arson039: 96%). The classifier reads the setting, not the
moment, so Fighting003 says "Fighting" from 0 s although the official event starts at 60.7 s. Explosion021 is the one clip whose
confidence rises through the event (0.31 → 0.81 over 5–8 s).

**Checks:** for all 14 clips, re-extracted whole-video vectors and probabilities equal the saved E3 replay exactly
(max |Δ| = 0.0; `build_summary.json`). GPU time on the MIG slice: 2.2–15.3 s per clip, about 5× faster than real time.
Pod log: `pod-logs/`.

**Not done:** bounding boxes. E3 has no localisation output. The stock detector cannot say *which* person is committing
the act, and D-71's frame-level anomaly scores time events convincingly on only 2 of the 8 candidate test clips that
have scores. Any box would therefore be decoration.

Rebuild: `scripts/build_showcase_timeline.py` (GPU pod), then `scripts/build_showcase_page.py` (CPU).
Revision of the timeline run: 3c6dad9.
