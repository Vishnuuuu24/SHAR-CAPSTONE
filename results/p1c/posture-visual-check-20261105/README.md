# Posture visual check, 2026-11-05

One-person informal inspection of cached YOLO11m-pose skeletons. Not ground truth, not posture-model
accuracy. It judges only whether the drawn skeleton follows the visible body.

- Input: `results/p1c/p1c-roi-cache/detections.jsonl`, sha256 in the `Data` line below.
- Script: `render_overlays.py <output_dir>`, run from the repo root. Sampling seed 20261105; at most one
  frame per video per sheet; person sheets are stratified by the height of the largest person in the frame.
- The JPGs are licensed video derivatives: keep them local and Git-ignored.

| Sheet (largest person height) | Good | Partial | Fail |
|---|---|---|---|
| under 40 px | 3 | 1 | 2 |
| 40-60 px | 4 | 2 | 0 |
| 60-80 px | 4 | 2 | 0 |
| 80-120 px | 5 | 0 | 1 |
| over 120 px | 4 | 1 | 1 |
| Total (30) | 20 | 6 | 4 |

Good = skeleton follows the body well enough to read coarse posture. Partial = correct on what is visible
but the body is hidden or tangled. Fail = wrong shape, wrong person mix, or nothing readable.
Problems seen: overhead camera (Normal f840, f5950, f3790), hidden body (Burglary017, Robbery043), crowd
mixing people (Shoplifting036), blur (Explosion040), motorcycle rider (RoadAccidents020), tangled crawling
pose (Abuse001 f1530), crossed shoulders from behind (Normal f65).

No-person sheet (12 frames): 9 looked empty, 2 clearly had people the model missed (Assault010 dormitory
room, Explosion042 distant crowd), 1 unclear (Arson028).

Data: detections.jsonl sha256 4a41a37f70089a2f91d14fc18e1e6227600b9d4d76dcf92192ce78d8422cafe2
Code revision at run: 4c710d8 plus the uncommitted script saved here.
