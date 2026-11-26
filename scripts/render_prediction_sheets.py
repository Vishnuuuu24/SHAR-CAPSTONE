#!/usr/bin/env python3
"""Render contact sheets of the D-53 classifier's wrong and right validation predictions (one per class each).

Reads the stored D-53 validation probabilities (no model is run) and 8 frames per chosen video from the D-34 frame
blob. Writes <output>/{wrong,correct}/<Class>__<video>.png, all_wrong.png, all_correct.png and index.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.prediction_sheets import pick_examples, render_sheet, spread, stack, video_summaries  # noqa: E402

PROBS = "results/p1c/p1c-late-fusion/dinov2_probs.csv"
ANNOTATION = ("Datasets/UCF Crime Official/annotations/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/"
              "Temporal_Anomaly_Annotation.txt")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probs", type=Path, default=REPO_ROOT / PROBS)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "results/p1c/prediction-sheets-d53")
    args = ap.parse_args(argv)
    from core.frame_batches import FRAME_INDEX, VIDEOS_CSV
    from core.p1c_background_mask import load_intervals
    from core.validation_frames import CLOSURE
    from data.video_blob import VideoBlobReader

    reader = VideoBlobReader(REPO_ROOT / CLOSURE, verify_digest=False)
    with (REPO_ROOT / VIDEOS_CSV).open(newline="") as h:
        span = {r["file_name"].rsplit(".", 1)[0]: (int(r["first_row_index"]), int(r["kept_frame_count"])) for r in csv.DictReader(h)}
    source_index = np.load(REPO_ROOT / FRAME_INDEX, mmap_mode="r")["source_frame_index"]
    ann = REPO_ROOT / ANNOTATION
    intervals = load_intervals(ann) if ann.exists() else {}

    summaries = video_summaries(args.probs)
    wrong, right = pick_examples(summaries)
    lines = ["# D-53 classifier: example predictions on validation videos", "",
             f"Source: `{PROBS}` (stored D-53 validation probabilities, 4 folds x 3 seeds; each video is seen by 3-12 runs).",
             f"{len(summaries)} videos, {sum(s['wrong_runs'] * 2 >= s['runs'] for s in summaries)} wrong in at least half their runs. "
             "Labels are video-level (D-40); 8 frames spread over the whole video; red border = inside the official interval.", ""]
    for kind, picks in (("wrong", wrong), ("correct", right)):
        (args.output / kind).mkdir(parents=True, exist_ok=True)
        sheets = []
        lines += [f"## {kind}", "", "| true | predicted | wrong runs | top probability | video | sheet |", "|---|---|---|---|---|---|"]
        for s in picks:
            first, n = span[s["video"]]
            pos = spread(n)
            nums = [int(source_index[first + p]) for p in pos]
            spans = intervals.get(s["video"], [])
            ev = [any(a <= f <= b for a, b in spans) for f in nums]
            note = ("red border = frame inside the official anomaly interval" if spans
                    else "no official interval for this video (labels are video-level)")
            img = render_sheet(s, [reader.frame(first + p) for p in pos], nums, ev, note)
            name = f"{kind}/{s['true']}__{s['video']}.png"
            img.save(args.output / name)
            sheets.append(img)
            lines.append(f"| {s['true']} | {s['majority_pred']} | {s['wrong_runs']}/{s['runs']} | "
                         f"{s['mean_probs'][s['majority_pred']]:.2f} | {s['video']} | `{name}` |")
        stack(sheets).save(args.output / f"all_{kind}.png")
        lines.append("")
    (args.output / "index.md").write_text("\n".join(lines) + "\n")
    print(f"{len(wrong)} wrong + {len(right)} correct sheets -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
