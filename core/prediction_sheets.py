"""Contact sheets of the D-53 classifier's validation predictions (visual evidence for the write-up, not a metric).

`video_summaries` folds the stored per-(fold, seed, video) probabilities into one record per video: how many runs saw
it, how many got it wrong, the majority predicted label and the run-mean probabilities. `pick_examples` takes, per true
class, the most confidently wrong video and the most confidently right one. `render_sheet` draws 8 frames spread over
the whole video from the D-34 frame blob, with the labels on top and a red border on frames inside an official
anomaly interval (where the video has one).
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

FRAMES_PER_SHEET = 8
THUMB = (240, 180)


def video_summaries(probs_csv: Path) -> list[dict]:
    rows = list(csv.DictReader(Path(probs_csv).open()))
    classes = [k[2:] for k in rows[0] if k.startswith("p_")]
    by = defaultdict(list)
    for r in rows:
        by[r["source_video_id"]].append(r)
    out = []
    for vid, rs in sorted(by.items()):
        true = rs[0]["true_label"]
        preds = Counter(r["pred_label"] for r in rs)
        mean = np.mean([[float(r[f"p_{c}"]) for c in classes] for r in rs], axis=0)
        out.append({"video": vid, "true": true, "runs": len(rs), "wrong_runs": sum(r["pred_label"] != true for r in rs),
                    "majority_pred": preds.most_common(1)[0][0], "mean_probs": dict(zip(classes, mean.tolist()))})
    return out


def pick_examples(summaries: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """Per true class: the most confidently wrong video (wrong in >= half its runs) and the most confidently right one."""

    wrong, right = [], []
    for cls in sorted({s["true"] for s in summaries}):
        vids = [s for s in summaries if s["true"] == cls]
        w = [s for s in vids if s["wrong_runs"] * 2 >= s["runs"] and s["majority_pred"] != cls]
        if w:
            wrong.append(max(w, key=lambda s: (s["wrong_runs"] / s["runs"], s["mean_probs"][s["majority_pred"]])))
        r = [s for s in vids if s["wrong_runs"] == 0]
        if r:
            right.append(max(r, key=lambda s: s["mean_probs"][cls]))
    return wrong, right


def spread(n: int, k: int = FRAMES_PER_SHEET) -> list[int]:
    """k positions spread evenly over 0..n-1, both ends included."""

    return sorted({int(round(x)) for x in np.linspace(0, n - 1, min(k, n))})


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_sheet(summary: dict, frames: Sequence[np.ndarray], frame_numbers: Sequence[int], in_event: Sequence[bool],
                 note: str = ""):
    from PIL import Image, ImageDraw

    tw, th = THUMB
    pad, head = 4, 74
    sheet = Image.new("RGB", (pad + len(frames) * (tw + pad), head + th + pad), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    big, small = _font(18), _font(13)
    top3 = sorted(summary["mean_probs"].items(), key=lambda kv: -kv[1])[:3]
    ok = summary["majority_pred"] == summary["true"]
    d.text((pad, 4), summary["video"], fill=(235, 235, 235), font=big)
    d.text((pad + 230, 4), f"TRUE: {summary['true']}", fill=(140, 200, 255), font=big)
    d.text((pad + 520, 4), f"PREDICTED: {summary['majority_pred']}", fill=(120, 220, 120) if ok else (255, 110, 110), font=big)
    d.text((pad, 30), f"wrong in {summary['wrong_runs']}/{summary['runs']} runs   |   top-3 mean probability: "
           + ",  ".join(f"{c} {p:.2f}" for c, p in top3), fill=(210, 210, 210), font=small)
    d.text((pad, 50), note or "red border = frame inside the official anomaly interval", fill=(170, 170, 170), font=small)
    for i, (fr, num, ev) in enumerate(zip(frames, frame_numbers, in_event)):
        x = pad + i * (tw + pad)
        thumb = Image.fromarray(fr).resize(THUMB)
        sheet.paste(thumb, (x, head))
        if ev:
            d.rectangle([x, head, x + tw - 1, head + th - 1], outline=(235, 40, 40), width=4)
        d.text((x + 5, head + 4), f"f{num}", fill=(255, 255, 0), font=small)
    return sheet


def stack(images: Sequence, gap: int = 6):
    from PIL import Image

    w = max(im.width for im in images)
    out = Image.new("RGB", (w, sum(im.height for im in images) + gap * (len(images) - 1)), (10, 10, 12))
    y = 0
    for im in images:
        out.paste(im, (0, y))
        y += im.height + gap
    return out
