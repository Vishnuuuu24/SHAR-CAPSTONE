#!/usr/bin/env python3
"""Assemble the local presentation page from saved showcase timelines (CPU only, no model run).

Reads `<dir>/timelines/<key>.json` written by scripts/build_showcase_timeline.py, copies each source MP4 and a few
tagged frames into `<dir>/media/` (licensed UCF pixels: git-ignored, never published), and fills
ui/showcase/template.html into `<dir>/index.html`. Open that file in a browser; no server is needed.

  .venv/bin/python scripts/build_showcase_page.py --dir results/demo/showcase-20261124 \\
      Explosion021_x264 RoadAccidents086_x264 Robbery044_x264
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.showcase_timeline import first_alert  # noqa: E402

E3 = REPO / "results/demo/e3-comparison-20261116"
ANNOTATIONS = REPO / "Datasets/UCF Crime Official/annotations/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt"
TEMPLATE = REPO / "ui/showcase/template.html"


def official_events(key: str, fps: float) -> list[list[float]]:
    """Official test-interval pairs in seconds, or [] when the video has none."""

    for line in ANNOTATIONS.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == f"{key}.mp4":
            n = [int(x) for x in parts[2:6]]
            return [[a / fps, b / fps] for a, b in ((n[0], n[1]), (n[2], n[3])) if a >= 0 and b >= 0]
    return []


def tagged_frames(src: Path, key: str, windows: list[dict], duration: float, media: Path, count: int) -> list[dict]:
    import cv2

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    for i in range(count):
        t = duration * (i + 0.5) / count
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, bgr = cap.read()
        if not ok:
            continue
        name = f"{key}-{i}.jpg"
        cv2.imwrite(str(media / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        current = [w for w in windows if w["end_s"] <= t]
        top = current[-1]["top3"][0] if current else None
        out.append({"t_s": t, "src": f"media/{name}", "label": top and top["label"], "prob": top and top["prob"]})
    cap.release()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--alert-threshold", type=float, default=0.5)
    ap.add_argument("--alert-run", type=int, default=3, help="consecutive one-second readings needed")
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args(argv)

    config = json.loads((E3 / "config.json").read_text())
    rows = {r["key"]: r for r in config["ucf"]}
    media = args.dir / "media"
    media.mkdir(parents=True, exist_ok=True)
    videos = []
    for key in args.keys:
        tl = json.loads((args.dir / "timelines" / f"{key}.json").read_text())
        if not tl["parity"]["same_top1"] or tl["parity"]["max_abs_prob_diff"] > 1e-6:
            raise SystemExit(f"{key}: whole-video result does not reproduce the saved E3 replay; refusing to show it")
        src = Path(rows[key]["path"])
        shutil.copyfile(src, media / f"{key}.mp4")
        heads = tl["members_used"]
        videos.append({
            "key": key, "label": tl["label"], "src": f"media/{key}.mp4", "duration_s": tl["duration_s"],
            "window_s": tl["windows"][0]["end_s"] - tl["windows"][0]["start_s"],
            "windows": [{k: w[k] for k in ("start_s", "end_s", "top3")} for w in tl["windows"]],
            "video": tl["video"], "alert": first_alert(tl["windows"], args.alert_threshold, args.alert_run),
            "events": official_events(key, tl["fps"]),
            "frames": tagged_frames(src, key, tl["windows"], tl["duration_s"], media, args.frames),
            "provenance": {
                "Video": key,
                "Members used": f"{heads} of 36 (folds {', '.join(map(str, tl['selected_folds']))}: none trained on this clip)",
                "Reproduces saved E3 replay": f"yes, max |Δp| = {tl['parity']['max_abs_prob_diff']:.1g}",
                "E3 config": tl["e3_config_sha256"][:16],
                "Source SHA-256": tl["source_sha256"][:16],
                "Code revision": tl["git_revision"][:10],
            },
        })
    data = {
        "alert_threshold": args.alert_threshold, "alert_run": args.alert_run,
        "chips": [
            "Model <b>E3</b>: DINOv2-L · DINOv2-B+SigLIP · DINOv2-B+V-JEPA 2",
            "14-class test: accuracy <b>46.2%</b> · macro-F1 <b>0.455</b> · top-3 <b>72.2%</b>",
            "Clips shown: <b>chosen examples it gets right</b>",
        ],
        "notes": [
            "These clips were picked because the model classifies them correctly. Across the full 14-class test it is "
            "right 46.2% of the time (macro-F1 0.455; correct class in its top three 72.2%). It is strongest on "
            "explosions, road accidents and shoplifting and weakest on Shooting, Assault, Fighting and Abuse.",
            "No answer here comes from a model member that trained on the clip. Each clip lists which members were used.",
            "Each second, the model reads the last ~5 s of frames. These are the same weights that produce the "
            "whole-video verdict, and the verdict is the reported prediction. Outputs were computed on the DGX GPU "
            "(about 5× faster than real time) and replay in sync with the video.",
            f"The alert appears once the same incident type has led for {args.alert_run} consecutive seconds at "
            f"≥{args.alert_threshold:.0%}. It names the <i>type</i> of incident. Training labels cover whole videos, "
            "so the model does not pinpoint when or where something happens, and it often recognises the setting "
            "before the event itself.",
            "No bounding boxes are drawn: the model classifies the whole frame and has no localisation output.",
        ],
        "videos": videos,
    }
    page = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data))
    (args.dir / "index.html").write_text(page)
    print(json.dumps({"page": str(args.dir / "index.html"),
                      "alerts": {v["key"]: v["alert"] for v in videos}, "events": {v["key"]: v["events"] for v in videos}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
