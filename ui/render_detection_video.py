#!/usr/bin/env python3
"""Render an mp4 of the stock YOLO11m-pose ROI teacher on one UCF video: person boxes plus the union ROI box."""
import argparse, subprocess
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

def render(src: Path, out: Path, weights: Path, max_frames: int, threads: int, scale: int, ffmpeg: str) -> int:
    torch.set_num_threads(threads)
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)) * scale, int(cap.get(4)) * scale
    raw = out.with_suffix(".raw.mp4")  # OpenCV here has no H.264; re-encode with ffmpeg below
    wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while n < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_CUBIC)
        r = model.predict(fr, verbose=False, conf=0.25, device="cpu")[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(boxes, r.boxes.conf.cpu().numpy()):
            cv2.rectangle(fr, (int(x1), int(y1)), (int(x2), int(y2)), (80, 220, 80), 2)
            cv2.putText(fr, f"person {c:.2f}", (int(x1), max(12, int(y1) - 4)), 0, 0.5, (80, 220, 80), 1)
        if len(boxes):
            x1, y1, x2, y2 = boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
            cv2.rectangle(fr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 140, 255), 2)
            tag = "ROI = union box"
        else:
            tag = "no person -> full-frame fallback"
        cv2.putText(fr, f"{tag} | frame {n}", (6, h - 8), 0, 0.55, (255, 255, 255), 1)
        wr.write(fr); n += 1
    wr.release()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    raw.unlink()
    return n

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--weights", type=Path, default=Path(".cache/weights/yolo11m-pose-29b17eaf.pt"))
    ap.add_argument("--max-frames", type=int, default=240); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--scale", type=int, default=2); ap.add_argument("--ffmpeg", default="ffmpeg")
    a = ap.parse_args()
    print("frames written:", render(a.src, a.out, a.weights, a.max_frames, a.threads, a.scale, a.ffmpeg))
