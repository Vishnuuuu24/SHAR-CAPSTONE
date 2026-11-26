"""Prototype "when + where" overlay for the live demo: the D-78 control anomaly detector decides WHEN, a motion
heuristic decides WHERE, and E3 (D-76) supplies WHAT. Owner choice 2026-11-25: motion-area box, prototype label.

WHEN: the D-78 `control` detector, seed 0, exactly as evaluated: DINOv2-B ++ SigLIP frame features (one frame per
15 source frames = every 3rd stride-5 frame), standardised with its saved training statistics, causal context
(deviation from the running mean), MLP, sigmoid, causal mean over 5 steps, alarm at >= its registered threshold
(99th percentile of held-out normal training frames). Official test read: frame AUC 0.813, event recall 0.718,
19.1 false alarms per hour (results/detect/followup-20261112/ucf-002/test-control.json). It is weak at finding
the moment inside crime videos (anomaly-video frame AUC about 0.61, D-71), so alarms are often early or late.

WHERE: not a learned quantity. The box is the largest region of frame-to-frame change during the alarm step,
smoothed over time. It marks activity, not "the crime"; the page says so.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

DETECTOR_DIR = "results/detect/followup-20261112/ucf-002"
VARIANT, SEED = "control", 0
STEP = 3                 # detector frame = every 3rd stride-5 frame (one per 15 source frames)
SMOOTH = 5               # causal mean window, as evaluated
MOTION_WIDTH = 160       # motion is computed on a 160-px-wide grey copy
MOTION_DIFF = 25         # grey-level change counted as motion (display heuristic, not a research threshold)
BOX_EMA = 0.5            # weight of the new box when smoothing box corners over steps


def causal_mean(scores: np.ndarray, window: int) -> np.ndarray:
    from core.detect_metrics import causal_mean as cm

    return cm(scores, window)


class Detector:
    """The evaluated D-78 control scorer (seed 0) with its own standardisation and alarm threshold."""

    def __init__(self, repo: Path, device="cpu"):
        import torch

        from core.detect_followup import FrameScorer
        from core.e3_inference import sha256_file

        root = repo / DETECTOR_DIR
        meta = json.loads((root / f"{VARIANT}-seed{SEED}.json").read_text())
        ckpt_path = root / f"{VARIANT}-seed{SEED}.pt"
        if sha256_file(ckpt_path) != meta["checkpoint_sha256"]:
            raise RuntimeError("detector checkpoint digest mismatch")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.mean = np.asarray(ck["mean"], dtype=np.float32)
        self.std = np.asarray(ck["std"], dtype=np.float32)
        self.model = FrameScorer(2 * len(self.mean))
        self.model.load_state_dict(ck["state"], strict=True)
        self.model.eval().to(device)
        self.device = device
        report = next(r for r in json.loads((root / f"test-{VARIANT}.json").read_text()) if r["seed"] == SEED)
        self.threshold = float(report["threshold"])
        self.test_report = {k: report[k] for k in ("frame_auc", "event_recall", "false_alarms_per_hour") if k in report}

    def scores(self, b: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Smoothed anomaly probability per detector step from per-frame DINOv2-B and SigLIP features ([N, 1536] each)."""

        from core.detect_model import add_context, score_frames

        x = np.concatenate([b, s], axis=1).astype(np.float16).astype(np.float32)  # stored-feature precision
        x = add_context((x - self.mean) / self.std, "causal")
        return causal_mean(score_frames(self.model, x, self.device), SMOOTH)


def detector_training_digests(repo: Path, ucf_rows: list[dict]) -> set[str]:
    """Content digests of every video in the detector's TRAINING split (its in-sample videos)."""

    by_path = {}
    for r in ucf_rows:
        for alias in r["aliases"]:
            by_path[alias] = r["sha256"]
    out = set()
    with (repo / "results/detect/features/rows.csv").open() as f:
        for row in csv.DictReader(f):
            if row["split"] == "train" and row["relative_path"] in by_path:
                out.add(by_path[row["relative_path"]])
    return out


class MotionTracker:
    """Feed every stride-5 RGB frame; keeps a per-frame motion box (normalised x0, y0, x1, y1) or None."""

    def __init__(self):
        self.prev = None
        self.boxes: list[tuple[float, float, float, float] | None] = []
        self.energy: list[float] = []

    def __call__(self, rgb: np.ndarray) -> None:
        import cv2

        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (MOTION_WIDTH, max(1, round(h * MOTION_WIDTH / w))), interpolation=cv2.INTER_AREA)
        grey = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)
        if self.prev is None:
            self.prev = grey
            self.boxes.append(None)
            self.energy.append(0.0)
            return
        mask = (cv2.absdiff(grey, self.prev) > MOTION_DIFF).astype(np.uint8)
        self.prev = grey
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        self.energy.append(float(mask.mean()))
        if n <= 1:
            self.boxes.append(None)
            return
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, bw, bh = (int(v) for v in stats[k, :4])
        gh, gw = grey.shape
        self.boxes.append((x / gw, y / gh, (x + bw) / gw, (y + bh) / gh))


def step_boxes(frame_boxes: list, steps: int) -> list:
    """One box per detector step: union of that step's frame boxes, then corner smoothing across steps."""

    out, prev = [], None
    for i in range(steps):
        group = [b for b in frame_boxes[i * STEP:(i + 1) * STEP] if b is not None]
        if group:
            a = np.array(group)
            box = np.array([a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()])
            prev = box if prev is None else BOX_EMA * box + (1 - BOX_EMA) * prev
        out.append(None if prev is None else [round(float(v), 4) for v in prev])
    return out


def alarm_segments(times: list[float], above: np.ndarray, step_s: float) -> list[dict]:
    """Maximal runs of alarmed steps as [start_s, end_s) display intervals."""

    segs, start = [], None
    for t, a in zip(times, above):
        if a and start is None:
            start = t
        if not a and start is not None:
            segs.append({"start_s": start, "end_s": t})
            start = None
    if start is not None:
        segs.append({"start_s": start, "end_s": times[-1] + step_s})
    return segs
