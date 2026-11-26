"""Inference with the final 14-class model, E3 (D-76), on one video file: the live demo's main model.

E3 = equal mean of three logistic arms over whole-video feature averages: DINOv2-L, DINOv2-B ++ SigLIP, and
DINOv2-B ++ V-JEPA 2 (16-frame clips), each trained per fold (4) and seed (3), so 36 members. This module loads
the reconstructed members from the E3 replay (`results/demo/e3-comparison-20261116`, digest-checked against
its export.json) and the four frozen encoders from the repository. At load, it checks that the repository's encoder
source files are byte-identical to the snapshot E3 ran with.

Frames: every 5th decoded frame, JPEG q95 round trip as in the evaluated frame blob (K-33), registered 224
preprocessing, no cap (the E3 deployment rule). For a file whose
*content* (SHA-256) is a UCF-Crime video, only the members of folds that never trained on it are used, which is
the held-out condition. Renaming the file does not change that. Other files use all 36 members.

The timeline scores consecutive ~5-second windows with the same members. It is a display aid: E3 learned from
whole-video labels and tends to read the setting rather than the moment (results/demo/showcase-20261124).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

from core.showcase_timeline import CLIP, FRAME_KEYS, pooled, trailing_windows

E3_DIR = "results/demo/e3-comparison-20261116"
STRIDE = 5
WINDOW = 32          # selected frames per timeline window: two V-JEPA clips, ~5.3 s at 30 fps
BATCH = 32           # frames per encoder batch, as in the E3 replay
TEST_SCORES = {"macro_f1": 0.4549, "accuracy": 0.462, "top3": 0.722, "source": "D-73 / D-76"}
ENCODER_SOURCES = (
    "core/p1c_roi.py", "data/p1c_loader.py", "scripts/build_p1c_feature_cache.py",
    "models/dinov2_backbone.py", "models/dinov2_large_backbone.py", "models/siglip_backbone.py",
    "models/vjepa2_backbone.py",
    "configs/experiments/p1c/P1C-VIDEO-DINOV2-B2-AR-FOLD1.yaml",
    "configs/experiments/p1c/P1C-VIDEO-SIGLIP-AR-FOLD1.yaml",
    "configs/experiments/p1c/P1C-VIDEO-VJEPA2-AR-FOLD1.yaml",
)
ENCODER_CONFIGS = {"B": ENCODER_SOURCES[7], "S": ENCODER_SOURCES[8], "VJ": ENCODER_SOURCES[9]}


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def e3_support(repo: Path):
    """The frozen E3 helpers (CONFIG, predict, content_membership, select_folds). Appended, never prepended, to
    sys.path so nothing in the repository is shadowed."""

    code = str(repo / E3_DIR / "code")
    if code not in sys.path:
        sys.path.append(code)
    import e3_support as support

    return support


def check_encoder_sources(repo: Path, snapshot: Path) -> None:
    for rel in ENCODER_SOURCES:
        if sha256_file(repo / rel) != sha256_file(snapshot / rel):
            raise RuntimeError(f"{rel} differs from the E3 snapshot; the encoders would not be E3's")


def load_members(repo: Path) -> dict:
    """The 36 reconstructed E3 logistic members, each checked against export.json and the parity record."""

    import joblib

    root = repo / E3_DIR
    export = json.loads((root / "export.json").read_text())
    if not json.loads((root / "cached_parity.json").read_text())["passed"]:
        raise RuntimeError("E3 cached-probability parity did not pass")
    members = {}
    for e in export["models"]:
        p = root / "models" / f"fold{e['fold']}-seed{e['seed']}-{e['arm']}.joblib"
        if sha256_file(p) != e["model_sha256"]:
            raise RuntimeError(f"estimator digest mismatch: {p.name}")
        members[(e["fold"], e["seed"], e["arm"])] = joblib.load(p)
    if len(members) != 36:
        raise RuntimeError(f"expected 36 E3 members, found {len(members)}")
    return members


def predict_batch(members: dict, arms: dict, vectors: dict[str, np.ndarray], folds: list[int], n_classes: int) -> np.ndarray:
    """Row-wise e3_support.predict for an (n, d) batch per encoder: mean class probability over selected members."""

    parts = []
    for (fold, _seed, arm), est in sorted(members.items()):
        if fold in folds:
            x = np.concatenate([vectors[k] for k in arms[arm]], axis=1)
            out = np.zeros((len(x), n_classes))
            out[:, est.classes_] = est.predict_proba(x)
            parts.append(out)
    return np.mean(parts, axis=0)


class E3Encoders:
    """The four frozen encoders and preprocessing exactly as e3_features.Extractor builds them."""

    def __init__(self, repo: Path, device: str = "cuda:0", jpeg: bool = False):
        import torch
        import yaml

        import scripts.build_p1c_feature_cache as loaders
        from data.p1c_loader import FramePreprocessing
        from models.dinov2_large_backbone import DINOv2ViTL14Reg, WEIGHTS_SHA256

        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        torch.backends.cudnn.allow_tf32 = True
        self.models, self.pre, self.weight_digests = {}, {}, {}
        for name, rel in ENCODER_CONFIGS.items():
            proto = yaml.safe_load((repo / rel).read_text())["fixed_protocol"]
            weights = dict(proto["backbone"]["weights"])
            path = repo / weights["path"]
            if sha256_file(path) != weights["sha256"]:
                raise RuntimeError(f"weight digest mismatch: {name}")
            weights["path"] = str(path)
            self.models[name] = loaders._BACKBONE_LOADERS[proto["backbone"]["family"]](weights).to(self.device).eval()
            self.pre[name] = FramePreprocessing.from_config(proto["preprocessing"])
            self.weight_digests[name] = weights["sha256"]
        self.models["L"] = DINOv2ViTL14Reg.from_official_checkpoint(repo).to(self.device).eval()
        self.pre["L"], self.weight_digests["L"] = self.pre["B"], WEIGHTS_SHA256
        self.jpeg = jpeg  # K-33: every evaluated feature came through the JPEG q95 frame blob

    def tensors(self, frame: np.ndarray):
        from PIL import Image

        from core.p1c_roi import preprocess_crop

        if self.jpeg:
            from core.demo_inference import registered_image

            image = registered_image(frame)
        else:
            image = Image.fromarray(frame, mode="RGB")
        return preprocess_crop(image, self.pre["B"]), preprocess_crop(image, self.pre["S"])


def extract_per_frame(enc, frames) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-frame B/L/S and per-clip VJ features with E3's encoders, batching and float16 rounding, unpooled.

    `enc` needs `.models`, `.device` and `.tensors(frame)` (E3Encoders or the frozen e3_features.Extractor).
    """

    import torch

    feats = {k: [] for k in FRAME_KEYS}
    clips, normalized, siglip, carry = [], [], [], []

    def run(key, x):
        return enc.models[key].pooled_features(x.to(enc.device)).to(torch.float16).float().cpu().numpy()

    def flush():
        nonlocal normalized, siglip, carry
        if not normalized:
            return
        x = torch.stack(normalized)
        feats["B"].append(run("B", x))
        feats["L"].append(run("L", x))
        feats["S"].append(run("S", torch.stack(siglip)))
        carry.extend(normalized)
        while len(carry) >= CLIP:
            count = min(len(carry) // CLIP, 2)
            clips.append(run("VJ", torch.stack([torch.stack(carry[i * CLIP:(i + 1) * CLIP]) for i in range(count)])))
            carry = carry[count * CLIP:]
        normalized, siglip = [], []

    with torch.inference_mode():
        for frame in frames:
            a, b = enc.tensors(frame)
            normalized.append(a)
            siglip.append(b)
            if len(normalized) == BATCH:
                flush()
        flush()
    if not clips:
        raise ValueError(f"the video is too short: E3 needs at least {CLIP} analysed frames ({CLIP * STRIDE} decoded)")
    return {k: np.concatenate(v) for k, v in feats.items()}, np.concatenate(clips)


class _Decoder:
    """Every STRIDE-th RGB frame of a video file, recording source indices, fps and decode time."""

    def __init__(self, path: Path, hook=None):
        import cv2

        self.hook = hook
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError("could not open the video (unsupported format or corrupt file)")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        self.indices, self.decoded, self.seconds = [], 0, 0.0

    def __iter__(self):
        import cv2

        try:
            while True:
                t = time.monotonic()
                ok, bgr = self.cap.read()
                if not ok:
                    break
                keep = self.decoded % STRIDE == 0
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if keep else None
                self.seconds += time.monotonic() - t
                if keep:
                    if self.hook is not None:
                        self.hook(rgb)
                    self.indices.append(self.decoded)
                    self.decoded += 1
                    yield rgb
                else:
                    self.decoded += 1
        finally:
            self.cap.release()


def _thumbs(path: Path, frame_indices: list[int], width: int = 240) -> list[str]:
    import cv2
    from PIL import Image

    cap, out = cv2.VideoCapture(str(path)), []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            out.append("")
            continue
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        im = im.resize((width, max(1, round(im.height * width / im.width))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        out.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    cap.release()
    return out


def _detector_exposure(digest: str, train: set[str], ucf: dict) -> str:
    if digest in train:
        return "in_sample"      # the detector trained on this video: its alarms here prove nothing
    return "held_out" if digest in ucf else "not_ucf"


class E3DemoModel:
    """Drop-in replacement for core.demo_inference.DemoModel (same predict() response, plus model fields)."""

    name = "E3 (D-76)"

    def __init__(self, repo_root: Path, device: str = "cuda:0", threads: int = 8, jpeg: bool = True,
                 detector: bool = True):
        # jpeg=True (K-33): uploads take the same JPEG q95 round trip as the evaluated frame blob. GPU check
        # 2026-11-25 (results/demo/e3-live-demo-20261124/detector_smoke.json): E3 probabilities then equal the
        # saved replay exactly (max |dp| 0.0 vs 0.041 without), detector scores within 0.0004.
        import torch

        torch.set_num_threads(threads)
        self.repo = Path(repo_root)
        self.support = e3_support(self.repo)
        cfg = self.support.CONFIG
        check_encoder_sources(self.repo, Path(cfg["code_root"]))
        self.members = load_members(self.repo)
        self.arms = cfg["members"]
        self.classes = json.loads((self.repo / E3_DIR / "class_names.json").read_text())
        self.encoders = E3Encoders(self.repo, device, jpeg=jpeg)
        self.device = self.encoders.device
        self.detector = None
        if detector:
            from core.demo_detector import Detector, detector_training_digests

            self.detector = Detector(self.repo, device=self.device)
            self.detector_train = detector_training_digests(self.repo, cfg["ucf"])
        self.seen = self.support.content_membership(json.loads((self.repo / E3_DIR / "training_membership.json").read_text()))
        self.ucf_labels: dict[str, set[str]] = {}
        for r in cfg["ucf"]:
            self.ucf_labels.setdefault(r["sha256"], set()).add(r["label"])

    def health(self) -> dict:
        out = {"model": self.name, "members": len(self.members), "test": TEST_SCORES, "jpeg_step": self.encoders.jpeg,
               "e3_config_sha256": self.support.CONFIG_SHA, "weights": self.encoders.weight_digests}
        if self.detector is not None:
            out["detector"] = {"name": "D-78 control seed 0", "threshold": self.detector.threshold, "test": self.detector.test_report}
        return out

    def membership(self, digest: str) -> tuple[list[int], str, dict | None]:
        """(folds to use, plain-language mode, expected-label record or None) for one file's content digest."""

        labels = self.ucf_labels.get(digest)
        if labels is None:
            return list(self.support.CONFIG["folds"]), "all 36 members (not a UCF-Crime video)", None
        folds, exposure = self.support.select_folds(digest, self.seen)
        seen_by = sorted(self.seen.get(digest, set()))
        if exposure == "held_out":
            mode = f"held-out members only: folds {folds} never trained on this video (the test condition)"
        else:
            mode = "WARNING: every fold trained on this video, so this is not a fair test"
        label = next(iter(labels)) if len(labels) == 1 else None
        return folds, mode, {"label": label, "conflicting_labels": sorted(labels) if label is None else None,
                             "seen_by_folds": seen_by, "held_out_folds": folds if exposure == "held_out" else []}

    def detect(self, per_frame: dict, tracker, idx: list[int], fps: float, digest: str) -> dict:
        """WHEN from the evaluated detector, WHERE from the motion heuristic; see core/demo_detector.py."""

        from core.demo_detector import STEP, alarm_segments, step_boxes

        scores = self.detector.scores(per_frame["B"][::STEP], per_frame["S"][::STEP])
        above = scores >= self.detector.threshold
        times = [idx[i] / fps for i in range(0, len(idx), STEP)][:len(scores)]
        boxes = step_boxes(tracker.boxes, len(scores))
        step_s = STEP * STRIDE / fps
        return {
            "threshold": self.detector.threshold, "step_s": step_s,
            "exposure": _detector_exposure(digest, self.detector_train, self.ucf_labels),
            "steps": [{"t_s": t, "score": round(float(sc), 4), "alarm": bool(a), "box": b if a else None}
                      for t, sc, a, b in zip(times, scores, above, boxes)],
            "segments": alarm_segments(times, above, step_s),
        }

    def predict(self, path: Path, filename: str) -> dict:
        t0 = time.monotonic()
        digest = sha256_file(path)
        folds, mode, expected = self.membership(digest)
        from core.demo_detector import MotionTracker

        tracker = MotionTracker() if self.detector is not None else None
        dec = _Decoder(path, hook=tracker)
        per_frame, per_clip = extract_per_frame(self.encoders, dec)
        t_ext = time.monotonic() - t0
        n, idx, fps = len(dec.indices), dec.indices, dec.fps

        probs = self.support.predict(self.members, pooled(per_frame, per_clip), folds, len(self.classes))
        order = np.argsort(-probs)

        wins = trailing_windows(n, WINDOW, WINDOW)
        batch = {k: [] for k in (*FRAME_KEYS, "VJ")}
        for s, e, c in wins:
            v = pooled(per_frame, per_clip, slice(s, e), c)
            for k in batch:
                batch[k].append(v[k])
        wp = predict_batch(self.members, self.arms, {k: np.stack(v) for k, v in batch.items()}, folds, len(self.classes))
        windows, prev_end = [], 0
        for (s, e, _c), p in zip(wins, wp):
            k = int(p.argmax())
            # Display span starts where the previous window ended, so the bar has no overlaps.
            windows.append({"start_s": idx[max(prev_end, s)] / fps, "end_s": (idx[e - 1] + 1) / fps,
                            "label": self.classes[k], "prob": float(p[k])})
            prev_end = e

        picks = np.unique(np.linspace(0, n - 1, min(8, n)).round().astype(int))
        images = _thumbs(path, [idx[i] for i in picks])
        thumbs = []
        for i, img in zip(picks, images):
            w = next((w for w, (s, e, _) in zip(windows, wins) if i < e), windows[-1])
            thumbs.append({"t_s": idx[i] / fps, "image": img, "window_label": w["label"]})
        pred = self.classes[int(order[0])]
        used = [f"fold{f}-seed{s}-{a}" for (f, s, a) in sorted(self.members) if f in folds]
        if expected is not None:
            expected["correct"] = None if expected["label"] is None else expected["label"] == pred
        return {
            "model": self.name, "filename": filename, "video_key": Path(filename).stem, "content_sha256": digest,
            "prediction": pred, "confidence": float(probs[order[0]]),
            "top5": [{"label": self.classes[int(i)], "prob": float(probs[i])} for i in order[:5]],
            "probs": {c: float(p) for c, p in zip(self.classes, probs)},
            "expected": expected, "heads_used": used, "mode": mode,
            "frames_used": n, "frame_stride": STRIDE, "fps": fps, "duration_s": (idx[-1] + 1) / fps,
            "window_frames": WINDOW, "windows": windows, "thumbnails": thumbs,
            "detector": None if self.detector is None else self.detect(per_frame, tracker, idx, fps, digest),
            "timing_s": {"decode": round(dec.seconds, 1), "features": round(t_ext - dec.seconds, 1),
                         "total": round(time.monotonic() - t0, 1)},
            "device": str(self.device),
        }
