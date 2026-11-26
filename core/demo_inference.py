"""Inference for the browser demo: the evaluated D-53 classifier applied to one uploaded video.

Model: frozen DINOv2-B (registered weights) -> CLS ++ mean-patch per frame -> the 12 D-53 `step_permuted` heads
(4 folds x 3 seeds) exported by `scripts/export_demo_heads.py`, probabilities averaged. Frames: every 5th decoded
frame (the D-34/D-35 stride), registered 224 preprocessing; very long videos are thinned evenly to `MAX_FRAMES`.
Scoring follows the evaluated rule: each head sees the video in its registered permuted order, split into 8-frame
chunks, mean logits, softmax.

Honesty rule: every UCF protocol video was a training or validation video for some heads. If the file's contents
(SHA-256) or its name identify a protocol video, only the heads of folds that never saw it are used (the D-57 test
condition), and the caller is told when no such head exists. Renaming a protocol copy therefore does not turn it into
unseen footage; a re-encoded or trimmed copy under a new name still does, because nothing can recognise it. A per-window timeline (consecutive 8-frame windows, unpermuted) is returned as a
diagnostic of which stretch looks like which class; it is not a localisation claim.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path

import numpy as np

STRIDE = 5
MAX_FRAMES = 2400
FRAMES = 8
HEADS_PATH = ".cache/demo/d53_heads.pt"
MEMBERSHIP_PATH = "results/demo/membership.json"
DIGESTS_PATH = "results/demo/protocol_digests.json"


def video_key(filename: str) -> str:
    """'Abuse013_x264.mp4' -> 'Abuse013_x264'; anything else -> its stem."""

    return re.sub(r"\.[A-Za-z0-9]+$", "", Path(filename).name)


def decode_frames(path: Path, stride: int = STRIDE, max_frames: int = MAX_FRAMES) -> tuple[list[np.ndarray], list[int], float]:
    """RGB frames at `stride` (source indices kept), thinned evenly to `max_frames`; returns (frames, indices, fps)."""

    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("could not open the video (unsupported format or corrupt file)")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frames, idx, i = [], [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            idx.append(i)
        i += 1
    cap.release()
    if not frames:
        raise ValueError("no frames could be decoded from the video")
    if len(frames) > max_frames:
        keep = np.unique(np.linspace(0, len(frames) - 1, max_frames).round().astype(int))
        frames, idx = [frames[k] for k in keep], [idx[k] for k in keep]
    return frames, idx, fps


def held_out_folds(key: str, membership: dict, folds=(1, 2, 3, 4)) -> tuple[list[int] | None, dict | None]:
    """(folds whose heads never saw the video, its protocol record) for a protocol video; (None, None) otherwise."""

    rec = membership.get("videos", {}).get(key)
    if rec is None:
        return None, None
    return [f for f in folds if f not in rec["seen_by_folds"]], rec


def resolve_video_key(key: str, digest: str, membership: dict, digests: dict) -> tuple[str, str | None]:
    """(protocol key, how it was recognised) from the contents first, then the name; (key, None) for unseen footage."""

    if digest in digests:
        return digests[digest], "file contents"
    if key in membership.get("videos", {}):
        return key, "file name"
    return key, None


def registered_image(frame: np.ndarray):
    """The frame as the evaluated pipeline saw it: every cached feature came through the D-35 JPEG blob, never raw decodes."""

    from PIL import Image

    from scripts.extract_p1c_video_frames import DEFAULT_JPEG_QUALITY, encode_jpeg_rgb

    return Image.open(io.BytesIO(encode_jpeg_rgb(frame, quality=DEFAULT_JPEG_QUALITY))).convert("RGB")


def _thumb(frame: np.ndarray, width: int = 240) -> str:
    from PIL import Image

    im = Image.fromarray(frame)
    im = im.resize((width, max(1, round(im.height * width / im.width))))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class DemoModel:
    def __init__(self, repo_root: Path, device: str = "cuda:0", threads: int = 8):
        import torch

        from core.frame_batches import FrameBatcher  # noqa: F401  (import check: same preprocessing stack)
        from scripts.build_combined_patch_cache import _load_backbone_and_preprocessing
        from scripts.run_p1c_xd_extra_probe import _build_head

        torch.set_num_threads(threads)
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.backbone, self.pre, self.weight_sha = _load_backbone_and_preprocessing(self.device)
        blob = torch.load(repo_root / HEADS_PATH, map_location="cpu", weights_only=False)
        self.classes = blob["classes"]
        self.heads = []
        for h in blob["heads"]:
            head = _build_head("step_permuted", len(self.classes), FRAMES)
            head.load_state_dict(h["state_dict"])
            self.heads.append({"fold": h["fold"], "seed": h["seed"], "head": head.eval()})
        self.export_info = {k: v for k, v in blob.items() if k not in ("heads", "classes")}
        self.membership = json.loads((repo_root / MEMBERSHIP_PATH).read_text())
        self.digests = json.loads((repo_root / DIGESTS_PATH).read_text())["digests"]

    def health(self) -> dict:
        return {"model": "D-53 (older demo model)", "members": len(self.heads),
                "test": {"macro_f1": 0.414, "accuracy": 0.418, "source": "D-57"}, **self.export_info}

    def features(self, frames: list[np.ndarray], batch: int = 64) -> np.ndarray:
        import torch

        from core.p1c_roi import preprocess_crop

        out = []
        with torch.no_grad():
            for s in range(0, len(frames), batch):
                x = torch.stack([preprocess_crop(registered_image(f), self.pre) for f in frames[s:s + batch]]).to(self.device)
                out.append(self.backbone.pooled_features(x).to(torch.float16).float().cpu().numpy())  # cache precision
        return np.concatenate(out)

    def predict(self, path: Path, filename: str) -> dict:
        import torch

        from core.p1c_head_variants import _chunk_sequence, _pad_sequence
        from core.p1c_oracle_interval import predict_video_probs
        from core.p1c_readiness import sha256_file
        from scripts.run_subset7_search import PERMUTE_BASE

        digest = sha256_file(path)
        t0 = time.time()
        frames, idx, fps = decode_frames(path)
        t_dec = time.time() - t0
        feats = self.features(frames)
        t_feat = time.time() - t0 - t_dec
        key, recognised_by = resolve_video_key(video_key(filename), digest, self.membership, self.digests)
        held, rec = held_out_folds(key, self.membership)
        if held is None:
            use, mode = self.heads, "all 12 heads (not a UCF protocol video)"
        elif held:
            use = [h for h in self.heads if h["fold"] in held]
            mode = f"held-out heads only: folds {held} never saw this video (the D-57 test condition)"
        else:
            use = self.heads
            mode = "WARNING: every fold trained or validated on this video, so this is not a fair test"
        per_head = []
        for h in use:
            rng = np.random.default_rng(PERMUTE_BASE["test"] + h["seed"])
            order = rng.permutation(len(feats))
            per_head.append(predict_video_probs(h["head"], feats[order], FRAMES))
        probs = np.mean(per_head, axis=0)
        order5 = np.argsort(-probs)[:5]

        windows = []
        chunks = _chunk_sequence(np.arange(len(feats)), FRAMES)
        with torch.no_grad():
            for c in chunks:
                seq, mask = _pad_sequence(feats[c], FRAMES)
                p = np.mean([torch.softmax(h["head"](torch.from_numpy(seq[None]), key_padding_mask=torch.from_numpy(mask[None]))[0], 0).numpy()
                             for h in use], axis=0)
                k = int(p.argmax())
                windows.append({"start_s": idx[c[0]] / fps, "end_s": idx[c[-1]] / fps, "label": self.classes[k], "prob": float(p[k])})
        picks = np.unique(np.linspace(0, len(frames) - 1, min(8, len(frames))).round().astype(int))
        thumbs = [{"t_s": idx[i] / fps, "image": _thumb(frames[i]), "window_label": windows[i // FRAMES]["label"]} for i in picks]
        pred = self.classes[int(order5[0])]
        return {
            "filename": filename, "video_key": key, "video_sha256": digest, "recognised_by": recognised_by,
            "fair": held is None or bool(held),
            "prediction": pred, "confidence": float(probs[order5[0]]),
            "top5": [{"label": self.classes[int(i)], "prob": float(probs[i])} for i in order5],
            "probs": {c: float(p) for c, p in zip(self.classes, probs)},
            "expected": None if rec is None else {"label": rec["label"], "correct": rec["label"] == pred,
                                                  "seen_by_folds": rec["seen_by_folds"], "held_out_folds": held},
            "heads_used": [f"fold{h['fold']}-seed{h['seed']}" for h in use], "mode": mode,
            "frames_used": len(frames), "frame_stride": STRIDE, "fps": fps, "duration_s": (idx[-1] + 1) / fps,
            "windows": windows, "thumbnails": thumbs,
            "timing_s": {"decode": round(t_dec, 1), "features": round(t_feat, 1), "total": round(time.time() - t0, 1)},
            "device": str(self.device),
        }
