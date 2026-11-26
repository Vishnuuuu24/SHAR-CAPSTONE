"""XD-Violence extra-training-video features (cross-dataset TRAINING-ONLY experiment, never evaluation).

Frames are decoded with cv2, JPEG round-tripped at the UCF extraction quality, resized/normalised by the registered
`core.p1c_roi.preprocess_crop`, and pooled by `models.dinov2_backbone.pool_cls_plus_mean_patch`, so a row is the same kind
of 1536-D vector as the registered UCF cache. XD labels map by name to UCF classes; they are never merged into the UCF headline.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: XD-Violence label code -> UCF class spelling (by name; B1 Fighting, B2 Shooting, B5 Abuse).
XD_CODE_TO_CLASS = {"B1": "Fighting", "B2": "Shooting", "B5": "Abuse"}
CLASS_TO_XD_CODE = {v: k for k, v in XD_CODE_TO_CLASS.items()}
FRAMES_PER_VIDEO = 8
CANDIDATE_STRIDE = 5
JPEG_QUALITY = 95  # results/p1c/p1c-video-frames/extract_closure.json
_LABEL_RE = re.compile(r"_label_([A-Za-z0-9-]+)\.mp4$")


def label_codes(file_name: str) -> list[str]:
    """Non-zero label codes in a file name, e.g. `..._label_B2-0-0.mp4` -> ['B2'] (`G` is normal, `0` is padding)."""

    match = _LABEL_RE.search(file_name)
    if match is None:
        raise ValueError(f"no `_label_<codes>.mp4` suffix in {file_name!r}")
    return [c for c in match.group(1).split("-") if c not in ("0", "")]


def class_for_video(folder_class: str, file_name: str) -> tuple[str, str]:
    """`(UCF class, xd_label_code)`; the folder is the single label and must agree with a code in the file name."""

    if folder_class not in CLASS_TO_XD_CODE:
        raise ValueError(f"folder class {folder_class!r} has no XD mapping; expected one of {sorted(CLASS_TO_XD_CODE)}")
    code = CLASS_TO_XD_CODE[folder_class]
    if code not in label_codes(file_name):
        raise ValueError(f"{file_name!r} codes {label_codes(file_name)} do not include {code} for folder {folder_class}")
    return folder_class, code


def stable_seed(name: str, base_seed: int = 0) -> int:
    """Per-name seed from sha256 (never python `hash()`, which is salted per process)."""

    return int.from_bytes(hashlib.sha256(f"{base_seed}:{name}".encode()).digest()[:4], "big")


def pick_frames(n_decoded: int, k: int, seed: int, stride: int = CANDIDATE_STRIDE) -> list[int]:
    """Source frame indices: `k` distinct seeded picks from every `stride`-th frame, ascending; repeat-cycle if fewer than `k`."""

    candidates = np.arange(0, n_decoded, stride)
    if len(candidates) == 0:
        raise ValueError("no candidate frames")
    if len(candidates) < k:
        return [int(x) for x in np.resize(candidates, k)]
    return sorted(int(x) for x in np.random.default_rng(seed).choice(candidates, size=k, replace=False))


def discover(root: Path) -> list[tuple[Path, str]]:
    """Sorted `(path, folder_class)` for every mp4 under `root/<Class>/`."""

    return sorted((p, p.parent.name) for p in Path(root).glob("*/*.mp4"))


def file_list_sha256(videos: Sequence[tuple[Path, str]], root: Path) -> str:
    lines = [f"{p.relative_to(root).as_posix()}\t{p.stat().st_size}" for p, _ in videos]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def decode_selected(path: Path, seed: int, k: int = FRAMES_PER_VIDEO) -> tuple[list[int], list[np.ndarray], float, int]:
    """`(frame indices, RGB uint8 frames after JPEG round trip, fps, decoded count)`; raises if undecodable.

    Pass 1 counts truly decoded frames (container frame counts are unreliable), pass 2 reads the picked ones.
    """

    import cv2

    from scripts.extract_p1c_video_frames import decode_jpeg_bytes, encode_jpeg_rgb

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    if n == 0:
        raise RuntimeError(f"no frames decoded from {path}")
    wanted = pick_frames(n, k, seed)
    need = set(wanted)
    got: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    for i in range(n):
        if i > max(need):
            break
        ok, bgr = cap.read()
        if not ok:
            break
        if i in need:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            got[i] = decode_jpeg_bytes(encode_jpeg_rgb(rgb, quality=JPEG_QUALITY))
    cap.release()
    if len(got) != len(need):
        raise RuntimeError(f"{path}: decoded only {len(got)}/{len(need)} picked frames")
    return wanted, [got[i] for i in wanted], fps, n


def pooled_features(frames: Sequence[np.ndarray], backbone: Any, preprocessing: Any, device: str = "cpu") -> np.ndarray:
    """`[len(frames), 1536]` float16 CLS++mean-patch features through the registered preprocessing."""

    import torch

    from models.dinov2_backbone import pool_cls_plus_mean_patch
    from scripts.build_p1c_patch_cache import _preprocess_batch

    with torch.no_grad():
        tensor = _preprocess_batch(np.stack(frames), preprocessing).to(device)
        pooled = pool_cls_plus_mean_patch(backbone(tensor), backbone.num_prefix_tokens)
    return pooled.to(torch.float16).cpu().numpy()


def _atomic_write(path: Path, writer) -> None:
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    tmp.replace(path)


def process_chunk(chunk: Sequence[tuple[Path, str]], parts_dir: Path, index: int, backbone: Any, preprocessing: Any,
                  base_seed: int = 0) -> None:
    """Featurise one chunk of videos and write `part-<index>.{npy,json}` atomically (resume = skip existing json)."""

    rows: list[dict[str, Any]] = []
    feats: list[np.ndarray] = []
    failed: list[dict[str, str]] = []
    for path, folder in chunk:
        try:
            cls, code = class_for_video(folder, path.name)
            frames_idx, frames, fps, n = decode_selected(path, stable_seed(path.name, base_seed))
            feats.append(pooled_features(frames, backbone, preprocessing))
        except Exception as exc:  # recorded in the manifest, never silent
            failed.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.extend({"video_id": path.stem, "class_name": cls, "xd_label_code": code, "source_frame_index": fi,
                     "fps": fps, "decoded_frames": n} for fi in frames_idx)
    array = np.concatenate(feats) if feats else np.zeros((0, 1536), np.float16)
    _atomic_write(parts_dir / f"part-{index:05d}.npy", lambda t: np.save(t.open("wb"), array))
    _atomic_write(parts_dir / f"part-{index:05d}.json", lambda t: t.write_text(json.dumps({"rows": rows, "failed": failed})))


def assemble(parts_dir: Path, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Concatenate parts in order into `features.npy` + `rows.jsonl`; returns `(rows, failed)`."""

    rows, failed, arrays = [], [], []
    for meta in sorted(parts_dir.glob("part-*.json")):
        info = json.loads(meta.read_text())
        arr = np.load(meta.with_suffix(".npy"))
        if len(arr) != len(info["rows"]):
            raise RuntimeError(f"{meta}: {len(arr)} feature rows vs {len(info['rows'])} metadata rows")
        rows += info["rows"]
        failed += info["failed"]
        arrays.append(arr)
    features = np.concatenate(arrays)
    np.save(out_dir / "features.npy", features)
    (out_dir / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return rows, failed


def load_cache(root: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    feats = np.load(Path(root) / "features.npy")
    rows = [json.loads(line) for line in (Path(root) / "rows.jsonl").read_text().splitlines() if line]
    if len(rows) != len(feats):
        raise ValueError(f"{root}: {len(rows)} rows vs {len(feats)} feature rows")
    return feats, rows
