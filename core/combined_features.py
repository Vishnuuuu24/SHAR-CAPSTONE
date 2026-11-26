"""Per-video DINOv2 token cache for SHAR-Combined (features only, every manifest row).

Each video contributes exactly ``FRAMES`` frames at the centres of ``FRAMES`` equal
temporal segments, decoded with cv2 (BGR->RGB, K-05), resized to 320x240 with
``cv2.INTER_AREA`` for EVERY source so UCF/XD/CUVA enter the backbone at one resolution,
then pushed through the caller-supplied forward (the headline backbone + preprocessing).
Storage is ``[n_videos, FRAMES, 1 + 256, 768]`` float16, token 0 = CLS, registers dropped.

A video that cannot be decoded is recorded in ``failures.csv`` and skipped; its feature
rows stay zero and ``rows.csv`` marks ``ok=0``. Nothing here scores or trains anything,
so building features for the sealed test split is allowed -- the probe never reads them.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

FRAMES = 8
FRAME_SIZE = (320, 240)  # (width, height) for cv2.resize
TOKENS = 257  # CLS + 16x16 patches
TOKEN_DIM = 768
ROW_FIELDS = ["index", "source_dataset", "source_video_id", "unified_label", "split", "fold",
              "ok", "n_decoded_frames", "frame_indices"]
FAILURE_FIELDS = ["index", "source_dataset", "source_video_id", "original_abs_path", "error"]


def frame_positions(n_frames: int, k: int = FRAMES) -> list[int]:
    """Centres of ``k`` equal segments of an ``n_frames`` video: ``floor((i + 0.5) * n / k)``.

    A video shorter than ``k`` repeats frames (each segment still maps to the frame it
    covers), so the result always has exactly ``k`` ascending indices in ``[0, n-1]``.
    """

    if n_frames < 1:
        raise ValueError("video has no frames")
    return [min(n_frames - 1, int((i + 0.5) * n_frames / k)) for i in range(k)]


def _read_positions(path: Path, positions: Sequence[int]) -> tuple[dict[int, np.ndarray], int]:
    """One sequential pass to EOF: keep the wanted frames, return them with the true decoded count (K-06)."""

    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    wanted = set(positions)
    got: dict[int, np.ndarray] = {}
    n = 0
    try:
        while cap.grab():
            if n in wanted:
                ok, bgr = cap.retrieve()
                if ok and bgr is not None:
                    got[n] = bgr
            n += 1
    finally:
        cap.release()
    return got, n


def decode_uniform_frames(path: Path, k: int = FRAMES) -> tuple[np.ndarray, list[int], int]:
    """``(frames[k, 240, 320, 3] uint8 RGB, frame indices, decoded count)``; raises if undecodable.

    ``CAP_PROP_FRAME_COUNT`` is only a hint (K-06): positions are first planned from it so
    the common case needs one decode pass, but the pass reads to EOF, and if the decoded
    count disagrees with the hint the positions are re-planned from the decoded count and
    a second pass is made.
    """

    import cv2

    cap = cv2.VideoCapture(str(path))
    hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    positions = frame_positions(hint, k) if hint > 0 else []
    got, n = _read_positions(path, positions)
    if n == 0:
        raise RuntimeError("no frames decoded")
    if n != hint or any(p not in got for p in positions):
        positions = frame_positions(n, k)
        got, n2 = _read_positions(path, positions)
        if n2 != n:
            raise RuntimeError(f"decoded count unstable between passes ({n} vs {n2})")
    missing = [p for p in positions if p not in got]
    if missing:
        raise RuntimeError(f"could not retrieve frames {missing}")
    frames = []
    for p in positions:
        rgb = cv2.cvtColor(got[p], cv2.COLOR_BGR2RGB)  # K-05: cv2 hands back BGR
        frames.append(cv2.resize(rgb, FRAME_SIZE, interpolation=cv2.INTER_AREA))
    return np.stack(frames), positions, n


def select_rows(rows: Sequence[dict], limit: int | None) -> list[dict]:
    """All rows, or with ``limit`` the first N taken round-robin across sources (smoke runs)."""

    if not limit:
        return list(rows)
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source_dataset"], []).append(r)
    out: list[dict] = []
    queues = list(by_source.values())
    while len(out) < limit and any(queues):
        for q in queues:
            if q and len(out) < limit:
                out.append(q.pop(0))
    return out


class VideoFrameDataset:
    """Map-style dataset: row index -> ``{index, frames | None, positions, n_decoded, error}``.

    Never raises from ``__getitem__`` -- a decode failure is returned as ``error`` so one
    corrupt file cannot kill a DataLoader worker or the run.
    """

    def __init__(self, rows: Sequence[dict], decoder: Callable[[Path], tuple[np.ndarray, list[int], int]] | None = None):
        self.rows = list(rows)
        self.decoder = decoder or decode_uniform_frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        try:
            frames, positions, n = self.decoder(Path(self.rows[i]["original_abs_path"]))
            return {"index": i, "frames": frames, "positions": positions, "n_decoded": n, "error": ""}
        except Exception as exc:  # noqa: BLE001 -- recorded in failures.csv, never fatal
            return {"index": i, "frames": None, "positions": [], "n_decoded": 0, "error": f"{type(exc).__name__}: {exc}"}


def collate_list(items: list[dict]) -> list[dict]:
    return items


def extract_cache(
    rows: Sequence[dict],
    out_dir: Path,
    forward: Callable[[np.ndarray], np.ndarray],
    *,
    token_shape: tuple[int, ...] = (TOKENS, TOKEN_DIM),
    frames: int = FRAMES,
    batch_videos: int = 4,
    num_workers: int = 0,
    decoder: Callable[[Path], tuple[np.ndarray, list[int], int]] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Stream features for every row into ``out_dir/features.float16.npy`` (memmap) plus rows/failures CSVs.

    ``forward(frames[n, 240, 320, 3] uint8) -> [n, *token_shape]`` runs the backbone; it is
    injected so the I/O and failure handling are testable without the 86M-parameter model.
    Returns ``{"memmap", "n_ok", "n_failed"}``; the caller finalises (hash) the memmap.
    """

    from torch.utils.data import DataLoader

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    memmap = np.lib.format.open_memmap(
        out_dir / "features.float16.npy", mode="w+", dtype=np.float16, shape=(n, frames, *token_shape))
    status: list[dict[str, Any]] = [{"ok": 0, "n_decoded_frames": 0, "frame_indices": ""} for _ in range(n)]
    n_ok = n_failed = 0
    loader = DataLoader(VideoFrameDataset(rows, decoder), batch_size=batch_videos, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_list)
    with (out_dir / "failures.csv").open("w", newline="", encoding="utf-8") as fh:
        failures = csv.DictWriter(fh, fieldnames=FAILURE_FIELDS)
        failures.writeheader()
        for batch in loader:
            good = [b for b in batch if b["frames"] is not None]
            for b in batch:
                if b["frames"] is None:
                    r = rows[b["index"]]
                    failures.writerow({"index": b["index"], "source_dataset": r["source_dataset"],
                                       "source_video_id": r["source_video_id"],
                                       "original_abs_path": r["original_abs_path"], "error": b["error"]})
                    n_failed += 1
            fh.flush()
            if not good:
                continue
            tokens = np.asarray(forward(np.concatenate([b["frames"] for b in good])))
            tokens = tokens.reshape(len(good), frames, *token_shape)
            if not np.isfinite(tokens).all():
                raise RuntimeError("non-finite backbone output; refusing to cache it")
            for j, b in enumerate(good):
                memmap[b["index"]] = tokens[j].astype(np.float16)
                status[b["index"]] = {"ok": 1, "n_decoded_frames": b["n_decoded"],
                                      "frame_indices": " ".join(map(str, b["positions"]))}
                n_ok += 1
            done = n_ok + n_failed
            if done % 100 < batch_videos:
                log(f"  {done}/{n} videos ({n_failed} failed)")
    with (out_dir / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"index": i, **{k: r[k] for k in ("source_dataset", "source_video_id", "unified_label", "split", "fold")},
                        **status[i]})
    return {"memmap": memmap, "n_ok": n_ok, "n_failed": n_failed}


def cls_plus_patches(tokens: Any, num_prefix_tokens: int) -> Any:
    """``[B, 1 + registers + patches, D] -> [B, 1 + patches, D]``: token 0 stays CLS, registers are dropped."""

    import torch

    return torch.cat([tokens[:, :1], tokens[:, num_prefix_tokens:]], dim=1)
