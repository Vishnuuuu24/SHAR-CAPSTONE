"""Weakly supervised frame scorers for the live detector (FINAL_PUSH_PLAN_2026-11-11 Step 4b).

Training sees VIDEO-level labels only (normal / anomaly, D-71). Every model maps one frame's features
to one anomaly score; they differ in the training loss and in what context the frame is given:

- `none`    the frame's standardised features;
- `causal`  those features plus their deviation from the mean of all frames seen so far in the video --
            available to a live system, which only ever knows the past;
- `offline` the features plus their deviation from the whole video's mean (needs the future).

The deviation feature follows this project's own N13 finding: removing a video's constant offset is what
made events rankable within a video. Models and epochs are chosen on held-out TRAINING videos scored at
video level; test intervals are never used for any choice.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from core.detect_data import DetectVideo

CONTEXTS = ("none", "causal", "offline")
TOPK_DIVISOR = 16          # video score = mean of the top ceil(T / 16) frame scores
SEQUENCE_LENGTH = 256      # frames per training video (uniform random subsample of longer videos)
SEGMENTS = 32              # Sultani et al. 2018: 32 temporal segments per video


@dataclass
class FeatureStore:
    """Frame features of the detection split, standardised with TRAINING-split statistics on access."""

    arrays: list[np.ndarray]                     # one [N, D_i] float16 array per backbone, row-aligned
    spans: dict[str, tuple[int, int]]            # relative_path -> [start, stop) rows
    source_frames: np.ndarray                    # [N] source frame index of each row
    mean: np.ndarray
    std: np.ndarray

    @property
    def dim(self) -> int:
        return int(sum(a.shape[1] for a in self.arrays))

    def frames(self, relative_path: str, context: str = "none") -> np.ndarray:
        start, stop = self.spans[relative_path]
        x = np.concatenate([np.asarray(a[start:stop], dtype=np.float32) for a in self.arrays], axis=1)
        return add_context((x - self.mean) / self.std, context)

    def source_frames_of(self, relative_path: str) -> np.ndarray:
        start, stop = self.spans[relative_path]
        return self.source_frames[start:stop]


def add_context(x: np.ndarray, context: str) -> np.ndarray:
    """`[T, D]` -> `[T, D]` (`none`) or `[T, 2D]` with the deviation-from-context-mean block appended."""

    if context == "none":
        return x
    if context == "causal":
        running = np.cumsum(x, axis=0, dtype=np.float64) / np.arange(1, len(x) + 1)[:, None]
        return np.concatenate([x, (x - running).astype(np.float32)], axis=1)
    if context == "offline":
        return np.concatenate([x, x - x.mean(axis=0, keepdims=True)], axis=1)
    raise ValueError(f"unknown context {context!r}")


def load_store(feature_dir: Path, backbones: Sequence[str], videos: Sequence[DetectVideo], *, chunk: int = 65536,
               standardise: bool = True) -> FeatureStore:
    """Open `<feature_dir>/<backbone>.float16.npy` + `rows.csv`; statistics come from the training split's rows only.

    `standardise=False` returns raw features (mean 0, std 1) for callers that fit their own scaler.
    """

    feature_dir = Path(feature_dir)
    arrays = [np.load(feature_dir / f"{name}.float16.npy", mmap_mode="r") for name in backbones]
    spans: dict[str, tuple[int, int]] = {}
    source_frames = np.empty(arrays[0].shape[0], dtype=np.int64)
    with (feature_dir / "rows.csv").open() as handle:
        reader = csv.reader(handle)
        header = next(reader)
        path_col, frame_col = header.index("relative_path"), header.index("source_frame_index")
        current, start, count = None, 0, 0
        for count, row in enumerate(reader):
            source_frames[count] = int(row[frame_col])
            if row[path_col] != current:
                if current is not None:
                    spans[current] = (start, count)
                if row[path_col] in spans:
                    raise ValueError(f"{row[path_col]} is not contiguous in rows.csv")
                current, start = row[path_col], count
        spans[current] = (start, count + 1)
    if count + 1 != arrays[0].shape[0] or any(a.shape[0] != arrays[0].shape[0] for a in arrays):
        raise ValueError("rows.csv and the feature arrays disagree on the number of rows")
    missing = [v.relative_path for v in videos if v.relative_path not in spans]
    if missing:
        raise ValueError(f"{len(missing)} split videos have no features, e.g. {missing[:3]}")
    if not standardise:
        dim = int(sum(a.shape[1] for a in arrays))
        return FeatureStore(arrays, spans, source_frames, np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32))
    train_rows = np.concatenate([np.arange(*spans[v.relative_path]) for v in videos if v.split == "train"])
    total, total_sq = 0.0, 0.0
    for begin in range(0, len(train_rows), chunk):
        rows = train_rows[begin:begin + chunk]
        block = np.concatenate([np.asarray(a[rows[0]:rows[-1] + 1][rows - rows[0]], dtype=np.float64) for a in arrays], axis=1)
        total, total_sq = total + block.sum(axis=0), total_sq + (block ** 2).sum(axis=0)
    mean = total / len(train_rows)
    std = np.sqrt(np.maximum(total_sq / len(train_rows) - mean ** 2, 1e-12))
    return FeatureStore(arrays, spans, source_frames, mean.astype(np.float32), std.astype(np.float32))


def split_heldout(videos: Sequence[DetectVideo], *, fraction: float = 0.1, seed: int = 0) -> tuple[list[DetectVideo], list[DetectVideo]]:
    """Training-split videos -> (fit, held-out), stratified by normal/anomaly. Held-out videos choose epochs and the alarm threshold."""

    rng = np.random.default_rng(seed)
    fit, held = [], []
    for flag in (False, True):
        group = sorted((v for v in videos if v.split == "train" and v.anomaly == flag), key=lambda v: v.relative_path)
        order = rng.permutation(len(group))
        cut = int(round(len(group) * fraction))
        held += [group[i] for i in order[:cut]]
        fit += [group[i] for i in order[cut:]]
    return fit, held


def _mlp(in_dim: int, hidden: Sequence[int], dropout: float):
    from torch import nn

    layers, width = [], in_dim
    for size in hidden:
        layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
        width = size
    return nn.Sequential(*layers, nn.Linear(width, 1))


def topk_video_score(frame_probs, mask):
    """Mean of the top `ceil(valid / TOPK_DIVISOR)` frame probabilities per video; `mask` is True at padded frames."""

    import torch

    valid = (~mask).sum(dim=1)
    k = torch.clamp((valid + TOPK_DIVISOR - 1) // TOPK_DIVISOR, min=1)
    ordered = frame_probs.masked_fill(mask, -1.0).sort(dim=1, descending=True).values
    keep = torch.arange(frame_probs.shape[1], device=frame_probs.device)[None, :] < k[:, None]
    return (ordered * keep).sum(dim=1) / k


def _subsample(length: int, target: int, rng: np.random.Generator) -> np.ndarray:
    if length <= target:
        return np.arange(length)
    edges = np.linspace(0, length, target + 1)
    return np.minimum((edges[:-1] + rng.random(target) * (edges[1:] - edges[:-1])).astype(np.int64), length - 1)


def _segment_means(x: np.ndarray, segments: int) -> np.ndarray:
    edges = np.linspace(0, len(x), segments + 1).astype(np.int64)
    return np.stack([x[a:max(b, a + 1)].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])


def score_frames(model, x: np.ndarray, device) -> np.ndarray:
    """Anomaly probability of every frame of one video."""

    import torch

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x).to(device)).squeeze(-1)).float().cpu().numpy()


def _video_auc(model, store: FeatureStore, videos: Sequence[DetectVideo], context: str, device, rule: str) -> float:
    from sklearn.metrics import roc_auc_score

    scores = []
    for video in videos:
        frames = score_frames(model, store.frames(video.relative_path, context), device)
        if rule == "max":
            scores.append(float(frames.max()))
        else:
            k = max(1, -(-len(frames) // TOPK_DIVISOR))
            scores.append(float(np.sort(frames)[-k:].mean()))
    return float(roc_auc_score([int(v.anomaly) for v in videos], scores))


def train_detector(store: FeatureStore, fit: Sequence[DetectVideo], held: Sequence[DetectVideo], *, loss: str, context: str = "none",
                   seed: int = 0, device="cpu", epochs: int = 60, patience: int = 10, batch_videos: int = 32,
                   learning_rate: float | None = None, log=print):
    """Train one frame scorer; returns `(model at its best held-out epoch, per-epoch log)`.

    `loss="topk"`: binary cross-entropy on the top-k mean frame probability of each video (the usual
    multiple-instance recipe). `loss="ranking"`: Sultani et al. 2018 -- 32 segment means per video, the
    highest anomalous segment must outrank the highest normal one, plus smoothness and sparsity terms.
    `learning_rate=None` keeps each loss's declared rate (AdamW 1e-4; Adagrad 0.01).
    """

    import copy

    import torch

    if loss not in ("topk", "ranking"):
        raise ValueError(f"unknown loss {loss!r}")
    if loss == "ranking" and context != "none":
        raise ValueError("the ranking baseline is defined without context features")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    in_dim = store.dim * (1 if context == "none" else 2)
    if loss == "topk":
        model = _mlp(in_dim, (512, 128), 0.5).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate or 1e-4, weight_decay=1e-2)
    else:
        model = _mlp(in_dim, (512, 32), 0.6).to(device)
        optimiser = torch.optim.Adagrad(model.parameters(), lr=learning_rate or 0.01, weight_decay=1e-3)
    anomalous = [v for v in fit if v.anomaly]
    normal = [v for v in fit if not v.anomaly]
    steps = max(1, min(len(anomalous), len(normal)) // batch_videos)
    length = SEGMENTS if loss == "ranking" else SEQUENCE_LENGTH

    def batch(videos: Sequence[DetectVideo]):
        xs = np.zeros((len(videos), length, in_dim), dtype=np.float32)
        mask = np.ones((len(videos), length), dtype=bool)
        for row, video in enumerate(videos):
            x = store.frames(video.relative_path, context)
            x = _segment_means(x, SEGMENTS) if loss == "ranking" else x[_subsample(len(x), length, rng)]
            xs[row, :len(x)], mask[row, :len(x)] = x, False
        return torch.from_numpy(xs).to(device), torch.from_numpy(mask).to(device)

    best_auc, best_state, best_epoch, stale, history = -1.0, None, -1, 0, []
    for epoch in range(epochs):
        model.train()
        order_a, order_n, running = rng.permutation(len(anomalous)), rng.permutation(len(normal)), 0.0
        for step in range(steps):
            picked_a = [anomalous[i] for i in order_a[step * batch_videos:(step + 1) * batch_videos]]
            picked_n = [normal[i] for i in order_n[step * batch_videos:(step + 1) * batch_videos]]
            x, mask = batch(picked_a + picked_n)
            probs = torch.sigmoid(model(x).squeeze(-1))
            if loss == "topk":
                target = torch.cat([torch.ones(len(picked_a)), torch.zeros(len(picked_n))]).to(device)
                value = torch.nn.functional.binary_cross_entropy(topk_video_score(probs, mask).clamp(1e-6, 1 - 1e-6), target)
            else:
                a, n = probs[:len(picked_a)], probs[len(picked_a):]
                hinge = torch.relu(1.0 - a.max(dim=1).values + n.max(dim=1).values).mean()
                smooth = ((a[:, 1:] - a[:, :-1]) ** 2).sum(dim=1).mean()
                value = hinge + 8e-5 * smooth + 8e-5 * a.sum(dim=1).mean()
            optimiser.zero_grad(set_to_none=True)
            value.backward()
            optimiser.step()
            running += float(value.detach())
        auc = _video_auc(model, store, held, context, device, "max" if loss == "ranking" else "topk")
        history.append({"epoch": epoch, "train_loss": running / steps, "heldout_video_auc": auc})
        if auc > best_auc + 1e-4:
            best_auc, best_state, best_epoch, stale = auc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    log({"loss": loss, "context": context, "seed": seed, "best_epoch": best_epoch, "heldout_video_auc": round(best_auc, 4), "epochs_run": len(history)})
    return model, history


def video_level_scores(store: FeatureStore, fit: Sequence[DetectVideo], score: Sequence[DetectVideo]) -> dict[str, float]:
    """Floor row: logistic regression on each video's mean feature; the one probability is given to every frame."""

    from sklearn.linear_model import LogisticRegression

    def means(videos):
        return np.stack([store.frames(v.relative_path).mean(axis=0) for v in videos])

    model = LogisticRegression(C=0.01, max_iter=3000).fit(means(fit), [int(v.anomaly) for v in fit])
    return {v.relative_path: float(p) for v, p in zip(score, model.predict_proba(means(score))[:, 1])}
