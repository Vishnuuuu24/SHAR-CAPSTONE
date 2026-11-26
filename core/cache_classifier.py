"""Tip-Adapter-style memory classifier on frozen features: one embedding per video, a kernel vote over training videos."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def video_embeddings(features, rows: Sequence[dict], chunk: int = 20000) -> tuple[list[str], list[str], np.ndarray]:
    """Per video: mean of L2-normalised frame features, re-normalised. `rows[i]` describes `features[i]`."""

    ids: list[str] = []
    where: dict[str, int] = {}
    labels: list[str] = []
    owner = np.empty(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        v = r["source_video_id"]
        if v not in where:
            where[v] = len(ids)
            ids.append(v)
            labels.append(r["label"])
        elif labels[where[v]] != r["label"]:
            raise ValueError(f"{v}: rows disagree on label")
        owner[i] = where[v]
    sums = np.zeros((len(ids), features.shape[1]), dtype=np.float64)
    for s in range(0, len(rows), chunk):
        x = np.asarray(features[s : s + chunk], dtype=np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        np.add.at(sums, owner[s : s + chunk], x)
    emb = sums / (np.linalg.norm(sums, axis=1, keepdims=True) + 1e-12)
    return ids, labels, emb.astype(np.float32)


def kernel_vote(query: np.ndarray, keys: np.ndarray, key_targets: Sequence[int], n_classes: int, beta: float) -> np.ndarray:
    """Class probabilities from exp(-beta * (1 - cosine)) affinities to every training video, summed per class."""

    affinity = np.exp(-beta * (1.0 - np.clip(query @ keys.T, -1.0, 1.0)))
    onehot = np.eye(n_classes)[np.asarray(key_targets)]
    votes = affinity @ onehot
    return votes / votes.sum(axis=1, keepdims=True)


def fuse(p_head: np.ndarray, p_cache: np.ndarray, weight: float) -> np.ndarray:
    return (1.0 - weight) * p_head + weight * p_cache
