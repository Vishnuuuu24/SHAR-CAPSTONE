"""Class-group specialists: global/local label mapping, hard routing, probability fusion (pure numpy)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


class GroupMap:
    """Maps between the global class index space and a specialist's local one (local order = `group_names` order)."""

    def __init__(self, global_classes: Sequence[str], group_names: Sequence[str]):
        if len(set(group_names)) != len(group_names) or len(group_names) < 2:
            raise ValueError("group needs >= 2 distinct classes")
        missing = [c for c in group_names if c not in global_classes]
        if missing:
            raise ValueError(f"unknown classes {missing}")
        self.global_classes, self.names = list(global_classes), list(group_names)
        self.idx = [self.global_classes.index(c) for c in self.names]

    def local_of(self, global_index: int) -> int:
        return self.idx.index(int(global_index))

    def in_group(self, global_indices) -> np.ndarray:
        return np.isin(np.asarray(global_indices), self.idx)

    def to_global(self, local_pred) -> np.ndarray:
        return np.asarray(self.idx)[np.asarray(local_pred, dtype=int)]


def route_hard(base_probs: np.ndarray, spec_probs: np.ndarray, gm: GroupMap) -> np.ndarray:
    """Final global predictions: the specialist decides only for videos the baseline sends into the group."""

    pred = base_probs.argmax(1).copy()
    sel = gm.in_group(pred)
    pred[sel] = gm.to_global(spec_probs[sel].argmax(1))
    return pred


def oracle_substitute(base_probs: np.ndarray, spec_probs: np.ndarray, truth: np.ndarray, gm: GroupMap) -> np.ndarray:
    """Diagnostic only: the (trained) specialist decides on videos whose TRUE class is in the group."""

    pred = base_probs.argmax(1).copy()
    sel = gm.in_group(truth)
    pred[sel] = gm.to_global(spec_probs[sel].argmax(1))
    return pred


def fuse_probs(base_probs: np.ndarray, spec_probs: np.ndarray, gm: GroupMap, weight: float) -> np.ndarray:
    """Keep the baseline's total probability mass on the group; redistribute it inside the group by a blend of the baseline's
    conditional distribution and the specialist's. Classes outside the group are untouched; rows still sum to 1."""

    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    out = np.array(base_probs, dtype=float, copy=True)
    mass = base_probs[:, gm.idx].sum(1, keepdims=True)
    cond = np.divide(base_probs[:, gm.idx], mass, out=np.full((len(base_probs), len(gm.idx)), 1.0 / len(gm.idx)), where=mass > 0)
    out[:, gm.idx] = mass * ((1.0 - weight) * cond + weight * spec_probs)
    return out


def split_holdout(video_ids_by_class: Mapping[str, Sequence[str]], fraction: float, seed_for) -> tuple[set[str], set[str]]:
    """Per-class seeded video-level split into (fit, holdout). `seed_for(cls)` -> int. Holdout has >= 2 videos when a class has >= 8."""

    fit, hold = set(), set()
    for cls, ids in video_ids_by_class.items():
        ids = sorted(ids)
        k = max(2 if len(ids) >= 8 else 1, int(round(fraction * len(ids))))
        order = np.random.default_rng(seed_for(cls)).permutation(len(ids))
        hold |= {ids[i] for i in order[:k]}
        fit |= {ids[i] for i in order[k:]}
    return fit, hold


def remap_sequences(seqs: Mapping[Any, tuple[np.ndarray, dict]], gm: GroupMap, keep: set[str] | None = None) -> dict:
    """Subset a `{(dataset, video): (sequence, row)}` dict to the group's classes with `target` remapped to local indices."""

    out = {}
    for key, (seq, row) in seqs.items():
        if row["label"] in gm.names and (keep is None or key[1] in keep):
            out[key] = (seq, {**row, "target": gm.names.index(row["label"])})
    return out
