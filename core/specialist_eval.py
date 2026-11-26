"""Paired evaluation of specialist strategies against the general model on identical (fold, seed) runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from core.confusion_groups import GATE_DELTA, macro_f1, per_class_f1

STRONG = ("Normal_Videos_event", "Explosion", "Robbery", "Arson")
RunKey = tuple[int, int]


def evaluate_system(classes: Sequence[str], base: Mapping[RunKey, tuple[np.ndarray, np.ndarray]],
                    cand: Mapping[RunKey, tuple[np.ndarray, np.ndarray]]) -> dict:
    """`base`/`cand`: {(fold, seed): (truth, pred)} over identical videos. Fold means average over seeds."""

    n = len(classes)
    if set(base) != set(cand):
        raise ValueError("run sets differ")
    fb, fc, dcls = defaultdict(list), defaultdict(list), np.zeros(n)
    for k in sorted(base):
        (tb, pb), (tc, pc) = base[k], cand[k]
        if not np.array_equal(tb, tc):
            raise ValueError(f"truth differs for run {k}")
        fb[k[0]].append(macro_f1(tb, pb, n)); fc[k[0]].append(macro_f1(tc, pc, n))
        dcls += per_class_f1(tc, pc, n) - per_class_f1(tb, pb, n)
    delta = {f: float(np.mean(fc[f]) - np.mean(fb[f])) for f in sorted(fb)}
    dcls /= len(base)
    return {"baseline_fold_mean": {f: float(np.mean(v)) for f, v in sorted(fb.items())},
            "candidate_fold_mean": {f: float(np.mean(v)) for f, v in sorted(fc.items())},
            "candidate_mean": float(np.mean([np.mean(v) for v in fc.values()])), "baseline_mean": float(np.mean([np.mean(v) for v in fb.values()])),
            "per_fold_delta": delta, "mean_delta": float(np.mean(list(delta.values()))),
            "folds_meeting_gate": int(sum(d >= GATE_DELTA for d in delta.values())), "passes_gate": all(d >= GATE_DELTA for d in delta.values()),
            "per_class_f1_delta": {c: float(dcls[i]) for i, c in enumerate(classes)},
            "strong_class_delta": {c: float(dcls[classes.index(c)]) for c in STRONG if c in classes},
            "worst_strong_class_delta": float(min(dcls[classes.index(c)] for c in STRONG if c in classes))}


def within_group_accuracy(base_probs: np.ndarray, spec_probs: np.ndarray, truth: np.ndarray, idx: Sequence[int]) -> dict:
    """On videos whose true class is in the group: specialist accuracy vs the baseline restricted to the same classes."""

    idx = list(idx)
    sel = np.isin(truth, idx)
    local_truth = np.array([idx.index(t) for t in truth[sel]])
    return {"n": int(sel.sum()), "specialist": float((spec_probs[sel].argmax(1) == local_truth).mean()),
            "baseline_restricted": float((base_probs[sel][:, idx].argmax(1) == local_truth).mean()), "chance": 1.0 / len(idx)}
