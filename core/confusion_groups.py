"""Confusion analysis, group selection and oracle ceilings for the divide-and-conquer study (classification only)."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np

from core.p1c_late_fusion import load_dinov2_probs

NORMAL = "Normal_Videos_event"
GATE_DELTA = 0.04
Run = dict  # {"truth": int[n], "pred": int[n], "probs": float[n, C], "videos": list[str]}


def macro_f1(truth: Sequence[int], pred: Sequence[int], n_classes: int) -> float:
    """Unweighted mean per-class F1 over all `n_classes` (a class with no TP scores 0), same as eval.metrics."""

    cm = confusion(truth, pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)))


def per_class_f1(truth, pred, n_classes: int) -> np.ndarray:
    cm = confusion(truth, pred, n_classes)
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)


def confusion(truth, pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(truth, dtype=int), np.asarray(pred, dtype=int)), 1)
    return cm


def load_runs(path) -> tuple[list[str], dict[tuple[int, int], Run]]:
    """`{(fold, seed): run}` from a probability CSV (fold, seed, source_video_id, true_label, p_*)."""

    classes, rows = load_dinov2_probs(path)
    by_run: dict[tuple[int, int], list] = defaultdict(list)
    for (fold, seed, vid), (label, p) in sorted(rows.items()):
        by_run[(fold, seed)].append((vid, classes.index(label), p))
    runs = {}
    for key, items in by_run.items():
        probs = np.stack([p for _v, _t, p in items])
        runs[key] = {"videos": [v for v, _t, _p in items], "truth": np.array([t for _v, t, _p in items]), "probs": probs,
                     "pred": probs.argmax(1)}
    return list(classes), runs


def baseline_report(classes: Sequence[str], runs: Mapping[tuple[int, int], Run]) -> dict:
    n = len(classes)
    folds = sorted({f for f, _s in runs})
    fold_means = {f: float(np.mean([macro_f1(r["truth"], r["pred"], n) for (ff, _s), r in runs.items() if ff == f])) for f in folds}
    return {"fold_mean_macro_f1": fold_means, "mean_over_runs": float(np.mean([macro_f1(r["truth"], r["pred"], n) for r in runs.values()])),
            "n_runs": len(runs), "videos_per_fold": {f: len({v for (ff, _s), r in runs.items() if ff == f for v in r["videos"]}) for f in folds}}


def fold_confusions(classes, runs) -> dict[int, np.ndarray]:
    """Directed confusion counts per fold, summed over seeds (rows true, columns predicted)."""

    out: dict[int, np.ndarray] = {}
    for (fold, _seed), r in runs.items():
        out[fold] = out.get(fold, 0) + confusion(r["truth"], r["pred"], len(classes))
    return out


def error_breakdown(cm: np.ndarray, classes: Sequence[str]) -> dict:
    normal = classes.index(NORMAL)
    off = cm.copy(); np.fill_diagonal(off, 0)
    total = int(off.sum())
    return {"errors": total, "support": int(cm.sum()), "accuracy": float(np.trace(cm) / cm.sum()),
            "to_normal": float(off[:, normal].sum() / total), "from_normal": float(off[normal].sum() / total),
            "anomaly_to_anomaly": float(off[np.ix_(*(2 * [[i for i in range(len(classes)) if i != normal]]))].sum() / total)}


def top_pairs(cm: np.ndarray, classes: Sequence[str], k: int = 12) -> list[dict]:
    support = cm.sum(1)
    pairs = [{"true": classes[i], "pred": classes[j], "count": int(cm[i, j]), "rate_of_true_class": float(cm[i, j] / support[i])}
             for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j]]
    return sorted(pairs, key=lambda p: -p["count"])[:k]


def pair_consistency(fold_cms: Mapping[int, np.ndarray], classes: Sequence[str], k: int = 12, top_per_fold: int = 12) -> list[dict]:
    """For the pooled top-k directed pairs: in how many folds is the pair also in that fold's own top list."""

    pooled = sum(fold_cms.values())
    tops = {f: {(p["true"], p["pred"]) for p in top_pairs(cm, classes, top_per_fold)} for f, cm in fold_cms.items()}
    return [{**p, "folds_in_top": sum((p["true"], p["pred"]) in t for t in tops.values()),
             "per_fold_count": {f: int(cm[classes.index(p["true"]), classes.index(p["pred"])]) for f, cm in sorted(fold_cms.items())}}
            for p in top_pairs(pooled, classes, k)]


def group_stats(cm: np.ndarray, group: Sequence[int]) -> dict:
    """`internal`: errors with true and predicted both in G; `leak`: true in G predicted outside; `coverage`: internal / all errors."""

    off = cm.copy(); np.fill_diagonal(off, 0)
    g = list(group)
    internal = int(off[np.ix_(g, g)].sum())
    from_g = int(off[g].sum())
    return {"internal": internal, "leak": from_g - internal, "closure": internal / from_g if from_g else 0.0,
            "coverage": internal / int(off.sum()) if off.sum() else 0.0}


def select_groups(cm: np.ndarray, classes: Sequence[str], sizes: Iterable[int] = (2, 3, 4), min_closure: float = 0.5,
                  min_internal: int = 6) -> dict[int, dict]:
    """Best anomaly-only group per size by coverage subject to closure >= `min_closure`.

    Rule is fixed in code and fed only the selection-side confusion matrix (never the scored fold's labels).
    """

    anomalies = [i for i, c in enumerate(classes) if c != NORMAL]
    out = {}
    for size in sizes:
        best = None
        for g in combinations(anomalies, size):
            s = group_stats(cm, g)
            if s["closure"] < min_closure or s["internal"] < min_internal:
                continue
            if best is None or (s["coverage"], -sum(g)) > (best[1]["coverage"], -sum(best[0])):
                best = (g, s)
        if best:
            out[size] = {"classes": [classes[i] for i in best[0]], **best[1]}
    return out


def oracle_predictions(truth: np.ndarray, pred: np.ndarray, group: Sequence[int], mode: str) -> np.ndarray:
    """Predictions after a PERFECT within-group specialist.

    `oracle_route`: every video whose true class is in G is corrected (routing uses the label; optimistic diagnostic).
    `hard_route`: only videos the baseline already assigns to G are corrected, and only when their true class is in G
    (a perfect specialist with no reject option cannot rescue a non-G video, and cannot turn a G prediction on a non-G video into a hit).
    """

    g = np.zeros(int(max(truth.max(), pred.max())) + 1, dtype=bool); g[list(group)] = True
    out = pred.copy()
    if mode == "oracle_route":
        sel = g[truth]
    elif mode == "hard_route":
        sel = g[pred] & g[truth]
    else:
        raise ValueError(mode)
    out[sel] = truth[sel]
    return out


def ceiling(classes, runs, group_names: Sequence[str], mode: str) -> dict:
    g = [classes.index(c) for c in group_names]
    n = len(classes)
    base, cand = defaultdict(list), defaultdict(list)
    cls_delta = np.zeros(n)
    for (fold, _s), r in runs.items():
        new = oracle_predictions(r["truth"], r["pred"], g, mode)
        base[fold].append(macro_f1(r["truth"], r["pred"], n)); cand[fold].append(macro_f1(r["truth"], new, n))
        cls_delta += per_class_f1(r["truth"], new, n) - per_class_f1(r["truth"], r["pred"], n)
    d = {f: float(np.mean(cand[f]) - np.mean(base[f])) for f in sorted(base)}
    return {"group": list(group_names), "mode": mode, "per_fold_delta": d, "mean_delta": float(np.mean(list(d.values()))),
            "folds_meeting_gate": sum(v >= GATE_DELTA for v in d.values()), "min_fold_delta": min(d.values()),
            "per_class_f1_delta": {classes[i]: float(cls_delta[i] / len(runs)) for i in range(n) if abs(cls_delta[i]) > 1e-12}}


def selection_confusion(classes, runs, scored_fold: int) -> np.ndarray:
    """Confusion of the OTHER folds' predictions, excluding any video that is also in `scored_fold`'s validation set.

    The AR folds' validation sets overlap (e.g. folds 1 and 2 share 35 videos), so this is what keeps the scored fold's
    labels out of its own group selection.
    """

    held = {v for (f, _s), r in runs.items() if f == scored_fold for v in r["videos"]}
    cm = np.zeros((len(classes),) * 2, dtype=np.int64)
    for (f, _s), r in runs.items():
        if f == scored_fold:
            continue
        keep = np.array([v not in held for v in r["videos"]], dtype=bool)
        cm += confusion(r["truth"][keep], r["pred"][keep], len(classes))
    return cm
