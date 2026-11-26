"""Validation-only failure analysis of the headline 14-class classifier from its stored per-video probabilities.

Everything here reads `(fold, seed, video, true, probs)` rows; nothing is retrained and no test data is touched.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from core.confusion_groups import NORMAL, macro_f1, per_class_f1
from core.subset7 import FAMILY


@dataclass(frozen=True)
class Predictions:
    classes: list[str]
    fold: np.ndarray       # [n] int
    seed: np.ndarray       # [n] int
    video: np.ndarray      # [n] str
    truth: np.ndarray      # [n] class index
    probs: np.ndarray      # [n, C]

    @property
    def pred(self) -> np.ndarray:
        return self.probs.argmax(1)


def load_predictions(path: Path) -> Predictions:
    rows = list(csv.DictReader(Path(path).open()))
    classes = sorted({r["true_label"] for r in rows})
    return Predictions(classes, np.array([int(r["fold"]) for r in rows]), np.array([int(r["seed"]) for r in rows]),
                       np.array([r["source_video_id"] for r in rows]), np.array([classes.index(r["true_label"]) for r in rows]),
                       np.array([[float(r["p_" + c]) for c in classes] for r in rows]))


def load_arm_predictions(path: Path, arm: str) -> Predictions:
    """One arm of an experiment `probs.csv` (columns arm, fold, seed, video, true, p_*), rows sorted by (fold, seed, video)."""

    rows = sorted((r for r in csv.DictReader(Path(path).open()) if r["arm"] == arm),
                  key=lambda r: (int(r["fold"]), int(r["seed"]), r["video"]))
    if not rows:
        raise ValueError(f"no rows for arm {arm!r} in {path}")
    classes = [k[2:] for k in rows[0] if k.startswith("p_")]
    return Predictions(classes, np.array([int(r["fold"]) for r in rows]), np.array([int(r["seed"]) for r in rows]),
                       np.array([r["video"] for r in rows]), np.array([classes.index(r["true"]) for r in rows]),
                       np.array([[float(r["p_" + c]) for c in classes] for r in rows]))


def headline(p: Predictions) -> dict:
    cells = sorted({(f, s) for f, s in zip(p.fold, p.seed)})
    mf = {c: macro_f1(p.truth[m], p.pred[m], len(p.classes)) for c in cells for m in [(p.fold == c[0]) & (p.seed == c[1])]}
    acc = {c: float((p.truth[m] == p.pred[m]).mean()) for c in cells for m in [(p.fold == c[0]) & (p.seed == c[1])]}
    per_fold = {int(f): float(np.mean([v for (ff, _), v in mf.items() if ff == f])) for f in sorted({c[0] for c in cells})}
    return {"macro_f1": float(np.mean(list(mf.values()))), "accuracy": float(np.mean(list(acc.values()))),
            "per_fold_macro_f1": per_fold, "seed_sd_within_fold": float(np.mean([
                np.std([v for (ff, _), v in mf.items() if ff == f], ddof=1) for f in per_fold])),
            "cells": len(cells)}


def per_class(p: Predictions) -> list[dict]:
    """Seed-averaged precision / recall / F1 per class (each fold-seed cell scored, then averaged)."""

    n, cells = len(p.classes), sorted({(f, s) for f, s in zip(p.fold, p.seed)})
    prec, rec, f1 = (np.zeros((len(cells), n)) for _ in range(3))
    for i, (f, s) in enumerate(cells):
        m = (p.fold == f) & (p.seed == s)
        t, y = p.truth[m], p.pred[m]
        for c in range(n):
            tp = float(((t == c) & (y == c)).sum())
            prec[i, c] = tp / max(1.0, float((y == c).sum()))
            rec[i, c] = tp / max(1.0, float((t == c).sum()))
        f1[i] = per_class_f1(t, y, n)
    support = np.array([len({v for v, t in zip(p.video, p.truth) if t == c}) for c in range(n)])
    return [{"class": p.classes[c], "family": FAMILY[p.classes[c]], "unique_val_videos": int(support[c]),
             "precision": float(prec[:, c].mean()), "recall": float(rec[:, c].mean()), "f1": float(f1[:, c].mean())} for c in range(n)]


def confusion_counts(p: Predictions) -> np.ndarray:
    """[true, pred] counts, summed over folds and averaged over seeds."""

    n, seeds = len(p.classes), sorted(set(p.seed.tolist()))
    cm = np.zeros((n, n))
    for t, y in zip(p.truth, p.pred):
        cm[t, y] += 1
    return cm / len(seeds)


def error_structure(p: Predictions) -> dict:
    cm = confusion_counts(p)
    n, ni = len(p.classes), p.classes.index(NORMAL)
    wrong = cm.sum() - np.trace(cm)
    crime = [c for c in range(n) if c != ni]
    crime_to_crime = sum(cm[a, b] for a in crime for b in crime if a != b)
    fam = [FAMILY[c] for c in p.classes]
    within = sum(cm[a, b] for a in crime for b in crime if a != b and fam[a] == fam[b])
    possible_within = sum(1 for a in crime for b in crime if a != b and fam[a] == fam[b])
    possible_all = len(crime) * (len(crime) - 1)
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            pairs.append({"pair": f"{p.classes[a]} <-> {p.classes[b]}", "count": float(cm[a, b] + cm[b, a]),
                          "a_to_b": float(cm[a, b]), "b_to_a": float(cm[b, a])})
    pairs.sort(key=lambda r: -r["count"])
    return {"errors": float(wrong), "crime_to_crime_share": float(crime_to_crime / wrong),
            "crime_to_normal_share": float(sum(cm[a, ni] for a in crime) / wrong),
            "normal_to_crime_share": float(sum(cm[ni, b] for b in crime) / wrong),
            "within_family_share_of_crime_errors": float(within / crime_to_crime),
            "within_family_share_if_errors_were_spread_evenly": float(possible_within / possible_all),
            "top_pairs": pairs[:12]}


def video_reliability(p: Predictions) -> dict:
    """Per unique validation video: share of its evaluations (folds x seeds) that were correct."""

    per = {}
    for v, t, y in zip(p.video, p.truth, p.pred):
        per.setdefault(v, [p.classes[t], []])[1].append(int(t == y))
    acc = {v: float(np.mean(x[1])) for v, x in per.items()}
    vals = np.array(list(acc.values()))
    by_class = {}
    for v, (c, _) in per.items():
        by_class.setdefault(c, []).append(acc[v])
    return {"videos": len(per), "always_right": int((vals == 1).sum()), "always_wrong": int((vals == 0).sum()),
            "sometimes_right": int(((vals > 0) & (vals < 1)).sum()),
            "share_of_errors_from_always_wrong_videos": float(sum(1 - a for v, a in acc.items() if a == 0) / max(1e-9, sum(1 - a for a in acc.values()))),
            "per_class_always_wrong": {c: int(sum(1 for a in x if a == 0)) for c, x in sorted(by_class.items())},
            "per_class_videos": {c: len(x) for c, x in sorted(by_class.items())}}


def topk_accuracy(p: Predictions, ks: Sequence[int] = (1, 2, 3, 5)) -> dict:
    rank = (p.probs > p.probs[np.arange(len(p.truth)), p.truth][:, None]).sum(1) + 1
    return {f"top{k}": float((rank <= k).mean()) for k in ks} | {"median_rank_of_true_class": float(np.median(rank))}


def calibration(p: Predictions, bins: int = 10) -> dict:
    conf, ok = p.probs.max(1), (p.pred == p.truth).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            rows.append({"confidence": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_confidence": float(conf[m].mean()), "accuracy": float(ok[m].mean())})
            ece += m.mean() * abs(conf[m].mean() - ok[m].mean())
    order = np.argsort(-conf)
    coverage = {f"{int(q * 100)}%": float(ok[order[: max(1, int(q * len(ok)))]].mean()) for q in (0.2, 0.4, 0.6, 0.8, 1.0)}
    return {"ece": float(ece), "mean_confidence_right": float(conf[ok == 1].mean()), "mean_confidence_wrong": float(conf[ok == 0].mean()),
            "share_of_errors_with_confidence_above_0.9": float(((conf > 0.9) & (ok == 0)).sum() / max(1, (ok == 0).sum())),
            "reliability": rows, "accuracy_when_keeping_most_confident": coverage}


def normal_vs_crime_auc(p: Predictions) -> float:
    ni = p.classes.index(NORMAL)
    score, pos = 1 - p.probs[:, ni], (p.truth != ni)
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cluster_bootstrap(p: Predictions, draws: int = 2000, seed: int = 0) -> dict:
    """Headline macro-F1 under resampling of videos, stratified by class within each fold.

    Each fold holds the same number of videos per class, so videos are redrawn within their class: a free
    resample of the fold would drop a whole class in about one draw in five and score it as F1 = 0.
    The same resampled videos are used for every seed of a fold.
    """

    rng = np.random.default_rng(seed)
    folds, seeds = sorted(set(p.fold.tolist())), sorted(set(p.seed.tolist()))
    cells, groups = {}, {}
    for f in folds:
        base = np.where((p.fold == f) & (p.seed == seeds[0]))[0]
        videos = p.video[base]
        cells[f] = [np.array([np.where((p.fold == f) & (p.seed == s) & (p.video == v))[0][0] for v in videos]) for s in seeds]
        groups[f] = [np.where(p.truth[base] == c)[0] for c in range(len(p.classes)) if (p.truth[base] == c).any()]
    out = []
    for _ in range(draws):
        per_fold = []
        for f in folds:
            pick = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups[f]])
            per_fold.append(np.mean([macro_f1(p.truth[ix[pick]], p.pred[ix[pick]], len(p.classes)) for ix in cells[f]]))
        out.append(np.mean(per_fold))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"ci95": [float(lo), float(hi)], "sd": float(np.std(out)), "draws": draws, "resampling": "within class, within fold"}


def _macro_f1_present(truth: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over the classes present in `truth` (equals `macro_f1` whenever every class is present)."""

    f1 = per_class_f1(truth, pred, n_classes)
    present = np.unique(truth)
    return float(f1[present].mean())


def paired_cluster_bootstrap(base: Predictions, cand: Predictions | None = None, draws: int = 2000, seed: int = 0) -> dict:
    """Mean-of-cell macro-F1 (and candidate-minus-base difference) under a global source-video bootstrap.

    Unique source videos are redrawn with replacement within their class across the whole evaluation set; a drawn
    video carries all of its fold/seed appearances, and both models are scored on the same draw. Each fold-seed cell is
    scored on its own appearances of the drawn videos (macro-F1 over the classes present in that cell), then cells are
    averaged as in the registered metric. This replaces `cluster_bootstrap`, which redrew videos separately inside
    each fold although validation videos recur across folds. It is conditional on the stored predictions: it does
    not include retraining variance or earlier method selection.
    """

    if cand is not None and (not np.array_equal(base.video, cand.video) or not np.array_equal(base.fold, cand.fold)
                             or not np.array_equal(base.seed, cand.seed) or not np.array_equal(base.truth, cand.truth)):
        raise ValueError("candidate and base must hold the same (fold, seed, video) rows in the same order")
    rng = np.random.default_rng(seed)
    n = len(base.classes)
    videos = sorted(set(base.video.tolist()))
    vclass = {v: int(base.truth[np.where(base.video == v)[0][0]]) for v in videos}
    by_class = {c: [v for v in videos if vclass[v] == c] for c in sorted(set(vclass.values()))}
    cells = sorted({(f, s) for f, s in zip(base.fold, base.seed)})
    rows_of = {(c, v): np.where((base.fold == c[0]) & (base.seed == c[1]) & (base.video == v))[0] for c in cells for v in videos}

    def score(p: Predictions, weights: dict) -> float:
        vals = []
        for c in cells:
            idx = np.concatenate([np.repeat(rows_of[(c, v)], w) for v, w in weights.items() if len(rows_of[(c, v)])] or [np.array([], int)])
            if len(idx):
                vals.append(_macro_f1_present(p.truth[idx], p.pred[idx], n))
        return float(np.mean(vals))

    observed = {v: 1 for v in videos}
    out_b, out_c = [], []
    for _ in range(draws):
        w: dict = {}
        for c, vs in by_class.items():
            for v in rng.choice(vs, size=len(vs), replace=True):
                w[v] = w.get(v, 0) + 1
        out_b.append(score(base, w))
        if cand is not None:
            out_c.append(score(cand, w))
    res = {"draws": draws, "unique_videos": len(videos), "resampling": "global, within class, source video carries all fold/seed rows",
           "base": {"observed": score(base, observed), "ci95": [float(x) for x in np.percentile(out_b, [2.5, 97.5])], "sd": float(np.std(out_b))}}
    if cand is not None:
        d = np.array(out_c) - np.array(out_b)
        res["candidate"] = {"observed": score(cand, observed), "ci95": [float(x) for x in np.percentile(out_c, [2.5, 97.5])]}
        res["difference"] = {"observed": res["candidate"]["observed"] - res["base"]["observed"],
                             "ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])], "sd": float(np.std(d)),
                             "share_of_draws_above_zero": float((d > 0).mean())}
    return res


def length_effect(p: Predictions, frames: Mapping[str, int]) -> dict:
    from scipy.stats import spearmanr

    err = np.array([int(t != y) for t, y in zip(p.truth, p.pred)])
    n = np.array([frames[v] for v in p.video], dtype=float)
    rho, pval = spearmanr(n, err)
    q = np.quantile(n, [0.25, 0.5, 0.75])
    bins = np.digitize(n, q)
    return {"spearman_frames_vs_error": float(rho), "p": float(pval),
            "error_rate_by_length_quartile": [float(err[bins == b].mean()) for b in range(4)],
            "quartile_edges_frames": [float(x) for x in q]}
