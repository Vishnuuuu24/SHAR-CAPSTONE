"""Domain-gap diagnostics between XD-Violence and UCF-Crime pooled DINOv2 video vectors (CPU, numpy/sklearn).

Pure functions on already-built video-level vectors; data loading lives in `scripts/diagnose_xd_domain_gap.py`.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

LR_C = 0.01  # strong L2 on standardised 1536-D features; fixed, not tuned


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def video_vectors(features: np.ndarray, video_ids: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Mean of each video's frame vectors, then L2-normalised. Returns (sorted ids, [V, D])."""

    ids = np.asarray(video_ids)
    order = sorted(set(video_ids))
    return order, unit(np.stack([np.asarray(features[ids == v], dtype=np.float64).mean(0) for v in order]))


def centroids(vecs: np.ndarray, labels: Sequence[str]) -> dict[str, np.ndarray]:
    lab = np.asarray(labels)
    return {c: unit(vecs[lab == c].mean(0)) for c in sorted(set(labels))}


def within_class_cos(vecs: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    cents = centroids(vecs, labels)
    lab = np.asarray(labels)
    return {c: float((vecs[lab == c] @ cents[c]).mean()) for c in cents}


def centroid_rank(src_centroid: np.ndarray, targets: Mapping[str, np.ndarray], same_name: str) -> dict:
    cos = {c: float(src_centroid @ v) for c, v in targets.items()}
    ranked = sorted(cos, key=lambda c: -cos[c])
    return {"cosine": cos, "ranking": ranked, "same_name_rank": ranked.index(same_name) + 1}


def _model():
    return LogisticRegression(C=LR_C, max_iter=2000)


def fit_predict(xtr, ytr, xte):
    """Standardise on train only, fit LR, return (labels, decision/proba matrix)."""

    sc = StandardScaler().fit(xtr)
    m = _model().fit(sc.transform(xtr), ytr)
    return m.predict(sc.transform(xte)), m.decision_function(sc.transform(xte)), m.classes_


def recall_report(y_true, y_pred, classes) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rec = {c: float((y_pred[y_true == c] == c).mean()) for c in classes if (y_true == c).any()}
    return {"n": int(len(y_true)), "accuracy": float((y_true == y_pred).mean()),
            "balanced_accuracy": float(np.mean(list(rec.values()))), "recall": rec}


def domain_separability(xa: np.ndarray, xb: np.ndarray, seed: int, folds: int = 5) -> dict:
    """5-fold stratified CV of LR telling set A (label 0) from set B (label 1); pooled out-of-fold scores."""

    x = np.vstack([xa, xb]); y = np.r_[np.zeros(len(xa), int), np.ones(len(xb), int)]
    score = np.zeros(len(y)); pred = np.zeros(len(y), int)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, y):
        p, d, _ = fit_predict(x[tr], y[tr], x[te])
        pred[te], score[te] = p, d
    return {"n_a": int(len(xa)), "n_b": int(len(xb)), "accuracy": float((pred == y).mean()),
            "roc_auc": float(roc_auc_score(y, score)), "chance_accuracy": float(max(len(xa), len(xb)) / len(y))}


def balanced_subsample(labels: Sequence[str], per_class: int, rng: np.random.Generator) -> np.ndarray:
    lab = np.asarray(labels)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(lab == c), per_class, replace=False) for c in sorted(set(labels))]))


def transfer(xtr, ytr, xte, yte) -> dict:
    p, _, classes = fit_predict(xtr, ytr, xte)
    return recall_report(yte, p, list(classes))


def leave_one_group_out(x, y, groups) -> dict:
    """Train on all groups but one, test on the held-out group; pooled predictions."""

    y, groups = np.asarray(y), np.asarray(groups)
    pred = np.empty(len(y), dtype=y.dtype)
    for g in sorted(set(groups)):
        te = groups == g
        pred[te] = fit_predict(x[~te], y[~te], x[te])[0]
    return recall_report(y, pred, sorted(set(y)))


def kfold_cv(x, y, seed: int, folds: int = 5) -> dict:
    y = np.asarray(y); pred = np.empty(len(y), dtype=y.dtype)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, y):
        pred[te] = fit_predict(x[tr], y[tr], x[te])[0]
    return recall_report(y, pred, sorted(set(y)))


def mean_reports(reports: Sequence[dict]) -> dict:
    """Mean and std over repeated subsamples of accuracy, balanced accuracy and per-class recall."""

    f = lambda vals: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return {"repeats": len(reports), "accuracy": f([r["accuracy"] for r in reports]),
            "balanced_accuracy": f([r["balanced_accuracy"] for r in reports]),
            "recall": {c: f([r["recall"][c] for r in reports]) for c in reports[0]["recall"]}}
