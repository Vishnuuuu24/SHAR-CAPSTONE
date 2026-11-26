"""Meaning-based 7-family grouping of the 14 UCF-Crime labels (D-72; FINAL_PUSH_PLAN_2026-11-11 Step 5).

The grouping follows the dataset authors' class definitions (who is harmed, whether property is taken,
fire) and was fixed before any family number was computed for a paper table. It is applied to the
14-class predictions as they are -- nothing is retrained -- so a family score is a regrouping of the
14-class result, never a separate model and never comparable with it as an improvement.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from core.confusion_groups import macro_f1, per_class_f1

FAMILY_OF = {
    "Abuse": "violence", "Assault": "violence", "Fighting": "violence", "Shooting": "violence",
    "Burglary": "theft", "Robbery": "theft", "Shoplifting": "theft", "Stealing": "theft",
    "Arson": "fire_blast", "Explosion": "fire_blast",
    "Vandalism": "vandalism", "Arrest": "arrest", "RoadAccidents": "road_accident", "Normal_Videos_event": "normal",
}
FAMILIES = sorted(set(FAMILY_OF.values()))


def family_index(class_names: Sequence[str]) -> np.ndarray:
    """Family position (in `FAMILIES`) of every class, in `class_names` order; raises on an unmapped class."""

    missing = [c for c in class_names if c not in FAMILY_OF]
    if missing:
        raise ValueError(f"classes without a family: {missing}")
    return np.asarray([FAMILIES.index(FAMILY_OF[c]) for c in class_names], dtype=np.int64)


def family_scores(truth: np.ndarray, predicted: np.ndarray, class_names: Sequence[str]) -> dict:
    """Accuracy, macro-F1 and per-family F1 after mapping 14-class truth and prediction to families."""

    mapping = family_index(class_names)
    t, p = mapping[np.asarray(truth, dtype=int)], mapping[np.asarray(predicted, dtype=int)]
    return {"accuracy": float((t == p).mean()), "macro_f1": macro_f1(t, p, len(FAMILIES)),
            "per_family_f1": dict(zip(FAMILIES, per_class_f1(t, p, len(FAMILIES)).tolist()))}


def family_scores_from_confusion(confusion: np.ndarray, class_names: Sequence[str]) -> dict:
    """The same scores from a stored 14x14 confusion matrix (rows = truth), for results without per-video predictions."""

    mapping = family_index(class_names)
    grouped = np.zeros((len(FAMILIES), len(FAMILIES)), dtype=np.float64)
    np.add.at(grouped, (mapping[:, None], mapping[None, :]), np.asarray(confusion, dtype=np.float64))
    tp = np.diag(grouped)
    denominator = grouped.sum(0) + grouped.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(FAMILIES)), where=denominator > 0)
    return {"accuracy": float(tp.sum() / grouped.sum()), "macro_f1": float(f1.mean()), "per_family_f1": dict(zip(FAMILIES, f1.tolist()))}


def top_k_accuracy(truth: np.ndarray, probabilities: np.ndarray, k: int = 3) -> float:
    """Share of videos whose true class is among the `k` most probable."""

    top = np.argsort(-np.asarray(probabilities), axis=1)[:, :k]
    return float((top == np.asarray(truth, dtype=int)[:, None]).any(axis=1).mean())
