"""Seven-class UCF-Crime subsets: crime-family novelty, candidate enumeration and the pre-registered winner rule."""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

SUBSET_SIZE = 7

#: Semantic crime family per class. A project judgement recorded 2026-10-24, not a published taxonomy.
FAMILY = {
    "Abuse": "violence", "Assault": "violence", "Fighting": "violence", "Shooting": "violence",
    "Burglary": "theft", "Robbery": "theft", "Shoplifting": "theft", "Stealing": "theft",
    "Arson": "fire", "Explosion": "fire",
    "Vandalism": "damage", "RoadAccidents": "traffic", "Arrest": "police",
    "Normal_Videos_event": "normal",
}
N_FAMILIES = len(set(FAMILY.values()))


def novelty(subset: Iterable[str]) -> int:
    """Number of distinct crime families a subset covers (0..7)."""

    return len({FAMILY[c] for c in subset})


def candidate_subsets(classes: Sequence[str], min_novelty: int, size: int = SUBSET_SIZE) -> list[tuple[str, ...]]:
    """Every `size`-class subset (in `classes` order) whose novelty is at least `min_novelty`."""

    missing = set(classes) - set(FAMILY)
    if missing:
        raise ValueError(f"no family registered for {sorted(missing)}")
    return [s for s in itertools.combinations(classes, size) if novelty(s) >= min_novelty]


def subset_id(subset: Sequence[str]) -> str:
    return "+".join(subset)


def aggregate(records: Iterable[Mapping], expected_folds: Sequence[int], expected_seeds: Sequence[int]) -> dict[str, dict]:
    """`{subset_id: {per_fold, mean, novelty, classes}}` from per-(subset, fold, seed) records; refuses incomplete subsets."""

    by = defaultdict(dict)
    for r in records:
        by[r["subset"]][(int(r["fold"]), int(r["seed"]))] = float(r["macro_f1"])
    need = {(f, s) for f in expected_folds for s in expected_seeds}
    out = {}
    for sid, vals in by.items():
        if set(vals) != need:
            raise ValueError(f"{sid}: have {sorted(vals)}, need {sorted(need)}")
        per_fold = {f: float(np.mean([vals[(f, s)] for s in expected_seeds])) for f in expected_folds}
        classes = sid.split("+")
        out[sid] = {"classes": classes, "novelty": novelty(classes), "per_fold": per_fold,
                    "mean": float(np.mean(list(per_fold.values())))}
    return out


def winner(table: Mapping[str, Mapping], min_novelty: int = 0) -> str:
    """Pre-registered rule: highest mean validation macro-F1; ties -> higher novelty -> lexicographic subset id."""

    pool = [sid for sid, r in table.items() if r["novelty"] >= min_novelty]
    if not pool:
        raise ValueError(f"no subset with novelty >= {min_novelty}")
    return min(pool, key=lambda sid: (-round(table[sid]["mean"], 12), -table[sid]["novelty"], sid))
