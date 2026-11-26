"""Pre-registered late fusion of per-video class probabilities (DINOv2 head A + VideoMAE arm B), equal weight."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.p1c_xd_probe import fold_means, gate_check, run_metrics

FUSION_WEIGHT = 0.5  # pre-registered; never tuned
HIGHLIGHT = ("Shooting", "Fighting", "Assault")
Key = tuple[int, int, str]  # (fold, seed, video id)


def _read(path: Path, arm_col: str | None, arm: str | None):
    with Path(path).open(newline="") as h:
        rd = csv.DictReader(h)
        classes = [c[2:] for c in rd.fieldnames if c.startswith("p_")]
        out: dict[Key, tuple[str, np.ndarray]] = {}
        for r in rd:
            if arm_col and r[arm_col] != arm:
                continue
            key = (int(r["fold"]), int(r["seed"]), r["source_video_id"])
            if key in out:
                raise ValueError(f"duplicate row {key}")
            out[key] = (r["true_label"], np.array([float(r["p_" + c]) for c in classes]))
    return classes, out


def load_videomae_probs(path: Path, arm: str):
    """`(class_names, {(fold, seed, video): (label, probs[14])})` for one arm of the VideoMAE per_video.csv."""

    classes, out = _read(path, "arm", arm)
    if not out:
        raise ValueError(f"no rows for arm {arm!r} in {path}")
    return classes, out


def load_dinov2_probs(path: Path):
    """Same shape, from the dinov2_probs.csv written by the CLI (columns fold, seed, source_video_id, true_label, p_*)."""

    return _read(path, None, None)


def align(a, b):
    """Check class order and (fold, seed, video) sets are identical; return keys in sorted order."""

    (ca, da), (cb, db) = a, b
    if list(ca) != list(cb):
        raise ValueError(f"class order differs: {ca} vs {cb}")
    if set(da) != set(db):
        raise ValueError(f"video sets differ: {len(set(da) ^ set(db))} keys in symmetric difference")
    for k in da:
        if da[k][0] != db[k][0]:
            raise ValueError(f"label mismatch for {k}: {da[k][0]} vs {db[k][0]}")
    return sorted(da)


def fuse(pa: np.ndarray, pb: np.ndarray, weight: float = FUSION_WEIGHT) -> np.ndarray:
    return weight * pa + (1.0 - weight) * pb


def _runs(name: str, classes: Sequence[str], probs: Mapping[Key, tuple[str, np.ndarray]]):
    by_run: dict[tuple[int, int], list[Key]] = defaultdict(list)
    for k in sorted(probs):
        by_run[(k[0], k[1])].append(k)
    runs = []
    for (fold, seed), keys in sorted(by_run.items()):
        truth = [classes.index(probs[k][0]) for k in keys]
        pred = [int(np.argmax(probs[k][1])) for k in keys]
        runs.append({"arm": name, "fold": fold, "seed": seed, "metrics": run_metrics(truth, pred, classes)})
    return runs


def fused_probs(a, b, keys: Sequence[Key], weight: float = FUSION_WEIGHT):
    return {k: (a[1][k][0], fuse(a[1][k][1], b[1][k][1], weight)) for k in keys}


def evaluate(systems: Mapping[str, tuple[Sequence[str], Mapping[Key, tuple[str, np.ndarray]]]], baseline: str, fused: Sequence[str]):
    """Macro-F1 per run / fold-mean per system; paired fused-minus-baseline deltas, gate, per-class F1 deltas."""

    classes = list(next(iter(systems.values()))[0])
    runs = {n: _runs(n, classes, p) for n, (_c, p) in systems.items()}
    allruns = [r for rs in runs.values() for r in rs]
    out: dict[str, Any] = {"systems": {}, "comparisons": {}}
    for n, rs in runs.items():
        fm = fold_means(allruns, n)
        out["systems"][n] = {"macro_f1_mean": float(np.mean([r["metrics"]["macro_f1"] for r in rs])), "fold_mean_macro_f1": fm,
                             "per_class_f1_mean": {c: float(np.mean([r["metrics"]["per_class"][c]["f1"] for r in rs])) for c in classes}}
    base = out["systems"][baseline]
    for n in fused:
        s = out["systems"][n]
        g = gate_check(base["fold_mean_macro_f1"], s["fold_mean_macro_f1"])
        out["comparisons"][f"{n}_minus_{baseline}"] = {
            "mean_delta": s["macro_f1_mean"] - base["macro_f1_mean"], "gate": g,
            "folds_positive": sum(d > 0 for d in g["per_fold_delta"].values()),
            "per_class_f1_delta": {c: s["per_class_f1_mean"][c] - base["per_class_f1_mean"][c] for c in classes},
            "highlight": {c: s["per_class_f1_mean"][c] - base["per_class_f1_mean"][c] for c in HIGHLIGHT if c in classes}}
    return out


def write_probs_csv(path: Path, classes: Sequence[str], probs: Mapping[Key, tuple[str, np.ndarray]], system: str | None = None) -> None:
    head = (["arm"] if system else []) + ["fold", "seed", "source_video_id", "true_label", "pred_label"] + ["p_" + c for c in classes]
    with Path(path).open("w", newline="") as h:
        w = csv.writer(h); w.writerow(head)
        for k in sorted(probs):
            lab, p = probs[k]
            w.writerow(([system] if system else []) + [k[0], k[1], k[2], lab, classes[int(np.argmax(p))]] + [f"{x:.6f}" for x in p])


def correctness_overlap(a, b, keys: Sequence[Key]) -> dict[str, float]:
    """Fraction of (fold, seed, video) rows both systems get right / only a / only b / neither (argmax vs label)."""

    classes = list(a[0])
    ra = np.array([int(np.argmax(a[1][k][1])) == classes.index(a[1][k][0]) for k in keys])
    rb = np.array([int(np.argmax(b[1][k][1])) == classes.index(b[1][k][0]) for k in keys])
    n = len(keys)
    return {"both": float((ra & rb).sum() / n), "only_a": float((ra & ~rb).sum() / n), "only_b": float((~ra & rb).sum() / n),
            "neither": float((~ra & ~rb).sum() / n), "n_rows": n,
            "phi": float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float("nan")}
