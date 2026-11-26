"""Helpers for the XD-extra training probe: extra-video selection, per-run metrics, aggregation, gate, leakage check."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from core.p1c_xd_extra import stable_seed
from eval.metrics import classification_metrics

TARGETS = ("Shooting", "Fighting", "Abuse")
ARM_K: dict[str, dict[str, int]] = {
    "baseline": {},
    "xd_k34": {"Shooting": 34, "Fighting": 34, "Abuse": 31},
    "xd_k102": {"Shooting": 102, "Fighting": 102, "Abuse": 31},
}
GATE_DELTA = 0.04
XD_DATASET = "xd_violence_train"


def select_extra_videos(videos_by_class: Mapping[str, Sequence[str]], arm: str, seed: int = 0) -> list[str]:
    """Prefix of a fixed per-class seeded shuffle, so k102 is a superset of k34 and every fold gets the same videos."""

    chosen: list[str] = []
    for cls, k in ARM_K[arm].items():
        ids = sorted(videos_by_class.get(cls, ()))
        order = np.random.default_rng(stable_seed(cls, seed)).permutation(len(ids))
        chosen += [ids[i] for i in order[: min(k, len(ids))]]
    return chosen


def extra_sequences(features: np.ndarray, rows: Sequence[Mapping[str, Any]], video_ids: Sequence[str], class_names: Sequence[str],
                    permute_seed: int):
    """`{(dataset, 'xd:<id>'): (sequence[8, 1536] float32, row)}` in the shape `train_sequence_head` consumes."""

    from core.p1c_head_variants import group_pooled_sequences_by_video

    wanted = set(video_ids)
    keep = [i for i, r in enumerate(rows) if r["video_id"] in wanted]
    sub_rows = [{"source_dataset": XD_DATASET, "source_video_id": "xd:" + rows[i]["video_id"],
                 "source_frame_index": int(rows[i]["source_frame_index"]), "target": class_names.index(rows[i]["class_name"]),
                 "label": rows[i]["class_name"]} for i in keep]
    return group_pooled_sequences_by_video(np.asarray(features[keep], dtype=np.float32), sub_rows,
                                           permute_frame_order=True, permute_seed=permute_seed)


def run_metrics(truth: Sequence[int], pred: Sequence[int], class_names: Sequence[str]) -> dict[str, Any]:
    m = classification_metrics([class_names[t] for t in truth], [class_names[p] for p in pred], class_names)
    other = [c for c in class_names if c not in TARGETS]
    return {"macro_f1": m["macro_f1"], "per_class": {c: {"f1": v["f1"], "recall": v["recall"], "support": v["support"]}
                                                       for c, v in m["per_class"].items()},
            "nontarget_mean_f1": float(np.mean([m["per_class"][c]["f1"] for c in other])), "confusion": m["confusion_matrix"]}


def _ms(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else float("nan"), "n": int(a.size)}


def fold_means(runs: Sequence[Mapping[str, Any]], arm: str) -> dict[int, float]:
    """Macro-F1 averaged over seeds, per fold."""

    by_fold: dict[int, list[float]] = defaultdict(list)
    for r in runs:
        if r["arm"] == arm:
            by_fold[int(r["fold"])].append(r["metrics"]["macro_f1"])
    return {f: float(np.mean(v)) for f, v in sorted(by_fold.items())}


def gate_check(base: Mapping[int, float], arm: Mapping[int, float], n_folds: int = 4, delta: float = GATE_DELTA) -> dict[str, Any]:
    """Pre-registered gate: fold-mean (over seeds) macro-F1 gain >= `delta` on every one of `n_folds` folds."""

    deltas = {f: arm[f] - base[f] for f in sorted(set(base) & set(arm))}
    return {"per_fold_delta": deltas, "threshold": delta, "folds_passing": sum(d >= delta for d in deltas.values()),
            "folds_compared": len(deltas), "passed": len(deltas) == n_folds and all(d >= delta for d in deltas.values())}


def aggregate(runs: Sequence[Mapping[str, Any]], class_names: Sequence[str]) -> dict[str, Any]:
    """Per-arm summary over run dicts `{arm, fold, seed, metrics}` (metrics from `run_metrics`)."""

    out: dict[str, Any] = {}
    arms = sorted({r["arm"] for r in runs}, key=list(ARM_K).index)
    base = fold_means(runs, "baseline") if "baseline" in arms else {}
    for arm in arms:
        rs = [r for r in runs if r["arm"] == arm]
        fm = fold_means(runs, arm)
        per_class = {c: {"f1": _ms([r["metrics"]["per_class"][c]["f1"] for r in rs]),
                         "recall": _ms([r["metrics"]["per_class"][c]["recall"] for r in rs])} for c in class_names}
        entry = {"macro_f1_over_runs": _ms([r["metrics"]["macro_f1"] for r in rs]), "fold_mean_macro_f1": fm,
                 "macro_f1_fold_averaged_sd_across_folds": _ms(list(fm.values())),
                 "per_class": per_class, "targets": {c: per_class[c] for c in TARGETS if c in per_class},
                 "nontarget_mean_f1_over_runs": _ms([r["metrics"]["nontarget_mean_f1"] for r in rs]),
                 "confusion_summed_over_folds_and_seeds": np.sum([r["metrics"]["confusion"] for r in rs], axis=0).tolist(),
                 "confusion_labels": list(class_names)}
        if arm != "baseline" and base:
            entry["gate"] = gate_check(base, fm)
            entry["per_class_f1_delta_vs_baseline"] = {c: per_class[c]["f1"]["mean"] - out["baseline"]["per_class"][c]["f1"]["mean"]
                                                       for c in class_names}
        out[arm] = entry
    return out


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def leakage_check(xd_means: np.ndarray, xd_ids: Sequence[str], val_means: np.ndarray, val_labels: Sequence[str],
                  val_ids: Sequence[str]) -> dict[str, Any]:
    """Max cosine of each XD video mean feature to any UCF validation video mean feature (raw and global-mean-centred)."""

    out: dict[str, Any] = {"n_xd_videos": len(xd_ids), "n_val_videos": len(val_ids)}
    centre = val_means.mean(axis=0, keepdims=True)
    for name, (a, b) in {"raw": (xd_means, val_means), "centred": (xd_means - centre, val_means - centre)}.items():
        sims = _unit(a) @ _unit(b).T
        best, mx = sims.argmax(axis=1), sims.max(axis=1)
        out[name] = {"max_cosine_quantiles": {str(q): float(np.quantile(mx, q)) for q in (0, 0.25, 0.5, 0.75, 0.95, 1)},
                     "count_above_0.95": int((mx > 0.95).sum()), "count_above_0.98": int((mx > 0.98).sum()),
                     "nearest_val_class_counts": dict(Counter(val_labels[i] for i in best)),
                     "top5": [{"xd_video": xd_ids[i], "cosine": float(mx[i]), "nearest_val_video": val_ids[best[i]],
                               "nearest_val_class": val_labels[best[i]]} for i in np.argsort(-mx)[:5]]}
    return out
