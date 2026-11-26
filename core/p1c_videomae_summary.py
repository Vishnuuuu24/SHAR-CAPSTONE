"""Cross-fold aggregation and the pre-registered gate for the VideoMAE partial fine-tune."""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Registered best system: STEP (frame order permuted) attention pooling, video-level macro-F1 (D-53).
REGISTERED_MEAN = 0.4599
GATE_DELTA = 0.04


def _sd(values: Sequence[float]) -> float:
    return st.stdev(values) if len(values) > 1 else 0.0


def load_registered_baseline(summary_path: Path) -> dict[int, float]:
    """Per-fold step_permuted video-level macro-F1 means from the registered summary."""

    import json

    per_fold = json.load(open(summary_path))["per_fold_per_head"]
    return {int(f): v["step_permuted"]["video_level_macro_f1_mean"] for f, v in per_fold.items()}


def gate(deltas_by_fold: Mapping[int, float], folds: Sequence[int], threshold: float = GATE_DELTA) -> dict[str, Any]:
    """Pre-registered gate: delta >= threshold on EVERY one of `folds` (and all folds present)."""

    present = [f for f in folds if f in deltas_by_fold]
    passed = [f for f in present if deltas_by_fold[f] >= threshold]
    return {"threshold": threshold, "folds_passed": passed, "n_passed": len(passed), "n_folds": len(folds),
            "complete": len(present) == len(folds), "met": len(present) == len(folds) and len(passed) == len(folds)}


def _fold_means(runs: Sequence[Mapping[str, Any]], key: str = "final_epoch_macro_f1") -> dict[int, float]:
    by: dict[int, list[float]] = {}
    for r in runs:
        by.setdefault(r["fold"], []).append(r[key])
    return {f: st.mean(v) for f, v in sorted(by.items())}


def aggregate(runs: Sequence[Mapping[str, Any]], class_names: Sequence[str], baseline_by_fold: Mapping[int, float] | None,
              folds: Sequence[int] = (1, 2, 3, 4)) -> dict[str, Any]:
    arms = sorted({r["arm"] for r in runs})
    out: dict[str, Any] = {"arms": {}, "paired_vs_k0": {}, "vs_registered_0.4599": {},
                           "comparison_note": "cross-protocol: the registered 0.4599 is STEP attention pooling over a video's "
                                              "DINOv2 frame features; these arms are VideoMAE clip-level. Compared, not ranked."}
    for arm in arms:
        sel = [r for r in runs if r["arm"] == arm]
        f1 = [r["final_epoch_macro_f1"] for r in sel]
        fm = _fold_means(sel)
        within = [_sd([r["final_epoch_macro_f1"] for r in sel if r["fold"] == f]) for f in fm]
        out["arms"][arm] = {
            "n_runs": len(sel), "trainable_parameters": sel[0]["trainable_parameters"],
            "final_epoch_macro_f1_mean": st.mean(f1), "sd_over_runs": _sd(f1),
            "sd_across_fold_means": _sd(list(fm.values())), "mean_within_fold_seed_sd": st.mean(within),
            "per_fold_mean": fm, "accuracy_mean": st.mean(r["final_epoch_accuracy"] for r in sel),
            "per_class_f1_mean": {c: st.mean(r["per_class_f1"][c] for r in sel) for c in class_names},
            "per_class_recall_mean": {c: st.mean(r["per_class_recall"][c] for r in sel) for c in class_names},
            "oracle_selected_macro_f1_mean_NOT_A_HEADLINE": st.mean(r["oracle_selected_epoch"]["val_macro_f1"] for r in sel),
            "mean_epoch_curve": [st.mean(r["epoch_curve"][e]["val_macro_f1"] for r in sel) for e in range(len(sel[0]["epoch_curve"]))],
            "wall_seconds_mean": st.mean(r["wall_seconds"] for r in sel),
        }
        if baseline_by_fold:
            d = {f: fm[f] - baseline_by_fold[f] for f in fm if f in baseline_by_fold}
            out["vs_registered_0.4599"][arm] = {"per_fold_delta": d, "mean_delta": st.mean(d.values()), "sd_across_folds": _sd(list(d.values())),
                                                "mean_vs_scalar_0.4599": st.mean(fm.values()) - REGISTERED_MEAN, "gate": gate(d, folds)}
    if "k0" in arms:
        base = out["arms"]["k0"]["per_fold_mean"]
        for arm in arms:
            if arm == "k0":
                continue
            fm = out["arms"][arm]["per_fold_mean"]
            d = {f: fm[f] - base[f] for f in fm if f in base}
            out["paired_vs_k0"][arm] = {"per_fold_delta": d, "mean_delta": st.mean(d.values()), "sd_across_folds": _sd(list(d.values())),
                                        "folds_positive": sum(v > 0 for v in d.values()),
                                        "per_class_f1_delta": {c: out["arms"][arm]["per_class_f1_mean"][c] - out["arms"]["k0"]["per_class_f1_mean"][c] for c in class_names},
                                        "gate": gate(d, folds)}
    return out


def write_per_video_csv(path: Path, runs: Sequence[Mapping[str, Any]], class_names: Sequence[str]) -> None:
    """One row per (arm, fold, seed, video): final-epoch mean-clip probabilities, for later late fusion."""

    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["arm", "fold", "seed", "source_video_id", "true_label", "pred_label", *[f"p_{c}" for c in class_names]])
        for r in sorted(runs, key=lambda r: (r["arm"], r["fold"], r["seed"])):
            for v in r["per_video"]:
                w.writerow([r["arm"], r["fold"], r["seed"], v["source_video_id"], v["true_label"], v["pred_label"], *[f"{p:.6f}" for p in v["probs"]]])
