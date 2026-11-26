"""Fail-closed aggregation for the three registered P1C validation seeds."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence


_SEEDS = (0, 1, 2)
_SUMMARY_METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _summary(values: Sequence[float]) -> dict[str, float]:
    if len(values) != len(_SEEDS):
        raise ValueError("three registered seed values are required")
    return {"mean": statistics.fmean(values), "sample_std": statistics.stdev(values), "minimum": min(values), "maximum": max(values)}


_CI_KEYS = frozenset(
    {"estimate", "lower_95", "upper_95", "iterations", "seed", "cluster_count", "resampling_unit"}
)


def _clustered_ci(value: object, field: str) -> dict[str, Any]:
    """Validate the exact shape ``eval.metrics.video_clustered_macro_f1_ci`` returns.

    Every other field in a seed report is validated before it reaches the
    aggregate; this one previously reached it via a bare ``.get()``, so a
    missing or malformed clustered CI was embedded verbatim instead of
    failing closed.
    """

    if not isinstance(value, Mapping) or set(value) != _CI_KEYS:
        raise ValueError(f"{field} must be a clustered-CI mapping with exactly {sorted(_CI_KEYS)}")
    estimate = _number(value.get("estimate"), f"{field}.estimate")
    lower = _number(value.get("lower_95"), f"{field}.lower_95")
    upper = _number(value.get("upper_95"), f"{field}.upper_95")
    for name, number in (("estimate", estimate), ("lower_95", lower), ("upper_95", upper)):
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{field}.{name} must be a macro-F1 value in [0,1]")
    if lower > upper:
        raise ValueError(f"{field}.lower_95 must not exceed {field}.upper_95")
    iterations = value.get("iterations")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 100:
        raise ValueError(f"{field}.iterations must be an integer >= 100")
    bootstrap_seed = value.get("seed")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError(f"{field}.seed must be a non-negative integer")
    cluster_count = value.get("cluster_count")
    if isinstance(cluster_count, bool) or not isinstance(cluster_count, int) or cluster_count < 2:
        raise ValueError(f"{field}.cluster_count must be an integer >= 2")
    if value.get("resampling_unit") != "source_video_id":
        raise ValueError(f"{field}.resampling_unit must equal 'source_video_id'")
    return {
        "estimate": estimate,
        "lower_95": lower,
        "upper_95": upper,
        "iterations": iterations,
        "seed": bootstrap_seed,
        "cluster_count": cluster_count,
        "resampling_unit": "source_video_id",
    }


def aggregate_validation_seeds(
    seed_reports: Sequence[Mapping[str, Any]], *, expected_experiment_id: str
) -> dict[str, Any]:
    """Summarize exactly three validation-only seed reports without pooling CIs.

    Cluster bootstrap intervals belong to each fitted model; pooling them across
    independently trained seeds would misrepresent the resampling unit. They
    are therefore retained verbatim per seed while model-to-model variability
    is reported as the three-seed sample standard deviation.
    """

    if len(seed_reports) != len(_SEEDS):
        raise ValueError("P1C aggregation requires exactly three seed reports")
    reports_by_seed: dict[int, Mapping[str, Any]] = {}
    for report in seed_reports:
        if report.get("experiment_id") != expected_experiment_id:
            raise ValueError("seed report experiment_id does not match the registered aggregate")
        seed = report.get("seed")
        if isinstance(seed, bool) or seed not in _SEEDS or seed in reports_by_seed:
            raise ValueError("seed reports must contain each registered seed exactly once")
        if report.get("test_evaluation_performed") is not False:
            raise ValueError("P1C validation aggregate refuses reports with Test evaluation")
        if not isinstance(report.get("selected_checkpoint"), str) or not report["selected_checkpoint"]:
            raise ValueError("each seed report must bind a validation-selected checkpoint")
        metrics = report.get("selected_checkpoint_validation_metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("each seed report needs selected_checkpoint_validation_metrics")
        if not isinstance(metrics.get("labels"), list) or not metrics["labels"]:
            raise ValueError("each seed report must register its ordered labels")
        reports_by_seed[seed] = report

    ordered = [reports_by_seed[seed] for seed in _SEEDS]
    labels = ordered[0]["selected_checkpoint_validation_metrics"]["labels"]
    if any(report["selected_checkpoint_validation_metrics"]["labels"] != labels for report in ordered[1:]):
        raise ValueError("seed reports disagree on the ordered class mapping")
    metric_summary = {
        name: _summary([_number(report["selected_checkpoint_validation_metrics"].get(name), f"{name} for seed {seed}") for seed, report in zip(_SEEDS, ordered, strict=True)])
        for name in _SUMMARY_METRICS
    }
    clustered_cis = [
        _clustered_ci(
            report.get("selected_checkpoint_validation_clustered_ci"),
            f"selected_checkpoint_validation_clustered_ci for seed {seed}",
        )
        for seed, report in zip(_SEEDS, ordered, strict=True)
    ]
    for seed, report, clustered_ci in zip(_SEEDS, ordered, clustered_cis, strict=True):
        macro_f1 = _number(
            report["selected_checkpoint_validation_metrics"].get("macro_f1"),
            f"macro_f1 for seed {seed}",
        )
        if not math.isclose(clustered_ci["estimate"], macro_f1, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"clustered-CI estimate must equal validation macro_f1 for seed {seed}")
    bootstrap_settings = {(ci["iterations"], ci["seed"]) for ci in clustered_cis}
    if len(bootstrap_settings) != 1:
        raise ValueError("seed reports disagree on clustered-CI bootstrap iterations or seed")
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        f1_values = []
        for seed, report in zip(_SEEDS, ordered, strict=True):
            values = report["selected_checkpoint_validation_metrics"].get("per_class", {}).get(label)
            if not isinstance(values, Mapping):
                raise ValueError(f"seed {seed} lacks per-class metrics for {label}")
            f1_values.append(_number(values.get("f1"), f"per-class F1 for {label}, seed {seed}"))
        per_class[str(label)] = _summary(f1_values)
    return {
        "kind": "P1C three-seed validation-only aggregate; not Test evaluation",
        "experiment_id": expected_experiment_id,
        "required_seeds": list(_SEEDS),
        "selection_rule": "best_validation_macro_f1",
        "test_evaluation_performed": False,
        "metrics": metric_summary,
        "per_class_f1": per_class,
        "seed_checkpoints": [
            {
                "seed": seed,
                "selected_checkpoint": report["selected_checkpoint"],
                "best_validation_macro_f1": _number(report.get("best_validation_macro_f1"), f"best_validation_macro_f1 for seed {seed}"),
                "video_clustered_macro_f1_ci": clustered_ci,
            }
            for seed, report, clustered_ci in zip(_SEEDS, ordered, clustered_cis, strict=True)
        ],
        "ci_policy": "Per-seed source-video clustered CIs are retained; they are not pooled across independently trained seeds.",
    }
