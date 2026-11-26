"""Aggregate the B-9 probe reports into one comparable artifact.

`core.p1c_probes` writes one JSON per fitted probe. B-9's question is not about
any single one of them -- it is whether *the best* classical probe on the frozen
features reaches the trained head. Answering that by eye across a directory of
seventeen report files is exactly how a wrong number gets quoted, so the comparison is
computed once, here, and written down.

Two fail-closed rules, both learned rather than invented:

- Every report must carry the same `cache_sha256`, config and validation row
  count. Reports built from two different caches are not comparable, and a
  directory is the easiest place for a stale one to survive unnoticed.
- A probe whose solver hit `max_iter` (`converged: false`) is never selected as
  the best. Its score is retained and reported -- rule 4 -- but a truncated fit
  is a lower bound on that probe, not a measurement of it, and promoting one to
  the headline would understate the features by an unknown amount.

A configuration fitted twice -- a default run and a later explicit sweep point
naming the same value -- collapses to one row rather than being ranked twice,
and the two scores must agree exactly or the collapse fails. A sweep directory
accumulates reruns, and two rows for one configuration would both inflate the
probe count and let a reader read a repeat as independent evidence. The equality
demand turns the accident into a check: these fits are separate processes, so
agreeing to the last bit is a free reproducibility result and disagreeing is a
finding.

Unlike `core.p1c_aggregation`, this aggregates across *probes*, not across the
three registered seeds: a probe is deterministic given its data and seed, so
there is no seed spread to summarise. The spread that matters here is the
source-video clustered CI each report already carries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Fields that must agree across every report before they may be compared.
_SHARED_FIELDS = ("cache_sha256", "config", "train_rows", "validation_rows", "train_stride")


def load_probe_reports(directory: Path) -> list[dict[str, Any]]:
    """Read every probe report in `directory`, sorted by filename for determinism."""

    reports = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "summary.json":
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append(report | {"report_file": path.name})
    if not reports:
        raise ValueError(f"no probe reports found in {directory}")
    return reports


def _require_agreement(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    shared = {}
    for field in _SHARED_FIELDS:
        values = {report[field] for report in reports}
        if len(values) != 1:
            raise ValueError(
                f"probe reports disagree on {field}: {sorted(map(str, values))}; "
                "they were not all fitted on the same cache and are not comparable"
            )
        shared[field] = values.pop()
    return shared


def _configuration_key(report: Mapping[str, Any]) -> str:
    """Identity of a fitted configuration, insensitive to `1` versus `1.0`.

    `--param C=1` and the estimator default `C=1.0` are the same fit; only JSON's
    int/float distinction makes them look different.
    """

    parameters = {
        key: float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        for key, value in report["parameters"].items()
    }
    return json.dumps(
        [report["probe"], report["scaling"], parameters], sort_keys=True, default=str
    )


def _collapse_duplicates(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per configuration, refusing a repeat that scored differently."""

    collapsed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.pop("configuration_key")
        existing = collapsed.get(key)
        if existing is None:
            collapsed[key] = row
            continue
        # Both scores, not just the headline one: the video-level metric is derived
        # from the same frame predictions, so two fits could in principle differ in
        # their predictions while coinciding on frame macro-F1. Checking the pair
        # costs nothing and makes the docstring's "must agree exactly" true as
        # written rather than true of one field.
        differing = [
            field for field in ("frame_macro_f1", "video_level_macro_f1")
            if existing[field] != row[field]
        ]
        if differing:
            raise ValueError(
                f"{existing['report_file']} and {row['report_file']} fit the same configuration "
                + "; ".join(
                    f"but scored {existing[field]} and {row[field]} on {field}" for field in differing
                )
                + "; a repeat of a deterministic fit must reproduce exactly"
            )
        existing.setdefault("duplicate_report_files", []).append(row["report_file"])
    return list(collapsed.values())


def _row(report: Mapping[str, Any]) -> dict[str, Any]:
    ci = report["video_clustered_macro_f1_ci"]
    return {
        "report_file": report["report_file"],
        "configuration_key": _configuration_key(report),
        "probe": report["probe"],
        "scaling": report["scaling"],
        "parameters": report["parameters"],
        "frame_macro_f1": report["frame_metrics"]["macro_f1"],
        "frame_accuracy": report["frame_metrics"]["accuracy"],
        "video_level_macro_f1": report["video_level_metrics"]["macro_f1"],
        "clustered_ci_lower_95": ci["lower_95"],
        "clustered_ci_upper_95": ci["upper_95"],
        "converged": report["converged"],
        "fit_seconds": report["fit_seconds"],
        "predict_seconds": report["predict_seconds"],
    }


def summarize_probe_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank the probes and, given the trained head's aggregate, state the gap.

    `reference` is the run summary the probes are being measured against -- the
    three-seed aggregate written by the training runner -- rather than a figure
    typed in here, so this artifact cannot drift from the run it quotes.
    """

    shared = _require_agreement(reports)
    rows = _collapse_duplicates([_row(report) for report in reports])
    rows.sort(key=lambda row: -row["frame_macro_f1"])
    eligible = [row for row in rows if row["converged"] is not False]
    if not eligible:
        raise ValueError("every probe hit its iteration ceiling; none may be reported as best")

    best = eligible[0]
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "kind": "classical probe sweep on the B-1 frozen-feature cache; validation-only; "
                "inherited-label score",
        "label_honesty": (
            "INHERITED-LABEL. AR fold 1 has no constructible event-only view (D-40); these "
            "probes inherit the cache's scope exactly and never read Test."
        ),
        "metric_name": "validation frame macro_f1 (inherited video-level labels, AR fold 1)",
        "probe_count": len(rows),
        "report_count": len(reports),
        "truncated_fit_count": sum(1 for row in rows if row["converged"] is False),
        "probes": rows,
        "best": best,
        **shared,
    }
    if reference is not None:
        reference_mean = reference["frame_macro_f1"]["mean"]
        payload["reference"] = {
            "experiment_id": reference["experiment_id"],
            "frame_macro_f1_mean": reference_mean,
            "frame_macro_f1_sd": reference["frame_macro_f1"]["sd"],
            "metric_name": reference["metric_name"],
        }
        payload["best_probe_minus_reference"] = best["frame_macro_f1"] - reference_mean
        payload["per_class_frame_f1_best_probe_minus_reference"] = _per_class_delta(
            best["report_file"], reports, reference
        )
    return payload


def _per_class_delta(
    report_file: str, reports: Sequence[Mapping[str, Any]], reference: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    """Per-class F1 of the best probe against the reference's per-seed mean.

    The macro headline can match while individual classes diverge in both
    directions; that pattern is the actual diagnostic B-9 was run for, so it is
    recorded rather than left for a reader to recompute.
    """

    best_report = next(report for report in reports if report["report_file"] == report_file)
    per_class = best_report["frame_metrics"]["per_class"]
    delta = {}
    for class_name, reference_stats in reference["per_class_frame_f1"].items():
        probe_f1 = per_class[class_name]["f1"]
        delta[class_name] = {
            "probe": probe_f1,
            "reference_mean": reference_stats["mean"],
            "reference_sd": reference_stats["sd"],
            "delta": probe_f1 - reference_stats["mean"],
        }
    return delta
