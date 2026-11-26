"""Tests for `core.p1c_probe_summary` (B-9 aggregation).

The two behaviours under test are the two ways this artifact could quietly
report a wrong number: comparing probes fitted on different caches, and letting
a truncated fit become the headline.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_probe_summary import (  # noqa: E402
    load_probe_reports,
    summarize_probe_reports,
)

CLASS_NAMES = ["cat", "dog"]


def _report(
    name: str,
    macro_f1: float,
    *,
    converged: bool | None = True,
    cache: str = "abc",
    parameters: dict | None = None,
) -> dict:
    return {
        "report_file": name,
        "cache_sha256": cache,
        "config": "configs/x.yaml",
        "train_rows": 100,
        "validation_rows": 20,
        "train_stride": 1,
        "probe": "logistic_regression",
        "scaling": "standardize",
        "parameters": {"C": 1.0} if parameters is None else parameters,
        "converged": converged,
        "fit_seconds": 1.0,
        "predict_seconds": 0.5,
        "frame_metrics": {
            "macro_f1": macro_f1,
            "accuracy": macro_f1,
            "per_class": {name: {"f1": macro_f1} for name in CLASS_NAMES},
        },
        "video_level_metrics": {"macro_f1": macro_f1},
        "video_clustered_macro_f1_ci": {"lower_95": macro_f1 - 0.1, "upper_95": macro_f1 + 0.1},
    }


REFERENCE = {
    "experiment_id": "REF-001",
    "metric_name": "validation frame macro_f1",
    "frame_macro_f1": {"mean": 0.35, "sd": 0.005},
    "per_class_frame_f1": {name: {"mean": 0.35, "sd": 0.005} for name in CLASS_NAMES},
}


class LoadTests(unittest.TestCase):
    def test_summary_is_not_read_back_as_a_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.json").write_text(json.dumps(_report("a.json", 0.3)), encoding="utf-8")
            (root / "summary.json").write_text("{}", encoding="utf-8")
            self.assertEqual([r["report_file"] for r in load_probe_reports(root)], ["a.json"])

    def test_empty_directory_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_probe_reports(Path(directory))


class SummarizeTests(unittest.TestCase):
    def test_probes_are_ranked_and_the_best_is_selected(self) -> None:
        payload = summarize_probe_reports(
            [_report("a.json", 0.30, parameters={"C": 0.1}),
             _report("b.json", 0.33, parameters={"C": 0.01}),
             _report("c.json", 0.31, parameters={"C": 1.0})],
            experiment_id="X",
        )
        self.assertEqual([row["frame_macro_f1"] for row in payload["probes"]], [0.33, 0.31, 0.30])
        self.assertEqual(payload["best"]["report_file"], "b.json")
        self.assertEqual(payload["probe_count"], 3)

    def test_a_truncated_fit_is_reported_but_never_selected(self) -> None:
        payload = summarize_probe_reports(
            [_report("a.json", 0.40, converged=False, parameters={"C": 10.0}),
             _report("b.json", 0.33, parameters={"C": 0.01})],
            experiment_id="X",
        )
        self.assertEqual(payload["best"]["report_file"], "b.json")
        self.assertEqual(payload["truncated_fit_count"], 1)
        # Kept visible under rule 4, just not promoted.
        self.assertIn(0.40, [row["frame_macro_f1"] for row in payload["probes"]])

    def test_knn_has_no_convergence_flag_and_stays_eligible(self) -> None:
        payload = summarize_probe_reports([_report("a.json", 0.33, converged=None)], experiment_id="X")
        self.assertEqual(payload["best"]["report_file"], "a.json")

    def test_all_truncated_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            summarize_probe_reports([_report("a.json", 0.4, converged=False)], experiment_id="X")

    def test_reports_from_two_caches_are_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            summarize_probe_reports(
                [_report("a.json", 0.3), _report("b.json", 0.3, cache="different", parameters={"C": 0.1})],
                experiment_id="X",
            )
        self.assertIn("cache_sha256", str(caught.exception))

    def test_one_configuration_fitted_twice_collapses_to_one_row(self) -> None:
        payload = summarize_probe_reports(
            # The estimator default and an explicit sweep point naming the same
            # value, exactly as run-a and run-b produced on the real cache.
            [_report("default.json", 0.33, parameters={"C": 1.0}),
             _report("sweep-C1.json", 0.33, parameters={"C": 1})],
            experiment_id="X",
        )
        self.assertEqual(payload["probe_count"], 1)
        self.assertEqual(payload["report_count"], 2)
        self.assertEqual(payload["probes"][0]["duplicate_report_files"], ["sweep-C1.json"])

    def test_a_repeat_that_scored_differently_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            summarize_probe_reports(
                [_report("a.json", 0.33, parameters={"C": 1.0}),
                 _report("b.json", 0.34, parameters={"C": 1})],
                experiment_id="X",
            )
        self.assertIn("must reproduce exactly", str(caught.exception))

    def test_different_configurations_are_not_collapsed(self) -> None:
        payload = summarize_probe_reports(
            [_report("a.json", 0.33, parameters={"C": 1.0}),
             _report("b.json", 0.31, parameters={"C": 0.1})],
            experiment_id="X",
        )
        self.assertEqual(payload["probe_count"], 2)

    def test_reference_gap_is_computed_from_the_reference_file(self) -> None:
        payload = summarize_probe_reports(
            [_report("a.json", 0.33)], experiment_id="X", reference=REFERENCE
        )
        self.assertAlmostEqual(payload["best_probe_minus_reference"], 0.33 - 0.35)
        self.assertAlmostEqual(
            payload["per_class_frame_f1_best_probe_minus_reference"]["cat"]["delta"], 0.33 - 0.35
        )

    def test_reference_is_optional(self) -> None:
        payload = summarize_probe_reports([_report("a.json", 0.33)], experiment_id="X")
        self.assertNotIn("best_probe_minus_reference", payload)


if __name__ == "__main__":
    unittest.main()
