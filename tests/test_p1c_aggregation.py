from __future__ import annotations

import unittest

from core.p1c_aggregation import aggregate_validation_seeds


def clustered_ci(f1: float, **overrides) -> dict:
    # The real shape production always emits: eval.metrics.video_clustered_macro_f1_ci.
    ci = {
        "estimate": f1,
        "lower_95": max(f1 - 0.1, 0.0),
        "upper_95": min(f1 + 0.1, 1.0),
        "iterations": 1000,
        "seed": 0,
        "cluster_count": 5,
        "resampling_unit": "source_video_id",
    }
    ci.update(overrides)
    return ci


def report(seed: int, f1: float, *, test: bool = False, clustered_ci_override: dict | None = None) -> dict:
    labels = ["Abuse", "Normal"]
    return {
        "experiment_id": "P1C-FRAME-GAP-FULL-001",
        "seed": seed,
        "selected_checkpoint": f"checkpoint_epoch_{seed:03d}.pt",
        "best_validation_macro_f1": f1,
        "test_evaluation_performed": test,
        "selected_checkpoint_validation_clustered_ci": (
            clustered_ci(f1) if clustered_ci_override is None else clustered_ci_override
        ),
        "selected_checkpoint_validation_metrics": {
            "labels": labels,
            "accuracy": f1 + 0.1,
            "macro_precision": f1 + 0.05,
            "macro_recall": f1 - 0.05,
            "macro_f1": f1,
            "per_class": {label: {"f1": f1} for label in labels},
        },
    }


class P1CAggregationTests(unittest.TestCase):
    def test_aggregates_registered_validation_seeds(self) -> None:
        aggregate = aggregate_validation_seeds([report(2, 0.9), report(0, 0.5), report(1, 0.7)], expected_experiment_id="P1C-FRAME-GAP-FULL-001")
        self.assertFalse(aggregate["test_evaluation_performed"])
        self.assertAlmostEqual(aggregate["metrics"]["macro_f1"]["mean"], 0.7)
        self.assertEqual([item["seed"] for item in aggregate["seed_checkpoints"]], [0, 1, 2])

    def test_rejects_missing_seed_or_test_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly once"):
            aggregate_validation_seeds([report(0, 0.5), report(0, 0.7), report(2, 0.9)], expected_experiment_id="P1C-FRAME-GAP-FULL-001")
        with self.assertRaisesRegex(ValueError, "Test"):
            aggregate_validation_seeds([report(0, 0.5), report(1, 0.7, test=True), report(2, 0.9)], expected_experiment_id="P1C-FRAME-GAP-FULL-001")

    def test_preserves_a_well_formed_clustered_ci_verbatim(self) -> None:
        aggregate = aggregate_validation_seeds(
            [report(0, 0.5), report(1, 0.7), report(2, 0.9)], expected_experiment_id="P1C-FRAME-GAP-FULL-001"
        )
        ci = aggregate["seed_checkpoints"][0]["video_clustered_macro_f1_ci"]
        self.assertEqual(ci, clustered_ci(0.5))

    def test_rejects_a_missing_clustered_ci(self) -> None:
        reports = [report(0, 0.5), report(1, 0.7), report(2, 0.9)]
        del reports[0]["selected_checkpoint_validation_clustered_ci"]
        with self.assertRaisesRegex(ValueError, "clustered-CI mapping"):
            aggregate_validation_seeds(reports, expected_experiment_id="P1C-FRAME-GAP-FULL-001")

    def test_rejects_a_null_clustered_ci(self) -> None:
        reports = [report(0, 0.5), report(1, 0.7), report(2, 0.9)]
        reports[0]["selected_checkpoint_validation_clustered_ci"] = None
        with self.assertRaisesRegex(ValueError, "clustered-CI mapping"):
            aggregate_validation_seeds(reports, expected_experiment_id="P1C-FRAME-GAP-FULL-001")

    def test_rejects_a_clustered_ci_missing_a_required_key(self) -> None:
        malformed = clustered_ci(0.5)
        del malformed["cluster_count"]
        reports = [report(0, 0.5, clustered_ci_override=malformed), report(1, 0.7), report(2, 0.9)]
        with self.assertRaisesRegex(ValueError, "clustered-CI mapping"):
            aggregate_validation_seeds(reports, expected_experiment_id="P1C-FRAME-GAP-FULL-001")

    def test_rejects_a_clustered_ci_with_lower_above_upper(self) -> None:
        malformed = clustered_ci(0.5, lower_95=0.6, upper_95=0.4)
        reports = [report(0, 0.5, clustered_ci_override=malformed), report(1, 0.7), report(2, 0.9)]
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            aggregate_validation_seeds(reports, expected_experiment_id="P1C-FRAME-GAP-FULL-001")

    def test_rejects_a_clustered_ci_with_too_few_bootstrap_iterations(self) -> None:
        malformed = clustered_ci(0.5, iterations=10)
        reports = [report(0, 0.5, clustered_ci_override=malformed), report(1, 0.7), report(2, 0.9)]
        with self.assertRaisesRegex(ValueError, "iterations"):
            aggregate_validation_seeds(reports, expected_experiment_id="P1C-FRAME-GAP-FULL-001")

    def test_rejects_a_clustered_ci_with_a_single_cluster(self) -> None:
        malformed = clustered_ci(0.5, cluster_count=1)
        reports = [report(0, 0.5, clustered_ci_override=malformed), report(1, 0.7), report(2, 0.9)]
        with self.assertRaisesRegex(ValueError, "cluster_count"):
            aggregate_validation_seeds(reports, expected_experiment_id="P1C-FRAME-GAP-FULL-001")

    def test_rejects_a_clustered_ci_with_the_wrong_resampling_unit(self) -> None:
        malformed = clustered_ci(0.5, resampling_unit="frame_index")
        reports = [report(0, 0.5, clustered_ci_override=malformed), report(1, 0.7), report(2, 0.9)]
        with self.assertRaisesRegex(ValueError, "resampling_unit"):
            aggregate_validation_seeds(reports, expected_experiment_id="P1C-FRAME-GAP-FULL-001")

    def test_rejects_a_ci_estimate_that_disagrees_with_validation_macro_f1(self) -> None:
        reports = [report(0, 0.5, clustered_ci_override=clustered_ci(0.01)), report(1, 0.7), report(2, 0.9)]
        with self.assertRaisesRegex(ValueError, "estimate must equal"):
            aggregate_validation_seeds(reports, expected_experiment_id="P1C-FRAME-GAP-FULL-001")

    def test_rejects_disagreeing_ci_bootstrap_settings(self) -> None:
        reports = [report(0, 0.5), report(1, 0.7, clustered_ci_override=clustered_ci(0.7, seed=99)), report(2, 0.9)]
        with self.assertRaisesRegex(ValueError, "bootstrap iterations or seed"):
            aggregate_validation_seeds(reports, expected_experiment_id="P1C-FRAME-GAP-FULL-001")
