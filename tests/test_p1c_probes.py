"""Tests for `core.p1c_probes` (B-9).

Synthetic, separable features rather than the real 1.6 GB cache: what is under
test is the probe harness -- scaling fitted on train only, video-preserving
striding, and scoring routed through the same metric functions a training run
uses -- not scikit-learn's estimators, which are not ours to test.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_probes import (  # noqa: E402
    apply_scaling,
    build_probe,
    fit_and_score_probe,
    stride_rows_within_videos,
)
from scripts.run_p1c_linear_probes import _sweep_points  # noqa: E402

CLASS_NAMES = ["cat", "dog", "bird"]


def _rows(video_frame_counts: list[tuple[str, int, int]]) -> list[dict]:
    """`[(video_id, target, frame_count), ...]` -> cache-shaped rows."""

    rows: list[dict] = []
    for video_id, target, count in video_frame_counts:
        for frame in range(count):
            rows.append({
                "source_dataset": "unit_test_dataset",
                "source_video_id": video_id,
                "source_frame_index": frame,
                "target": target,
                "label": CLASS_NAMES[target],
            })
    return rows


def _separable_features(rows: list[dict], noise: float = 0.01, seed: int = 0) -> np.ndarray:
    """One well-separated cluster per class, so a working probe must score 1.0."""

    rng = np.random.default_rng(seed)
    centres = np.eye(len(CLASS_NAMES), dtype=np.float32) * 10.0
    targets = np.asarray([row["target"] for row in rows])
    return centres[targets] + rng.normal(0.0, noise, size=(len(rows), len(CLASS_NAMES))).astype(np.float32)


class StrideTests(unittest.TestCase):
    def test_stride_one_keeps_every_row(self) -> None:
        rows = _rows([("v0", 0, 5), ("v1", 1, 3)])
        np.testing.assert_array_equal(stride_rows_within_videos(rows, 1), np.arange(8))

    def test_stride_counts_within_each_video_not_globally(self) -> None:
        """A short video must not be dropped because a long one consumed the phase."""

        rows = _rows([("v0", 0, 5), ("v1", 1, 2), ("v2", 2, 5)])
        keep = stride_rows_within_videos(rows, 3)
        kept_videos = {rows[i]["source_video_id"] for i in keep}
        self.assertEqual(kept_videos, {"v0", "v1", "v2"})
        # v0 frames 0 and 3; v1 frame 0; v2 frames 0 and 3.
        self.assertEqual([rows[i]["source_frame_index"] for i in keep], [0, 3, 0, 0, 3])

    def test_stride_rejects_zero(self) -> None:
        with self.assertRaises(ValueError):
            stride_rows_within_videos(_rows([("v0", 0, 2)]), 0)


class ScalingTests(unittest.TestCase):
    def test_standardize_fits_on_train_only(self) -> None:
        """Validation must be transformed by TRAIN statistics, never its own."""

        train = np.array([[0.0], [2.0]], dtype=np.float32)
        validation = np.array([[10.0], [12.0]], dtype=np.float32)
        scaled_train, scaled_validation = apply_scaling(train, validation, "standardize")
        self.assertAlmostEqual(float(scaled_train.mean()), 0.0, places=5)
        # If the scaler had been fitted on validation too, this would be ~0.
        self.assertGreater(float(scaled_validation.mean()), 5.0)

    def test_l2_scaling_gives_unit_rows(self) -> None:
        train = np.array([[3.0, 4.0]], dtype=np.float32)
        scaled_train, _ = apply_scaling(train, train, "l2")
        self.assertAlmostEqual(float(np.linalg.norm(scaled_train[0])), 1.0, places=5)

    def test_none_scaling_is_a_pass_through(self) -> None:
        train = np.array([[3.0, 4.0]], dtype=np.float32)
        scaled_train, _ = apply_scaling(train, train, "none")
        np.testing.assert_array_equal(scaled_train, train)

    def test_unknown_scaling_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            apply_scaling(np.zeros((1, 1)), np.zeros((1, 1)), "whitening")


class BuildProbeTests(unittest.TestCase):
    def test_unknown_probe_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            build_probe("random_forest", seed=0)

    def test_overrides_reach_the_registered_parameters(self) -> None:
        _estimator, params = build_probe("knn", seed=0, n_neighbors=7)
        self.assertEqual(params["n_neighbors"], 7)


class SweepPointTests(unittest.TestCase):
    """`--sweep` exists so one cache load pays for several fits; if it named the
    output files wrong, a sweep would overwrite its own results silently."""

    def test_unswept_run_is_a_single_unsuffixed_point(self) -> None:
        self.assertEqual(_sweep_points(None), [(None, None, "")])

    def test_each_value_gets_its_own_suffix(self) -> None:
        self.assertEqual(
            _sweep_points("C=0.01,1"),
            [("C", 0.01, "-C0.01"), ("C", 1, "-C1")],
        )

    def test_missing_equals_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            _sweep_points("C0.01")


class FitAndScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train_rows = _rows([(f"train{i:02d}", i % 3, 12) for i in range(9)])
        self.validation_rows = _rows([(f"val{i:02d}", i % 3, 8) for i in range(6)])
        self.train_features = _separable_features(self.train_rows, seed=0)
        self.validation_features = _separable_features(self.validation_rows, seed=1)

    def _fit(self, probe: str, **kwargs):
        return fit_and_score_probe(
            self.train_features, self.train_rows,
            self.validation_features, self.validation_rows, CLASS_NAMES,
            probe=probe, bootstrap_iterations=100, **kwargs,
        )

    def test_separable_features_are_classified_perfectly(self) -> None:
        for probe in ("logistic_regression", "linear_svm", "knn"):
            with self.subTest(probe=probe):
                result = self._fit(probe)
                self.assertAlmostEqual(result.frame_metrics["macro_f1"], 1.0, places=6)
                self.assertAlmostEqual(result.video_level_metrics["macro_f1"], 1.0, places=6)

    def test_video_level_counts_videos_not_frames(self) -> None:
        result = self._fit("logistic_regression")
        self.assertEqual(result.validation_rows, 48)
        self.assertEqual(result.video_count, 6)
        self.assertEqual(result.train_rows, 108)

    def test_clustered_ci_brackets_the_point_estimate(self) -> None:
        result = self._fit("logistic_regression")
        ci = result.video_clustered_macro_f1_ci
        self.assertLessEqual(ci["lower_95"], result.frame_metrics["macro_f1"])
        self.assertGreaterEqual(ci["upper_95"], result.frame_metrics["macro_f1"])

    def test_convergence_is_reported_for_iterative_solvers(self) -> None:
        self.assertIs(self._fit("logistic_regression").converged, True)
        # k-NN has no iteration count to report; None means "not applicable",
        # which must not be confused with "did not converge".
        self.assertIsNone(self._fit("knn").converged)

    def test_truncated_fit_is_reported_as_not_converged(self) -> None:
        self.assertIs(self._fit("logistic_regression", max_iter=1).converged, False)

    def test_mismatched_features_and_rows_are_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            fit_and_score_probe(
                self.train_features[:-1], self.train_rows,
                self.validation_features, self.validation_rows, CLASS_NAMES,
                probe="logistic_regression", bootstrap_iterations=100,
            )
        self.assertIn("train features/rows disagree", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
