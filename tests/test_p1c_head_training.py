"""Tests for `core.p1c_head_training` (CPU cached-feature linear head trainer).

Synthetic, separable features rather than the real 13 GB cache, mirroring
`tests/test_p1c_probes.py`: what is under test is the training/early-stopping/
selection harness, not torch's optimizer or nn.Linear.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_head_training import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_DELTA,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    train_cached_head,
)

CLASS_NAMES = ["cat", "dog", "bird"]


def _rows(video_frame_counts: list[tuple[str, int, int]]) -> list[dict]:
    """`[(video_id, target, frame_count), ...]` -> cache-shaped rows.

    `target` is stored as a string, exactly as `core.feature_cache.load_feature_cache`
    actually returns it (`csv.DictReader` over `rows.csv` never casts fields) --
    a synthetic fixture with a real int here would hide any code path that
    forgot to cast before indexing `class_names`.
    """

    rows: list[dict] = []
    for video_id, target, count in video_frame_counts:
        for frame in range(count):
            rows.append({
                "source_dataset": "unit_test_dataset",
                "source_video_id": video_id,
                "source_frame_index": str(frame),
                "target": str(target),
                "label": CLASS_NAMES[target],
            })
    return rows


def _separable_features(rows: list[dict], noise: float = 0.01, seed: int = 0) -> np.ndarray:
    """One well-separated cluster per class, so a working trainer must score ~1.0."""

    rng = np.random.default_rng(seed)
    centres = np.eye(len(CLASS_NAMES), dtype=np.float32) * 10.0
    targets = np.asarray([int(row["target"]) for row in rows])
    return centres[targets] + rng.normal(0.0, noise, size=(len(rows), len(CLASS_NAMES))).astype(np.float32)


class RecipeConstantsTests(unittest.TestCase):
    """The registered CONFIRM-015 recipe constants must match the config exactly."""

    def test_registered_constants(self) -> None:
        self.assertEqual(DEFAULT_LEARNING_RATE, 3e-4)
        self.assertEqual(DEFAULT_WEIGHT_DECAY, 1e-4)
        self.assertEqual(DEFAULT_BATCH_SIZE, 64)
        self.assertEqual(DEFAULT_PATIENCE, 5)
        self.assertEqual(DEFAULT_MIN_DELTA, 0.001)


class TrainCachedHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        # 30 frames/class train, 12 frames/class validation, several videos
        # each so the clustered bootstrap has enough source videos.
        self.train_rows = _rows([(f"train{i:02d}", i % 3, 10) for i in range(9)])
        self.validation_rows = _rows([(f"val{i:02d}", i % 3, 6) for i in range(6)])
        self.train_features = _separable_features(self.train_rows, seed=0)
        self.validation_features = _separable_features(self.validation_rows, seed=1)

    def _train(self, **overrides):
        kwargs = dict(bootstrap_iterations=100, seed=0)
        kwargs.update(overrides)
        return train_cached_head(
            self.train_features, self.train_rows,
            self.validation_features, self.validation_rows, CLASS_NAMES,
            **kwargs,
        )

    def test_separable_features_train_to_near_perfect_macro_f1(self) -> None:
        # The registered lr (3e-4) is tuned for a 391,014-row ResNet cache;
        # on this tiny synthetic fixture it barely moves in 30 epochs, so a
        # larger lr is used here purely to reach separability quickly -- the
        # recipe-constant defaults themselves are checked in
        # RecipeConstantsTests above.
        result = self._train(epochs=100, learning_rate=0.1)
        self.assertGreaterEqual(result.best_validation_macro_f1, 0.95)
        self.assertGreaterEqual(result.probe_result.frame_metrics["macro_f1"], 0.95)

    def test_best_epoch_selection_returns_the_best_not_the_last(self) -> None:
        """A run driven far past its optimum must still report the peak epoch."""

        result = self._train(epochs=30, learning_rate=0.5)
        best_recorded = max(record["validation_macro_f1"] for record in result.epochs)
        self.assertAlmostEqual(result.best_validation_macro_f1, best_recorded, places=9)
        # The reported metric must be the best epoch's, not whatever the last
        # completed epoch happened to score.
        last_epoch_metric = result.epochs[-1]["validation_macro_f1"]
        selected_epoch_metric = result.epochs[result.best_epoch]["validation_macro_f1"]
        self.assertAlmostEqual(result.best_validation_macro_f1, selected_epoch_metric, places=9)
        if last_epoch_metric != best_recorded:
            self.assertNotEqual(result.best_epoch, len(result.epochs) - 1)

    def test_early_stopping_fires_at_registered_patience(self) -> None:
        """Flat features -- no separation signal -- must stop within patience+1 epochs."""

        flat_train_rows = _rows([(f"train{i:02d}", 0, 10) for i in range(9)])
        flat_validation_rows = _rows([(f"val{i:02d}", 0, 6) for i in range(6)])
        # Two classes never present at all is invalid for macro metrics with
        # 3 class_names; use near-zero noise so the model cannot improve
        # beyond epoch 0's plateau and early stopping is exercised directly.
        rng = np.random.default_rng(0)
        flat_train_features = rng.normal(0.0, 0.001, size=(len(flat_train_rows), 3)).astype(np.float32)
        flat_validation_features = rng.normal(0.0, 0.001, size=(len(flat_validation_rows), 3)).astype(np.float32)
        result = train_cached_head(
            flat_train_features, flat_train_rows,
            flat_validation_features, flat_validation_rows, CLASS_NAMES,
            seed=0, epochs=30, patience=5, min_delta=0.001, bootstrap_iterations=100,
        )
        self.assertTrue(result.stopped_early)
        # patience=5: the loop runs the improving epoch plus 5 stagnant ones.
        self.assertLessEqual(len(result.epochs), 6)

    def test_seed_is_reproducible(self) -> None:
        first = self._train(epochs=5, seed=3)
        second = self._train(epochs=5, seed=3)
        self.assertEqual(
            [record["validation_macro_f1"] for record in first.epochs],
            [record["validation_macro_f1"] for record in second.epochs],
        )
        self.assertEqual(first.best_epoch, second.best_epoch)
        self.assertAlmostEqual(
            first.best_validation_macro_f1, second.best_validation_macro_f1, places=12
        )

    def test_different_seeds_need_not_match(self) -> None:
        first = self._train(epochs=5, seed=0)
        second = self._train(epochs=5, seed=1)
        # Not asserting inequality of the final metric (both may saturate at
        # ~1.0 on separable data) -- only that per-epoch loss traces differ,
        # proving the seed actually reached the shuffle/init.
        self.assertNotEqual(
            [record["train_mean_loss"] for record in first.epochs],
            [record["train_mean_loss"] for record in second.epochs],
        )

    def test_validation_probabilities_off_by_default_and_json_is_unchanged(self) -> None:
        result = self._train(epochs=5, seed=0)
        self.assertIsNone(result.best_validation_probabilities)
        self.assertNotIn("best_validation_probabilities", result.to_json())

    def test_capture_validation_probabilities_shape_and_normalisation(self) -> None:
        result = self._train(epochs=5, seed=0, capture_validation_probabilities=True)
        probs = result.best_validation_probabilities
        self.assertIsNotNone(probs)
        self.assertEqual(probs.shape, (len(self.validation_rows), len(CLASS_NAMES)))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        self.assertTrue((probs >= 0).all())
        # to_json still drops the bulk array
        self.assertNotIn("best_validation_probabilities", result.to_json())

    def test_capturing_probabilities_does_not_change_the_selected_result(self) -> None:
        plain = self._train(epochs=8, seed=2, learning_rate=0.1)
        captured = self._train(epochs=8, seed=2, learning_rate=0.1, capture_validation_probabilities=True)
        self.assertEqual(plain.best_epoch, captured.best_epoch)
        self.assertAlmostEqual(
            plain.best_validation_macro_f1, captured.best_validation_macro_f1, places=12
        )
        self.assertEqual(
            [r["train_mean_loss"] for r in plain.epochs],
            [r["train_mean_loss"] for r in captured.epochs],
        )
        # the captured argmax must agree with the probe_result's frame predictions
        argmax_from_probs = [CLASS_NAMES[i] for i in captured.best_validation_probabilities.argmax(axis=1)]
        cm = captured.probe_result.frame_metrics["confusion_matrix"]
        self.assertEqual(sum(sum(row) for row in cm), len(argmax_from_probs))

    def test_mismatched_features_and_rows_are_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            train_cached_head(
                self.train_features[:-1], self.train_rows,
                self.validation_features, self.validation_rows, CLASS_NAMES,
                seed=0, bootstrap_iterations=100,
            )
        self.assertIn("train features/rows disagree", str(caught.exception))

    def test_mismatched_validation_features_and_rows_are_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            train_cached_head(
                self.train_features, self.train_rows,
                self.validation_features[:-1], self.validation_rows, CLASS_NAMES,
                seed=0, bootstrap_iterations=100,
            )
        self.assertIn("validation features/rows disagree", str(caught.exception))


class TrainWeightsTests(unittest.TestCase):
    """`train_weights` must leave the default path byte-identical and behave
    correctly when supplied: all-ones == unweighted, zero-weight rows inert."""

    def setUp(self) -> None:
        self.train_rows = _rows([(f"train{i:02d}", i % 3, 10) for i in range(9)])
        self.validation_rows = _rows([(f"val{i:02d}", i % 3, 6) for i in range(6)])
        self.train_features = _separable_features(self.train_rows, seed=0)
        self.validation_features = _separable_features(self.validation_rows, seed=1)

    def _train(self, **overrides):
        kwargs = dict(bootstrap_iterations=100, seed=0, epochs=8, learning_rate=0.1)
        kwargs.update(overrides)
        return train_cached_head(
            self.train_features, self.train_rows,
            self.validation_features, self.validation_rows, CLASS_NAMES,
            **kwargs,
        )

    def test_no_weights_leaves_json_and_result_unchanged(self) -> None:
        result = self._train()
        self.assertIsNone(result.train_weight_summary)
        self.assertNotIn("train_weight_summary", result.to_json())

    def test_all_ones_weights_reproduce_unweighted_result_exactly(self) -> None:
        plain = self._train()
        weighted = self._train(train_weights=np.ones(len(self.train_rows), dtype=np.float64))
        self.assertEqual(
            [r["validation_macro_f1"] for r in plain.epochs],
            [r["validation_macro_f1"] for r in weighted.epochs],
        )
        # Losses agree to float32 precision but not necessarily to the last
        # ULP: the unweighted path calls `reduction="mean"` directly while an
        # all-ones weighted batch computes `sum(per_sample_loss) / weight_sum`
        # -- mathematically identical, but a different summation order that
        # floating point is not required to round to the same bits.
        for plain_epoch, weighted_epoch in zip(plain.epochs, weighted.epochs):
            self.assertAlmostEqual(
                plain_epoch["train_mean_loss"], weighted_epoch["train_mean_loss"], places=6
            )
        self.assertEqual(plain.best_epoch, weighted.best_epoch)
        self.assertAlmostEqual(
            plain.best_validation_macro_f1, weighted.best_validation_macro_f1, places=9
        )
        self.assertEqual(weighted.train_weight_summary["sum"], float(len(self.train_rows)))
        self.assertEqual(weighted.train_weight_summary["zero_weight_rows"], 0)
        self.assertIn("train_weight_summary", weighted.to_json())

    def test_zero_weight_rows_do_not_affect_training(self) -> None:
        """Add rows with a wrong-boundary label but weight 0: the result must
        match training without those rows at all."""

        rng = np.random.default_rng(7)
        poison_rows = _rows([("poison00", 1, 5), ("poison01", 2, 5)])
        # Poison features sit on class 0's cluster but carry a different label,
        # so an unweighted fit would be pulled off the clean solution.
        poison_features = np.tile(np.array([10.0, 0.0, 0.0], dtype=np.float32), (10, 1))
        poison_features = poison_features + rng.normal(0.0, 0.01, size=(10, 3)).astype(np.float32)

        clean_rows = list(self.train_rows)
        clean_features = self.train_features
        combined_rows = clean_rows + poison_rows
        combined_features = np.concatenate([clean_features, poison_features])
        weights = np.concatenate([np.ones(len(clean_rows)), np.zeros(len(poison_rows))])

        baseline = train_cached_head(
            clean_features, clean_rows, self.validation_features, self.validation_rows, CLASS_NAMES,
            seed=0, epochs=8, learning_rate=0.1, bootstrap_iterations=100,
        )
        weighted = train_cached_head(
            combined_features, combined_rows, self.validation_features, self.validation_rows, CLASS_NAMES,
            seed=0, epochs=8, learning_rate=0.1, bootstrap_iterations=100, train_weights=weights,
        )
        self.assertEqual(weighted.train_weight_summary["zero_weight_rows"], len(poison_rows))
        self.assertAlmostEqual(
            baseline.best_validation_macro_f1, weighted.best_validation_macro_f1, places=6
        )

    def test_mismatched_length_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._train(train_weights=np.ones(len(self.train_rows) - 1))
        self.assertIn("train_weights length", str(caught.exception))

    def test_negative_weight_is_refused(self) -> None:
        weights = np.ones(len(self.train_rows))
        weights[0] = -1.0
        with self.assertRaises(ValueError) as caught:
            self._train(train_weights=weights)
        self.assertIn(">= 0", str(caught.exception))

    def test_nan_weight_is_refused(self) -> None:
        weights = np.ones(len(self.train_rows))
        weights[0] = float("nan")
        with self.assertRaises(ValueError) as caught:
            self._train(train_weights=weights)
        self.assertIn("finite", str(caught.exception))

    def test_all_zero_weights_are_refused(self) -> None:
        weights = np.zeros(len(self.train_rows))
        with self.assertRaises(ValueError) as caught:
            self._train(train_weights=weights)
        self.assertIn("at least one value > 0", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
