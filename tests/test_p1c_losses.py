"""Tests for `core.p1c_losses` (P3B imbalance-loss ablation) and its
backward-compatibility guard on `core.p1c_head_training.train_cached_head`.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_losses import (  # noqa: E402
    LOSS_NAMES,
    build_loss,
    class_counts,
    focal_loss,
    inverse_frequency_weights,
    log_prior_adjustment,
)
from core.p1c_head_training import train_cached_head  # noqa: E402

CLASS_NAMES = ["cat", "dog", "bird"]


def _rows(video_frame_counts: list[tuple[str, int, int]]) -> list[dict]:
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
    rng = np.random.default_rng(seed)
    centres = np.eye(len(CLASS_NAMES), dtype=np.float32) * 10.0
    targets = np.asarray([int(row["target"]) for row in rows])
    return centres[targets] + rng.normal(0.0, noise, size=(len(rows), len(CLASS_NAMES))).astype(np.float32)


class ClassCountsTests(unittest.TestCase):
    def test_counts_match_label_order(self) -> None:
        rows = _rows([("v0", 0, 5), ("v1", 1, 2), ("v2", 2, 7)])
        counts = class_counts(rows, CLASS_NAMES)
        np.testing.assert_array_equal(counts, [5, 2, 7])


class InverseFrequencyWeightsTests(unittest.TestCase):
    def test_balanced_counts_yield_all_ones(self) -> None:
        counts = np.array([272, 272, 272], dtype=np.int64)
        weights = inverse_frequency_weights(counts)
        np.testing.assert_allclose(weights, [1.0, 1.0, 1.0], atol=1e-10)

    def test_imbalanced_counts_are_not_uniform(self) -> None:
        counts = np.array([10, 100, 1000], dtype=np.int64)
        weights = inverse_frequency_weights(counts)
        self.assertFalse(np.allclose(weights, weights[0]))
        # normalised to mean 1.0
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=10)

    def test_unnormalised_is_plain_inverse_frequency(self) -> None:
        counts = np.array([1, 2, 4], dtype=np.int64)
        weights = inverse_frequency_weights(counts, normalise=False)
        np.testing.assert_allclose(weights, [1.0, 0.5, 0.25])

    def test_zero_count_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            inverse_frequency_weights(np.array([0, 5, 5]))


class LogPriorAdjustmentTests(unittest.TestCase):
    def test_equal_counts_give_a_constant_vector(self) -> None:
        counts = np.array([272, 272, 272], dtype=np.int64)
        adjustment = log_prior_adjustment(counts)
        self.assertTrue(np.allclose(adjustment, adjustment[0]))
        self.assertAlmostEqual(float(adjustment[0]), float(np.log(1.0 / 3.0)), places=10)

    def test_unequal_counts_give_a_nonconstant_vector(self) -> None:
        counts = np.array([10, 100, 1000], dtype=np.int64)
        adjustment = log_prior_adjustment(counts)
        self.assertFalse(np.allclose(adjustment, adjustment[0]))

    def test_zero_count_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            log_prior_adjustment(np.array([0, 5, 5]))


class FocalLossTests(unittest.TestCase):
    def test_gamma_zero_equals_cross_entropy(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(16, 4)
        targets = torch.randint(0, 4, (16,))
        focal = focal_loss(logits, targets, gamma=0.0, reduction="none")
        ce = F.cross_entropy(logits, targets, reduction="none")
        torch.testing.assert_close(focal, ce)

    def test_downweights_an_easy_example_relative_to_a_hard_one(self) -> None:
        # "easy": model confidently correct. "hard": model confidently wrong.
        easy_logits = torch.tensor([[10.0, 0.0]])
        hard_logits = torch.tensor([[0.0, 10.0]])
        target = torch.tensor([0])
        logits = torch.cat([easy_logits, hard_logits], dim=0)
        targets = torch.cat([target, target], dim=0)
        focal = focal_loss(logits, targets, gamma=2.0, reduction="none")
        ce = F.cross_entropy(logits, targets, reduction="none")
        easy_ratio = float(focal[0] / ce[0])
        hard_ratio = float(focal[1] / ce[1])
        self.assertLess(easy_ratio, hard_ratio)
        # the easy example's focal weight should be reduced to near-nothing
        self.assertLess(easy_ratio, 0.01)

    def test_finite_for_a_saturated_logit(self) -> None:
        logits = torch.tensor([[50.0, -50.0, -50.0]])
        targets = torch.tensor([0])
        loss = focal_loss(logits, targets, gamma=2.0, reduction="none")
        self.assertTrue(torch.isfinite(loss).all())
        # the correct class is saturated-confident, so the loss must be ~0
        self.assertLess(float(loss[0]), 1e-6)

    def test_saturated_wrong_class_is_also_finite(self) -> None:
        logits = torch.tensor([[-50.0, 50.0, -50.0]])
        targets = torch.tensor([0])
        loss = focal_loss(logits, targets, gamma=2.0, reduction="none")
        self.assertTrue(torch.isfinite(loss).all())

    def test_alpha_scales_per_class(self) -> None:
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([0, 1])
        alpha = torch.tensor([2.0, 0.5])
        unweighted = focal_loss(logits, targets, gamma=1.0, reduction="none")
        weighted = focal_loss(logits, targets, gamma=1.0, alpha=alpha, reduction="none")
        torch.testing.assert_close(weighted, unweighted * alpha[targets])

    def test_mean_reduction_matches_manual_mean(self) -> None:
        torch.manual_seed(1)
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        per_example = focal_loss(logits, targets, gamma=2.0, reduction="none")
        reduced = focal_loss(logits, targets, gamma=2.0, reduction="mean")
        torch.testing.assert_close(reduced, per_example.mean())

    def test_invalid_reduction_is_refused(self) -> None:
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        with self.assertRaises(ValueError):
            focal_loss(logits, targets, gamma=2.0, reduction="sum")


class BuildLossTests(unittest.TestCase):
    def test_cross_entropy_matches_functional_cross_entropy(self) -> None:
        torch.manual_seed(2)
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        counts = np.array([272, 272, 272])
        criterion = build_loss("cross_entropy", counts=counts, gamma=2.0, reduction="none")
        torch.testing.assert_close(criterion(logits, targets), F.cross_entropy(logits, targets, reduction="none"))

    def test_weighted_cross_entropy_matches_cross_entropy_under_balanced_counts(self) -> None:
        torch.manual_seed(3)
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        counts = np.array([272, 272, 272])
        ce = build_loss("cross_entropy", counts=counts, reduction="none")
        weighted = build_loss("weighted_cross_entropy", counts=counts, reduction="none")
        torch.testing.assert_close(weighted(logits, targets), ce(logits, targets))

    def test_balanced_softmax_matches_cross_entropy_under_equal_priors(self) -> None:
        torch.manual_seed(4)
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        counts = np.array([272, 272, 272])
        ce = build_loss("cross_entropy", counts=counts, reduction="none")
        balanced = build_loss("balanced_softmax", counts=counts, reduction="none")
        torch.testing.assert_close(balanced(logits, targets), ce(logits, targets))

    def test_weighted_cross_entropy_diverges_from_cross_entropy_under_imbalance(self) -> None:
        torch.manual_seed(5)
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        counts = np.array([10, 100, 1000])
        ce = build_loss("cross_entropy", counts=counts, reduction="none")
        weighted = build_loss("weighted_cross_entropy", counts=counts, reduction="none")
        self.assertFalse(torch.allclose(weighted(logits, targets), ce(logits, targets)))

    def test_unknown_loss_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            build_loss("not_a_loss", counts=np.array([1, 1]))

    def test_invalid_reduction_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            build_loss("cross_entropy", counts=np.array([1, 1]), reduction="sum")

    def test_loss_names_has_all_four(self) -> None:
        self.assertEqual(LOSS_NAMES, ("cross_entropy", "weighted_cross_entropy", "focal", "balanced_softmax"))


class TrainCachedHeadBackwardCompatibilityTests(unittest.TestCase):
    """The backward-compatibility guard: `loss="cross_entropy"` (explicit)
    must reproduce calling `train_cached_head` with no `loss` argument at
    all, on the same seed."""

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

    def test_explicit_cross_entropy_matches_default(self) -> None:
        default = self._train()
        explicit = self._train(loss="cross_entropy")
        self.assertEqual(
            [r["train_mean_loss"] for r in default.epochs],
            [r["train_mean_loss"] for r in explicit.epochs],
        )
        self.assertEqual(
            [r["validation_macro_f1"] for r in default.epochs],
            [r["validation_macro_f1"] for r in explicit.epochs],
        )
        self.assertEqual(default.best_epoch, explicit.best_epoch)
        self.assertAlmostEqual(
            default.best_validation_macro_f1, explicit.best_validation_macro_f1, places=12
        )
        self.assertEqual(default.recipe["loss"], explicit.recipe["loss"])
        self.assertEqual(default.recipe["loss"], "cross_entropy_mean_unweighted")

    def test_weighted_cross_entropy_matches_cross_entropy_under_balanced_train_set(self) -> None:
        # setUp's train set is 30 rows/class -- exactly balanced, same as the
        # real fold-1 pipeline this ablation targets (D-46, 272 rows/class).
        ce = self._train(loss="cross_entropy")
        weighted = self._train(loss="weighted_cross_entropy")
        self.assertEqual(
            [r["validation_macro_f1"] for r in ce.epochs],
            [r["validation_macro_f1"] for r in weighted.epochs],
        )
        for ce_epoch, weighted_epoch in zip(ce.epochs, weighted.epochs):
            self.assertAlmostEqual(ce_epoch["train_mean_loss"], weighted_epoch["train_mean_loss"], places=6)
        self.assertEqual(ce.best_epoch, weighted.best_epoch)
        self.assertAlmostEqual(ce.best_validation_macro_f1, weighted.best_validation_macro_f1, places=9)

    def test_unknown_loss_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._train(loss="not_a_loss")

    def test_focal_gamma_is_recorded_in_recipe(self) -> None:
        result = self._train(loss="focal", focal_gamma=3.5)
        self.assertEqual(result.recipe["loss_name"], "focal")
        self.assertEqual(result.recipe["focal_gamma"], 3.5)
        self.assertIn("loss_name", result.to_json()["recipe"])


if __name__ == "__main__":
    unittest.main()
