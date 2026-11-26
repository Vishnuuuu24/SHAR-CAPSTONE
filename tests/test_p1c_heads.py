"""Tests for `core.p1c_heads` (P3A visual head comparison) and `core.p1c_head_variants`.

Synthetic, small fixtures throughout -- what is under test is head shape
contracts, gradient flow, and the new trainer's batching/scoring logic, never
torch's own conv/attention kernels or the real 32.5 GB patch cache.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_heads import (  # noqa: E402
    GapLinearHead,
    MlpHead,
    MultiScaleSeHead,
    StepHead,
)
from core.p1c_head_variants import (  # noqa: E402
    group_pooled_sequences_by_video,
    load_pooled_features,
    pooled_vector_from_tokens,
    train_grid_head,
    train_sequence_head,
    train_vector_head,
)

CLASS_NAMES = [f"class{i}" for i in range(14)]


class GapLinearHeadTests(unittest.TestCase):
    def test_matches_bare_linear_on_the_same_pooled_vector(self) -> None:
        torch.manual_seed(0)
        pooled = torch.randn(5, 1536)
        head = GapLinearHead(in_dim=1536, num_classes=14)
        reference = nn.Linear(1536, 14)
        reference.load_state_dict(head.linear.state_dict())
        np.testing.assert_allclose(
            head(pooled).detach().numpy(), reference(pooled).detach().numpy(), atol=1e-6
        )

    def test_wrong_input_width_raises(self) -> None:
        head = GapLinearHead(in_dim=1536)
        with self.assertRaises(ValueError):
            head(torch.randn(2, 100))

    def test_parameter_count(self) -> None:
        head = GapLinearHead(in_dim=1536, num_classes=14)
        self.assertEqual(head.parameter_count(), 1536 * 14 + 14)


class MlpHeadTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        head = MlpHead(in_dim=1536, hidden=32, num_classes=14)
        logits = head(torch.randn(3, 1536))
        self.assertEqual(logits.shape, (3, 14))

    def test_parameter_count_reported(self) -> None:
        head = MlpHead(in_dim=1536, hidden=32, num_classes=14)
        expected = (1536 * 32 + 32) + (32 * 14 + 14)
        self.assertEqual(head.parameter_count(), expected)


class MultiScaleSeHeadTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        head = MultiScaleSeHead(token_dim=768, grid=(16, 16), width=32)
        logits = head(torch.randn(4, 256, 768))
        self.assertEqual(logits.shape, (4, 14))

    def test_grid_product_assertion(self) -> None:
        head = MultiScaleSeHead(token_dim=768, grid=(16, 16), width=32)
        with self.assertRaises(ValueError):
            head(torch.randn(2, 255, 768))
        with self.assertRaises(ValueError):
            head(torch.randn(2, 257, 768))

    def test_wrong_token_dim_raises(self) -> None:
        head = MultiScaleSeHead(token_dim=768, grid=(16, 16), width=32)
        with self.assertRaises(ValueError):
            head(torch.randn(2, 256, 100))

    def test_se_gate_is_bounded_and_gates_channels_not_space(self) -> None:
        head = MultiScaleSeHead(token_dim=768, grid=(16, 16), width=32)
        head.eval()
        tokens = torch.randn(2, 256, 768)
        _logits, debug = head(tokens, return_debug=True)
        gate = debug["se_gate"]
        self.assertEqual(gate.shape, (2, 32))  # one scalar per (batch, channel) -- never per spatial position
        self.assertTrue(bool((gate > 0).all()))
        self.assertTrue(bool((gate < 1).all()))

    def test_parameter_count_reported(self) -> None:
        head = MultiScaleSeHead(token_dim=768, grid=(16, 16), width=32, se_reduction=8)
        self.assertEqual(head.parameter_count(), 31282)


class StepHeadTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        head = StepHead(token_dim=1536, frames=8, depth=1)
        logits = head(torch.randn(3, 8, 1536))
        self.assertEqual(logits.shape, (3, 14))

    def test_shorter_sequence_with_mask_runs(self) -> None:
        head = StepHead(token_dim=1536, frames=8, depth=1)
        sequence = torch.randn(2, 5, 1536)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[:, 3:] = True
        logits = head(sequence, key_padding_mask=mask)
        self.assertEqual(logits.shape, (2, 14))

    def test_sequence_longer_than_window_raises(self) -> None:
        head = StepHead(token_dim=1536, frames=8, depth=1)
        with self.assertRaises(ValueError):
            head(torch.randn(1, 9, 1536))

    def test_order_sensitive(self) -> None:
        """Shuffling the frame axis must change the logits -- STEP's whole point."""

        torch.manual_seed(0)
        head = StepHead(token_dim=1536, frames=8, depth=1)
        head.eval()
        sequence = torch.randn(1, 8, 1536)
        shuffled = sequence[:, torch.randperm(8), :]
        with torch.no_grad():
            original_logits = head(sequence)
            shuffled_logits = head(shuffled)
        self.assertFalse(torch.allclose(original_logits, shuffled_logits, atol=1e-6))

    def test_parameter_count_reported(self) -> None:
        head = StepHead(token_dim=1536, frames=8, depth=1, num_classes=14)
        qkv = 3 * (1536 * 1536 + 1536)
        position = 8 * 1536
        cls_token = 1536
        classifier = 1536 * 14 + 14
        self.assertEqual(head.parameter_count(), qkv + position + cls_token + classifier)


class GradientFlowTests(unittest.TestCase):
    """`loss.backward()` must give every parameter of every head a non-None grad."""

    def _assert_all_grads_populated(self, head: nn.Module, logits: torch.Tensor, batch: int) -> None:
        target = torch.randint(0, 14, (batch,))
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, f"{name} received no gradient")

    def test_gap_linear(self) -> None:
        head = GapLinearHead(in_dim=1536)
        self._assert_all_grads_populated(head, head(torch.randn(4, 1536)), 4)

    def test_mlp(self) -> None:
        head = MlpHead(in_dim=1536, hidden=32)
        self._assert_all_grads_populated(head, head(torch.randn(4, 1536)), 4)

    def test_multiscale_se(self) -> None:
        head = MultiScaleSeHead(token_dim=768, grid=(16, 16), width=32)
        self._assert_all_grads_populated(head, head(torch.randn(4, 256, 768)), 4)

    def test_step_full_window(self) -> None:
        # A full (unpadded) window so every position embedding row is used.
        head = StepHead(token_dim=1536, frames=8, depth=1)
        self._assert_all_grads_populated(head, head(torch.randn(4, 8, 1536)), 4)


class PooledVectorHelperTests(unittest.TestCase):
    def test_pooled_vector_is_cls_concat_mean_patch(self) -> None:
        cls = np.array([[1.0, 2.0]], dtype=np.float32)
        patch = np.array([[[0.0, 0.0], [2.0, 4.0]]], dtype=np.float32)  # mean -> [1.0, 2.0]
        pooled = pooled_vector_from_tokens(cls, patch)
        np.testing.assert_allclose(pooled, [[1.0, 2.0, 1.0, 2.0]])

    def test_load_pooled_features_streams_in_chunks(self) -> None:
        rng = np.random.default_rng(0)
        cls = rng.normal(size=(10, 8)).astype(np.float16)
        patch = rng.normal(size=(10, 4, 8)).astype(np.float16)
        indices = np.array([9, 0, 3, 7])
        pooled = load_pooled_features(cls, patch, indices, chunk_size=2)
        expected = pooled_vector_from_tokens(cls[indices], patch[indices])
        np.testing.assert_allclose(pooled, expected, atol=1e-3)


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


def _separable_pooled(rows: list[dict], noise: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = np.eye(len(CLASS_NAMES), dtype=np.float32)[:, : ]
    # Tile the 14-D one-hot centre up to 1536 dims so distinct classes stay
    # separable in the full pooled width the real heads expect.
    tiled = np.tile(centres, (1, (1536 // len(CLASS_NAMES)) + 1))[:, :1536] * 10.0
    targets = np.asarray([int(row["target"]) for row in rows])
    return tiled[targets] + rng.normal(0.0, noise, size=(len(rows), 1536)).astype(np.float32)


class TrainVectorHeadEndToEndTests(unittest.TestCase):
    def test_trains_and_early_stops(self) -> None:
        train_rows = _rows([(f"train{i:02d}", i % 14, 4) for i in range(28)])
        validation_rows = _rows([(f"val{i:02d}", i % 14, 3) for i in range(14)])
        train_pooled = _separable_pooled(train_rows, seed=0)
        validation_pooled = _separable_pooled(validation_rows, seed=1)

        head = GapLinearHead(in_dim=1536, num_classes=14)
        result = train_vector_head(
            head, train_pooled, train_rows, validation_pooled, validation_rows, CLASS_NAMES,
            seed=0, epochs=1, bootstrap_iterations=100,
        )
        self.assertEqual(len(result.epochs), 1)
        self.assertGreaterEqual(result.best_validation_macro_f1, 0.0)
        self.assertEqual(result.recipe["video_level_aggregation"], "majority_vote")

    def test_early_stopping_fires_on_flat_features(self) -> None:
        train_rows = _rows([(f"train{i:02d}", 0, 4) for i in range(9)])
        validation_rows = _rows([(f"val{i:02d}", 0, 3) for i in range(6)])
        rng = np.random.default_rng(0)
        train_pooled = rng.normal(0.0, 0.001, size=(len(train_rows), 1536)).astype(np.float32)
        validation_pooled = rng.normal(0.0, 0.001, size=(len(validation_rows), 1536)).astype(np.float32)

        head = GapLinearHead(in_dim=1536, num_classes=14)
        result = train_vector_head(
            head, train_pooled, train_rows, validation_pooled, validation_rows, CLASS_NAMES,
            seed=0, epochs=30, patience=5, min_delta=0.001, bootstrap_iterations=100,
        )
        self.assertTrue(result.stopped_early)
        self.assertLessEqual(len(result.epochs), 6)


class TrainGridHeadEndToEndTests(unittest.TestCase):
    def test_trains_from_a_memmap(self) -> None:
        train_rows = _rows([(f"train{i:02d}", i % 14, 2) for i in range(28)])
        validation_rows = _rows([(f"val{i:02d}", i % 14, 1) for i in range(14)])
        rng = np.random.default_rng(0)
        grid = rng.normal(size=(len(train_rows) + len(validation_rows), 256, 768)).astype(np.float16)
        train_indices = np.arange(len(train_rows))
        validation_indices = np.arange(len(train_rows), len(train_rows) + len(validation_rows))

        head = MultiScaleSeHead(token_dim=768, grid=(16, 16), width=16)
        result = train_grid_head(
            head, grid, train_indices, train_rows, validation_indices, validation_rows, CLASS_NAMES,
            seed=0, epochs=1, batch_size=8, bootstrap_iterations=100,
        )
        self.assertEqual(len(result.epochs), 1)
        self.assertEqual(result.recipe["video_level_aggregation"], "majority_vote")


class TrainSequenceHeadEndToEndTests(unittest.TestCase):
    def test_trains_and_reports_direct_step_aggregation(self) -> None:
        train_rows = _rows([(f"train{i:02d}", i % 14, 8) for i in range(28)])
        validation_rows = _rows([(f"val{i:02d}", i % 14, 10) for i in range(14)])  # > frames, exercises chunking
        pooled_train = _separable_pooled(train_rows, seed=0)
        pooled_validation = _separable_pooled(validation_rows, seed=1)

        train_sequences = group_pooled_sequences_by_video(pooled_train, train_rows)
        validation_sequences = group_pooled_sequences_by_video(pooled_validation, validation_rows)

        head = StepHead(token_dim=1536, frames=8, depth=1, num_classes=14)
        result = train_sequence_head(
            head, train_sequences, validation_sequences, CLASS_NAMES,
            frames=8, seed=0, epochs=1, batch_size=8, bootstrap_iterations=100,
        )
        self.assertEqual(len(result.epochs), 1)
        self.assertEqual(result.recipe["video_level_aggregation"], "step_direct_cls_prediction")
        # One prediction per video: frame_metrics and video_level_metrics collapse to the same number.
        self.assertAlmostEqual(
            result.probe_result.frame_metrics["macro_f1"],
            result.probe_result.video_level_metrics["macro_f1"],
            places=12,
        )
        self.assertEqual(result.probe_result.video_count, 14)

    def test_disagreeing_target_within_a_video_is_refused(self) -> None:
        rows = _rows([("v0", 0, 3)])
        rows[1]["target"] = "1"  # corrupt: same video, disagreeing label
        pooled = np.zeros((3, 1536), dtype=np.float32)
        with self.assertRaises(ValueError):
            group_pooled_sequences_by_video(pooled, rows)


if __name__ == "__main__":
    unittest.main()
