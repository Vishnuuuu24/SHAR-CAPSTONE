"""Tests for `core.p1c_temporal_features` (temporal-window features).

Synthetic fixtures only, mirroring `tests/test_p1c_probes.py`'s style.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_temporal_features import build_temporal_features  # noqa: E402


def _rows(video_frame_counts: list[tuple[str, int]]) -> list[dict]:
    """`[(video_id, frame_count), ...]` -> cache-shaped rows, contiguous per video."""

    rows: list[dict] = []
    for video_id, count in video_frame_counts:
        for frame in range(count):
            rows.append({
                "source_dataset": "unit_test_dataset",
                "source_video_id": video_id,
                "source_frame_index": frame * 5,
                "target": "0",
                "label": "class0",
            })
    return rows


class ShapeTests(unittest.TestCase):
    def test_baseline_shape_matches_input(self) -> None:
        rows = _rows([("v0", 6)])
        features = np.arange(6 * 4, dtype=np.float16).reshape(6, 4)
        out = build_temporal_features(features, rows, "baseline", radius=2)
        self.assertEqual(out.shape, (6, 4))
        np.testing.assert_allclose(out, features.astype(np.float32))

    def test_concat_width_is_5d(self) -> None:
        rows = _rows([("v0", 10)])
        features = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
        out = build_temporal_features(features, rows, "concat", radius=2)
        self.assertEqual(out.shape, (10, 15))

    def test_mean_width_is_d(self) -> None:
        rows = _rows([("v0", 10)])
        features = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
        out = build_temporal_features(features, rows, "mean", radius=2)
        self.assertEqual(out.shape, (10, 3))

    def test_mean_std_width_is_2d(self) -> None:
        rows = _rows([("v0", 10)])
        features = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
        out = build_temporal_features(features, rows, "mean_std", radius=2)
        self.assertEqual(out.shape, (10, 6))

    def test_row_count_always_equals_input(self) -> None:
        rows = _rows([("v0", 7), ("v1", 3)])
        features = np.random.default_rng(0).normal(size=(10, 2)).astype(np.float32)
        for variant in ("baseline", "concat", "mean", "mean_std", "diff"):
            out = build_temporal_features(features, rows, variant, radius=1)
            self.assertEqual(out.shape[0], 10)

    def test_diff_width_is_2d(self) -> None:
        rows = _rows([("v0", 10)])
        features = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
        out = build_temporal_features(features, rows, "diff", radius=2)
        self.assertEqual(out.shape, (10, 6))

    def test_unknown_variant_rejected(self) -> None:
        rows = _rows([("v0", 3)])
        features = np.zeros((3, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            build_temporal_features(features, rows, "median", radius=1)


class BoundaryTests(unittest.TestCase):
    def test_window_never_crosses_video_boundary(self) -> None:
        """Two adjacent 3-frame videos; the boundary row's concat window must
        use only its own video's frames, never the other video's."""

        rows = _rows([("v0", 3), ("v1", 3)])
        # Distinct, easily-identified feature values per video: v0 frames are
        # 0,1,2; v1 frames are 100,101,102.
        features = np.array([[0.0], [1.0], [2.0], [100.0], [101.0], [102.0]], dtype=np.float32)
        out = build_temporal_features(features, rows, "concat", radius=1)
        # Row 2 is v0's last frame: window is [1,2,2] (clamped), never touching v1.
        np.testing.assert_allclose(out[2], [1.0, 2.0, 2.0])
        # Row 3 is v1's first frame: window is [100,100,101] (clamped), never touching v0.
        np.testing.assert_allclose(out[3], [100.0, 100.0, 101.0])

    def test_edge_clamping_repeats_edge_frame(self) -> None:
        rows = _rows([("v0", 5)])
        features = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        out = build_temporal_features(features, rows, "concat", radius=2)
        # First row: true window is [-2,-1,0,1,2] -> clamped to [0,0,0,1,2].
        np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0, 1.0, 2.0])
        # Last row: true window is [2,3,4,5,6] -> clamped to [2,3,4,4,4].
        np.testing.assert_allclose(out[4], [2.0, 3.0, 4.0, 4.0, 4.0])

    def test_concat_centre_block_equals_baseline(self) -> None:
        rows = _rows([("v0", 8)])
        features = np.random.default_rng(1).normal(size=(8, 3)).astype(np.float32)
        out = build_temporal_features(features, rows, "concat", radius=2)
        centre_block = out[:, 2 * 3 : 3 * 3]  # radius=2 -> centre is the 3rd of 5 blocks
        np.testing.assert_allclose(centre_block, features)

    def test_mean_of_constant_feature_equals_the_constant(self) -> None:
        rows = _rows([("v0", 6)])
        features = np.full((6, 4), 7.0, dtype=np.float32)
        out = build_temporal_features(features, rows, "mean", radius=2)
        np.testing.assert_allclose(out, features)

    def test_mean_std_of_constant_feature_has_zero_std(self) -> None:
        rows = _rows([("v0", 6)])
        features = np.full((6, 4), 7.0, dtype=np.float32)
        out = build_temporal_features(features, rows, "mean_std", radius=2)
        np.testing.assert_allclose(out[:, :4], features)
        np.testing.assert_allclose(out[:, 4:], np.zeros((6, 4)), atol=1e-6)


class DiffVariantTests(unittest.TestCase):
    def test_first_row_of_each_video_has_zero_difference_half(self) -> None:
        rows = _rows([("v0", 4), ("v1", 3)])
        features = np.random.default_rng(0).normal(size=(7, 3)).astype(np.float32)
        out = build_temporal_features(features, rows, "diff", radius=2)
        np.testing.assert_allclose(out[0, 3:], np.zeros(3))  # v0 first row
        np.testing.assert_allclose(out[4, 3:], np.zeros(3))  # v1 first row

    def test_appearance_half_equals_baseline(self) -> None:
        rows = _rows([("v0", 6), ("v1", 4)])
        features = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
        out = build_temporal_features(features, rows, "diff", radius=2)
        baseline = build_temporal_features(features, rows, "baseline", radius=2)
        np.testing.assert_allclose(out[:, :3], baseline)

    def test_difference_is_current_minus_previous_within_same_video(self) -> None:
        """Two adjacent videos; the boundary row's diff must use only its own
        video's previous frame, never the other video's."""

        rows = _rows([("v0", 3), ("v1", 3)])
        features = np.array([[0.0], [1.0], [2.0], [100.0], [101.0], [102.0]], dtype=np.float32)
        out = build_temporal_features(features, rows, "diff", radius=2)
        # Interior rows: current - previous within the same video.
        np.testing.assert_allclose(out[1], [1.0, 1.0 - 0.0])
        np.testing.assert_allclose(out[2], [2.0, 2.0 - 1.0])
        np.testing.assert_allclose(out[4], [101.0, 101.0 - 100.0])
        np.testing.assert_allclose(out[5], [102.0, 102.0 - 101.0])
        # v1's first row must not use v0's last frame as its previous frame.
        np.testing.assert_allclose(out[3], [100.0, 0.0])

    def test_radius_is_ignored_for_diff(self) -> None:
        rows = _rows([("v0", 6)])
        features = np.random.default_rng(2).normal(size=(6, 3)).astype(np.float32)
        out_r0 = build_temporal_features(features, rows, "diff", radius=0)
        out_r5 = build_temporal_features(features, rows, "diff", radius=5)
        np.testing.assert_allclose(out_r0, out_r5)


if __name__ == "__main__":
    unittest.main()
