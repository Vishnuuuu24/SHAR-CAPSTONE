"""Tests for `core.p1c_flow` (the RAFT flow-descriptor pooling, pure NumPy).

Synthetic constant/zero flow fields throughout -- what is under test is the
pooling arithmetic (grid boundaries, orientation binning, normalisation,
NaN-avoidance), never RAFT itself or any real flow.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_flow import (  # noqa: E402
    DEFAULT_GRID,
    DEFAULT_ORIENTATION_BINS,
    DESCRIPTOR_LENGTH,
    descriptor_field_names,
    descriptor_length,
    flow_descriptor,
    flow_descriptor_batch,
)

HEIGHT, WIDTH = 240, 320


def _constant_flow(u: float, v: float, height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    field = np.empty((2, height, width), dtype=np.float32)
    field[0] = u
    field[1] = v
    return field


class DescriptorLengthTests(unittest.TestCase):
    def test_default_length_is_162(self) -> None:
        self.assertEqual(DESCRIPTOR_LENGTH, 162)
        self.assertEqual(descriptor_length(), 162)

    def test_length_formula_for_other_grid_and_bins(self) -> None:
        self.assertEqual(descriptor_length(grid=(2, 3), orientation_bins=6), 2 * 3 * (2 + 6) + 2)

    def test_field_names_length_matches_descriptor_length(self) -> None:
        names = descriptor_field_names()
        self.assertEqual(len(names), DESCRIPTOR_LENGTH)
        self.assertEqual(len(set(names)), len(names), "field names must be unique")

    def test_field_names_length_matches_for_other_grid_and_bins(self) -> None:
        grid, bins = (2, 3), 6
        names = descriptor_field_names(grid=grid, orientation_bins=bins)
        self.assertEqual(len(names), descriptor_length(grid=grid, orientation_bins=bins))

    def test_actual_descriptor_matches_declared_length(self) -> None:
        flow = _constant_flow(1.0, 0.0)
        descriptor = flow_descriptor(flow)
        self.assertEqual(descriptor.shape, (DESCRIPTOR_LENGTH,))
        self.assertEqual(descriptor.dtype, np.float32)


class ZeroFlowTests(unittest.TestCase):
    def test_zero_flow_gives_zero_magnitudes_and_no_nan(self) -> None:
        flow = _constant_flow(0.0, 0.0)
        descriptor = flow_descriptor(flow)
        self.assertFalse(np.any(np.isnan(descriptor)))
        rows, cols = DEFAULT_GRID
        stride = 2 + DEFAULT_ORIENTATION_BINS
        for cell in range(rows * cols):
            mean_mag = descriptor[cell * stride]
            std_mag = descriptor[cell * stride + 1]
            histogram = descriptor[cell * stride + 2 : cell * stride + stride]
            self.assertEqual(mean_mag, 0.0)
            self.assertEqual(std_mag, 0.0)
            np.testing.assert_array_equal(histogram, np.zeros(DEFAULT_ORIENTATION_BINS, dtype=np.float32))
        self.assertEqual(descriptor[-2], 0.0)  # global mean magnitude
        self.assertEqual(descriptor[-1], 0.0)  # global p95 magnitude


class OrientationBinTests(unittest.TestCase):
    def test_constant_rightward_flow_bin_and_mean_magnitude(self) -> None:
        speed = 3.0
        flow = _constant_flow(speed, 0.0)
        descriptor = flow_descriptor(flow)
        rows, cols = DEFAULT_GRID
        stride = 2 + DEFAULT_ORIENTATION_BINS
        rightward_bin = DEFAULT_ORIENTATION_BINS // 2  # documented: angle 0 -> bins/2
        for cell in range(rows * cols):
            mean_mag = descriptor[cell * stride]
            histogram = descriptor[cell * stride + 2 : cell * stride + stride]
            self.assertAlmostEqual(float(mean_mag), speed, places=5)
            expected_hist = np.zeros(DEFAULT_ORIENTATION_BINS, dtype=np.float32)
            expected_hist[rightward_bin] = 1.0
            np.testing.assert_allclose(histogram, expected_hist, atol=1e-6)
        self.assertAlmostEqual(float(descriptor[-2]), speed, places=5)

    def test_constant_upward_flow_bin(self) -> None:
        # Image/array convention: v positive is DOWNWARD, so upward motion is v < 0.
        speed = 2.0
        flow = _constant_flow(0.0, -speed)
        descriptor = flow_descriptor(flow)
        stride = 2 + DEFAULT_ORIENTATION_BINS
        upward_bin = DEFAULT_ORIENTATION_BINS // 4  # documented: angle -pi/2 -> bins/4
        histogram = descriptor[2 : 2 + DEFAULT_ORIENTATION_BINS]  # first cell
        expected_hist = np.zeros(DEFAULT_ORIENTATION_BINS, dtype=np.float32)
        expected_hist[upward_bin] = 1.0
        np.testing.assert_allclose(histogram, expected_hist, atol=1e-6)

    def test_flipping_sign_rotates_histogram_by_half_the_bins(self) -> None:
        # A non-axis-aligned direction, to check the general shift property.
        flow = _constant_flow(1.0, 0.6)
        flipped = _constant_flow(-1.0, -0.6)
        descriptor = flow_descriptor(flow)
        flipped_descriptor = flow_descriptor(flipped)
        stride = 2 + DEFAULT_ORIENTATION_BINS
        half = DEFAULT_ORIENTATION_BINS // 2
        histogram = descriptor[2 : 2 + DEFAULT_ORIENTATION_BINS]
        flipped_histogram = flipped_descriptor[2 : 2 + DEFAULT_ORIENTATION_BINS]
        rotated = np.roll(histogram, half)
        np.testing.assert_allclose(flipped_histogram, rotated, atol=1e-6)


class ScalingInvarianceTests(unittest.TestCase):
    def test_scaling_field_by_two_doubles_magnitude_but_not_orientation_histogram(self) -> None:
        rng = np.random.default_rng(0)
        flow = rng.normal(size=(2, HEIGHT, WIDTH)).astype(np.float32)
        scaled = flow * 2.0

        descriptor = flow_descriptor(flow)
        scaled_descriptor = flow_descriptor(scaled)

        stride = 2 + DEFAULT_ORIENTATION_BINS
        rows, cols = DEFAULT_GRID
        for cell in range(rows * cols):
            mean_mag = descriptor[cell * stride]
            scaled_mean_mag = scaled_descriptor[cell * stride]
            std_mag = descriptor[cell * stride + 1]
            scaled_std_mag = scaled_descriptor[cell * stride + 1]
            histogram = descriptor[cell * stride + 2 : cell * stride + stride]
            scaled_histogram = scaled_descriptor[cell * stride + 2 : cell * stride + stride]
            self.assertAlmostEqual(float(scaled_mean_mag), float(mean_mag) * 2.0, places=4)
            self.assertAlmostEqual(float(scaled_std_mag), float(std_mag) * 2.0, places=4)
            np.testing.assert_allclose(scaled_histogram, histogram, atol=1e-5)
        self.assertAlmostEqual(float(scaled_descriptor[-2]), float(descriptor[-2]) * 2.0, places=4)
        self.assertAlmostEqual(float(scaled_descriptor[-1]), float(descriptor[-1]) * 2.0, places=4)


class BatchTests(unittest.TestCase):
    def test_batch_equals_looping_single_version(self) -> None:
        rng = np.random.default_rng(1)
        flows = rng.normal(size=(5, 2, HEIGHT, WIDTH)).astype(np.float32)
        batch_result = flow_descriptor_batch(flows)
        looped = np.stack([flow_descriptor(flow) for flow in flows])
        np.testing.assert_allclose(batch_result, looped, atol=1e-6)
        self.assertEqual(batch_result.shape, (5, DESCRIPTOR_LENGTH))

    def test_batch_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            flow_descriptor_batch(np.zeros((5, 3, HEIGHT, WIDTH), dtype=np.float32))


class ShapeValidationTests(unittest.TestCase):
    def test_single_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            flow_descriptor(np.zeros((3, HEIGHT, WIDTH), dtype=np.float32))

    def test_non_dividing_grid_does_not_crash(self) -> None:
        # 240/4=60 divides evenly but 320/3 does not -- exercise the remainder case.
        flow = _constant_flow(1.0, 1.0)
        descriptor = flow_descriptor(flow, grid=(4, 3))
        self.assertEqual(descriptor.shape, (descriptor_length(grid=(4, 3)),))
        self.assertFalse(np.any(np.isnan(descriptor)))


if __name__ == "__main__":
    unittest.main()
