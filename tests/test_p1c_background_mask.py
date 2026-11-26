"""Tests for `core.p1c_background_mask` (background-tolerant loss weights).

Synthetic fixtures only, mirroring `tests/test_p1c_temporal_features.py`'s style.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_background_mask import background_weight_counts, background_weights  # noqa: E402


def _row(video_id: str, frame_index: int, label: str) -> dict:
    return {
        "source_dataset": "unit_test_dataset",
        "source_video_id": video_id,
        "source_frame_index": frame_index,
        "label": label,
    }


class BackgroundWeightsTests(unittest.TestCase):
    def setUp(self) -> None:
        # "shooting_ann": annotated, event spans [50, 100]. Frames outside are
        # known background.
        self.rows = [
            _row("shooting_ann", 10, "Shooting"),   # before span: background
            _row("shooting_ann", 60, "Shooting"),   # inside span: event
            _row("shooting_ann", 100, "Shooting"),  # span end inclusive: event
            _row("shooting_ann", 150, "Shooting"),  # after span: background
            _row("shooting_unann", 5, "Shooting"),  # unannotated video: kept
            _row("other_label_vid", 10, "Robbery"), # different label, video id
                                                     # happens to be annotated
                                                     # under Shooting -- must
                                                     # stay untouched
            _row("normal_vid", 10, "Normal"),
        ]
        self.intervals = {"shooting_ann": [(50, 100)], "other_label_vid": [(0, 5)]}

    def test_only_annotated_label_out_of_span_rows_are_zeroed(self) -> None:
        weights = background_weights(self.rows, self.intervals, label="Shooting")
        expected = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        np.testing.assert_array_equal(weights, expected)

    def test_in_span_rows_keep_weight_one(self) -> None:
        weights = background_weights(self.rows, self.intervals, label="Shooting")
        self.assertEqual(weights[1], 1.0)
        self.assertEqual(weights[2], 1.0)

    def test_other_labels_keep_weight_one_even_if_video_id_is_in_intervals(self) -> None:
        weights = background_weights(self.rows, self.intervals, label="Shooting")
        # Row index 5 is label="Robbery" on video "other_label_vid", which IS
        # an annotated key in `intervals` (for Shooting) -- must not be zeroed.
        self.assertEqual(weights[5], 1.0)

    def test_unannotated_video_of_the_label_keeps_all_ones(self) -> None:
        weights = background_weights(self.rows, self.intervals, label="Shooting")
        self.assertEqual(weights[4], 1.0)

    def test_empty_intervals_returns_all_ones(self) -> None:
        """Regression contract: no official intervals -> nothing is zeroed."""

        weights = background_weights(self.rows, {}, label="Shooting")
        np.testing.assert_array_equal(weights, np.ones(len(self.rows)))

    def test_counts(self) -> None:
        counts = background_weight_counts(self.rows, self.intervals, label="Shooting")
        self.assertEqual(counts["annotated_videos"], 1)  # only shooting_ann
        self.assertEqual(counts["rows_zeroed"], 2)
        self.assertEqual(counts["rows_kept"], 3)  # 5 Shooting rows total - 2 zeroed


if __name__ == "__main__":
    unittest.main()
