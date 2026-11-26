"""Tests for `core.p1c_video_sampling` (per-video temporal-coverage sampling).

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

from core.p1c_video_sampling import sample_rows_per_video  # noqa: E402


def _rows(video_frame_counts: list[tuple[str, int, int]]) -> list[dict]:
    """`[(video_id, target, frame_count), ...]` -> cache-shaped rows."""

    rows: list[dict] = []
    for video_id, target, count in video_frame_counts:
        for frame in range(count):
            rows.append({
                "source_dataset": "unit_test_dataset",
                "source_video_id": video_id,
                "source_frame_index": frame * 5,
                "target": str(target),
                "label": f"class{target}",
            })
    return rows


class DeterministicSamplingTests(unittest.TestCase):
    def test_reproducible(self) -> None:
        rows = _rows([("v0", 0, 100)])
        first = sample_rows_per_video(rows, 10, seed=None)
        second = sample_rows_per_video(rows, 10, seed=None)
        np.testing.assert_array_equal(first, second)

    def test_spans_first_and_last_frame(self) -> None:
        rows = _rows([("v0", 0, 100)])
        keep = sample_rows_per_video(rows, 10, seed=None)
        self.assertEqual(keep[0], 0)
        self.assertEqual(keep[-1], 99)

    def test_short_video_keeps_everything(self) -> None:
        rows = _rows([("v0", 0, 5)])
        keep = sample_rows_per_video(rows, 10, seed=None)
        np.testing.assert_array_equal(keep, np.arange(5))

    def test_sorted_ascending(self) -> None:
        rows = _rows([("v0", 0, 50), ("v1", 1, 50)])
        keep = sample_rows_per_video(rows, 7, seed=None)
        self.assertTrue(np.all(np.diff(keep) > 0))

    def test_every_video_still_represented(self) -> None:
        rows = _rows([("v0", 0, 500), ("v1", 1, 3), ("v2", 2, 40)])
        keep = sample_rows_per_video(rows, 8, seed=None)
        kept_videos = {rows[i]["source_video_id"] for i in keep}
        self.assertEqual(kept_videos, {"v0", "v1", "v2"})

    def test_rejects_nonpositive_frames_per_video(self) -> None:
        with self.assertRaises(ValueError):
            sample_rows_per_video(_rows([("v0", 0, 5)]), 0, seed=None)


class SeededSamplingTests(unittest.TestCase):
    def test_different_seeds_differ(self) -> None:
        rows = _rows([("v0", 0, 200)])
        a = sample_rows_per_video(rows, 10, seed=1)
        b = sample_rows_per_video(rows, 10, seed=2)
        self.assertFalse(np.array_equal(a, b))

    def test_stays_within_video(self) -> None:
        rows = _rows([("v0", 0, 50), ("v1", 1, 50)])
        keep = sample_rows_per_video(rows, 6, seed=3)
        for index in keep:
            self.assertIn(rows[index]["source_video_id"], {"v0", "v1"})
        # Every kept index must actually belong to the video whose span it
        # falls in -- i.e. jitter never crossed the v0/v1 boundary at row 50.
        for index in keep:
            if index < 50:
                self.assertEqual(rows[index]["source_video_id"], "v0")
            else:
                self.assertEqual(rows[index]["source_video_id"], "v1")

    def test_covers_the_span(self) -> None:
        """Seeded sampling still spreads across the video, not clustered at one end."""

        rows = _rows([("v0", 0, 300)])
        keep = sample_rows_per_video(rows, 10, seed=7)
        self.assertLess(keep[0], 40)
        self.assertGreater(keep[-1], 260)

    def test_short_video_keeps_everything_even_seeded(self) -> None:
        rows = _rows([("v0", 0, 4)])
        keep = sample_rows_per_video(rows, 10, seed=5)
        np.testing.assert_array_equal(keep, np.arange(4))


class ProvenancePreservationTests(unittest.TestCase):
    def test_video_ids_and_labels_preserved(self) -> None:
        rows = _rows([("v0", 0, 40), ("v1", 1, 40), ("v2", 2, 40)])
        for seed in (None, 0, 1):
            keep = sample_rows_per_video(rows, 6, seed=seed)
            for index in keep:
                original = rows[index]
                self.assertIn(original["source_video_id"], {"v0", "v1", "v2"})
                # target/label are consistent with which video the row came from.
                expected_target = {"v0": "0", "v1": "1", "v2": "2"}[original["source_video_id"]]
                self.assertEqual(original["target"], expected_target)


class PerClassBudgetTests(unittest.TestCase):
    def test_mapped_label_gets_its_own_budget(self) -> None:
        rows = _rows([("v0", 0, 100), ("v1", 1, 100)])
        keep = sample_rows_per_video(rows, 8, seed=None, frames_per_video_by_label={"class1": 32})
        v0_count = sum(1 for i in keep if rows[i]["source_video_id"] == "v0")
        v1_count = sum(1 for i in keep if rows[i]["source_video_id"] == "v1")
        self.assertEqual(v0_count, 8)
        self.assertEqual(v1_count, 32)

    def test_unmapped_label_uses_default(self) -> None:
        rows = _rows([("v0", 0, 100)])
        keep = sample_rows_per_video(rows, 8, seed=None, frames_per_video_by_label={"class9": 32})
        self.assertEqual(len(keep), 8)

    def test_unknown_label_in_mapping_is_simply_unused(self) -> None:
        rows = _rows([("v0", 0, 100)])
        keep_plain = sample_rows_per_video(rows, 8, seed=None)
        keep_mapped = sample_rows_per_video(
            rows, 8, seed=None, frames_per_video_by_label={"class_does_not_exist": 32}
        )
        np.testing.assert_array_equal(keep_plain, keep_mapped)

    def test_mapping_preserves_determinism(self) -> None:
        rows = _rows([("v0", 0, 100), ("v1", 1, 100)])
        first = sample_rows_per_video(rows, 8, seed=None, frames_per_video_by_label={"class1": 32})
        second = sample_rows_per_video(rows, 8, seed=None, frames_per_video_by_label={"class1": 32})
        np.testing.assert_array_equal(first, second)

    def test_mapping_preserves_sortedness(self) -> None:
        rows = _rows([("v0", 0, 100), ("v1", 1, 100)])
        keep = sample_rows_per_video(rows, 8, seed=3, frames_per_video_by_label={"class1": 32})
        self.assertTrue(np.all(np.diff(keep) > 0))

    def test_mapping_preserves_short_video_rule(self) -> None:
        rows = _rows([("v0", 0, 5)])
        keep = sample_rows_per_video(rows, 8, seed=None, frames_per_video_by_label={"class0": 32})
        np.testing.assert_array_equal(keep, np.arange(5))

    def test_rejects_zero_budget(self) -> None:
        rows = _rows([("v0", 0, 20)])
        with self.assertRaises(ValueError):
            sample_rows_per_video(rows, 8, seed=None, frames_per_video_by_label={"class0": 0})

    def test_rejects_negative_budget(self) -> None:
        rows = _rows([("v0", 0, 20)])
        with self.assertRaises(ValueError):
            sample_rows_per_video(rows, 8, seed=None, frames_per_video_by_label={"class0": -5})

    def test_rejects_non_int_budget(self) -> None:
        rows = _rows([("v0", 0, 20)])
        with self.assertRaises(ValueError):
            sample_rows_per_video(rows, 8, seed=None, frames_per_video_by_label={"class0": 8.5})

    def test_rejects_non_string_key(self) -> None:
        rows = _rows([("v0", 0, 20)])
        with self.assertRaises(ValueError):
            sample_rows_per_video(rows, 8, seed=None, frames_per_video_by_label={0: 8})

    def test_omitting_mapping_reproduces_existing_behaviour_exactly(self) -> None:
        rows = _rows([("v0", 0, 500), ("v1", 1, 3), ("v2", 2, 40)])
        for seed in (None, 0, 1, 7):
            without_kwarg = sample_rows_per_video(rows, 8, seed=seed)
            with_none = sample_rows_per_video(rows, 8, seed=seed, frames_per_video_by_label=None)
            np.testing.assert_array_equal(without_kwarg, with_none)


if __name__ == "__main__":
    unittest.main()
