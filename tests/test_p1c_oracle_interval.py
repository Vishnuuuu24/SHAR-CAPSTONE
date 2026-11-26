import unittest

import numpy as np

from core.p1c_oracle_interval import (
    fps_from_cache, frames_in_spans, frames_to_seconds, interval_iou, mcnemar_exact, overlap_length, sample_evenly,
    select_event_positions, spans_overlap, union_spans,
)

FRAMES = np.arange(0, 100, 5)


class IntervalSelectionTests(unittest.TestCase):
    def test_inside_and_outside_partition_the_frames(self):
        spans = [(20, 45)]
        inside = select_event_positions(FRAMES, spans, inside=True)
        outside = select_event_positions(FRAMES, spans, inside=False)
        self.assertEqual(list(FRAMES[inside]), [20, 25, 30, 35, 40, 45])
        self.assertEqual(len(inside) + len(outside), len(FRAMES))
        self.assertFalse(set(inside) & set(outside))

    def test_two_spans_and_empty(self):
        self.assertEqual(int(frames_in_spans(FRAMES, [(0, 5), (90, 95)]).sum()), 4)
        self.assertEqual(len(select_event_positions(FRAMES, [(200, 300)], inside=True)), 0)

    def test_fewer_than_k_repeats_deterministically(self):
        pool = np.array([3, 4, 5])
        a, b = sample_evenly(pool, 8, seed=1), sample_evenly(pool, 8, seed=1)
        self.assertEqual(len(a), 8)
        self.assertEqual(set(a), {3, 4, 5})
        np.testing.assert_array_equal(a, b)


class SamplerTests(unittest.TestCase):
    def test_even_and_seed_only_changes_order(self):
        pool = np.arange(80)
        a, b = sample_evenly(pool, 8, seed=0), sample_evenly(pool, 8, seed=1)
        self.assertEqual(sorted(a), sorted(b))
        self.assertEqual(sorted(a), [5, 15, 25, 35, 45, 55, 65, 75])
        self.assertEqual(len(set(a)), 8)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            sample_evenly(np.array([], dtype=int), 8, 0)


class SpanTests(unittest.TestCase):
    def test_iou_and_overlap(self):
        self.assertAlmostEqual(interval_iou([(0, 10)], [(5, 15)]), 5 / 15)
        self.assertEqual(interval_iou([(0, 1)], [(2, 3)]), 0.0)
        self.assertEqual(interval_iou([], []), 0.0)
        self.assertAlmostEqual(overlap_length([(0, 10), (8, 12)], [(5, 20)]), 7)
        self.assertEqual(union_spans([(0, 3), (2, 5), (7, 8)]), [(0, 5), (7, 8)])
        self.assertTrue(spans_overlap((1, 2), [(0, 1.5)]))
        self.assertFalse(spans_overlap((1, 2), [(2, 3)]))

    def test_fps_conversion(self):
        fps = fps_from_cache(295, 10.0)
        self.assertEqual(fps, 30.0)
        self.assertEqual(frames_to_seconds([(30, 60)], fps), [(1.0, 2.0)])


class McNemarTests(unittest.TestCase):
    def test_toy_table(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertEqual(mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 5), 2 / 32)
        self.assertAlmostEqual(mcnemar_exact(1, 9), 2 * (1 + 10) / 1024)


class TrainEventRowsTests(unittest.TestCase):
    def test_covered_video_gets_in_event_rows_others_unchanged(self):
        from scripts.run_p1c_oracle_interval_probe import _train_event_rows

        def mk(v, f):
            return {"source_video_id": v, "source_frame_index": str(f), "label": "Burglary", "target": "4"}

        full = [mk("A", f) for f in range(0, 100, 5)] + [mk("B", f) for f in range(0, 50, 5)]
        control = [mk("A", f) for f in range(0, 100, 12)] + [mk("B", 0)]
        rows, info = _train_event_rows(full, control, {"A": [(20, 24)]})
        a = [int(r["source_frame_index"]) for r in rows if r["source_video_id"] == "A"]
        self.assertEqual(len(a), 8)
        self.assertEqual(set(a), {20})
        self.assertEqual([r for r in rows if r["source_video_id"] == "B"], [mk("B", 0)])
        self.assertEqual(info["covered_train_videos"], 1)
        self.assertEqual(info["covered_train_videos_with_fewer_than_8_event_frames"], 1)


if __name__ == "__main__":
    unittest.main()


class GlanceSpanTest(unittest.TestCase):
    def test_window_uses_each_videos_fps_and_merges_overlaps(self):
        from core.p1c_oracle_interval import glance_spans

        spans = glance_spans({"a": [100, 150, 1000], "b": [10]}, {"a": 30.0, "b": 15.0}, 2.0)
        self.assertEqual(spans["a"], [(40, 210), (940, 1060)])
        self.assertEqual(spans["b"], [(0, 40)])

    def test_load_glances_sorts_per_video(self):
        import tempfile, os
        from core.p1c_oracle_interval import load_glances

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as h:
            h.write("video-id,glance\nX,30\nX,10\nY,5\n")
        try:
            self.assertEqual(load_glances(h.name), {"X": [10, 30], "Y": [5]})
        finally:
            os.unlink(h.name)
