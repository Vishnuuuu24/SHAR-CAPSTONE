import unittest

import numpy as np

from scripts.build_p1c_vjepa2_clip_cache import select_spread, shuffled_order
from scripts.run_p1c_vjepa2_probe import _align_train, _same_clips


def clips(n_videos, per):
    return [(list(range(16)), f"v{v}", 16 * k, 16 * k + 15, v % 3) for v in range(n_videos) for k in range(per)]


class Vjepa2Pipeline(unittest.TestCase):
    def test_select_spread_keeps_n_per_video_in_order(self):
        kept = select_spread(clips(3, 40), 8)
        self.assertEqual(len(kept), 24)
        for v in range(3):
            starts = [c[2] for c in kept if c[1] == f"v{v}"]
            self.assertEqual(len(starts), 8)
            self.assertEqual(starts, sorted(starts))
            self.assertGreater(starts[-1] - starts[0], 16 * 20)  # spread across the video, not the first 8

    def test_select_spread_is_stable_under_reselection(self):
        once = select_spread(clips(2, 40), 8)
        self.assertEqual(select_spread(once, 8), once)  # the probe re-selects from the subsampled cache

    def test_shuffled_order_is_deterministic_permutation(self):
        a, b = shuffled_order(5), shuffled_order(5)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(sorted(a.tolist()), list(range(16)))
        self.assertFalse(np.array_equal(shuffled_order(5), shuffled_order(6)))

    def test_align_train_orders_and_refuses_missing(self):
        rows = [{"source_video_id": "v", "source_frame_index": i} for i in (0, 16, 32)]
        feats = np.arange(3)[:, None].astype(np.float32)
        wanted = [rows[2], rows[0]]
        self.assertEqual(_align_train(feats, rows, wanted, "t").ravel().tolist(), [2.0, 0.0])
        with self.assertRaises(ValueError):
            _align_train(feats, rows[:2], wanted, "t")
        with self.assertRaises(ValueError):
            _same_clips(rows, rows[:2], "t")


if __name__ == "__main__":
    unittest.main()
