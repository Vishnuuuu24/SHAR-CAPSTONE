import unittest

import numpy as np

from core.showcase_timeline import CLIP, first_alert, pooled, trailing_windows


class TrailingWindowTests(unittest.TestCase):
    def test_windows_are_causal_hold_whole_clips_and_close_on_last_frame(self):
        n = 100
        wins = trailing_windows(n, width=32, step=6)
        self.assertEqual(wins[0][:2], (0, 32))
        self.assertEqual(wins[-1][1], n)
        for s, e, clips in wins:
            self.assertLessEqual(e - s, 32)
            self.assertTrue(clips)
            for k in clips:
                self.assertTrue(s <= k * CLIP and (k + 1) * CLIP <= e)

    def test_short_video_gets_one_window(self):
        self.assertEqual(trailing_windows(20, 32, 6), [(0, 20, [0])])

    def test_full_pool_matches_plain_mean(self):
        rng = np.random.default_rng(0)
        f = {k: rng.normal(size=(40, 3)).astype(np.float16).astype(np.float32) for k in ("B", "L", "S")}
        c = rng.normal(size=(2, 4)).astype(np.float32)
        v = pooled(f, c)
        np.testing.assert_allclose(v["B"], f["B"].mean(0), rtol=1e-6)
        np.testing.assert_allclose(pooled(f, c, slice(0, 16), [0])["VJ"], c[0])


class FirstAlertTests(unittest.TestCase):
    @staticmethod
    def w(end, label, prob):
        return {"end_s": end, "top3": [{"label": label, "prob": prob}]}

    def test_needs_a_consecutive_run_of_one_class_over_threshold(self):
        ws = [self.w(1, "Explosion", .4), self.w(2, "Explosion", .6), self.w(3, "Arson", .7),
              self.w(4, "Explosion", .8), self.w(5, "Explosion", .8), self.w(6, "Explosion", .9)]
        self.assertEqual(first_alert(ws, .5, 3), {"label": "Explosion", "at_s": 6})

    def test_normal_never_alerts(self):
        self.assertIsNone(first_alert([self.w(i, "Normal_Videos_event", .99) for i in range(5)], .5, 2))


if __name__ == "__main__":
    unittest.main()
