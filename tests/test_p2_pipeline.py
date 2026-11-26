import unittest

import numpy as np

from core.p2_pipeline import conditions, process_frames, sample_key, summarise
from core.p2a_conventions import load_frozen_p2a_conventions

CONV = load_frozen_p2a_conventions("configs/p2a_conventions.yaml")


class P2PipelineTest(unittest.TestCase):
    def test_fifty_conditions_with_unique_names(self):
        cs = conditions(CONV)
        self.assertEqual(len(cs), 50)
        self.assertEqual(len({c.name for c in cs}), 50)
        self.assertEqual(cs[0].name, "identity__identity")
        self.assertIn("gaussian_25.0__nlm", {c.name for c in cs})
        self.assertIn("low_light_g2.2_s15.0__median", {c.name for c in cs})

    def test_process_is_deterministic_and_clean_identity_is_lossless(self):
        rng = np.random.default_rng(0)
        img = rng.random((24, 32, 3), dtype=np.float32)
        key = sample_key("Abuse/Abuse001_x264.mp4", 5, "ab" * 32, 0)
        cs = {c.name: c for c in conditions(CONV)}
        u8, ps, ss = process_frames([img], [key], cs["identity__identity"], CONV)
        self.assertEqual(ps[0], float("inf"))
        self.assertAlmostEqual(ss[0], 1.0, places=5)
        np.testing.assert_array_equal(u8[0], np.rint(img * 255).astype(np.uint8))
        a = process_frames([img], [key], cs["gaussian_25.0__median"], CONV)
        b = process_frames([img], [key], cs["gaussian_25.0__median"], CONV)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertLess(a[1][0], 40.0)

    def test_summarise_deltas_and_recovery(self):
        t = {"identity__identity": {"macro_f1": 0.5, "accuracy": 0.5},
             "gaussian_25.0__identity": {"macro_f1": 0.3, "accuracy": 0.3},
             "gaussian_25.0__median": {"macro_f1": 0.4, "accuracy": 0.45}}
        s = summarise(t, "identity__identity")
        self.assertAlmostEqual(s["gaussian_25.0__median"]["macro_f1_delta_vs_clean"], -0.1)
        self.assertAlmostEqual(s["gaussian_25.0__median"]["macro_f1_delta_vs_unrestored"], 0.1)
        self.assertAlmostEqual(s["gaussian_25.0__median"]["macro_f1_recovered_fraction"], 0.5)
        self.assertIsNone(s["identity__identity"]["macro_f1_recovered_fraction"])


if __name__ == "__main__":
    unittest.main()
