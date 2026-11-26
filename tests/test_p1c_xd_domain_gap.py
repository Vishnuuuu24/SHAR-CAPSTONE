import unittest

import numpy as np

from core import p1c_xd_domain_gap as g


def _blobs(rng, centers, n, noise=0.3):
    x = np.vstack([c + noise * rng.standard_normal((n, len(c))) for c in centers])
    return x, np.repeat(np.arange(len(centers)), n)


class DomainGapTests(unittest.TestCase):
    def test_video_vectors_are_unit_means(self):
        f = np.array([[1.0, 0], [1, 0], [0, 2.0]])
        ids, v = g.video_vectors(f, ["a", "a", "b"])
        self.assertEqual(ids, ["a", "b"])
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0)
        np.testing.assert_allclose(v[0], [1, 0])

    def test_centroid_rank(self):
        t = {"A": np.array([1.0, 0]), "B": np.array([0, 1.0])}
        r = g.centroid_rank(np.array([0.1, 0.9]), t, "A")
        self.assertEqual(r["ranking"], ["B", "A"])
        self.assertEqual(r["same_name_rank"], 2)

    def test_separability_high_for_shifted_and_chance_for_identical(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((60, 20)); b = rng.standard_normal((60, 20))
        self.assertLess(g.domain_separability(a, b, 0)["roc_auc"], 0.75)
        self.assertGreater(g.domain_separability(a, b + 3.0, 0)["roc_auc"], 0.99)

    def test_transfer_and_recall(self):
        rng = np.random.default_rng(1)
        x, y = _blobs(rng, [np.eye(10)[0] * 3, np.eye(10)[1] * 3], 30)
        lab = np.where(y == 0, "p", "q")
        r = g.transfer(x[::2], lab[::2], x[1::2], lab[1::2])
        self.assertGreater(r["accuracy"], 0.95)
        self.assertEqual(set(r["recall"]), {"p", "q"})

    def test_balanced_subsample_and_logo(self):
        lab = np.array(["a"] * 10 + ["b"] * 4)
        idx = g.balanced_subsample(lab, 3, np.random.default_rng(0))
        self.assertEqual((lab[idx] == "a").sum(), 3); self.assertEqual((lab[idx] == "b").sum(), 3)
        rng = np.random.default_rng(2)
        x, y = _blobs(rng, [np.eye(8)[0] * 3, np.eye(8)[1] * 3], 20)
        lab = np.where(y == 0, "p", "q"); groups = np.tile([1, 2, 3, 4], 10)
        self.assertGreater(g.leave_one_group_out(x, lab, groups)["accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
