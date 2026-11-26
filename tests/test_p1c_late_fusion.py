import unittest

import numpy as np

from core import p1c_late_fusion as lf

C = ["Shooting", "Fighting", "Assault"]


def sysdict(rows):
    # rows: {(fold, seed, vid): (label, [p...])}
    return C, {k: (l, np.array(p, float)) for k, (l, p) in rows.items()}


class LateFusionTests(unittest.TestCase):
    def test_fuse_equal_weight(self):
        self.assertTrue(np.allclose(lf.fuse(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])), [0.5, 0.5, 0]))

    def test_align_mismatch_raises(self):
        a = sysdict({(1, 0, "v1"): ("Shooting", [1, 0, 0])})
        b = sysdict({(1, 0, "v2"): ("Shooting", [1, 0, 0])})
        with self.assertRaises(ValueError):
            lf.align(a, b)
        with self.assertRaises(ValueError):
            lf.align(a, (C[::-1], a[1]))
        with self.assertRaises(ValueError):
            lf.align(a, sysdict({(1, 0, "v1"): ("Fighting", [1, 0, 0])}))

    def test_delta_and_gate(self):
        rows_a, rows_b = {}, {}
        for f in (1, 2):
            for i, lab in enumerate(C):
                v = (f, 0, f"v{i}")
                onehot = [0, 0, 0]; onehot[i] = 1
                wrong = [0, 0, 0]; wrong[(i + 1) % 3] = 0.6; wrong[i] = 0.4
                rows_a[v] = (lab, wrong)  # A always wrong, true class second
                rows_b[v] = (lab, [x * 0.9 + 0.05 for x in onehot])  # B right
        a, b = sysdict(rows_a), sysdict(rows_b)
        keys = lf.align(a, b)
        systems = {"A": a, "B": b, "A+B": (C, lf.fused_probs(a, b, keys))}
        ev = lf.evaluate(systems, "A", ["B", "A+B"])
        self.assertAlmostEqual(ev["systems"]["B"]["macro_f1_mean"], 1.0)
        # fused true class 0.5*0.4+0.5*0.95=0.675 > 0.3 on A's wrong class
        self.assertAlmostEqual(ev["systems"]["A+B"]["macro_f1_mean"], 1.0)
        comp = ev["comparisons"]["B_minus_A"]
        self.assertEqual(comp["folds_positive"], 2)
        self.assertFalse(comp["gate"]["passed"])  # only 2 of 4 folds present
        self.assertAlmostEqual(comp["highlight"]["Shooting"], 1.0)

    def test_correctness_overlap(self):
        a = sysdict({(1, 0, "v1"): ("Shooting", [1, 0, 0]), (1, 0, "v2"): ("Fighting", [0, 1, 0]),
                     (1, 0, "v3"): ("Assault", [1, 0, 0]), (1, 0, "v4"): ("Assault", [1, 0, 0])})
        b = sysdict({(1, 0, "v1"): ("Shooting", [1, 0, 0]), (1, 0, "v2"): ("Fighting", [1, 0, 0]),
                     (1, 0, "v3"): ("Assault", [0, 0, 1]), (1, 0, "v4"): ("Assault", [0, 1, 0])})
        o = lf.correctness_overlap(a, b, lf.align(a, b))
        self.assertEqual((o["both"], o["only_a"], o["only_b"], o["neither"]), (0.25, 0.25, 0.25, 0.25))


if __name__ == "__main__":
    unittest.main()
