import unittest

import numpy as np

from core.specialist_eval import evaluate_system, within_group_accuracy


class SpecialistEvalTests(unittest.TestCase):
    def test_paired_delta_and_gate(self):
        classes = ["A", "B", "Normal_Videos_event"]
        truth = np.array([0, 1, 2, 0, 1, 2])
        bad, good = np.array([1, 0, 2, 1, 0, 2]), truth.copy()
        base = {(f, s): (truth, bad) for f in (1, 2) for s in (0, 1)}
        cand = {(f, s): (truth, good) for f in (1, 2) for s in (0, 1)}
        r = evaluate_system(classes, base, cand)
        self.assertTrue(r["passes_gate"]); self.assertEqual(r["folds_meeting_gate"], 2)
        self.assertGreater(r["mean_delta"], 0.04)
        same = evaluate_system(classes, base, base)
        self.assertEqual(same["mean_delta"], 0.0); self.assertFalse(same["passes_gate"])

    def test_mismatched_truth_or_runs_rejected(self):
        classes = ["A", "B"]
        a = {(1, 0): (np.array([0, 1]), np.array([0, 1]))}
        with self.assertRaises(ValueError):
            evaluate_system(classes, a, {(1, 0): (np.array([1, 0]), np.array([0, 1]))})
        with self.assertRaises(ValueError):
            evaluate_system(classes, a, {(2, 0): a[(1, 0)]})

    def test_within_group_accuracy(self):
        base = np.array([[.6, .3, .1], [.3, .2, .5], [.5, .3, .2]])
        spec = np.array([[.9, .1], [.2, .8], [.4, .6]])
        r = within_group_accuracy(base, spec, np.array([0, 1, 2]), [0, 1])
        self.assertEqual(r["n"], 2)
        self.assertAlmostEqual(r["specialist"], 1.0)
        self.assertAlmostEqual(r["baseline_restricted"], 0.5)


if __name__ == "__main__":
    unittest.main()
