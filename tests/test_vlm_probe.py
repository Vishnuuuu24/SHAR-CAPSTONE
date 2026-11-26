import unittest

import numpy as np

from core.vlm_probe import CLASS_TEXT, LETTERS, build_question, evaluate, letter_probs_to_classes, option_order

CLASSES = sorted(CLASS_TEXT)


class VlmProbeTest(unittest.TestCase):
    def test_orders_and_question(self):
        rev = option_order(CLASSES, "reverse")
        self.assertEqual(rev[0], "Vandalism")
        q = build_question(rev, 8)
        self.assertIn("A. Vandalism", q)
        self.assertIn("N. Abuse", q)
        self.assertIn("H. Normal activity (no crime)", build_question(option_order(CLASSES, "forward"), 8))

    def test_letter_probs_map_back_to_classes(self):
        logits = np.full(len(LETTERS), -10.0)
        logits[0] = 5.0  # letter A
        fwd = letter_probs_to_classes(logits, option_order(CLASSES, "forward"), CLASSES)
        rev = letter_probs_to_classes(logits, option_order(CLASSES, "reverse"), CLASSES)
        self.assertEqual(CLASSES[int(fwd.argmax())], "Abuse")
        self.assertEqual(CLASSES[int(rev.argmax())], "Vandalism")
        self.assertAlmostEqual(fwd.sum(), 1.0)

    def test_evaluate_perfect_vlm_beats_baseline(self):
        n = len(CLASSES)
        truth = np.arange(n)
        base_probs = np.full((n, n), 1.0 / n)
        base_probs[:, 0] += 0.1  # baseline always predicts class 0
        runs = {(f, 0): {"videos": [f"v{i}" for i in range(n)], "truth": truth, "probs": base_probs} for f in (1, 2)}
        vlm = {f"v{i}": np.eye(n)[i] for i in range(n)}
        out = evaluate(CLASSES, runs, vlm, 0.5)
        self.assertAlmostEqual(out["vlm_alone"]["candidate_mean"], 1.0)
        self.assertTrue(out["fusion_w0.5"]["passes_gate"])

    def test_evaluate_refuses_missing_video(self):
        n = len(CLASSES)
        runs = {(1, 0): {"videos": ["a"], "truth": np.array([0]), "probs": np.ones((1, n)) / n}}
        with self.assertRaises(KeyError):
            evaluate(CLASSES, runs, {}, 0.5)


if __name__ == "__main__":
    unittest.main()
