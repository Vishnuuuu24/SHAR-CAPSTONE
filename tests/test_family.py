"""D-72 family grouping and top-k accuracy."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from core import family

CLASSES = sorted(family.FAMILY_OF)
REPO_ROOT = Path(__file__).resolve().parents[1]


class FamilyTest(unittest.TestCase):
    def test_all_fourteen_labels_map_to_seven_families(self) -> None:
        self.assertEqual(len(family.FAMILY_OF), 14)
        self.assertEqual(len(family.FAMILIES), 7)
        self.assertEqual(sum(v == "violence" for v in family.FAMILY_OF.values()), 4)
        self.assertEqual(sum(v == "theft" for v in family.FAMILY_OF.values()), 4)
        with self.assertRaises(ValueError):
            family.family_index(["Abuse", "Loitering"])

    def test_a_within_family_error_is_correct_at_family_level(self) -> None:
        truth = np.array([CLASSES.index("Assault"), CLASSES.index("Robbery"), CLASSES.index("Arson")])
        predicted = np.array([CLASSES.index("Fighting"), CLASSES.index("Stealing"), CLASSES.index("Arrest")])
        scores = family.family_scores(truth, predicted, CLASSES)
        self.assertAlmostEqual(scores["accuracy"], 2 / 3)
        self.assertEqual(scores["per_family_f1"]["violence"], 1.0)

    def test_confusion_route_equals_prediction_route(self) -> None:
        rng = np.random.default_rng(0)
        truth, predicted = rng.integers(0, 14, 500), rng.integers(0, 14, 500)
        confusion = np.zeros((14, 14), dtype=int)
        np.add.at(confusion, (truth, predicted), 1)
        a, b = family.family_scores(truth, predicted, CLASSES), family.family_scores_from_confusion(confusion, CLASSES)
        self.assertAlmostEqual(a["accuracy"], b["accuracy"])
        self.assertAlmostEqual(a["macro_f1"], b["macro_f1"])

    def test_stored_headline_test_confusions_regroup_to_the_audited_figures(self) -> None:
        path = REPO_ROOT / "results/p1c/headline-14class-test/test.json"
        if not path.exists():
            self.skipTest("headline test artifact not present")
        stored = json.loads(path.read_text())
        classes = list(stored["test_per_class_f1"])
        pooled = sum(np.array(r["test_confusion"]) for r in stored["runs"])
        scores = family.family_scores_from_confusion(pooled, classes)
        self.assertAlmostEqual(scores["accuracy"], 0.5570, places=3)
        self.assertAlmostEqual(scores["macro_f1"], 0.5208, places=3)

    def test_top_k(self) -> None:
        probabilities = np.array([[0.5, 0.3, 0.2, 0.0], [0.1, 0.2, 0.3, 0.4]])
        self.assertEqual(family.top_k_accuracy(np.array([2, 0]), probabilities, k=3), 0.5)
        self.assertEqual(family.top_k_accuracy(np.array([0, 3]), probabilities, k=1), 1.0)


if __name__ == "__main__":
    unittest.main()
