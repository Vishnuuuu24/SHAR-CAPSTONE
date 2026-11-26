import unittest

from core.subset7 import FAMILY, aggregate, candidate_subsets, novelty, winner

CLASSES = sorted(FAMILY)


class Subset7Test(unittest.TestCase):
    def test_candidate_counts_match_family_structure(self):
        self.assertEqual(len(candidate_subsets(CLASSES, 7)), 4 * 4 * 2)
        self.assertEqual(len(candidate_subsets(CLASSES, 6)), 560)
        self.assertTrue(all(len(s) == 7 for s in candidate_subsets(CLASSES, 6)))

    def test_novelty(self):
        self.assertEqual(novelty(["Abuse", "Assault", "Fighting", "Shooting"]), 1)
        self.assertEqual(novelty(["Arrest", "Explosion", "Fighting", "Normal_Videos_event", "RoadAccidents", "Shoplifting", "Vandalism"]), 7)

    def test_aggregate_refuses_incomplete(self):
        recs = [{"subset": "A+B", "fold": 1, "seed": 0, "macro_f1": 0.5}]
        with self.assertRaises(ValueError):
            aggregate(recs, [1, 2], [0])

    def test_winner_rule_and_tiebreak(self):
        table = {
            "x": {"mean": 0.70, "novelty": 6}, "y": {"mean": 0.70, "novelty": 7},
            "z": {"mean": 0.69, "novelty": 7}, "w": {"mean": 0.71, "novelty": 5},
        }
        self.assertEqual(winner(table), "w")
        self.assertEqual(winner(table, min_novelty=6), "y")
        self.assertEqual(winner(table, min_novelty=7), "y")

    def test_aggregate_means(self):
        recs = [{"subset": "Arson+Arrest", "fold": f, "seed": s, "macro_f1": v}
                for f, s, v in [(1, 0, .4), (1, 1, .6), (2, 0, .8), (2, 1, .8)]]
        t = aggregate(recs, [1, 2], [0, 1])["Arson+Arrest"]
        self.assertAlmostEqual(t["per_fold"][1], 0.5)
        self.assertAlmostEqual(t["mean"], 0.65)


if __name__ == "__main__":
    unittest.main()
