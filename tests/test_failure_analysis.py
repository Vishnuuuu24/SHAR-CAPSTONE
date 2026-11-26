import unittest

import numpy as np

from core import failure_analysis as fa
from core.subset7 import FAMILY

CLASSES = sorted(FAMILY)


def _preds(truth, pred, folds=None, seeds=None, conf=0.9):
    n, c = len(truth), len(CLASSES)
    probs = np.full((n, c), (1 - conf) / (c - 1))
    probs[np.arange(n), pred] = conf
    return fa.Predictions(CLASSES, np.array(folds if folds is not None else [1] * n), np.array(seeds if seeds is not None else [0] * n),
                          np.array([f"v{i}" for i in range(n)]), np.array(truth), probs)


class FailureAnalysis(unittest.TestCase):
    def test_perfect_predictions(self):
        t = list(range(14)) * 2
        p = _preds(t, t)
        h = fa.headline(p)
        self.assertAlmostEqual(h["macro_f1"], 1.0)
        self.assertAlmostEqual(h["accuracy"], 1.0)
        self.assertEqual(fa.topk_accuracy(p)["top1"], 1.0)
        self.assertEqual(fa.video_reliability(p)["always_wrong"], 0)
        self.assertTrue(all(r["recall"] == 1.0 for r in fa.per_class(p)))

    def test_error_structure_shares(self):
        ni = CLASSES.index("Normal_Videos_event")
        a, b = CLASSES.index("Abuse"), CLASSES.index("Assault")      # same family: violence
        r = CLASSES.index("Robbery")
        truth = [a, a, a, a, r, ni]
        pred = [b, b, ni, a, a, r]                                    # 2 within-family, 1 crime->normal, 1 cross-family, 1 normal->crime
        e = fa.error_structure(_preds(truth, pred))
        self.assertAlmostEqual(e["errors"], 5.0)
        self.assertAlmostEqual(e["crime_to_crime_share"], 3 / 5)
        self.assertAlmostEqual(e["crime_to_normal_share"], 1 / 5)
        self.assertAlmostEqual(e["normal_to_crime_share"], 1 / 5)
        self.assertAlmostEqual(e["within_family_share_of_crime_errors"], 2 / 3)
        self.assertEqual(e["top_pairs"][0]["pair"], "Abuse <-> Assault")

    def test_calibration_and_selective_accuracy(self):
        truth = [0, 1, 2, 3]
        p = _preds(truth, [0, 1, 2, 4], conf=0.95)                    # 3 right, 1 confidently wrong
        c = fa.calibration(p)
        self.assertAlmostEqual(c["mean_confidence_right"], 0.95)
        self.assertAlmostEqual(c["share_of_errors_with_confidence_above_0.9"], 1.0)
        self.assertAlmostEqual(c["ece"], abs(0.95 - 0.75))
        self.assertAlmostEqual(c["accuracy_when_keeping_most_confident"]["100%"], 0.75)

    def test_normal_vs_crime_auc_extremes(self):
        ni = CLASSES.index("Normal_Videos_event")
        good = _preds([0, 1, ni, ni], [0, 1, ni, ni])
        bad = _preds([0, 1, ni, ni], [ni, ni, 0, 1])
        self.assertAlmostEqual(fa.normal_vs_crime_auc(good), 1.0)
        self.assertAlmostEqual(fa.normal_vs_crime_auc(bad), 0.0)

    def test_bootstrap_is_tight_when_perfect(self):
        t = list(range(14)) * 4
        b = fa.cluster_bootstrap(_preds(t, t), draws=50)
        self.assertAlmostEqual(b["ci95"][0], 1.0, places=9)           # a class is never dropped, so perfect stays perfect

    def test_top_pairs_use_seed_average(self):
        t = [0, 0, 0, 0]                                              # two videos, each evaluated by two seeds
        p = _preds(t, [1, 1, 1, 1], seeds=[0, 0, 1, 1])
        self.assertAlmostEqual(fa.error_structure(p)["errors"], 2.0)  # 4 wrong evaluations / 2 seeds


class PairedClusterBootstrap(unittest.TestCase):
    def _two_fold(self, pred_fn):
        # 28 videos (2 per class); each appears in folds 1 and 2 (overlapping folds), seeds 0 and 1
        truth, folds, seeds, vids = [], [], [], []
        for f in (1, 2):
            for s in (0, 1):
                for v in range(28):
                    truth.append(v % 14); folds.append(f); seeds.append(s); vids.append(f"v{v}")
        pred = [pred_fn(t, v) for t, v in zip(truth, vids)]
        p = _preds(truth, pred, folds, seeds)
        return fa.Predictions(p.classes, p.fold, p.seed, np.array(vids), p.truth, p.probs)

    def test_observed_equals_registered_metric_and_perfect_is_tight(self):
        perfect = self._two_fold(lambda t, v: t)
        r = fa.paired_cluster_bootstrap(perfect, draws=30)
        self.assertAlmostEqual(r["base"]["observed"], fa.headline(perfect)["macro_f1"])
        self.assertAlmostEqual(r["base"]["ci95"][0], 1.0)

    def test_identical_models_have_zero_difference(self):
        a = self._two_fold(lambda t, v: t if int(v[1:]) % 3 else (t + 1) % 14)
        r = fa.paired_cluster_bootstrap(a, a, draws=50)
        self.assertEqual(r["difference"]["ci95"], [0.0, 0.0])

    def test_better_candidate_has_positive_difference(self):
        base = self._two_fold(lambda t, v: t if int(v[1:]) % 2 else (t + 1) % 14)   # half the videos wrong
        cand = self._two_fold(lambda t, v: t)                                        # all right
        r = fa.paired_cluster_bootstrap(base, cand, draws=200)
        self.assertGreater(r["difference"]["ci95"][0], 0.0)
        self.assertEqual(r["difference"]["share_of_draws_above_zero"], 1.0)

    def test_mismatched_rows_are_refused(self):
        a = self._two_fold(lambda t, v: t)
        b = fa.Predictions(a.classes, a.fold, a.seed, a.video[::-1].copy(), a.truth, a.probs)
        with self.assertRaises(ValueError):
            fa.paired_cluster_bootstrap(a, b, draws=5)


if __name__ == "__main__":
    unittest.main()
