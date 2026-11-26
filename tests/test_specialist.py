import unittest

import numpy as np

from core.confusion_groups import macro_f1, oracle_predictions
from core.specialist import GroupMap, fuse_probs, oracle_substitute, remap_sequences, route_hard, split_holdout

CLASSES = ["A", "B", "C", "D", "N"]


class SpecialistTests(unittest.TestCase):
    def setUp(self):
        self.gm = GroupMap(CLASSES, ["C", "A", "B"])  # local order differs from global order on purpose

    def test_mapping_round_trip(self):
        self.assertEqual(self.gm.idx, [2, 0, 1])
        self.assertEqual(self.gm.to_global([0, 1, 2]).tolist(), [2, 0, 1])
        self.assertEqual(self.gm.local_of(0), 1)
        self.assertEqual(self.gm.in_group([0, 3, 4, 2]).tolist(), [True, False, False, True])
        with self.assertRaises(ValueError):
            GroupMap(CLASSES, ["A", "Z"])

    def test_hard_routing_only_changes_group_predictions(self):
        base = np.array([[.7, .1, .1, .05, .05],   # in group -> specialist decides
                         [.0, .0, .0, .9, .1],     # D -> untouched
                         [.0, .0, .0, .1, .9]])    # N -> untouched
        spec = np.array([[.1, .1, .8], [.8, .1, .1], [.8, .1, .1]])  # local [C, A, B]
        self.assertEqual(route_hard(base, spec, self.gm).tolist(), [1, 3, 4])

    def test_oracle_uses_truth_not_prediction(self):
        base = np.array([[.1, .1, .1, .6, .1]])  # baseline says D, truth is B
        spec = np.array([[.1, .1, .8]])           # specialist says local 2 = B
        self.assertEqual(oracle_substitute(base, spec, np.array([1]), self.gm).tolist(), [1])
        self.assertEqual(oracle_substitute(base, spec, np.array([3]), self.gm).tolist(), [3])

    def test_fusion_conserves_group_mass_and_normalisation(self):
        rng = np.random.default_rng(0)
        base = rng.dirichlet(np.ones(5), size=20)
        spec = rng.dirichlet(np.ones(3), size=20)
        for w in (0.0, 0.5, 1.0):
            out = fuse_probs(base, spec, self.gm, w)
            np.testing.assert_allclose(out.sum(1), 1.0)
            np.testing.assert_allclose(out[:, self.gm.idx].sum(1), base[:, self.gm.idx].sum(1))
            np.testing.assert_allclose(out[:, [3, 4]], base[:, [3, 4]])
        np.testing.assert_allclose(fuse_probs(base, spec, self.gm, 0.0), base)

    def test_fusion_w1_follows_specialist_inside_group(self):
        base = np.array([[.3, .1, .1, .4, .1]])
        spec = np.array([[0.0, 0.0, 1.0]])  # local B
        out = fuse_probs(base, spec, self.gm, 1.0)
        self.assertEqual(int(out.argmax()), 1)  # group mass 0.5 all moves to B, which beats untouched D at 0.4
        self.assertAlmostEqual(out[0, 3], 0.4)
        self.assertAlmostEqual(out[0, 1], 0.5)

    def test_holdout_is_video_disjoint_and_seeded(self):
        ids = {"A": [f"a{i}" for i in range(30)], "B": [f"b{i}" for i in range(12)]}
        fit, hold = split_holdout(ids, 0.25, lambda c: ord(c))
        self.assertFalse(fit & hold)
        self.assertEqual(len(fit | hold), 42)
        self.assertEqual(sum(v.startswith("a") for v in hold), 8)
        self.assertEqual((fit, hold), split_holdout(ids, 0.25, lambda c: ord(c)))

    def test_remap_sequences(self):
        seqs = {("d", "v1"): (np.zeros((1, 2)), {"label": "A", "target": 0}), ("d", "v2"): (np.zeros((1, 2)), {"label": "N", "target": 4}),
                ("d", "v3"): (np.zeros((1, 2)), {"label": "C", "target": 2})}
        out = remap_sequences(seqs, self.gm)
        self.assertEqual({k[1]: v[1]["target"] for k, v in out.items()}, {"v1": 1, "v3": 0})
        self.assertEqual(set(k[1] for k in remap_sequences(seqs, self.gm, keep={"v1"})), {"v1"})

    def test_macro_f1_matches_eval_metrics_and_oracle_modes(self):
        from eval.metrics import classification_metrics
        t, p = np.array([0, 1, 2, 3, 4, 4]), np.array([1, 1, 0, 3, 4, 2])
        self.assertAlmostEqual(macro_f1(t, p, 5), classification_metrics([CLASSES[i] for i in t], [CLASSES[i] for i in p], CLASSES)["macro_f1"])
        g = [0, 1, 2]
        self.assertEqual(oracle_predictions(t, p, g, "oracle_route").tolist(), [0, 1, 2, 3, 4, 2])
        self.assertEqual(oracle_predictions(t, p, g, "hard_route").tolist(), [0, 1, 2, 3, 4, 2])


if __name__ == "__main__":
    unittest.main()
