import json
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
E3 = REPO / "results/demo/e3-comparison-20261116"
HAVE_E3 = (E3 / "export.json").exists() and (E3 / "answers/ucf/Robbery044_x264.json").exists()


@unittest.skipUnless(HAVE_E3, "E3 replay artifacts are local-only")
class E3MemberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.e3_inference import e3_support, load_members

        cls.support = e3_support(REPO)
        cls.members = load_members(REPO)
        cls.classes = json.loads((E3 / "class_names.json").read_text())

    def saved(self, key):
        a = json.loads((E3 / "answers/ucf" / f"{key}.json").read_text())
        with np.load(E3 / "vectors" / f"{a['source_sha256']}.npz") as d:
            return a, {k: d[k] for k in ("B", "L", "S", "VJ")}

    def test_batch_matches_frozen_predict_and_saved_answers(self):
        from core.e3_inference import predict_batch

        keys = ["Robbery044_x264", "Explosion021_x264", "Abuse001_x264"]
        answers, vecs = zip(*(self.saved(k) for k in keys))
        for a, v in zip(answers, vecs):
            p = self.support.predict(self.members, v, a["selected_folds"], len(self.classes))
            np.testing.assert_allclose(p, [a["probs"][c] for c in self.classes], atol=1e-12)
            batch = {k: np.stack([v[k], v[k]]) for k in v}
            pb = predict_batch(self.members, self.support.CONFIG["members"], batch, a["selected_folds"], len(self.classes))
            # Batched float32 matrix products round differently from one row (measured 1.2e-7); windows only.
            np.testing.assert_allclose(pb[0], p, atol=1e-6)
            np.testing.assert_allclose(pb[1], p, atol=1e-6)

    def test_membership_uses_content_not_name(self):
        from core.e3_inference import E3DemoModel

        m = E3DemoModel.__new__(E3DemoModel)  # membership() needs no encoders
        m.support = self.support
        m.seen = self.support.content_membership(json.loads((E3 / "training_membership.json").read_text()))
        m.ucf_labels = {}
        for r in self.support.CONFIG["ucf"]:
            m.ucf_labels.setdefault(r["sha256"], set()).add(r["label"])
        a, _ = self.saved("Robbery044_x264")
        folds, mode, exp = m.membership(a["source_sha256"])
        self.assertEqual(folds, a["selected_folds"])
        self.assertEqual(exp["label"], "Robbery")
        self.assertTrue(mode.startswith("held-out"))
        folds, mode, exp = m.membership("0" * 64)
        self.assertEqual(folds, [1, 2, 3, 4])
        self.assertIsNone(exp)


if __name__ == "__main__":
    unittest.main()
