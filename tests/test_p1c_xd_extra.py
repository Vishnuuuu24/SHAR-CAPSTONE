import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.p1c_xd_extra import class_for_video, file_list_sha256, label_codes, pick_frames, stable_seed
from core.p1c_xd_probe import aggregate, extra_sequences, gate_check, leakage_check, run_metrics, select_extra_videos

NAMES = ["Abuse", "Fighting", "Normal_Videos_event", "Shooting"]


class LabelAndSamplingTests(unittest.TestCase):
    def test_label_mapping(self):
        self.assertEqual(label_codes("X_label_B2-0-0.mp4"), ["B2"])
        self.assertEqual(class_for_video("Shooting", "X_label_B2-0-0.mp4"), ("Shooting", "B2"))
        self.assertEqual(class_for_video("Abuse", "x_label_B5-0-0.mp4"), ("Abuse", "B5"))
        self.assertEqual(class_for_video("Fighting", "x_label_B1-B2-0.mp4")[1], "B1")
        with self.assertRaises(ValueError):
            class_for_video("Shooting", "X_label_B1-0-0.mp4")  # folder disagrees with file code
        with self.assertRaises(ValueError):
            class_for_video("Riot", "X_label_B1-0-0.mp4")
        with self.assertRaises(ValueError):
            label_codes("no_label.mp4")

    def test_stable_seed_is_hash_based_and_fixed(self):
        self.assertEqual(stable_seed("a.mp4"), int.from_bytes(hashlib.sha256(b"0:a.mp4").digest()[:4], "big"))
        self.assertNotEqual(stable_seed("a.mp4"), stable_seed("b.mp4"))
        self.assertNotEqual(stable_seed("a.mp4", 0), stable_seed("a.mp4", 1))

    def test_pick_frames_deterministic_distinct_on_stride(self):
        a, b = pick_frames(500, 8, 7), pick_frames(500, 8, 7)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 8)
        self.assertTrue(all(i % 5 == 0 and i < 500 for i in a))
        self.assertEqual(a, sorted(a))
        self.assertNotEqual(a, pick_frames(500, 8, 8))

    def test_pick_frames_short_video_cycles(self):
        self.assertEqual(pick_frames(15, 8, 3), [0, 5, 10, 0, 5, 10, 0, 5])
        with self.assertRaises(ValueError):
            pick_frames(0, 8, 0)

    def test_file_list_hash_depends_on_names_and_sizes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "Abuse").mkdir()
            p = root / "Abuse" / "a_label_B5-0-0.mp4"
            p.write_bytes(b"123")
            h1 = file_list_sha256([(p, "Abuse")], root)
            p.write_bytes(b"1234")
            self.assertNotEqual(h1, file_list_sha256([(p, "Abuse")], root))


class ProbeHelperTests(unittest.TestCase):
    def test_selection_is_prefix_superset_and_deterministic(self):
        vids = {"Shooting": [f"s{i}" for i in range(200)], "Fighting": [f"f{i}" for i in range(150)], "Abuse": [f"a{i}" for i in range(31)]}
        k34, k102 = select_extra_videos(vids, "xd_k34"), select_extra_videos(vids, "xd_k102")
        self.assertEqual(len(k34), 34 + 34 + 31)
        self.assertEqual(len(k102), 102 + 102 + 31)
        self.assertTrue(set(k34) <= set(k102))
        self.assertEqual(k34, select_extra_videos(vids, "xd_k34"))
        self.assertEqual(select_extra_videos(vids, "baseline"), [])
        self.assertEqual(len(select_extra_videos({"Abuse": ["a1", "a2"]}, "xd_k34")), 2)

    def test_extra_sequences_shape_prefix_and_label(self):
        rows = [{"video_id": "v1", "class_name": "Shooting", "source_frame_index": 5 * i} for i in range(8)]
        rows += [{"video_id": "v2", "class_name": "Abuse", "source_frame_index": 5 * i} for i in range(8)]
        feats = np.random.default_rng(0).normal(size=(16, 1536)).astype(np.float16)
        seqs = extra_sequences(feats, rows, ["v1"], NAMES, 0)
        ((ds, vid), (seq, row)), = seqs.items()
        self.assertEqual(vid, "xd:v1")
        self.assertEqual(seq.shape, (8, 1536))
        self.assertEqual(row["target"], NAMES.index("Shooting"))

    def test_gate(self):
        base = {1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4}
        ok = gate_check(base, {1: 0.45, 2: 0.441, 3: 0.5, 4: 0.46})
        self.assertTrue(ok["passed"])
        bad = gate_check(base, {1: 0.45, 2: 0.43, 3: 0.5, 4: 0.46})
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["folds_passing"], 3)
        self.assertFalse(gate_check({1: 0.4}, {1: 0.9})["passed"])  # needs all 4 folds

    def test_aggregate_and_metrics(self):
        truth = [0, 1, 2, 3, 3]
        perfect = run_metrics(truth, truth, NAMES)
        worse = run_metrics(truth, [0, 1, 2, 2, 2], NAMES)
        self.assertEqual(perfect["macro_f1"], 1.0)
        self.assertEqual(worse["per_class"]["Shooting"]["recall"], 0.0)
        runs = [{"arm": "baseline", "fold": f, "seed": s, "metrics": worse} for f in (1, 2) for s in (0, 1)]
        runs += [{"arm": "xd_k34", "fold": f, "seed": s, "metrics": perfect} for f in (1, 2) for s in (0, 1)]
        agg = aggregate(runs, NAMES)
        self.assertEqual(agg["xd_k34"]["macro_f1_over_runs"]["n"], 4)
        self.assertAlmostEqual(agg["xd_k34"]["per_class_f1_delta_vs_baseline"]["Shooting"], 1.0)
        self.assertEqual(agg["xd_k34"]["gate"]["folds_compared"], 2)
        self.assertEqual(np.asarray(agg["baseline"]["confusion_summed_over_folds_and_seeds"]).sum(), 20)

    def test_leakage_counts(self):
        val = np.eye(4, 8, dtype=np.float32) + 5.0
        xd = np.stack([val[1] * 2, np.r_[np.ones(4), -np.ones(4)]])
        out = leakage_check(xd, ["x1", "x2"], val, NAMES, ["v0", "v1", "v2", "v3"])
        self.assertEqual(out["raw"]["count_above_0.98"], 1)
        self.assertEqual(out["raw"]["top5"][0]["nearest_val_video"], "v1")
        self.assertEqual(out["raw"]["nearest_val_class_counts"]["Fighting"], 1)


if __name__ == "__main__":
    unittest.main()
