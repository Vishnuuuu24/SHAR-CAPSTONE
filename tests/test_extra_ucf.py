import unittest

import numpy as np

from core.extra_ucf_videos import balanced_extra_subset, blob_rows, protocol_video_ids, select_extra_videos
from core.head_experiment import ARMS, GATED, extra_sequences


def _csv_rows():
    rows = []
    for cls, n in (("Burglary", 6), ("Robbery", 5), ("Abuse", 3)):
        for i in range(1, n + 1):
            rows.append({"file_name": f"{cls}{i:03d}_x264.mp4", "source_directory": cls, "decode_status": "ok",
                         "first_row_index": str(i * 100), "kept_frame_count": "40"})
    return rows


class Selection(unittest.TestCase):
    def test_extra_videos_exclude_the_protocol_and_other_classes(self):
        protocol = {"Burglary001_x264", "Burglary002_x264", "Robbery001_x264", "Abuse001_x264"}
        extra = select_extra_videos(_csv_rows(), protocol, classes=("Burglary", "Robbery"))
        names = [v["video"] for v in extra]
        self.assertEqual(names, sorted(names))
        self.assertFalse(set(names) & protocol)
        self.assertEqual({v["label"] for v in extra}, {"Burglary", "Robbery"})
        self.assertEqual(len(extra), 4 + 4)

    def test_bad_decode_status_is_refused(self):
        rows = _csv_rows()
        rows[0]["decode_status"] = "failed"
        with self.assertRaises(ValueError):
            select_extra_videos(rows, set(), classes=("Burglary",))

    def test_blob_rows_follow_the_frame_index(self):
        fi = np.zeros(400, dtype=[("byte_offset", "<i8"), ("byte_length", "<i8"), ("video_id", "<i8"), ("source_frame_index", "<i8")])
        fi["source_frame_index"][100:140] = np.arange(40) * 5
        out = blob_rows({"video": "Burglary001_x264", "label": "Burglary", "first_row": 100, "frames": 40}, fi)
        self.assertEqual(len(out), 40)
        self.assertEqual(out[0]["blob_row"], 100)
        self.assertEqual(out[-1]["blob_row"], 139)
        self.assertEqual([r["source_frame_index"] for r in out[:3]], [0, 5, 10])

    def test_balanced_subset_is_seeded_and_capped(self):
        labels = {f"b{i}": "Burglary" for i in range(10)} | {f"r{i}": "Robbery" for i in range(3)}
        keep = balanced_extra_subset(labels, 4, seed=1)
        self.assertEqual(sum(1 for v in keep if labels[v] == "Burglary"), 4)
        self.assertEqual(sum(1 for v in keep if labels[v] == "Robbery"), 3)    # fewer than asked: all of them
        self.assertEqual(keep, balanced_extra_subset(labels, 4, seed=1))
        self.assertNotEqual(keep, balanced_extra_subset(labels, 4, seed=2))


class Sequences(unittest.TestCase):
    def test_extra_sequences_have_the_trainers_shape(self):
        rng = np.random.default_rng(0)
        rows, feats = [], []
        for c, cls in enumerate(("Burglary", "Robbery")):
            for v in range(5):
                for f in range(8):
                    rows.append({"source_dataset": "ucf", "source_video_id": f"{cls}{v}", "source_frame_index": f * 5,
                                 "label": cls, "target": c})
                    feats.append(rng.normal(size=12))
        seqs = extra_sequences(np.array(feats, dtype=np.float32), rows, per_class=3, seed=0)
        self.assertEqual(len(seqs), 6)
        for (dataset, video), (seq, row) in seqs.items():
            self.assertEqual(dataset, "ucf")
            self.assertEqual(seq.shape, (8, 12))
            self.assertEqual(row["label"], video.rstrip("01234"))


class Registration(unittest.TestCase):
    def test_extra_arms_are_registered_and_gated(self):
        for arm in ("extra34", "extra_all", "extra_all_w"):
            self.assertIn(arm, ARMS)
            self.assertIn(arm, GATED)
        self.assertTrue(ARMS["extra_all_w"]["weighted"])
        self.assertNotIn("weighted", ARMS["extra_all"])

    @unittest.skipUnless(__import__("pathlib").Path("results/p1b/p1b-video-ar-fold1/train_manifest.csv").exists(), "manifests absent")
    def test_protocol_has_700_videos(self):
        from pathlib import Path
        self.assertEqual(len(protocol_video_ids(Path("."))), 700)


if __name__ == "__main__":
    unittest.main()
