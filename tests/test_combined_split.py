import csv
import tempfile
import unittest
from pathlib import Path

from core import combined_dataset as cd
from core import combined_split as cs


def row(src, label, vid, test=False, uni=None):
    return {"source_dataset": src, "source_label": label, "unified_label": uni or label,
            "source_video_id": vid, "relative_link_path": f"{src}/{label}/{vid}.mp4",
            "original_abs_path": f"/x/{vid}.mp4", "size_bytes": "3", "split_in_source": "Test" if test else "Train",
            "is_ucf_test": str(test).lower(), "trainable": str(not test).lower(), "notes": ""}


def fake_rows():
    rows = [row(cd.UCF, "Fighting", f"F{i:03d}", test=i < 5) for i in range(50)]
    rows += [row(cd.UCF, "Abuse", f"A{i:03d}", test=i < 30) for i in range(50)]
    for m in range(20):  # 20 movies x 3 clips; movie 0 also shoots
        for c in range(3):
            rows.append(row(cd.XD, "Fighting", f"Fighting_Movie{m}___00-0{c}-00_label_B1", uni="Fighting"))
    rows.append(row(cd.XD, "Shooting", "Shooting_Movie0___01-00-00_label_B2", uni="Shooting"))
    rows.append(row(cd.XD, "Shooting", "Shooting_Other___01-00-00_label_B2", uni="Shooting"))
    return rows


class SplitTests(unittest.TestCase):
    def test_group_id(self):
        r = row(cd.XD, "Fighting", "Fighting_Bad.Boys.1995___01-07-01_01-08-13_label_B1-0-0")
        self.assertEqual(cs.group_id(r), "xd_violence:Bad.Boys.1995")
        self.assertEqual(cs.group_id(row(cd.UCF, "Abuse", "Abuse001")), "ucf_crime_official:Abuse001")

    def test_deterministic_and_targets(self):
        a, b = cs.assign_splits(fake_rows()), cs.assign_splits(fake_rows())
        self.assertEqual(a, b)
        t = lambda lab, src: sum(r["split"] == "test" for r in a
                                 if r["source_label"] == lab and r["source_dataset"] == src)
        self.assertEqual(t("Fighting", cd.UCF), 20)
        self.assertEqual(t("Abuse", cd.UCF), 20)
        # official test (5) all included for Fighting
        self.assertTrue(all(r["split"] == "test" for r in a if r["is_ucf_test"] == "true" and r["source_label"] == "Fighting"))
        # Abuse has 30 official test: sampled 20, remainder becomes trainable surplus (own benchmark)
        left = [r for r in a if r["source_label"] == "Abuse" and r["split"] == "train_pool" and r["is_ucf_test"] == "true"]
        self.assertEqual(len(left), 10)
        self.assertTrue(all(r["trainable"] == "true" and r["fold"] != "" for r in left))
        self.assertNotEqual(cs.assign_splits(fake_rows(), seed=1), a)

    def test_groups_and_folds(self):
        a = cs.assign_splits(fake_rows())
        cs.assert_groups_unsplit(a)
        xd = [r for r in a if r["source_dataset"] == cd.XD]
        by = {}
        for r in xd:
            by.setdefault(r["group_id"], set()).add(r["split"])
        self.assertTrue(all(len(s) == 1 for s in by.values()))
        self.assertGreaterEqual(sum(r["split"] == "test" for r in xd), 3)
        folds = {r["fold"] for r in a if r["split"] == "train_pool" and r["trainable"] == "true"}
        self.assertEqual(folds, {"1", "2", "3", "4"})
        self.assertTrue(all(r["fold"] == "" for r in a if r["split"] == "test"))
        self.assertTrue(all(r["trainable"] == "false" for r in a if r["split"] == "test"))
        bad = [dict(r) for r in a if r["group_id"].endswith("Movie1")]
        bad[0]["split"] = "test"
        with self.assertRaises(AssertionError):
            cs.assert_groups_unsplit(bad)

    def test_loader_never_returns_test(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            with (out / "manifest.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cd.FIELDS)
                w.writeheader()
                w.writerows(fake_rows())
            s = cs.split_manifest(out)
            self.assertEqual(s["test_links"], len([p for p in (out / "test").rglob("*") if p.is_symlink()]))
            m = out / "manifest.csv"
            for f in (1, 2, 3, 4):
                tr, va = cs.load_trainable(m, f)
                self.assertTrue(tr and va)
                self.assertFalse({r["split"] for r in tr + va} - {"train_pool"})
                self.assertFalse({r["relative_link_path"] for r in tr} & {r["relative_link_path"] for r in va})
            with self.assertRaises(ValueError):
                cs.load_trainable(m, 5)
            self.assertEqual(cs.split_manifest(out)["manifest_sha256"], s["manifest_sha256"])
        with self.assertRaises(AssertionError):
            cs.assert_test_sealed([{"split": "test", "relative_link_path": "x"}])
        with self.assertRaises(AssertionError):
            cs.assert_test_sealed([{"split": "train_pool", "test_sealed": "true", "relative_link_path": "x"}])


if __name__ == "__main__":
    unittest.main()
