import csv
import tempfile
import unittest
from pathlib import Path

from core import combined_dataset as cd


def _touch(p: Path, n: int = 3) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * n)


def make_fake(root: Path) -> None:
    r = root / "UCF Crime Official" / "raw" / "UCF_Crimes"
    (r / "Anomaly_Detection_splits").mkdir(parents=True)
    train = [f"Abuse/Abuse00{i}_x264.mp4" for i in (1, 2)] + ["Shooting/Shooting001_x264.mp4"]
    test = ["Abuse/Abuse003_x264.mp4", "Testing_Normal_Videos_Anomaly/Normal_Videos_003_x264.mp4"]
    (r / "Anomaly_Detection_splits" / "Anomaly_Train.txt").write_text("\n".join(train) + "\n")
    (r / "Anomaly_Detection_splits" / "Anomaly_Test.txt").write_text("\n".join(test) + "\n")
    for rel in train + test:
        _touch(r / "Videos" / rel)
    _touch(r / "Videos" / "z_Normal_Videos_event" / "Normal_Videos_003_x264.mp4")
    for c in ("Shooting", "Fighting"):
        _touch(root / "XD-Violence" / "train" / c / f"{c}_a.mp4")


def sel():
    return {cd.UCF: ["Abuse", "Shooting", "Normal_Videos_event"], cd.XD: ["Shooting"], cd.RWF: ["Fight"]}


def rows_of(out: Path):
    with (out / "manifest.csv").open() as fh:
        return list(csv.DictReader(fh))


class CombinedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "Datasets"
        make_fake(self.root)
        self.out = self.root / "SHAR-Combined"

    def tearDown(self):
        self.tmp.cleanup()

    def test_unmapped_class_fails(self):
        with self.assertRaises(cd.UnmappedClassError):
            cd.build(self.root, self.out, {cd.UCF: ["Littering"]})
        self.assertFalse(self.out.exists())

    def test_test_flag_and_not_trainable(self):
        cd.build(self.root, self.out, sel())
        rows = rows_of(self.out)
        cd.assert_no_ucf_test_trainable(rows)
        test = [r for r in rows if r["source_dataset"] == cd.UCF and r["split_in_source"] == "Test"]
        self.assertEqual(len(test), 2)  # Abuse003 + K-11 Normal duplicate
        self.assertTrue(all(r["is_ucf_test"] == "true" and r["trainable"] == "false" for r in test))
        train = [r for r in rows if r["split_in_source"] == "Train"]
        self.assertTrue(all(r["trainable"] == "true" for r in train))
        bad = dict(test[0], trainable="true")
        with self.assertRaises(AssertionError):
            cd.assert_no_ucf_test_trainable([bad])

    def test_idempotent_and_refuses_foreign_files(self):
        cd.build(self.root, self.out, sel())
        first = (self.out / "manifest.csv").read_bytes()
        cd.build(self.root, self.out, sel())
        self.assertEqual(first, (self.out / "manifest.csv").read_bytes())
        links = [p for p in self.out.rglob("*") if p.is_symlink()]
        self.assertEqual(len(links), 6)
        self.assertTrue(all(p.exists() for p in links))
        (self.out / "stray.txt").write_text("hi")
        with self.assertRaises(RuntimeError):
            cd.build(self.root, self.out, sel())

    def test_rwf_noop_and_present(self):
        s = cd.build(self.root, self.out, sel())
        self.assertNotIn(cd.RWF, s["counts_source_x_class"])
        _touch(self.root / "RWF-2000" / "train" / "Fight" / "f1.mp4")
        s = cd.build(self.root, self.out, sel())
        self.assertEqual(s["counts_source_x_class"][cd.RWF], {"Fight": 1})
        row = [r for r in rows_of(self.out) if r["source_dataset"] == cd.RWF][0]
        self.assertEqual((row["source_label"], row["unified_label"]), ("Fight", "Fighting"))


if __name__ == "__main__":
    unittest.main()
