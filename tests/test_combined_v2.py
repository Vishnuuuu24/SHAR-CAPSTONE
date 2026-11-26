import csv
import json
import tempfile
import unittest
from pathlib import Path

from core import combined_dataset as cd
from core import combined_split as cs
from test_combined_split import fake_rows, row


def v1_assign(rows):
    a = cs.assign_splits(rows)
    return {r["relative_link_path"]: (r["split"], r["fold"]) for r in a
            if r["source_dataset"] == cd.UCF and r["source_label"] in ("Fighting", "Abuse")}


class StabilityTests(unittest.TestCase):
    def test_old_assignments_survive_new_classes_and_surplus(self):
        base = fake_rows()
        old = v1_assign(base)
        # new UCF classes (own RNG per source,class) must not move Fighting/Abuse
        extra = [row(cd.UCF, "Arson", f"R{i:03d}", test=i < 8) for i in range(50)]
        extra += [row(cd.CUVA, "Fighting", f"{i:05d}", uni="Fighting") for i in range(40)]
        new = v1_assign(base + extra)
        # Surplus official-Test rows (Abuse: 30 official, 20 sealed) become trainable in
        # their own folds, but sealed-test rows and previously trainable folds are unchanged.
        for k, (split, fold) in old.items():
            if split == "test" or fold != "":
                self.assertEqual(new[k], (split, fold), k)
        a = cs.assign_splits(base + extra)
        left = [r for r in a if r["source_label"] == "Abuse" and r["split"] == "train_pool" and r["is_ucf_test"] == "true"]
        self.assertEqual(len(left), 10)
        self.assertTrue(all(r["trainable"] == "true" and r["fold"] != "" for r in left))


class AppendedTierTests(unittest.TestCase):
    def test_appended_rows_do_not_move_earlier_folds(self):
        rows = [row(cd.UCF, "Normal", f"N{i:03d}", test=i < 30, uni="Normal") for i in range(60)]
        base = {r["relative_link_path"]: (r["split"], r["fold"]) for r in cs.assign_splits(rows)}
        extra = [row(cd.UCF, "Normal", f"T{i:03d}", uni="Normal") for i in range(80)]
        for r in extra:
            r["notes"] = cd.APPENDED_V2
        new = cs.assign_splits(rows + extra)
        for r in new:
            if r["relative_link_path"] in base:
                self.assertEqual((r["split"], r["fold"]), base[r["relative_link_path"]])
        app = [r for r in new if cd.APPENDED_V2 in r["notes"]]
        self.assertTrue(all(r["split"] == "train_pool" and r["fold"] != "" for r in app))
        self.assertEqual(len({r["fold"] for r in app}), 4)


class TwinTests(unittest.TestCase):
    def test_twins_share_split_and_guard_raises(self):
        rows = [row(cd.UCF, "Normal", f"N{i:03d}", test=True, uni="Normal") for i in range(60)]
        twins = []
        for r in rows[:50]:  # same stem+size, different dir => same group_id
            t = dict(r, relative_link_path=r["relative_link_path"].replace("/Normal/", "/Normal2/"))
            twins.append(t)
        a = cs.assign_splits(rows + twins)
        by = {}
        for r in a:
            by.setdefault(r["group_id"], set()).add((r["split"], r["fold"]))
        self.assertTrue(all(len(v) == 1 for v in by.values()))
        bad = [dict(r) for r in a[:1]] + [dict(a[60], split="test", fold="")]
        bad[0]["split"] = "train_pool"
        bad[1]["relative_link_path"] = a[0]["relative_link_path"].replace("/Normal/", "/Z/")
        with self.assertRaises(AssertionError):
            cs.assert_no_twin_straddle(bad)


class CuvaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        recs = []
        cats = {1: 20, 4: 20, 5: 20, 6: 20, 7: 6, 9: 4}
        n = 0
        for c, k in cats.items():
            for _ in range(k):
                n += 1
                name = f"{n:05d}.mp4"
                d = self.root / "CUVA" / "raw" / f"group_{n % 2}"
                d.mkdir(parents=True, exist_ok=True)
                (d / name).write_bytes(b"x")
                recs.append({"task": "Classification", "visual_input": name, "output": f"[{c}-1,5]"})
        (self.root / "CUVA" / "raw" / "group_0" / "99999.mp4").write_bytes(b"x")
        recs.append({"task": "Classification", "visual_input": "99999.mp4", "output": "枪战"})
        for i, t in enumerate(["枪战", "枪击", "逮捕", "校园霸凌"] * 3):
            nm = f"8{i:04d}.mp4"
            (self.root / "CUVA" / "raw" / "group_1" / nm).write_bytes(b"x")
            recs.append({"task": "Classification", "visual_input": nm, "output": t})
        recs.append({"task": "Other", "visual_input": "00001.mp4", "output": "ignored"})
        a = self.root / "CUVA" / "annotations"
        a.mkdir(parents=True)
        (a / cd.CUVA_ANNOTATION).write_text(json.dumps(recs))

    def tearDown(self):
        self.tmp.cleanup()

    def test_mapping_and_nontrainable_unmapped(self):
        self.assertEqual(cd.parse_cuva_label("[7-1,5]"), "Theft")
        self.assertEqual(cd.parse_cuva_label("junk"), cd.CUVA_UNPARSED)
        out = self.root / "SHAR-Combined"
        cd.build(self.root, out, {cd.CUVA: cd.DEFAULT_CLASSES[cd.CUVA]})
        with (out / "manifest.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        by = {r["source_label"]: r for r in rows}
        self.assertEqual(by["Traffic Accidents"]["unified_label"], "RoadAccidents")
        for lab in ("Theft", "Fire", cd.CUVA_UNPARSED):
            self.assertEqual((by[lab]["unified_label"], by[lab]["trainable"]), ("", "false"))
        self.assertIn("no UCF counterpart", by["Theft"]["notes"])
        cs.split_manifest(out)
        with (out / "manifest.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            if r["unified_label"] == "":
                self.assertEqual((r["split"], r["trainable"], r["fold"]), ("unlabeled_pool", "false", ""))
        self.assertEqual(sum(r["split"] == "test" for r in rows if r["source_label"] == "Fighting"), 3)
        self.assertEqual(len({r["group_id"] for r in rows}), len(rows))
        for f in (1, 2, 3, 4):
            tr, va = cs.load_trainable(out / "manifest.csv", f)
            self.assertTrue(all(r["split"] == "train_pool" and r["unified_label"] != "" for r in tr + va))
            self.assertFalse({r["source_label"] for r in tr + va} & {"Theft", "Fire", cd.CUVA_UNPARSED})
        by = {}
        for r in rows:
            by.setdefault(r["source_label"], []).append(r)
        self.assertEqual({r["unified_label"] for r in by["FreeText:Shooting"]}, {"Shooting"})
        self.assertEqual({r["unified_label"] for r in by["FreeText:Arrest"]}, {"Arrest"})
        self.assertIn("free-text annotator label, translated", by["FreeText:Arrest"][0]["notes"])
        self.assertTrue(all(r["unified_label"] == "" and r["trainable"] == "false"
                            for r in by[cd.CUVA_UNPARSED] if "校园霸凌" in r["notes"]))
        self.assertIn("(unmapped)", json.loads((out / "split_summary.json").read_text())["table_source_x_unified_label"][cd.CUVA])


if __name__ == "__main__":
    unittest.main()
