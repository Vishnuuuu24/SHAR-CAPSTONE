import unittest

from core.xd_dataset import XD_CLASSES, assign_validation, group_of, labels_of


class XdDatasetTest(unittest.TestCase):
    def test_labels_and_group(self):
        stem = "Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6"
        self.assertEqual(labels_of(stem), ["Explosion", "Shooting", "Car accident"])
        self.assertEqual(group_of(stem), "Bad.Boys.1995")
        self.assertEqual(group_of("v=abc__#1_label_A"), "v=abc")
        with self.assertRaises(ValueError):
            labels_of("x__#1_label_B3-0-0")

    def test_validation_is_grouped_and_covers_every_class(self):
        rows = []
        for i, cls in enumerate(XD_CLASSES * 10):
            rows.append({"split": "train", "group_id": f"g{i % 23}", "unified_label": cls})
        chosen = assign_validation(rows, 0.1, 0)
        val = [r for r in rows if r["group_id"] in chosen]
        for cls in XD_CLASSES:
            self.assertTrue(any(r["unified_label"] == cls for r in val))
        self.assertTrue(all(r["group_id"] in chosen for r in rows if r["group_id"] in {v["group_id"] for v in val}))


if __name__ == "__main__":
    unittest.main()
