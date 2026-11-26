import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from core import combined_probe as cp
from core.combined_dataset import CUVA, UCF, XD
from core.combined_split import SPLIT_FIELDS

DIM = 768


def mrow(src, label, vid, split="train_pool", fold="1", trainable="true", unified=None):
    unified = label if unified is None else unified
    return {**{k: "" for k in SPLIT_FIELDS}, "source_dataset": src, "source_label": label, "unified_label": unified,
            "source_video_id": vid, "relative_link_path": f"{src}/{label}/{vid}.mp4", "original_abs_path": f"/x/{vid}.mp4",
            "size_bytes": "1", "trainable": trainable, "split": split, "fold": fold,
            "test_sealed": str(split == "test").lower(), "group_id": f"{src}:{vid}", "is_ucf_test": "false"}


def fake_manifest_rows():
    rows = []
    for ci, c in enumerate(cp.CLASS_NAMES):
        for i in range(8):
            rows.append(mrow(UCF, c, f"u_{c}_{i}", fold=str(i % 4 + 1)))
        rows.append(mrow(UCF, c, f"u_{c}_T", split="test", fold="", trainable="false"))
    for c in ("Shooting", "Fighting", "Abuse"):
        for i in range(8):
            rows.append(mrow(XD, c, f"x_{c}_{i}", fold=str(i % 4 + 1)))
        rows.append(mrow(XD, c, f"x_{c}_T", split="test", fold="", trainable="false"))
    for c in ("Robbery", "Fighting"):
        for i in range(8):
            rows.append(mrow(CUVA, c, f"c_{c}_{i}", fold=str(i % 4 + 1)))
    rows.append(mrow(CUVA, "Unparsed", "c_unlabeled", split="unlabeled_pool", fold="", trainable="false", unified=""))
    rows.append(mrow(UCF, "Abuse", "u_untrainable", fold="1", trainable="false"))  # train_pool but untrainable
    return rows


def write_fixture(d: Path, fail=()):
    rows = fake_manifest_rows()
    with (d / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SPLIT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    cache = d / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(len(cp.CLASS_NAMES), DIM)).astype(np.float32)
    feats = np.zeros((len(rows), 8, 3, DIM), np.float16)
    with (cache / "rows.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "source_dataset", "source_video_id", "unified_label", "split", "fold", "ok"])
        for i, r in enumerate(rows):
            ok = r["source_video_id"] not in fail
            if r["unified_label"] and ok:
                feats[i] = centers[cp.CLASS_NAMES.index(r["unified_label"])] + rng.normal(scale=0.5, size=(8, 3, DIM))
            w.writerow([i, r["source_dataset"], r["source_video_id"], r["unified_label"], r["split"], r["fold"], int(ok)])
    np.save(cache / "features.float16.npy", feats)
    from core.p1c_readiness import sha256_file

    (cache / "manifest.json").write_text(json.dumps({"combined_manifest_sha256": sha256_file(d / "manifest.csv"),
                                                     "features_sha256": "deadbeef"}))
    return d / "manifest.csv", cache, rows


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manifest, self.cache, self.rows = write_fixture(Path(self.tmp.name))
        _f, self.index, _m = cp.load_cache(self.cache, self.manifest)

    def tearDown(self):
        self.tmp.cleanup()

    def test_never_test_untrainable_or_unlabeled(self):
        for arm in cp.ARMS:
            for fold in (1, 2, 3, 4):
                tr, va, _ = cp.fold_rows(self.manifest, fold, arm, self.index)
                for r in tr + va:
                    self.assertEqual(r["split"], "train_pool")
                    self.assertEqual(r["trainable"], "true")
                    self.assertNotEqual(r["unified_label"], "")
                    self.assertNotEqual(r["source_video_id"], "u_untrainable")
                    self.assertFalse(r["source_video_id"].endswith("_T"))
                self.assertTrue(all(int(r["fold"]) != fold for r in tr))
                self.assertTrue(all(int(r["fold"]) == fold for r in va))

    def test_ucf_only_excludes_other_sources_but_validation_is_shared(self):
        tr_u, va_u, _ = cp.fold_rows(self.manifest, 2, "ucf_only", self.index)
        tr_a, va_a, _ = cp.fold_rows(self.manifest, 2, "all_sources", self.index)
        self.assertEqual({r["source_dataset"] for r in tr_u}, {UCF})
        self.assertEqual({r["source_dataset"] for r in tr_a}, {UCF, XD, CUVA})
        self.assertEqual([r["source_video_id"] for r in va_u], [r["source_video_id"] for r in va_a])
        self.assertEqual({r["source_dataset"] for r in va_a}, {UCF, XD, CUVA})
        self.assertEqual({r["source_video_id"] for r in tr_u} - {r["source_video_id"] for r in tr_a}, set())

    def test_failed_cache_rows_are_dropped_and_counted(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, cache, _ = write_fixture(Path(d), fail={"x_Fighting_1"})
            _f, index, _m = cp.load_cache(cache, manifest)
            self.assertNotIn((XD, "x_Fighting_1"), index)
            tr, va, counts = cp.fold_rows(manifest, 3, "all_sources", index)  # x_Fighting_1 has fold 2
            tr2, va2, counts2 = cp.fold_rows(manifest, 1, "all_sources", index)
            self.assertEqual(counts["train_missing_from_cache"] + counts2["val_missing_from_cache"], 1 + 0)
            self.assertNotIn("x_Fighting_1", {r["source_video_id"] for r in tr + va + tr2 + va2})

    def test_stale_cache_refused(self):
        (self.cache / "manifest.json").write_text(json.dumps({"combined_manifest_sha256": "0" * 64, "features_sha256": "x"}))
        with self.assertRaises(ValueError):
            cp.load_cache(self.cache, self.manifest)


class ReportingTests(unittest.TestCase):
    def test_per_source_parts_and_present_class_macro(self):
        val = [mrow(UCF, c, f"u{i}") for i, c in enumerate(cp.CLASS_NAMES)]
        val += [mrow(XD, "Shooting", "x0"), mrow(XD, "Fighting", "x1"), mrow(CUVA, "Robbery", "c0")]
        preds = {}
        for r in val:
            p = np.zeros(14)
            p[cp.CLASS_NAMES.index(r["unified_label"])] = 1.0
            preds[(r["source_dataset"], r["source_video_id"])] = p
        preds[(XD, "x1")] = np.eye(14)[cp.CLASS_NAMES.index("Shooting")]  # wrong
        parts = cp.evaluate_parts(preds, val)
        self.assertEqual(set(parts), {"ucf", "xd", "cuva"})
        self.assertEqual((parts["ucf"]["n_videos"], parts["xd"]["n_videos"], parts["cuva"]["n_videos"]), (14, 2, 1))
        self.assertAlmostEqual(parts["ucf"]["macro_f1"], 1.0)
        self.assertEqual(parts["xd"]["classes_present"], ["Fighting", "Shooting"])
        self.assertAlmostEqual(parts["xd"]["macro_f1"], (0.0 + 2 * 0.5 / 1.5) / 2)  # Fighting 0, Shooting P=.5 R=1
        self.assertAlmostEqual(parts["cuva"]["macro_f1"], 1.0)

    def test_summary_gate_and_delta(self):
        def run(arm, fold, seed, f1):
            m = {"macro_f1": f1, "per_class_f1": {c: f1 for c in cp.CLASS_NAMES}}
            return {"arm": arm, "fold": fold, "seed": seed, "parts": {"ucf": m}}

        runs = [run("ucf_only", f, s, 0.50) for f in (1, 2, 3, 4) for s in (0, 1)]
        runs += [run("all_sources", f, s, 0.55 if f < 4 else 0.52) for f in (1, 2, 3, 4) for s in (0, 1)]
        out = cp.summarize(runs)
        cmp = out["comparison_all_sources_minus_ucf_only"]
        self.assertEqual(cmp["gate_on_ucf_part"]["folds_passing"], 3)
        self.assertFalse(cmp["gate_on_ucf_part"]["passed"])
        self.assertAlmostEqual(cmp["ucf_mean_delta_over_folds"], 0.0425)
        self.assertEqual(set(cmp["highlight_per_class_f1_delta"]), set(cp.HIGHLIGHT))

    def test_inverse_frequency_weights(self):
        w = cp.inverse_frequency_weights([0, 0, 0, 1], 3)
        self.assertAlmostEqual(w[0], 4 / 9)
        self.assertAlmostEqual(w[1], 4 / 3)
        self.assertEqual(w[2], 1.0)


class EndToEndTests(unittest.TestCase):
    def test_run_writes_outputs_and_never_touches_test_rows(self):
        import torch

        torch.set_num_threads(2)
        from scripts.run_combined_probe import run

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            manifest, cache, _ = write_fixture(d)
            out = d / "out"
            summary = run(Namespace(manifest=manifest, cache_root=cache, output=out, folds=[1], seeds=[0],
                                    arms=["ucf_only", "all_sources"], epochs=1, threads=2))
            for name in ("config.json", "summary.json", "per_video.csv"):
                self.assertTrue((out / name).exists())
            with (out / "per_video.csv").open() as fh:
                rows = list(csv.DictReader(fh))
            self.assertTrue(rows)
            self.assertFalse(any(r["source_video_id"].endswith("_T") or r["source_video_id"] in ("u_untrainable", "c_unlabeled") for r in rows))
            self.assertAlmostEqual(sum(float(rows[0]["p_" + c]) for c in cp.CLASS_NAMES), 1.0, places=4)
            self.assertEqual({r["source_dataset"] for r in rows}, {UCF, XD, CUVA})
            self.assertIn("ucf", summary["arms"]["all_sources"]["parts"])
            config = json.loads((out / "config.json").read_text())
            self.assertIn("inverse-frequency", config["class_balancing"])
            self.assertEqual(set(config["seeds"]), {0})


if __name__ == "__main__":
    unittest.main()
