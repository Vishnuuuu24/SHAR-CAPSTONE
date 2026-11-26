"""Video-level split integrity for the P1C Action-Recognition folds (backlog Fix #8 / #15).

The dataset is frames drawn from videos. If two frames of the same video land in
different splits the model effectively validates on its training data, because
frames within a video are near-duplicates (P1C-VIDEO-FRAME-BASELINE-011: one
video contributes hundreds of them). The split must therefore be decided at the
*video* level and never the frame level.

These tests assert that property directly against the artifacts every P1C run
reads -- the P1B fold manifests and the P1C feature caches built from them -- so
a future manifest regeneration or cache rebuild that broke it would fail here
rather than silently inflate a score. They are pure file reads: no GPU, no
network, fast.

If a fold's manifests or caches are simply absent (a fresh checkout) the
relevant test skips rather than fails -- it is guarding artifacts, not requiring
them.
"""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
P1B_DIR = REPO_ROOT / "results" / "p1b"
P1C_DIR = REPO_ROOT / "results" / "p1c"
FOLDS = (1, 2, 3, 4)
SPLITS = ("train", "validation", "test")


def _manifest_dir(fold: int) -> Path:
    return P1B_DIR / f"p1b-video-ar-fold{fold}"


def _video_keys(manifest_csv: Path) -> set[tuple[str, str]]:
    with manifest_csv.open(encoding="utf-8", newline="") as handle:
        return {
            (row["source_dataset"], row["source_video_id"])
            for row in csv.DictReader(handle)
        }


def _video_label_map(manifest_csv: Path) -> dict[tuple[str, str], str]:
    labels: dict[tuple[str, str], str] = {}
    with manifest_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source_dataset"], row["source_video_id"])
            labels[key] = row["label"]
    return labels


class FoldManifestSplitTests(unittest.TestCase):
    def _require_fold(self, fold: int) -> dict[str, Path]:
        paths = {split: _manifest_dir(fold) / f"{split}_manifest.csv" for split in SPLITS}
        if not all(path.exists() for path in paths.values()):
            self.skipTest(f"fold {fold} manifests not present")
        return paths

    def test_train_validation_test_video_sets_are_pairwise_disjoint(self) -> None:
        for fold in FOLDS:
            paths = self._require_fold(fold)
            keys = {split: _video_keys(path) for split, path in paths.items()}
            with self.subTest(fold=fold):
                self.assertEqual(keys["train"] & keys["validation"], set(), "train/validation video overlap")
                self.assertEqual(keys["train"] & keys["test"], set(), "train/test video overlap")
                self.assertEqual(keys["validation"] & keys["test"], set(), "validation/test video overlap")

    def test_a_video_carries_one_label_everywhere_it_appears(self) -> None:
        for fold in FOLDS:
            paths = self._require_fold(fold)
            merged: dict[tuple[str, str], str] = {}
            for path in paths.values():
                for key, label in _video_label_map(path).items():
                    if key in merged:
                        self.assertEqual(merged[key], label, f"{key} has conflicting labels in fold {fold}")
                    merged[key] = label

    def test_video_level_class_balance_is_preserved_per_split(self) -> None:
        # The AR folds are balanced by construction at the video level
        # (34 train / 4 validation / 12 test videos per class, 14 classes);
        # frame counts are imbalanced only because videos differ in length.
        expected = {"train": 34, "validation": 4, "test": 12}
        for fold in FOLDS:
            paths = self._require_fold(fold)
            for split, path in paths.items():
                counts: dict[str, int] = {}
                for label in _video_label_map(path).values():
                    counts[label] = counts.get(label, 0) + 1
                with self.subTest(fold=fold, split=split):
                    self.assertEqual(len(counts), 14, "expected all 14 classes")
                    self.assertEqual(set(counts.values()), {expected[split]})


class FeatureCacheProvenanceTests(unittest.TestCase):
    """Every cached feature row keeps its source video id, and stays inside its split."""

    CACHE_GLOBS = ("p1c-frozen-feature-cache-ar-fold{fold}", "p1c-dinov2-feature-cache-ar-fold{fold}")

    def _cache_rows(self, cache_split_dir: Path) -> list[dict[str, str]]:
        with (cache_split_dir / "rows.csv").open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_cache_rows_retain_source_video_id_and_match_their_manifest_split(self) -> None:
        checked = 0
        for fold in FOLDS:
            manifest_dir = _manifest_dir(fold)
            if not manifest_dir.exists():
                continue
            for template in self.CACHE_GLOBS:
                cache_root = P1C_DIR / template.format(fold=fold)
                if not cache_root.exists():
                    continue
                for split in ("train", "validation"):
                    split_dir = cache_root / split
                    manifest_csv = manifest_dir / f"{split}_manifest.csv"
                    if not (split_dir / "rows.csv").exists() or not manifest_csv.exists():
                        continue
                    manifest_keys = _video_keys(manifest_csv)
                    rows = self._cache_rows(split_dir)
                    self.assertTrue(rows, f"{split_dir} has no rows")
                    cache_keys = set()
                    for row in rows:
                        self.assertTrue(row["source_video_id"], f"empty source_video_id in {split_dir}")
                        cache_keys.add((row["source_dataset"], row["source_video_id"]))
                    leaked = cache_keys - manifest_keys
                    with self.subTest(cache=str(cache_root), split=split):
                        self.assertEqual(leaked, set(), f"cache videos absent from the {split} manifest")
                    checked += 1
        if checked == 0:
            self.skipTest("no feature caches present to check")

    def test_manifest_json_records_the_matching_split(self) -> None:
        checked = 0
        for fold in FOLDS:
            for template in self.CACHE_GLOBS:
                cache_root = P1C_DIR / template.format(fold=fold)
                for split in ("train", "validation"):
                    manifest_json = cache_root / split / "manifest.json"
                    if not manifest_json.exists():
                        continue
                    payload = json.loads(manifest_json.read_text(encoding="utf-8"))
                    with self.subTest(cache=str(cache_root), split=split):
                        self.assertEqual(payload["split"], split)
                        self.assertIn(f"ar-fold{fold}", payload["source_manifest_path"])
                    checked += 1
        if checked == 0:
            self.skipTest("no feature caches present to check")


if __name__ == "__main__":
    unittest.main()
