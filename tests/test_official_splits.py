"""Tests for `data.official_splits`.

Synthetic fixtures cover structure and the failure paths. The real-corpus tests
are skipped when `Datasets/` is absent, since that tree is local-only and
Git-ignored -- they must not turn a fresh clone red.
"""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.official_splits import (  # noqa: E402
    AR_NORMAL_CLASS,
    AR_NORMAL_LOCAL_DIR,
    SplitEntry,
    local_directory_for,
    read_action_recognition_split,
    read_anomaly_detection_split,
    read_class_ids,
)

REAL_CORPUS = REPO_ROOT / "Datasets" / "UCF Crime Official" / "raw" / "UCF_Crimes"

AR_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    AR_NORMAL_CLASS, "RoadAccidents", "Robbery", "Shooting", "Shoplifting",
    "Stealing", "Vandalism",
]


def _write_ar_fixture(root: Path, *, per_class_train: int = 38, per_class_test: int = 12) -> None:
    splits = root / "Action_Regnition_splits"
    splits.mkdir(parents=True)
    # Upstream lines carry a trailing space; reproduce it so the parser is
    # tested against the real file shape rather than a tidied one.
    (splits / "ClassIDs.txt").write_text(
        "".join(f"{name} {i + 1} \n" for i, name in enumerate(AR_CLASSES)),
        encoding="utf-8",
    )
    for fold in (1, 2, 3, 4):
        for stem, count, offset in (
            ("train", per_class_train, 0),
            ("test", per_class_test, 900),
        ):
            lines = [
                f"{cls}/{cls}{offset + i:03d}_x264.mp4 \n"
                for cls in AR_CLASSES
                for i in range(count)
            ]
            (splits / f"{stem}_{fold:03d}.txt").write_text("".join(lines), encoding="utf-8")


def _write_ad_fixture(root: Path, *, train: int = 1610, test: int = 290) -> None:
    splits = root / "Anomaly_Detection_splits"
    splits.mkdir(parents=True)
    (splits / "Anomaly_Train.txt").write_text(
        "".join(f"Abuse/Abuse{i:04d}_x264.mp4 \n" for i in range(train)), encoding="utf-8"
    )
    (splits / "Anomaly_Test.txt").write_text(
        "".join(f"Arson/Arson{i:04d}_x264.mp4 \n" for i in range(test)), encoding="utf-8"
    )


class SyntheticSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_action_recognition_fold_parses_and_is_balanced(self) -> None:
        _write_ar_fixture(self.root)
        split = read_action_recognition_split(self.root, fold=1)
        self.assertEqual(len(split["train"]), 532)
        self.assertEqual(len(split["test"]), 168)
        self.assertEqual(len({e.class_name for e in split["train"]}), 14)

    def test_unbalanced_fold_is_refused(self) -> None:
        # A truncated or hand-edited split file must fail loudly rather than
        # produce a quietly skewed experiment. Drop a single line from one
        # class so the file stays well-formed and only the balance breaks.
        _write_ar_fixture(self.root)
        path = self.root / "Action_Regnition_splits" / "train_001.txt"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[1:]), encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            read_action_recognition_split(self.root, fold=1)
        self.assertIn("balanced", str(caught.exception))

    def test_uniformly_smaller_but_balanced_fold_is_accepted(self) -> None:
        # The parser validates the invariant, not the corpus-specific 38/12.
        _write_ar_fixture(self.root, per_class_train=3, per_class_test=2)
        split = read_action_recognition_split(self.root, fold=1)
        self.assertEqual(len(split["train"]), 42)
        self.assertEqual(len(split["test"]), 28)

    def test_invalid_fold_number_is_refused(self) -> None:
        _write_ar_fixture(self.root)
        with self.assertRaises(ValueError):
            read_action_recognition_split(self.root, fold=5)

    def test_class_ids_must_number_fourteen(self) -> None:
        _write_ar_fixture(self.root)
        path = self.root / "Action_Regnition_splits" / "ClassIDs.txt"
        path.write_text("Abuse 1 \n", encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            read_class_ids(self.root / "Action_Regnition_splits")
        self.assertIn("14", str(caught.exception))

    def test_anomaly_detection_parses(self) -> None:
        _write_ad_fixture(self.root)
        split = read_anomaly_detection_split(self.root)
        self.assertEqual(len(split["train"]), 1610)
        self.assertEqual(len(split["test"]), 290)

    def test_anomaly_detection_total_is_enforced(self) -> None:
        _write_ad_fixture(self.root, train=1000)
        with self.assertRaises(ValueError) as caught:
            read_anomaly_detection_split(self.root)
        self.assertIn("1900", str(caught.exception))

    def test_normal_class_maps_to_its_local_directory(self) -> None:
        entry = SplitEntry(
            relative_path=f"{AR_NORMAL_CLASS}/Normal_Videos_317_x264.mp4",
            directory=AR_NORMAL_CLASS,
            video_id="Normal_Videos_317_x264",
            class_name=AR_NORMAL_CLASS,
        )
        self.assertEqual(local_directory_for(entry), AR_NORMAL_LOCAL_DIR)

    def test_other_directories_are_not_remapped(self) -> None:
        entry = SplitEntry("Abuse/Abuse001_x264.mp4", "Abuse", "Abuse001_x264", "Abuse")
        self.assertEqual(local_directory_for(entry), "Abuse")


@unittest.skipUnless(REAL_CORPUS.is_dir(), "official corpus not present locally")
class RealCorpusTests(unittest.TestCase):
    """Guards against the official files drifting from what D-34/D-35 assume."""

    def test_all_four_action_recognition_folds_parse(self) -> None:
        for fold in (1, 2, 3, 4):
            split = read_action_recognition_split(REAL_CORPUS, fold=fold)
            self.assertEqual(len(split["train"]), 532, f"fold {fold}")
            self.assertEqual(len(split["test"]), 168, f"fold {fold}")
            # Pin the official per-class figures here rather than in the
            # parser, which checks only the balance invariant.
            for name, expected in (("train", 38), ("test", 12)):
                counts = Counter(e.class_name for e in split[name])
                self.assertEqual(set(counts.values()), {expected}, f"fold {fold} {name}")

    def test_action_recognition_test_folds_are_disjoint(self) -> None:
        seen: set[str] = set()
        for fold in (1, 2, 3, 4):
            paths = {
                e.relative_path
                for e in read_action_recognition_split(REAL_CORPUS, fold=fold)["test"]
            }
            self.assertEqual(seen & paths, set(), f"fold {fold} test overlaps an earlier fold")
            seen |= paths
        self.assertEqual(len(seen), 672)

    def test_every_split_video_exists_on_disk(self) -> None:
        videos = REAL_CORPUS / "Videos"
        entries = list(read_action_recognition_split(REAL_CORPUS, fold=1)["train"])
        entries += read_anomaly_detection_split(REAL_CORPUS)["test"]
        missing = [
            e.relative_path
            for e in entries
            if not (videos / local_directory_for(e) / f"{e.video_id}.mp4").is_file()
        ]
        self.assertEqual(missing, [])

    def test_normal_videos_event_is_absent_from_anomaly_splits(self) -> None:
        # The two protocols must not be conflated: the AR Normal class does not
        # participate in the anomaly-detection task at all.
        ad = read_anomaly_detection_split(REAL_CORPUS)
        for split, entries in ad.items():
            offenders = [e.relative_path for e in entries if e.directory == AR_NORMAL_CLASS]
            self.assertEqual(offenders, [], f"{split} contains AR Normal videos")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
