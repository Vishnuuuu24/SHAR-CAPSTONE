"""End-to-end tests for manifest construction from a video blob.

These build a real (tiny) blob with the real extractor and real split files in
the official on-disk shape, then run the real builder over them. Nothing is
mocked, so a change that breaks the extractor/builder contract fails here
rather than at the first calibration.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.official_splits import SplitEntry  # noqa: E402
from data.video_manifests import (  # noqa: E402
    MANIFEST_FIELDS,
    assert_consistent_label_vocabulary,
    assert_no_video_leakage,
    carve_validation,
)
from scripts.build_p1c_video_manifests import main as build_main  # noqa: E402
from scripts.extract_p1c_video_frames import extract_videos  # noqa: E402

CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    "Normal_Videos_event", "RoadAccidents", "Robbery", "Shooting", "Shoplifting",
    "Stealing", "Vandalism",
]
LOCAL_DIR = {"Normal_Videos_event": "z_Normal_Videos_event"}

HEIGHT, WIDTH, FRAMES, STRIDE = 32, 32, 10, 5
PER_CLASS_TRAIN, PER_CLASS_TEST = 3, 2


class BuildManifestsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        self.ucf_root = self.tmp / "UCF_Crimes"
        videos_root = self.ucf_root / "Videos"
        splits_root = self.ucf_root / "Action_Regnition_splits"
        splits_root.mkdir(parents=True)

        (splits_root / "ClassIDs.txt").write_text(
            "".join(f"{name} {i + 1} \n" for i, name in enumerate(CLASSES)), encoding="utf-8"
        )

        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # red; a channel swap would be visible downstream

        train_lines, test_lines = [], []
        for cls in CLASSES:
            local = LOCAL_DIR.get(cls, cls)
            (videos_root / local).mkdir(parents=True, exist_ok=True)
            for i in range(PER_CLASS_TRAIN + PER_CLASS_TEST):
                stem = f"{cls}{i:03d}_x264"
                self._write_video(videos_root / local / f"{stem}.mp4", frame)
                line = f"{cls}/{stem}.mp4 \n"
                (train_lines if i < PER_CLASS_TRAIN else test_lines).append(line)

        for fold in (1, 2, 3, 4):
            (splits_root / f"train_{fold:03d}.txt").write_text("".join(train_lines), encoding="utf-8")
            (splits_root / f"test_{fold:03d}.txt").write_text("".join(test_lines), encoding="utf-8")

        self.blob_dir = self.tmp / "blob"
        extract_videos(
            source_root=videos_root,
            output_dir=self.blob_dir,
            directories=None,
            stride=STRIDE,
            expected_height=HEIGHT,
            expected_width=WIDTH,
        )
        self.closure = self.blob_dir / "extract_closure.json"
        self.out = self.tmp / "manifests"

    @staticmethod
    def _write_video(path: Path, frame_rgb: np.ndarray) -> None:
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT)
        )
        if not writer.isOpened():  # pragma: no cover - environment dependent
            raise unittest.SkipTest("cv2.VideoWriter unavailable")
        for _ in range(FRAMES):
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        writer.release()

    def _build(self, **overrides: object) -> dict:
        argv = [
            "--protocol", "action_recognition", "--fold", "1",
            "--closure", str(self.closure),
            "--ucf-root", str(self.ucf_root),
            "--output-dir", str(self.out),
        ]
        for key, value in overrides.items():
            argv += [f"--{key.replace('_', '-')}", str(value)]
        self.assertEqual(build_main(argv), 0)
        return json.loads((self.out / "manifest_closure.json").read_text(encoding="utf-8"))

    def _rows(self, split: str) -> list[dict[str, str]]:
        with (self.out / f"{split}_manifest.csv").open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_manifests_have_the_registered_schema(self) -> None:
        self._build()
        for split in ("train", "validation", "test"):
            with (self.out / f"{split}_manifest.csv").open(encoding="utf-8") as handle:
                header = handle.readline().strip().split(",")
            self.assertEqual(header, MANIFEST_FIELDS, split)

    def test_every_row_is_video_inherited(self) -> None:
        self._build()
        for split in ("train", "validation", "test"):
            scopes = {row["label_scope"] for row in self._rows(split)}
            self.assertEqual(scopes, {"video_inherited"}, split)

    def test_no_source_video_crosses_a_split(self) -> None:
        self._build()
        seen: dict[str, str] = {}
        for split in ("train", "validation", "test"):
            for row in self._rows(split):
                previous = seen.setdefault(row["source_video_id"], split)
                self.assertEqual(previous, split, f"{row['source_video_id']} leaked")

    def test_frame_counts_match_the_blob_stride(self) -> None:
        closure = self._build()
        expected_per_video = -(-FRAMES // STRIDE)
        for split in ("train", "validation", "test"):
            rows = self._rows(split)
            videos = {row["source_video_id"] for row in rows}
            self.assertEqual(len(rows), len(videos) * expected_per_video, split)
            self.assertEqual(closure["splits"][split]["row_count"], len(rows))

    def test_all_fourteen_classes_survive_the_validation_carve(self) -> None:
        self._build()
        for split in ("train", "validation", "test"):
            labels = {row["label"] for row in self._rows(split)}
            self.assertEqual(len(labels), 14, f"{split} lost a class: {sorted(labels)}")

    def test_normal_class_resolves_to_its_local_directory(self) -> None:
        self._build()
        rows = [r for r in self._rows("train") + self._rows("validation")
                if r["label"] == "Normal_Videos_event"]
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(row["filepath"].startswith("z_Normal_Videos_event/"), row["filepath"])

    def test_file_digest_is_the_source_video_digest(self) -> None:
        self._build()
        with (self.blob_dir / "videos.csv").open(encoding="utf-8", newline="") as handle:
            digests = {
                r["file_name"][: -len(".mp4")]: r["file_sha256"] for r in csv.DictReader(handle)
            }
        for row in self._rows("train"):
            self.assertEqual(row["file_digest"], digests[row["source_video_id"]])

    def test_build_is_deterministic_for_a_seed(self) -> None:
        first = self._build()
        second = self._build()
        for split in ("train", "validation", "test"):
            self.assertEqual(
                first["splits"][split]["sha256"], second["splits"][split]["sha256"], split
            )

    def test_closure_records_protocol_and_blob_identity(self) -> None:
        closure = self._build()
        self.assertEqual(closure["protocol"], "action_recognition")
        self.assertEqual(closure["fold"], 1)
        self.assertEqual(closure["stride"], STRIDE)
        blob = json.loads(self.closure.read_text(encoding="utf-8"))
        self.assertEqual(closure["blob_sha256"], blob["blob_sha256"])

    def test_leakage_check_rejects_a_shared_video(self) -> None:
        shared = SplitEntry("Abuse/Abuse001_x264.mp4", "Abuse", "Abuse001_x264", "Abuse")
        with self.assertRaises(ValueError) as caught:
            assert_no_video_leakage({"train": [shared], "test": [shared]})
        self.assertIn("Abuse001_x264", str(caught.exception))

    def test_mismatched_label_vocabulary_is_refused(self) -> None:
        # The real anomaly-detection protocol trips this: its Normal videos sit
        # in Training_/Testing_Normal_Videos_Anomaly, so train and test would
        # carry different labels.
        train = [SplitEntry("A/A1.mp4", "A", "A1", "TrainOnlyNormal")]
        test = [SplitEntry("B/B1.mp4", "B", "B1", "TestOnlyNormal")]
        with self.assertRaises(ValueError) as caught:
            assert_consistent_label_vocabulary({"train": train, "test": test})
        message = str(caught.exception)
        self.assertIn("label vocabularies differ", message)
        self.assertIn("label-mapping decision", message)

    def test_matching_label_vocabulary_is_accepted(self) -> None:
        train = [SplitEntry("A/A1.mp4", "A", "A1", "Normal")]
        test = [SplitEntry("A/A2.mp4", "A", "A2", "Normal")]
        assert_consistent_label_vocabulary({"train": train, "test": test})

    def test_carve_refuses_a_degenerate_fraction(self) -> None:
        entries = [SplitEntry("A/A1.mp4", "A", "A1", "A")]
        for bad in (0.0, 1.0, -0.1):
            with self.assertRaises(ValueError):
                carve_validation(entries, fraction=bad, seed=0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
