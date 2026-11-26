"""Tests for `data.video_dataset.P1CVideoFrameDataset`.

Builds a real blob and real manifests, then reads through the dataset, so the
extractor -> manifest -> dataset chain is exercised end to end.
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.augmentation import resolve_augmentation_policy  # noqa: E402
from data.p1c_loader import FramePreprocessing  # noqa: E402
from data.video_dataset import P1CVideoFrameDataset  # noqa: E402
from scripts.build_p1c_video_manifests import main as build_main  # noqa: E402
from scripts.extract_p1c_video_frames import extract_videos  # noqa: E402

CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    "Normal_Videos_event", "RoadAccidents", "Robbery", "Shooting", "Shoplifting",
    "Stealing", "Vandalism",
]
LOCAL_DIR = {"Normal_Videos_event": "z_Normal_Videos_event"}
SRC_H, SRC_W, FRAMES, STRIDE = 48, 64, 10, 5
TRAIN_PER_CLASS, TEST_PER_CLASS = 3, 2

PREPROCESSING = FramePreprocessing(
    resize_height=64,
    resize_width=64,
    interpolation="bilinear",
    color_mode="RGB",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


class VideoDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        ucf_root = tmp / "UCF_Crimes"
        videos_root = ucf_root / "Videos"
        splits_root = ucf_root / "Action_Regnition_splits"
        splits_root.mkdir(parents=True)
        (splits_root / "ClassIDs.txt").write_text(
            "".join(f"{n} {i + 1} \n" for i, n in enumerate(CLASSES)), encoding="utf-8"
        )

        # Pure red, so a channel swap anywhere in the chain is unmistakable.
        frame = np.zeros((SRC_H, SRC_W, 3), dtype=np.uint8)
        frame[:, :, 0] = 255

        train_lines, test_lines = [], []
        for cls_name in CLASSES:
            local = LOCAL_DIR.get(cls_name, cls_name)
            (videos_root / local).mkdir(parents=True, exist_ok=True)
            for i in range(TRAIN_PER_CLASS + TEST_PER_CLASS):
                stem = f"{cls_name}{i:03d}_x264"
                writer = cv2.VideoWriter(
                    str(videos_root / local / f"{stem}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (SRC_W, SRC_H),
                )
                if not writer.isOpened():  # pragma: no cover
                    raise unittest.SkipTest("cv2.VideoWriter unavailable")
                for _ in range(FRAMES):
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
                line = f"{cls_name}/{stem}.mp4 \n"
                (train_lines if i < TRAIN_PER_CLASS else test_lines).append(line)

        for fold in (1, 2, 3, 4):
            (splits_root / f"train_{fold:03d}.txt").write_text("".join(train_lines), encoding="utf-8")
            (splits_root / f"test_{fold:03d}.txt").write_text("".join(test_lines), encoding="utf-8")

        cls.blob_dir = tmp / "blob"
        extract_videos(
            source_root=videos_root,
            output_dir=cls.blob_dir,
            directories=None,
            stride=STRIDE,
            expected_height=SRC_H,
            expected_width=SRC_W,
        )
        cls.closure = cls.blob_dir / "extract_closure.json"
        cls.manifests = tmp / "manifests"
        assert build_main([
            "--protocol", "action_recognition", "--fold", "1",
            "--closure", str(cls.closure), "--ucf-root", str(ucf_root),
            "--output-dir", str(cls.manifests),
        ]) == 0

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _dataset(self, split: str = "train") -> P1CVideoFrameDataset:
        return P1CVideoFrameDataset(
            self.manifests / f"{split}_manifest.csv",
            closure_path=self.closure,
            preprocessing=PREPROCESSING,
            class_names=CLASSES,
        )

    def test_length_matches_the_manifest(self) -> None:
        dataset = self._dataset()
        expected_per_video = -(-FRAMES // STRIDE)
        # 14 classes x 3 train videos, minus whatever the validation carve took.
        self.assertEqual(len(dataset) % expected_per_video, 0)
        self.assertGreater(len(dataset), 0)

    def test_sample_shape_and_dtype(self) -> None:
        sample = self._dataset()[0]
        self.assertEqual(sample["image"].shape, (3, 64, 64))
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertIsInstance(sample["target"], int)
        self.assertIn(sample["target"], range(14))

    def test_pixels_are_rgb_through_the_whole_chain(self) -> None:
        sample = self._dataset()[0]
        # Undo normalisation and check red still dominates.
        mean = torch.tensor(PREPROCESSING.mean).view(3, 1, 1)
        std = torch.tensor(PREPROCESSING.std).view(3, 1, 1)
        pixels = (sample["image"] * std + mean).clamp(0, 1)
        r, g, b = (pixels[c].mean().item() for c in range(3))
        self.assertGreater(r, 0.8, f"red too low: {r}")
        self.assertLess(b, 0.25, f"blue too high: {b}")

    def test_metadata_round_trips_from_the_manifest(self) -> None:
        sample = self._dataset()[0]
        self.assertEqual(sample["split"], "train")
        self.assertEqual(sample["source_dataset"], "ucf_crime_official_video_frames")
        self.assertTrue(sample["filepath"].endswith(".mp4"))
        self.assertEqual(len(sample["file_digest"]), 64)
        self.assertEqual(sample["source_frame_index"] % STRIDE, 0)

    def test_targets_cover_all_fourteen_classes(self) -> None:
        dataset = self._dataset()
        targets = {dataset[i]["target"] for i in range(len(dataset))}
        self.assertEqual(targets, set(range(14)))

    def test_unknown_label_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            P1CVideoFrameDataset(
                self.manifests / "train_manifest.csv",
                closure_path=self.closure,
                preprocessing=PREPROCESSING,
                class_names=CLASSES[:-1],  # drop Vandalism
            )
        self.assertIn("class_names", str(caught.exception))

    def test_every_row_resolves_to_a_blob_row(self) -> None:
        # Construction raises if any (video, frame) pair is missing, so simply
        # building all three splits is the assertion.
        for split in ("train", "validation", "test"):
            self.assertGreater(len(self._dataset(split)), 0, split)

    def test_source_video_groups_supports_the_leakage_check(self) -> None:
        train = self._dataset("train")
        validation = self._dataset("validation")
        test = self._dataset("test")
        train_groups = train.source_video_groups()
        self.assertTrue(train_groups)
        self.assertIsInstance(next(iter(train_groups)), tuple)
        # This is exactly what the runners assert before calibrating.
        self.assertEqual(train_groups & validation.source_video_groups(), frozenset())
        self.assertEqual(train_groups & test.source_video_groups(), frozenset())
        self.assertEqual(
            validation.source_video_groups() & test.source_video_groups(), frozenset()
        )


class VideoDatasetAugmentationTests(unittest.TestCase):
    """K-22: augmentation must be structurally impossible on validation/test here too.

    Unlike `P1CFrameDataset`, this class has no manifest-driven split field of
    its own, so its guard rests entirely on the caller-supplied `split`
    keyword -- these tests pin that it actually refuses.
    """

    _POLICY = resolve_augmentation_policy(
        {
            "name": "standard_frame",
            "rationale": "Attack the near-duplicate-frame problem this video-blob dataset exists to serve.",
            "random_resized_crop": {"scale_min": 0.7, "scale_max": 1.0, "ratio_min": 0.75, "ratio_max": 1.333},
            "horizontal_flip_probability": 0.5,
            "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.0},
        }
    )

    @classmethod
    def setUpClass(cls) -> None:
        # A dedicated, noisy-content corpus rather than reusing
        # `VideoDatasetTests`'s solid-red fixture: a uniform colour survives
        # crop/flip/jitter unchanged regardless of seed, which would make the
        # per-frame-distinctness test below pass trivially even with a
        # broken (video-constant) seed.
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        ucf_root = tmp / "UCF_Crimes"
        videos_root = ucf_root / "Videos"
        splits_root = ucf_root / "Action_Regnition_splits"
        splits_root.mkdir(parents=True)
        (splits_root / "ClassIDs.txt").write_text(
            "".join(f"{n} {i + 1} \n" for i, n in enumerate(CLASSES)), encoding="utf-8"
        )
        rng = np.random.default_rng(0)
        train_lines, test_lines = [], []
        for cls_name in CLASSES:
            local = LOCAL_DIR.get(cls_name, cls_name)
            (videos_root / local).mkdir(parents=True, exist_ok=True)
            for i in range(TRAIN_PER_CLASS + TEST_PER_CLASS):
                stem = f"{cls_name}{i:03d}_x264"
                writer = cv2.VideoWriter(
                    str(videos_root / local / f"{stem}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (SRC_W, SRC_H),
                )
                if not writer.isOpened():  # pragma: no cover
                    raise unittest.SkipTest("cv2.VideoWriter unavailable")
                # The SAME noisy frame content repeated across every frame of
                # this video (real near-duplicate videos, just like the ones
                # this experiment targets) -- isolates whether the applied
                # augmentation differs per frame from whether the raw pixels
                # happen to differ, which a per-frame-random fixture would not.
                frame = rng.integers(0, 256, size=(SRC_H, SRC_W, 3), dtype=np.uint8)
                for _ in range(FRAMES):
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
                line = f"{cls_name}/{stem}.mp4 \n"
                (train_lines if i < TRAIN_PER_CLASS else test_lines).append(line)
        for fold in (1, 2, 3, 4):
            (splits_root / f"train_{fold:03d}.txt").write_text("".join(train_lines), encoding="utf-8")
            (splits_root / f"test_{fold:03d}.txt").write_text("".join(test_lines), encoding="utf-8")
        cls.blob_dir = tmp / "blob"
        extract_videos(
            source_root=videos_root, output_dir=cls.blob_dir, directories=None,
            stride=STRIDE, expected_height=SRC_H, expected_width=SRC_W,
        )
        cls.closure = cls.blob_dir / "extract_closure.json"
        cls.manifests = tmp / "manifests"
        assert build_main([
            "--protocol", "action_recognition", "--fold", "1",
            "--closure", str(cls.closure), "--ucf-root", str(ucf_root),
            "--output-dir", str(cls.manifests),
        ]) == 0

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _dataset(self, split: str, **kwargs) -> P1CVideoFrameDataset:
        return P1CVideoFrameDataset(
            self.manifests / f"{split}_manifest.csv", closure_path=self.closure,
            preprocessing=PREPROCESSING, class_names=CLASSES, split=split, **kwargs,
        )

    def test_train_split_accepts_a_real_policy(self) -> None:
        dataset = self._dataset("train", augmentation=self._POLICY, augmentation_run_seed=5)
        self.assertEqual(dataset.augmentation, self._POLICY)

    def test_validation_split_refuses_a_real_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "may only be registered for split=train"):
            self._dataset("validation", augmentation=self._POLICY)

    def test_test_split_refuses_a_real_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "may only be registered for split=train"):
            self._dataset("test", augmentation=self._POLICY)

    def test_omitted_split_refuses_a_real_policy(self) -> None:
        # `split` defaults to None, which is also not "train": a caller must
        # explicitly assert train, never rely on an implicit default.
        with self.assertRaisesRegex(ValueError, "may only be registered for split=train"):
            P1CVideoFrameDataset(
                self.manifests / "train_manifest.csv", closure_path=self.closure,
                preprocessing=PREPROCESSING, class_names=CLASSES, augmentation=self._POLICY,
            )

    def test_no_augmentation_argument_is_unaffected_by_split(self) -> None:
        for split in ("train", "validation", "test"):
            self.assertTrue(self._dataset(split).augmentation.is_none)

    def test_distinct_frames_of_the_same_video_get_distinct_augmentations(self) -> None:
        # The bug this dataset's seeding specifically has to avoid: if the
        # seed only depended on the shared video filepath/digest, every frame
        # sampled from one source video would receive the identical crop/flip/
        # jitter, leaving them exactly the near-duplicates augmentation exists
        # to break up.
        dataset = self._dataset("train", augmentation=self._POLICY, augmentation_run_seed=3)
        by_video: dict[str, list[torch.Tensor]] = {}
        for index in range(len(dataset)):
            sample = dataset[index]
            by_video.setdefault(sample["source_video_id"], []).append(sample["image"])
        multi_frame_videos = [images for images in by_video.values() if len(images) > 1]
        self.assertTrue(multi_frame_videos, "fixture produced no multi-frame video to exercise this")
        for images in multi_frame_videos:
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    self.assertFalse(torch.equal(images[i], images[j]), (i, j))

    def test_same_run_seed_and_epoch_reproduce_the_same_augmentation(self) -> None:
        first = self._dataset("train", augmentation=self._POLICY, augmentation_run_seed=9)
        second = self._dataset("train", augmentation=self._POLICY, augmentation_run_seed=9)
        for index in range(len(first)):
            self.assertTrue(torch.equal(first[index]["image"], second[index]["image"]), index)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class PickleTests(unittest.TestCase):
    """The dataset crosses a process boundary by pickle, so its size matters.

    `build_p1c_frame_dataloader` sets the "spawn" start method, which means
    every worker receives a *pickled copy* of the dataset rather than sharing
    the parent's pages. Two things therefore have to hold, and neither is
    visible from single-process use:

      1. The pickle must not drag the blob along. The reader holds a
         `np.memmap` over an 84.3 GiB file, and `np.memmap` inherits
         `ndarray`'s pickling, which serialises mapped bytes *by value*. That
         defect OOM-killed CALIBRATION-006 (K-15).
      2. The per-row storage must stay compact, or six workers each pay for it.

    These assert both against a real blob and real manifests.
    """

    @classmethod
    def setUpClass(cls) -> None:
        VideoDatasetTests.setUpClass()
        cls.blob_dir, cls.closure, cls.manifests = (
            VideoDatasetTests.blob_dir, VideoDatasetTests.closure, VideoDatasetTests.manifests
        )

    @classmethod
    def tearDownClass(cls) -> None:
        VideoDatasetTests.tearDownClass()

    def _dataset(self) -> P1CVideoFrameDataset:
        return P1CVideoFrameDataset(
            self.manifests / "train_manifest.csv", closure_path=self.closure,
            preprocessing=PREPROCESSING, class_names=CLASSES,
        )

    def test_pickle_is_far_smaller_than_the_blob(self) -> None:
        dataset = self._dataset()
        blob_bytes = (self.blob_dir / "frames.jpegblob").stat().st_size
        payload = pickle.dumps(dataset)
        self.assertLess(
            len(payload), blob_bytes,
            "the pickled dataset is at least as large as the blob itself, which means "
            "the memmap is being serialised by value -- the CALIBRATION-006 OOM",
        )
        # Independent of blob size: nothing here should be megabytes.
        self.assertLess(len(payload), 1 << 20, f"pickle is {len(payload)} bytes")

    def test_unpickled_dataset_yields_identical_samples(self) -> None:
        dataset = self._dataset()
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertEqual(len(restored), len(dataset))
        for index in (0, len(dataset) // 2, len(dataset) - 1):
            original, copy = dataset[index], restored[index]
            self.assertTrue(torch.equal(original["image"], copy["image"]), index)
            for field in ("target", "source_dataset", "source_video_id",
                          "source_frame_index", "filepath", "file_digest", "split"):
                self.assertEqual(original[field], copy[field], f"{field} at {index}")

    def test_reader_reopens_the_blob_after_unpickling(self) -> None:
        # The memmap is dropped on the way out and must be restored on demand,
        # not left as None for the first __getitem__ to trip over.
        restored = pickle.loads(pickle.dumps(self._dataset()))
        self.assertIsNotNone(restored[0]["image"])
        self.assertEqual(len(restored._reader.index), restored._reader.closure.kept_frame_count)
