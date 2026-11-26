"""Tests for `core.p1c_roi` (ROI study geometry, planning, and preprocessing).

Synthetic fixtures only, no blob or detector access -- mirrors
`tests/test_p1c_video_sampling.py`'s style.
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_roi import (  # noqa: E402
    RoiRecord,
    crop_to_box,
    plan_roi_frames,
    preprocess_crop,
    read_roi_records_csv,
    union_person_box,
    write_roi_records_csv,
)
from data.augmentation import NONE_AUGMENTATION  # noqa: E402
from data.p1c_loader import FramePreprocessing, _INTERPOLATION  # noqa: E402


class UnionPersonBoxTests(unittest.TestCase):
    def test_no_detections_above_threshold_returns_none(self) -> None:
        boxes = np.array([[10.0, 10.0, 20.0, 20.0]])
        confidences = np.array([0.1])
        result = union_person_box(
            boxes, confidences, conf_threshold=0.25, frame_width=320, frame_height=240
        )
        self.assertIsNone(result)

    def test_empty_detections_returns_none(self) -> None:
        boxes = np.zeros((0, 4))
        confidences = np.zeros((0,))
        result = union_person_box(
            boxes, confidences, conf_threshold=0.25, frame_width=320, frame_height=240
        )
        self.assertIsNone(result)

    def test_single_box_union_equals_itself_before_margin(self) -> None:
        # margin_fraction=0 isolates the union step from the expansion step.
        boxes = np.array([[50.0, 60.0, 150.0, 160.0]])
        confidences = np.array([0.9])
        result = union_person_box(
            boxes, confidences, conf_threshold=0.25, frame_width=320, frame_height=240,
            margin_fraction=0.0,
        )
        self.assertEqual(result, (50, 60, 150, 160))

    def test_union_of_two_boxes_is_the_bounding_box(self) -> None:
        boxes = np.array([[10.0, 10.0, 50.0, 50.0], [100.0, 20.0, 140.0, 80.0]])
        confidences = np.array([0.9, 0.9])
        result = union_person_box(
            boxes, confidences, conf_threshold=0.25, frame_width=320, frame_height=240,
            margin_fraction=0.0,
        )
        self.assertEqual(result, (10, 10, 140, 80))

    def test_low_confidence_box_excluded_from_union(self) -> None:
        boxes = np.array([[10.0, 10.0, 50.0, 50.0], [200.0, 200.0, 220.0, 220.0]])
        confidences = np.array([0.9, 0.1])
        result = union_person_box(
            boxes, confidences, conf_threshold=0.25, frame_width=320, frame_height=240,
            margin_fraction=0.0,
        )
        self.assertEqual(result, (10, 10, 50, 50))

    def test_margin_expands_box_by_fraction_of_its_own_size(self) -> None:
        # Union box is 100 wide x 40 tall; margin_fraction=0.10 adds 10 on
        # each side in x and 4 on each side in y.
        boxes = np.array([[100.0, 100.0, 200.0, 140.0]])
        confidences = np.array([0.9])
        result = union_person_box(
            boxes, confidences, conf_threshold=0.25, frame_width=320, frame_height=240,
            margin_fraction=0.10,
        )
        self.assertEqual(result, (90, 96, 210, 144))

    def test_margin_clamped_to_frame_bounds(self) -> None:
        boxes = np.array([[0.0, 0.0, 20.0, 20.0]])
        confidences = np.array([0.9])
        result = union_person_box(
            boxes, confidences, conf_threshold=0.25, frame_width=320, frame_height=240,
            margin_fraction=1.0,
        )
        x0, y0, x1, y1 = result
        self.assertEqual((x0, y0), (0, 0))
        self.assertLessEqual(x1, 320)
        self.assertLessEqual(y1, 240)

    def test_box_at_far_edge_clamped_and_never_degenerate(self) -> None:
        boxes = np.array([[319.0, 239.0, 320.0, 240.0]])
        confidences = np.array([0.9])
        result = union_person_box(
            boxes, confidences, conf_threshold=0.25, frame_width=320, frame_height=240,
            margin_fraction=0.0,
        )
        x0, y0, x1, y1 = result
        self.assertGreater(x1, x0)
        self.assertGreater(y1, y0)
        self.assertLessEqual(x1, 320)
        self.assertLessEqual(y1, 240)

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            union_person_box(
                np.zeros((2, 4)), np.zeros((1,)), conf_threshold=0.25,
                frame_width=320, frame_height=240,
            )


class CropToBoxTests(unittest.TestCase):
    def test_crops_to_the_exact_box(self) -> None:
        image = Image.new("RGB", (320, 240), color=(1, 2, 3))
        cropped = crop_to_box(image, (10, 20, 110, 120))
        self.assertEqual(cropped.size, (100, 100))

    def test_degenerate_box_raises(self) -> None:
        image = Image.new("RGB", (320, 240))
        with self.assertRaises(ValueError):
            crop_to_box(image, (50, 50, 50, 60))


def _fixture_preprocessing() -> FramePreprocessing:
    return FramePreprocessing(
        resize_height=224,
        resize_width=224,
        interpolation="bilinear",
        color_mode="RGB",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


class PreprocessCropTests(unittest.TestCase):
    def test_full_frame_matches_the_dataset_own_tensor(self) -> None:
        """A synthetic full-frame crop must equal the dataset's own arithmetic.

        Builds the P1CVideoFrameDataset.__getitem__ tensor path by hand
        (rather than constructing the dataset, which needs a real blob
        closure) from the same NONE_AUGMENTATION.apply + float32/255 +
        mean/std steps, and asserts `preprocess_crop` reproduces it exactly.
        """

        rng = np.random.default_rng(0)
        pixels = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
        image = Image.fromarray(pixels, mode="RGB")
        preprocessing = _fixture_preprocessing()

        expected_image = NONE_AUGMENTATION.apply(
            image,
            resize_width=preprocessing.resize_width,
            resize_height=preprocessing.resize_height,
            resample=_INTERPOLATION[preprocessing.interpolation],
            seed=0,
        )
        import torch

        expected_pixels = np.asarray(expected_image, dtype=np.float32)
        expected_tensor = torch.from_numpy(np.ascontiguousarray(expected_pixels)).permute(2, 0, 1).div_(255.0)
        mean = torch.tensor(preprocessing.mean, dtype=expected_tensor.dtype).view(3, 1, 1)
        std = torch.tensor(preprocessing.std, dtype=expected_tensor.dtype).view(3, 1, 1)
        expected_tensor = (expected_tensor - mean) / std

        actual_tensor = preprocess_crop(image, preprocessing)
        self.assertTrue(torch.equal(actual_tensor, expected_tensor))
        self.assertEqual(tuple(actual_tensor.shape), (3, 224, 224))

    def test_wrong_color_mode_is_converted(self) -> None:
        image = Image.new("L", (320, 240), color=128)
        preprocessing = _fixture_preprocessing()
        tensor = preprocess_crop(image, preprocessing)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))


class RoiRecordTests(unittest.TestCase):
    def _sample_records(self) -> list[RoiRecord]:
        return [
            RoiRecord(
                source_video_id="Shooting001_x264",
                source_frame_index=100,
                target=10,
                label="Shooting",
                person_count=2,
                box_x0=10,
                box_y0=20,
                box_x1=110,
                box_y1=220,
                box_area_fraction=0.4167,
                roi_fallback=0,
                max_person_conf=0.87,
            ),
            RoiRecord(
                source_video_id="Normal_Videos123_x264",
                source_frame_index=5,
                target=7,
                label="Normal_Videos_event",
                person_count=0,
                box_x0=-1,
                box_y0=-1,
                box_x1=-1,
                box_y1=-1,
                box_area_fraction=0.0,
                roi_fallback=1,
                max_person_conf=0.0,
            ),
        ]

    def test_fallback_row_requires_sentinel_box(self) -> None:
        with self.assertRaises(ValueError):
            RoiRecord(
                source_video_id="v", source_frame_index=0, target=0, label="Abuse",
                person_count=0, box_x0=0, box_y0=0, box_x1=10, box_y1=10,
                box_area_fraction=0.1, roi_fallback=1, max_person_conf=0.0,
            )

    def test_invalid_roi_fallback_raises(self) -> None:
        with self.assertRaises(ValueError):
            RoiRecord(
                source_video_id="v", source_frame_index=0, target=0, label="Abuse",
                person_count=1, box_x0=0, box_y0=0, box_x1=10, box_y1=10,
                box_area_fraction=0.1, roi_fallback=2, max_person_conf=0.5,
            )

    def test_csv_round_trip(self) -> None:
        records = self._sample_records()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            write_roi_records_csv(path, records)
            reloaded = read_roi_records_csv(path)
        self.assertEqual(reloaded, records)

    def test_csv_round_trip_preserves_header_order(self) -> None:
        records = self._sample_records()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            write_roi_records_csv(path, records)
            with path.open() as handle:
                header = next(csv.reader(handle))
        self.assertEqual(
            header,
            [
                "source_video_id", "source_frame_index", "target", "label", "person_count",
                "box_x0", "box_y0", "box_x1", "box_y1", "box_area_fraction", "roi_fallback",
                "max_person_conf",
            ],
        )


def _cache_rows(video_frame_counts: list[tuple[str, str, int]]) -> list[dict]:
    """`[(video_id, label, frame_count), ...]` -> cache-shaped rows, frames stride 5."""

    rows: list[dict] = []
    for video_id, label, count in video_frame_counts:
        for frame in range(count):
            rows.append({
                "row_index": str(len(rows)),
                "source_dataset": "unit_test_dataset",
                "source_video_id": video_id,
                "source_frame_index": str(frame * 5),
                "target": "0",
                "label": label,
            })
    return rows


def _write_fold_root(root: Path, train_rows: list[dict], validation_rows: list[dict]) -> None:
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class PlanRoiFramesTests(unittest.TestCase):
    def test_train_sampled_and_validation_kept_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fold1"
            train_rows = _cache_rows([("v0", "Abuse", 40)])
            validation_rows = _cache_rows([("v1", "Abuse", 3)])
            _write_fold_root(root, train_rows, validation_rows)

            plan = plan_roi_frames([root], frames_per_video=8, sampling_seed=0)

            train_frames = {frame for video, frame in plan if video == "v0"}
            validation_frames = {frame for video, frame in plan if video == "v1"}
            self.assertEqual(len(train_frames), 8)
            self.assertEqual(validation_frames, {0, 5, 10})

    def test_dedup_across_folds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root1 = Path(tmp) / "fold1"
            root2 = Path(tmp) / "fold2"
            # Same video/frame appears as validation in fold1 and train in fold2.
            shared_validation = _cache_rows([("shared", "Abuse", 3)])
            _write_fold_root(root1, _cache_rows([("v0", "Abuse", 5)]), shared_validation)
            _write_fold_root(root2, shared_validation, _cache_rows([("v2", "Abuse", 2)]))

            plan = plan_roi_frames([root1, root2], frames_per_video=8, sampling_seed=0)

            shared_count = sum(1 for video, _frame in plan if video == "shared")
            self.assertEqual(shared_count, 3)  # not 6 -- deduplicated, not doubled

    def test_result_is_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fold1"
            _write_fold_root(root, _cache_rows([("b", "Abuse", 3), ("a", "Abuse", 3)]), _cache_rows([("c", "Abuse", 1)]))
            plan = plan_roi_frames([root], frames_per_video=8, sampling_seed=0)
            self.assertEqual(plan, sorted(plan))

    def test_short_video_keeps_every_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fold1"
            _write_fold_root(root, _cache_rows([("short", "Abuse", 3)]), _cache_rows([("v", "Abuse", 1)]))
            plan = plan_roi_frames([root], frames_per_video=8, sampling_seed=0)
            train_frames = {frame for video, frame in plan if video == "short"}
            self.assertEqual(train_frames, {0, 5, 10})

    def test_deterministic_across_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fold1"
            _write_fold_root(root, _cache_rows([("v0", "Abuse", 50)]), _cache_rows([("v1", "Abuse", 2)]))
            first = plan_roi_frames([root], frames_per_video=8, sampling_seed=0)
            second = plan_roi_frames([root], frames_per_video=8, sampling_seed=0)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
