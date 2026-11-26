from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.extract_p1c_video_frames import (
    decode_jpeg_bytes,
    discover_videos,
    extract_videos,
    read_frame_record,
)


def _write_synthetic_video(path: Path, frames_true_rgb: list[np.ndarray], *, fps: float = 30.0) -> None:
    """Write a tiny MP4 whose decoded frames reproduce ``frames_true_rgb``.

    ``cv2.VideoWriter.write`` follows the same universal OpenCV convention as
    ``imwrite``/``imencode``: it treats the array handed to it as BGR. To make
    a fixture whose *true* colour (in the R, G, B sense) is known, each frame
    is converted from true RGB to BGR before being written, mirroring exactly
    what ``cv2.VideoCapture.read()`` will hand back (BGR) when the extractor
    later reads it.
    """

    height, width = frames_true_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"test fixture VideoWriter failed to open: {path}")
    try:
        for frame_rgb in frames_true_rgb:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _solid_frame(height: int, width: int, rgb: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = rgb[0]
    frame[..., 1] = rgb[1]
    frame[..., 2] = rgb[2]
    return frame


def _noisy_frame(height: int, width: int, seed: int) -> np.ndarray:
    """A per-pixel pseudo-random frame so offset/index bugs cannot hide behind a flat colour."""

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _read_frame_index(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _read_videos_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class DiscoverVideosTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "Abuse").mkdir()
        (self.root / "Arrest").mkdir()
        (self.root / "Abuse" / "Abuse002_x264.mp4").write_bytes(b"")
        (self.root / "Abuse" / "Abuse001_x264.mp4").write_bytes(b"")
        (self.root / "Arrest" / "Arrest001_x264.mp4").write_bytes(b"")
        (self.root / "Arrest" / "notes.txt").write_bytes(b"")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_discovery_is_sorted_by_directory_then_filename_and_ignores_non_mp4(self) -> None:
        videos = discover_videos(self.root, None)
        self.assertEqual(
            [video.relative_path for video in videos],
            ["Abuse/Abuse001_x264.mp4", "Abuse/Abuse002_x264.mp4", "Arrest/Arrest001_x264.mp4"],
        )

    def test_directory_order_given_on_the_command_line_does_not_affect_discovery_order(self) -> None:
        forward = discover_videos(self.root, ["Abuse", "Arrest"])
        backward = discover_videos(self.root, ["Arrest", "Abuse"])
        self.assertEqual([v.relative_path for v in forward], [v.relative_path for v in backward])

    def test_missing_directory_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "configured source directory is missing"):
            discover_videos(self.root, ["DoesNotExist"])


class ExtractVideosTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_root = self.root / "corpus"
        (self.source_root / "Abuse").mkdir(parents=True)
        (self.source_root / "Arrest").mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_round_trip_and_channel_order_is_rgb_not_bgr(self) -> None:
        # A pure, strongly asymmetric colour: real red is (255, 0, 0). A BGR/RGB
        # swap anywhere in the pipeline would turn this into real blue instead,
        # which the assertions below would catch.
        true_red = _solid_frame(12, 16, (255, 0, 0))
        video_path = self.source_root / "Abuse" / "Abuse001_x264.mp4"
        _write_synthetic_video(video_path, [true_red] * 3)

        output_dir = self.root / "out"
        closure = extract_videos(
            source_root=self.source_root,
            directories=["Abuse"],
            output_dir=output_dir,
            stride=1,
            jpeg_quality=95,
            expected_height=12,
            expected_width=16,
        )
        self.assertIsNotNone(closure)
        self.assertEqual(closure["color_order"], "RGB")

        frame_index = _read_frame_index(output_dir / "frame_index.npy")
        self.assertEqual(len(frame_index), 3)
        blob_path = output_dir / "frames.jpegblob"
        decoded = read_frame_record(
            blob_path,
            byte_offset=int(frame_index[0]["byte_offset"]),
            byte_length=int(frame_index[0]["byte_length"]),
        )
        mean = decoded.reshape(-1, 3).mean(axis=0)
        # Real red decoded as RGB: channel 0 (R) high, channels 1/2 (G, B) low.
        self.assertGreater(mean[0], 200.0)
        self.assertLess(mean[1], 15.0)
        self.assertLess(mean[2], 15.0)
        # A JPEG q95 round-trip of a flat colour is close but not required to
        # be bit-identical; this is a tolerance check, not an equality check.
        self.assertTrue(np.all(np.abs(decoded.astype(int) - true_red.astype(int)) <= 12))

    def test_stride_is_honoured(self) -> None:
        frames = [_noisy_frame(8, 8, seed=index) for index in range(23)]
        video_path = self.source_root / "Abuse" / "Abuse001_x264.mp4"
        _write_synthetic_video(video_path, frames)

        output_dir = self.root / "out"
        extract_videos(
            source_root=self.source_root,
            directories=["Abuse"],
            output_dir=output_dir,
            stride=5,
            expected_height=8,
            expected_width=8,
        )
        frame_index = _read_frame_index(output_dir / "frame_index.npy")
        videos = _read_videos_csv(output_dir / "videos.csv")
        decoded_frame_count = int(videos[0]["decoded_frame_count"])
        self.assertEqual(list(frame_index["source_frame_index"]), list(range(0, decoded_frame_count, 5)))
        self.assertEqual(int(videos[0]["kept_frame_count"]), math.ceil(decoded_frame_count / 5))
        self.assertEqual(len(frame_index), math.ceil(decoded_frame_count / 5))

    def test_offset_index_is_contiguous_and_consistent(self) -> None:
        frames_a = [_noisy_frame(10, 10, seed=100 + index) for index in range(7)]
        frames_b = [_noisy_frame(10, 10, seed=200 + index) for index in range(4)]
        _write_synthetic_video(self.source_root / "Abuse" / "Abuse001_x264.mp4", frames_a)
        _write_synthetic_video(self.source_root / "Arrest" / "Arrest001_x264.mp4", frames_b)

        output_dir = self.root / "out"
        extract_videos(
            source_root=self.source_root,
            directories=["Abuse", "Arrest"],
            output_dir=output_dir,
            stride=2,
            expected_height=10,
            expected_width=10,
        )
        frame_index = _read_frame_index(output_dir / "frame_index.npy")
        offsets = frame_index["byte_offset"]
        lengths = frame_index["byte_length"]
        self.assertTrue(np.all(np.diff(offsets) >= 0))
        self.assertTrue(np.array_equal(offsets[:-1] + lengths[:-1], offsets[1:]))
        blob_bytes = (output_dir / "frames.jpegblob").stat().st_size
        self.assertEqual(int(offsets[-1] + lengths[-1]), blob_bytes)

    def test_resuming_across_two_invocations_matches_a_single_shot_run(self) -> None:
        frames_a = [_noisy_frame(8, 8, seed=10 + index) for index in range(6)]
        frames_b = [_noisy_frame(8, 8, seed=20 + index) for index in range(5)]
        _write_synthetic_video(self.source_root / "Abuse" / "Abuse001_x264.mp4", frames_a)
        _write_synthetic_video(self.source_root / "Arrest" / "Arrest001_x264.mp4", frames_b)

        resumed_dir = self.root / "resumed"
        first_call = extract_videos(
            source_root=self.source_root,
            directories=["Abuse", "Arrest"],
            output_dir=resumed_dir,
            stride=3,
            expected_height=8,
            expected_width=8,
            max_new_videos=1,
        )
        self.assertIsNone(first_call)  # interrupted: one of two videos done, no closure yet
        self.assertFalse((resumed_dir / "extract_closure.json").is_file())
        second_call = extract_videos(
            source_root=self.source_root,
            directories=["Abuse", "Arrest"],
            output_dir=resumed_dir,
            stride=3,
            expected_height=8,
            expected_width=8,
            max_new_videos=1,
        )
        self.assertIsNotNone(second_call)

        single_shot_dir = self.root / "single_shot"
        extract_videos(
            source_root=self.source_root,
            directories=["Abuse", "Arrest"],
            output_dir=single_shot_dir,
            stride=3,
            expected_height=8,
            expected_width=8,
        )

        self.assertEqual(
            (resumed_dir / "frames.jpegblob").read_bytes(),
            (single_shot_dir / "frames.jpegblob").read_bytes(),
        )
        resumed_index = _read_frame_index(resumed_dir / "frame_index.npy")
        single_shot_index = _read_frame_index(single_shot_dir / "frame_index.npy")
        self.assertTrue(np.array_equal(resumed_index, single_shot_index))
        resumed_videos = _read_videos_csv(resumed_dir / "videos.csv")
        single_shot_videos = _read_videos_csv(single_shot_dir / "videos.csv")
        for row_resumed, row_single in zip(resumed_videos, single_shot_videos):
            for key in row_resumed:
                self.assertEqual(row_resumed[key], row_single[key], key)

    def test_closure_refuses_to_overwrite_an_existing_closure(self) -> None:
        _write_synthetic_video(self.source_root / "Abuse" / "Abuse001_x264.mp4", [_noisy_frame(6, 6, seed=1)])
        output_dir = self.root / "out"
        extract_videos(
            source_root=self.source_root,
            directories=["Abuse"],
            output_dir=output_dir,
            stride=1,
            expected_height=6,
            expected_width=6,
        )
        with self.assertRaisesRegex(FileExistsError, "already closed"):
            extract_videos(
                source_root=self.source_root,
                directories=["Abuse"],
                output_dir=output_dir,
                stride=1,
                expected_height=6,
                expected_width=6,
            )

    def test_geometry_mismatch_against_corpus_wide_expectation_aborts_by_default(self) -> None:
        _write_synthetic_video(self.source_root / "Abuse" / "Abuse001_x264.mp4", [_noisy_frame(6, 6, seed=1)])
        output_dir = self.root / "out"
        with self.assertRaisesRegex(ValueError, "differs from the corpus-wide expectation"):
            extract_videos(
                source_root=self.source_root,
                directories=["Abuse"],
                output_dir=output_dir,
                stride=1,
                expected_height=99,
                expected_width=99,
            )

    def test_geometry_mismatch_is_recorded_and_continues_when_allowed(self) -> None:
        _write_synthetic_video(self.source_root / "Abuse" / "Abuse001_x264.mp4", [_noisy_frame(6, 6, seed=1)])
        output_dir = self.root / "out"
        closure = extract_videos(
            source_root=self.source_root,
            directories=["Abuse"],
            output_dir=output_dir,
            stride=1,
            expected_height=99,
            expected_width=99,
            allow_heterogeneous_geometry=True,
        )
        self.assertEqual(len(closure["heterogeneous_geometry_videos"]), 1)
        videos = _read_videos_csv(output_dir / "videos.csv")
        self.assertEqual(videos[0]["decode_status"], "heterogeneous_geometry")

    def test_decode_helpers_round_trip_a_known_image(self) -> None:
        array = _solid_frame(4, 4, (10, 20, 30))
        from scripts.extract_p1c_video_frames import encode_jpeg_rgb

        payload = encode_jpeg_rgb(array, quality=95)
        decoded = decode_jpeg_bytes(payload)
        self.assertEqual(decoded.shape, array.shape)
        self.assertTrue(np.all(np.abs(decoded.astype(int) - array.astype(int)) <= 6))


if __name__ == "__main__":
    unittest.main()
