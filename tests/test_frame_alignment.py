from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from data.frame_alignment import (
    best_policy_result,
    compare_frame,
    read_kaggle_png,
    read_source_video_frame,
    resize_to_match,
)


def _make_source_video(path: Path, *, frame_count: int, height: int, width: int) -> list[np.ndarray]:
    """Write a synthetic video with a distinct, textured pattern per frame.

    Each frame is deterministic per-frame noise (not a flat color) so SSIM has
    real local structure to discriminate a correct match from a wrong one.
    """

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("test fixture could not open a synthetic VideoWriter")
    frames = []
    for index in range(frame_count):
        rng = np.random.default_rng(index)
        frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        writer.write(frame[:, :, ::-1])  # BGR for OpenCV's writer
        frames.append(frame)
    writer.release()
    return frames


class FrameAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.video_path = Path(self.temp_dir.name) / "Abuse001_x264.mp4"
        self.height, self.width = 240, 320
        _make_source_video(self.video_path, frame_count=20, height=self.height, width=self.width)

    def _kaggle_png_for(self, frame_index: int) -> Path:
        decoded = read_source_video_frame(self.video_path, frame_index)
        resized = resize_to_match(decoded, (64, 64), policy="stretch_to_target", interpolation="area")
        png_path = Path(self.temp_dir.name) / f"Abuse001_x264_{frame_index}.png"
        bgr_uint8 = cv2.cvtColor((resized * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(png_path), bgr_uint8)
        return png_path

    def test_read_source_video_frame_returns_canonical_float32_rgb(self) -> None:
        frame = read_source_video_frame(self.video_path, 0)
        self.assertEqual(frame.dtype, np.float32)
        self.assertEqual(frame.shape, (self.height, self.width, 3))
        self.assertTrue(np.all(frame >= 0.0) and np.all(frame <= 1.0))

    def test_read_source_video_frame_rejects_an_out_of_range_index(self) -> None:
        with self.assertRaises(ValueError):
            read_source_video_frame(self.video_path, 999)

    def test_correctly_matched_frame_scores_near_perfect_similarity(self) -> None:
        png_path = self._kaggle_png_for(7)
        result = compare_frame(
            png_path,
            self.video_path,
            source_video_id="Abuse001_x264",
            source_frame_index=7,
            resize_policy="stretch_to_target",
            interpolation="area",
        )
        self.assertGreater(result.ssim, 0.99)
        self.assertGreater(result.psnr, 40.0)

    def test_misattributed_frame_index_scores_far_lower_similarity(self) -> None:
        # The PNG genuinely is frame 7; comparing it against frame 3's decoded
        # video content is exactly the failure mode a real index-scaling bug
        # would produce, and must score far below the correct match.
        png_path = self._kaggle_png_for(7)
        correct = compare_frame(
            png_path,
            self.video_path,
            source_video_id="Abuse001_x264",
            source_frame_index=7,
            resize_policy="stretch_to_target",
            interpolation="area",
        )
        wrong = compare_frame(
            png_path,
            self.video_path,
            source_video_id="Abuse001_x264",
            source_frame_index=3,
            resize_policy="stretch_to_target",
            interpolation="area",
        )
        self.assertGreater(correct.ssim, wrong.ssim + 0.3)

    def test_best_policy_result_prefers_the_policy_that_actually_matches(self) -> None:
        png_path = self._kaggle_png_for(7)
        best = best_policy_result(
            png_path,
            self.video_path,
            source_video_id="Abuse001_x264",
            source_frame_index=7,
            interpolation="area",
        )
        self.assertEqual(best.resize_policy, "stretch_to_target")
        self.assertGreater(best.ssim, 0.99)

    def test_center_crop_then_resize_preserves_source_aspect_content(self) -> None:
        decoded = read_source_video_frame(self.video_path, 0)
        cropped = resize_to_match(decoded, (64, 64), policy="center_crop_then_resize", interpolation="area")
        self.assertEqual(cropped.shape, (64, 64, 3))
        self.assertTrue(np.all(cropped >= 0.0) and np.all(cropped <= 1.0))

    def test_resize_to_match_rejects_unregistered_policy_or_interpolation(self) -> None:
        decoded = read_source_video_frame(self.video_path, 0)
        with self.assertRaises(ValueError):
            resize_to_match(decoded, (64, 64), policy="nearest_neighbor_guess", interpolation="area")
        with self.assertRaises(ValueError):
            resize_to_match(decoded, (64, 64), policy="stretch_to_target", interpolation="bicubic")

    def test_read_kaggle_png_rejects_a_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_kaggle_png(Path(self.temp_dir.name) / "does_not_exist.png")


if __name__ == "__main__":
    unittest.main()
