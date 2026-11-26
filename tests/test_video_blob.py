"""Tests for the D-34/D-35 video blob read side (`data.video_blob`).

These build a real blob by running the extractor over a synthetic video, then
read it back, so the two sides are tested against each other rather than against
a hand-rolled fixture that could encode the same mistake twice.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import pagecache  # noqa: E402
from data import video_blob  # noqa: E402
from data.video_blob import (  # noqa: E402
    VideoBlobClosure,
    VideoBlobReader,
    reset_verified_blob_cache,
)
from scripts.extract_p1c_video_frames import extract_videos  # noqa: E402


def _write_synthetic_video(
    path: Path, frames_true_rgb: list[np.ndarray], *, fps: float = 30.0
) -> None:
    """Write an MP4 whose decoded frames reproduce ``frames_true_rgb``.

    ``cv2.VideoWriter.write`` treats its input as BGR, so a fixture with known
    *true* RGB colour must be converted on the way in -- mirroring what
    ``VideoCapture.read()`` will hand back when the extractor reads it.
    """
    height, width = frames_true_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():  # pragma: no cover - environment dependent
        raise unittest.SkipTest("cv2.VideoWriter could not open an mp4v writer")
    for frame in frames_true_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


class VideoBlobReaderTests(unittest.TestCase):
    HEIGHT = 48
    WIDTH = 64
    FRAMES = 20
    STRIDE = 5

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        source_root = self.tmp / "Videos"
        (source_root / "Abuse").mkdir(parents=True)

        # Pure red: a BGR/RGB transposition turns this into pure blue, which no
        # structural check would notice.
        red = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        red[:, :, 0] = 255
        _write_synthetic_video(
            source_root / "Abuse" / "Abuse001_x264.mp4", [red] * self.FRAMES
        )

        self.output_dir = self.tmp / "out"
        extract_videos(
            source_root=source_root,
            output_dir=self.output_dir,
            directories=["Abuse"],
            stride=self.STRIDE,
            expected_height=self.HEIGHT,
            expected_width=self.WIDTH,
        )
        self.closure_path = self.output_dir / "extract_closure.json"
        # Each test must see a cold cache, or a digest-mismatch test could
        # pass only because an earlier test already verified the same file.
        reset_verified_blob_cache()
        self.addCleanup(reset_verified_blob_cache)

    def test_closure_loads_and_records_registered_parameters(self) -> None:
        closure = VideoBlobClosure.load(self.closure_path)
        self.assertEqual(closure.stride, self.STRIDE)
        self.assertEqual(closure.jpeg_quality, 95)
        self.assertEqual(closure.color_order, "RGB")
        self.assertEqual(
            closure.expected_image_shape, (self.HEIGHT, self.WIDTH, 3)
        )
        self.assertEqual(closure.video_count, 1)

    def test_frames_decode_as_rgb_not_bgr(self) -> None:
        reader = VideoBlobReader(self.closure_path)
        frame = reader.frame(0)
        self.assertEqual(frame.shape, (self.HEIGHT, self.WIDTH, 3))
        self.assertEqual(frame.dtype, np.uint8)
        mean_r, mean_g, mean_b = frame.reshape(-1, 3).mean(axis=0)
        # Red dominant, blue negligible. Reversed if a channel swap crept in.
        self.assertGreater(mean_r, 200, f"red channel too low: {mean_r}")
        self.assertLess(mean_b, 60, f"blue channel too high: {mean_b}")
        self.assertGreater(mean_r, mean_b + 100)

    def test_row_count_and_stride_agree_with_provenance(self) -> None:
        reader = VideoBlobReader(self.closure_path)
        expected_rows = -(-self.FRAMES // self.STRIDE)  # ceil
        self.assertEqual(len(reader), expected_rows)
        observed = [reader.provenance(i)["source_frame_index"] for i in range(len(reader))]
        self.assertEqual(observed, list(range(0, self.FRAMES, self.STRIDE)))

    def test_payload_slices_are_contiguous_and_cover_the_blob(self) -> None:
        reader = VideoBlobReader(self.closure_path)
        offsets = reader.index["byte_offset"]
        lengths = reader.index["byte_length"]
        for i in range(len(reader) - 1):
            self.assertEqual(int(offsets[i]) + int(lengths[i]), int(offsets[i + 1]))
        self.assertEqual(
            int(offsets[-1]) + int(lengths[-1]), reader.closure.blob_bytes
        )

    def test_rows_for_video_returns_every_row_of_the_only_video(self) -> None:
        reader = VideoBlobReader(self.closure_path)
        rows = reader.rows_for_video(0)
        self.assertEqual(rows.tolist(), list(range(len(reader))))

    def test_digest_mismatch_is_refused(self) -> None:
        closure = json.loads(self.closure_path.read_text(encoding="utf-8"))
        closure["blob_sha256"] = "0" * 64
        self.closure_path.write_text(json.dumps(closure), encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            VideoBlobReader(self.closure_path)
        self.assertIn("sha256", str(caught.exception))

    def test_truncated_blob_is_refused_before_digest_work(self) -> None:
        blob = self.output_dir / "frames.jpegblob"
        with blob.open("r+b") as handle:
            handle.truncate(blob.stat().st_size - 16)
        with self.assertRaises(ValueError) as caught:
            VideoBlobReader(self.closure_path, verify_digest=False)
        self.assertIn("bytes", str(caught.exception))

    def test_non_rgb_color_order_is_refused(self) -> None:
        closure = json.loads(self.closure_path.read_text(encoding="utf-8"))
        closure["color_order"] = "BGR"
        self.closure_path.write_text(json.dumps(closure), encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            VideoBlobClosure.load(self.closure_path)
        self.assertIn("color_order", str(caught.exception))

    def test_second_reader_reuses_the_verification_but_a_changed_blob_does_not(self) -> None:
        VideoBlobReader(self.closure_path)  # first verification
        VideoBlobReader(self.closure_path)  # cached, must still succeed
        # Corrupting the blob changes its identity key, so verification reruns
        # and fails -- the cache must not mask a changed file.
        blob = self.output_dir / "frames.jpegblob"
        data = bytearray(blob.read_bytes())
        data[-1] ^= 0xFF
        blob.write_bytes(bytes(data))
        with self.assertRaises(ValueError) as caught:
            VideoBlobReader(self.closure_path)
        self.assertIn("sha256", str(caught.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class Sha256FileTests(unittest.TestCase):
    """`_sha256_file` drops page cache as it reads; the digest must not change.

    The drop hints fire every `pagecache.DROP_INTERVAL_BYTES`, which is 64 MiB in
    production. These tests shrink it so a small fixture crosses the boundary
    many times, exercising the hinting path rather than skipping straight to
    the final flush.
    """

    def _roundtrip(self, payload: bytes, *, interval: int, chunk: int) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            path.write_bytes(payload)
            original = pagecache.DROP_INTERVAL_BYTES
            pagecache.DROP_INTERVAL_BYTES = interval
            try:
                return video_blob._sha256_file(path, chunk_bytes=chunk)
            finally:
                pagecache.DROP_INTERVAL_BYTES = original

    def test_digest_matches_hashlib_across_chunk_and_interval_sizes(self) -> None:
        payload = bytes(random.Random(0).getrandbits(8) for _ in range(100_000))
        expected = hashlib.sha256(payload).hexdigest()
        # Interval smaller than, equal to, and larger than the read chunk, plus
        # sizes that do and do not divide the payload evenly.
        for interval, chunk in ((1, 1), (512, 1024), (4096, 4096), (7919, 512), (1 << 20, 1 << 20)):
            with self.subTest(interval=interval, chunk=chunk):
                self.assertEqual(self._roundtrip(payload, interval=interval, chunk=chunk), expected)

    def test_empty_file_hashes_without_a_trailing_hint(self) -> None:
        self.assertEqual(
            self._roundtrip(b"", interval=64, chunk=64),
            hashlib.sha256(b"").hexdigest(),
        )
