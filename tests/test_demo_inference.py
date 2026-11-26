import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.demo_inference import decode_frames, held_out_folds, video_key


class DemoInference(unittest.TestCase):
    def test_video_key(self):
        self.assertEqual(video_key("Abuse013_x264.mp4"), "Abuse013_x264")
        self.assertEqual(video_key("/x/y/Normal_Videos_478_x264.MP4"), "Normal_Videos_478_x264")
        self.assertEqual(video_key("clip"), "clip")

    def test_held_out_folds(self):
        m = {"videos": {"A_x264": {"label": "Abuse", "seen_by_folds": [1, 3]}, "B_x264": {"label": "Arson", "seen_by_folds": [1, 2, 3, 4]}}}
        self.assertEqual(held_out_folds("A_x264", m)[0], [2, 4])
        self.assertEqual(held_out_folds("B_x264", m)[0], [])
        self.assertEqual(held_out_folds("other", m), (None, None))

    def test_decode_stride_and_thinning(self):
        import cv2

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "v.avi"
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
            for i in range(53):
                w.write(np.full((48, 64, 3), i * 4 % 255, dtype=np.uint8))
            w.release()
            frames, idx, fps = decode_frames(path)
            self.assertEqual(idx, list(range(0, 53, 5)))
            self.assertEqual(frames[0].shape, (48, 64, 3))
            self.assertAlmostEqual(fps, 30.0, places=1)
            _f, idx2, _ = decode_frames(path, max_frames=4)
            self.assertEqual(len(idx2), 4)
            self.assertEqual((idx2[0], idx2[-1]), (0, 50))

    def test_unreadable_video_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(b"not a video")
            f.flush()
            with self.assertRaises(ValueError):
                decode_frames(Path(f.name))


if __name__ == "__main__":
    unittest.main()
