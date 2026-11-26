import csv
import unittest
from pathlib import Path

import numpy as np

from core.demo_detector import STEP, MotionTracker, alarm_segments, step_boxes

REPO = Path(__file__).resolve().parents[1]
FEATS = REPO / "results/detect/features"
SAVED = REPO / "results/detect/followup-20261112/ucf-002/test-control-seed0.npz"
HAVE = (FEATS / "dinov2b.float16.npy").exists() and SAVED.exists()


class OverlayHelperTests(unittest.TestCase):
    def test_alarm_segments(self):
        t = [0.0, 0.5, 1.0, 1.5, 2.0]
        self.assertEqual(alarm_segments(t, np.array([0, 1, 1, 0, 1], bool), 0.5),
                         [{"start_s": 0.5, "end_s": 1.5}, {"start_s": 2.0, "end_s": 2.5}])

    def test_step_boxes_union_then_smooth(self):
        frames = [None, (0.1, 0.1, 0.2, 0.2), (0.3, 0.3, 0.4, 0.4)] + [None] * STEP
        boxes = step_boxes(frames, 2)
        self.assertEqual(boxes[0], [0.1, 0.1, 0.4, 0.4])
        self.assertEqual(boxes[1], [0.1, 0.1, 0.4, 0.4])  # no motion: hold the last box

    def test_motion_tracker_finds_moving_square(self):
        tr = MotionTracker()
        a = np.zeros((240, 320, 3), np.uint8)
        b = a.copy()
        b[100:140, 200:260] = 255
        tr(a)
        tr(b)
        self.assertIsNone(tr.boxes[0])
        x0, y0, x1, y1 = tr.boxes[1]
        self.assertTrue(0.55 < x0 < 0.65 and 0.35 < y0 < 0.45 and 0.78 < x1 < 0.86 and 0.55 < y1 < 0.63, tr.boxes[1])


@unittest.skipUnless(HAVE, "detector features are local-only")
class DetectorParityTests(unittest.TestCase):
    def test_reproduces_saved_official_test_scores(self):
        from core.demo_detector import Detector

        det = Detector(REPO)
        rel = "Explosion/Explosion021_x264.mp4"
        with (FEATS / "rows.csv").open() as f:
            rows = [int(r["row"]) for r in csv.DictReader(f) if r["relative_path"] == rel]
        b = np.load(FEATS / "dinov2b.float16.npy", mmap_mode="r")[rows].astype(np.float32)
        s = np.load(FEATS / "siglip.float16.npy", mmap_mode="r")[rows].astype(np.float32)
        got = det.scores(b, s)
        want = np.load(SAVED)[rel]
        np.testing.assert_allclose(got, want, atol=1e-5)
        self.assertGreater(det.threshold, 0.5)


if __name__ == "__main__":
    unittest.main()
