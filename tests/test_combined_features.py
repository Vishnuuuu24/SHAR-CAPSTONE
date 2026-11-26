import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core import combined_features as cf


def _read(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def write_video(path: Path, n_frames: int, bgr=(0, 0, 255)) -> None:
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(n_frames):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:] = bgr
        frame[:, : min(63, i * 3)] = (i * 10 % 255, 255, 0)  # moving edge so frames differ
        w.write(frame)
    w.release()


class PositionTests(unittest.TestCase):
    def test_segment_centres(self):
        self.assertEqual(cf.frame_positions(80), [5, 15, 25, 35, 45, 55, 65, 75])

    def test_short_video_repeats_but_stays_in_range(self):
        pos = cf.frame_positions(3)
        self.assertEqual(len(pos), 8)
        self.assertEqual(pos, sorted(pos))
        self.assertTrue(0 <= min(pos) and max(pos) <= 2)
        self.assertEqual(cf.frame_positions(1), [0] * 8)

    def test_empty_video_raises(self):
        with self.assertRaises(ValueError):
            cf.frame_positions(0)


class DecodeTests(unittest.TestCase):
    def test_decode_shape_rgb_and_resize(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.mp4"
            write_video(p, 40)
            frames, pos, n = cf.decode_uniform_frames(p)
            self.assertEqual(n, 40)
            self.assertEqual(frames.shape, (8, 240, 320, 3))
            self.assertEqual(frames.dtype, np.uint8)
            self.assertEqual(pos, cf.frame_positions(40))
            # the right edge of every frame is the pure-red BGR background -> R channel high after BGR->RGB (K-05)
            right = frames[0, :, -1].mean(0)  # frame 0 of 8: edge has barely moved
            self.assertGreater(right[0], 200)
            self.assertLess(right[2], 60)

    def test_short_video_gives_eight_frames(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "short.mp4"
            write_video(p, 3)
            frames, pos, n = cf.decode_uniform_frames(p)
            self.assertEqual((frames.shape[0], len(pos), n), (8, 8, 3))

    def test_corrupt_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.mp4"
            p.write_bytes(b"not a video" * 50)
            with self.assertRaises(Exception):
                cf.decode_uniform_frames(p)


class SelectionTests(unittest.TestCase):
    def test_limit_is_round_robin(self):
        rows = [{"source_dataset": s, "i": i} for s in "abc" for i in range(3)]
        got = cf.select_rows(rows, 5)
        self.assertEqual([r["source_dataset"] for r in got], ["a", "b", "c", "a", "b"])
        self.assertEqual(len(cf.select_rows(rows, None)), 9)

    def test_cls_plus_patches_drops_registers(self):
        import torch

        tokens = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)  # CLS + 4 registers + 2 patches
        out = cf.cls_plus_patches(tokens, 5)
        self.assertEqual(tuple(out.shape), (2, 3, 3))
        self.assertTrue(torch.equal(out[:, 0], tokens[:, 0]))
        self.assertTrue(torch.equal(out[:, 1:], tokens[:, 5:]))


class ExtractCacheTests(unittest.TestCase):
    def test_failures_are_recorded_and_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = []
            for i, name in enumerate(["a", "bad", "c"]):
                p = d / f"{name}.mp4"
                if name == "bad":
                    p.write_bytes(b"garbage" * 100)
                else:
                    write_video(p, 24)
                rows.append({"source_dataset": "cuva", "source_video_id": name, "unified_label": "Fighting",
                             "split": "test", "fold": "", "original_abs_path": str(p)})
            rows.append({**rows[0], "source_video_id": "missing", "original_abs_path": str(d / "nope.mp4")})
            calls = []

            def forward(frames):
                calls.append(frames.shape)
                return np.ones((frames.shape[0], 5, 4), np.float32)

            out = d / "cache"
            res = cf.extract_cache(rows, out, forward, token_shape=(5, 4), batch_videos=2, num_workers=0, log=lambda s: None)
            self.assertEqual((res["n_ok"], res["n_failed"]), (2, 2))
            feats = np.load(out / "features.float16.npy")
            self.assertEqual(feats.shape, (4, 8, 5, 4))
            self.assertTrue(np.isfinite(feats.astype(np.float32)).all())
            self.assertTrue((feats[0] == 1).all() and (feats[2] == 1).all())
            self.assertTrue((feats[1] == 0).all() and (feats[3] == 0).all())  # failed rows stay zero
            fails = _read(out / "failures.csv")
            self.assertEqual([f["source_video_id"] for f in fails], ["bad", "missing"])
            status = _read(out / "rows.csv")
            self.assertEqual([r["ok"] for r in status], ["1", "0", "1", "0"])
            self.assertEqual(status[0]["frame_indices"], " ".join(map(str, cf.frame_positions(24))))
            self.assertTrue(all(s[1:] == (240, 320, 3) for s in calls))

    def test_non_finite_output_refused(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_video(d / "a.mp4", 10)
            rows = [{"source_dataset": "cuva", "source_video_id": "a", "unified_label": "", "split": "x", "fold": "",
                     "original_abs_path": str(d / "a.mp4")}]
            with self.assertRaises(RuntimeError):
                cf.extract_cache(rows, d / "c", lambda f: np.full((f.shape[0], 5, 4), np.nan), token_shape=(5, 4), log=lambda s: None)


if __name__ == "__main__":
    unittest.main()
