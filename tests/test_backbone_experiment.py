import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from core.backbone_cache import align, blob_rows_for, frame_key, union_keys
from core.backbone_experiment import ARMS, CONTRAST, inner_monitor_videos
from core.p1c_heads import ProjectedStepHead


class ProjectedHead(unittest.TestCase):
    def test_step_head_init_is_independent_of_input_width(self):
        torch.manual_seed(3)
        b = ProjectedStepHead(1536, frames=8, num_classes=14, projection_seed=10003)
        torch.manual_seed(3)
        l = ProjectedStepHead(2048, frames=8, num_classes=14, projection_seed=10003)
        for (nb, pb), (nl, pl) in zip(b.step.named_parameters(), l.step.named_parameters()):
            self.assertEqual(nb, nl)
            self.assertTrue(torch.equal(pb, pl), msg=nb)
        self.assertEqual(tuple(b.projection.weight.shape), (1536, 1536))
        self.assertEqual(tuple(l.projection.weight.shape), (1536, 2048))

    def test_forward_shape_and_mask(self):
        torch.manual_seed(0)
        h = ProjectedStepHead(2048, frames=8, num_classes=14)
        x = torch.rand(3, 8, 2048)
        mask = torch.zeros(3, 8, dtype=torch.bool)
        mask[0, 6:] = True
        self.assertEqual(tuple(h(x, key_padding_mask=mask).shape), (3, 14))

    def test_projection_seed_is_deterministic(self):
        a = ProjectedStepHead(64, width=32, projection_seed=5)
        b = ProjectedStepHead(64, width=32, projection_seed=5)
        self.assertTrue(torch.equal(a.projection.weight, b.projection.weight))


class Monitor(unittest.TestCase):
    def test_fixed_stratified_and_refuses_small_classes(self):
        targets = {f"v{i}": i % 3 for i in range(30)}
        m = inner_monitor_videos(targets, 4, seed=2)
        self.assertEqual(len(m), 12)
        self.assertEqual(sorted(sum(1 for v in m if targets[v] == c) for c in range(3)), [4, 4, 4])
        self.assertEqual(m, inner_monitor_videos(targets, 4, seed=2))
        with self.assertRaises(ValueError):
            inner_monitor_videos({"a": 0, "b": 0, "c": 1}, 2, seed=0)

    def test_registered_arms(self):
        self.assertEqual(CONTRAST, ("L_proj", "B_proj"))
        self.assertTrue(set(CONTRAST) <= set(ARMS))


class FrameBookkeeping(unittest.TestCase):
    def test_union_blob_rows_and_alignment(self):
        rows_a = [{"source_video_id": "V1", "source_frame_index": 5}, {"source_video_id": "V2", "source_frame_index": 0}]
        rows_b = [{"source_video_id": "V1", "source_frame_index": 5}, {"source_video_id": "V1", "source_frame_index": 10}]
        keys = union_keys([rows_a, rows_b])
        self.assertEqual(keys, [("V1", 5), ("V1", 10), ("V2", 0)])
        fi = np.zeros(20, dtype=[("byte_offset", "<i8"), ("byte_length", "<i8"), ("video_id", "<i8"), ("source_frame_index", "<i8")])
        fi["source_frame_index"][4:7] = [0, 5, 10]     # V1 at rows 4..6
        fi["source_frame_index"][10:12] = [0, 5]       # V2 at rows 10..11
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "videos.csv"
            with p.open("w", newline="") as h:
                w = csv.writer(h)
                w.writerow(["file_name", "first_row_index", "kept_frame_count"])
                w.writerow(["V1.mp4", 4, 3])
                w.writerow(["V2.mp4", 10, 2])
            self.assertEqual(blob_rows_for(keys, p, fi).tolist(), [5, 6, 10])
        feats = np.arange(3)[:, None] * np.ones((1, 4))
        np.testing.assert_array_equal(align(feats, keys, rows_b)[:, 0], [0, 1])
        with self.assertRaises(KeyError):
            align(feats, keys, [{"source_video_id": "V9", "source_frame_index": 0}])
        self.assertEqual(frame_key({"source_video_id": "V1", "source_frame_index": "5"}), ("V1", 5))


if __name__ == "__main__":
    unittest.main()
