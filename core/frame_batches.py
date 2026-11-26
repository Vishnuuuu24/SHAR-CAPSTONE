"""Registered-preprocessing image batches straight from the D-34 frame blob (shared by D-65/D-67 extractors).

`FrameBatcher(rows)` decodes blob rows on a thread pool and applies the registered preprocessing (224 bilinear,
ImageNet normalisation, via `core.p1c_roi.preprocess_crop`), returning one `[N, 3, 224, 224]` float tensor.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEOS_CSV = "results/p1c/p1c-video-frames/videos.csv"
FRAME_INDEX = "results/p1c/p1c-video-frames/frame_index.npy"


class FrameBatcher:
    def __init__(self, threads: int = 8, repo_root: Path = REPO_ROOT):
        import yaml

        from core.validation_frames import CLOSURE
        from data.p1c_loader import FramePreprocessing
        from data.video_blob import VideoBlobReader
        from scripts.build_combined_patch_cache import HEADLINE_CONFIG

        protocol = yaml.safe_load(HEADLINE_CONFIG.read_text())["fixed_protocol"]
        self.preprocessing = FramePreprocessing.from_config(protocol["preprocessing"])
        self.reader = VideoBlobReader(repo_root / CLOSURE, verify_digest=False)
        self.pool = ThreadPoolExecutor(threads)
        self.repo_root = repo_root

    def blob_rows(self, keys) -> np.ndarray:
        from core.backbone_cache import blob_rows_for

        return blob_rows_for(keys, self.repo_root / VIDEOS_CSV, np.load(self.repo_root / FRAME_INDEX, mmap_mode="r"))

    def __call__(self, rows: np.ndarray):
        import torch
        from PIL import Image

        from core.p1c_roi import preprocess_crop

        frames = list(self.pool.map(self.reader.frame, np.asarray(rows).tolist()))
        return torch.stack(list(self.pool.map(lambda fr: preprocess_crop(Image.fromarray(fr, mode="RGB"), self.preprocessing), frames)))
