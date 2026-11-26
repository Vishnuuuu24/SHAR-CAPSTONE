#!/usr/bin/env python3
"""Build the frozen V-JEPA 2 ViT-L clip-feature cache for one fold and split.

Thin wrapper over `scripts/build_p1c_clip_cache.py`: same 16-row clip tiling, same registered 224 preprocessing, same
rows.csv provenance, so a V-JEPA 2 row is directly comparable to the VideoMAE row for the same clip. The backbone is
registered in `build_p1c_feature_cache._BACKBONE_LOADERS`; this wrapper adds two switches:

    --frame-order shuffled   permutes the 16 frames of every clip with a fixed per-clip seed. A cache built this way,
                             probed identically, shows how much of the score survives without temporal order.
    --train-clips-per-video N  (default 8; 0 = every clip) for the train split only, keep the N temporally spread clips the
                             probe actually trains on (`sample_rows_per_video`, the same call the probe makes), about
                             a quarter of the cost of tiling everything. The validation split is always complete.

Everything else is `build_p1c_clip_cache.py`'s own argument list (`--config --split --output-dir ...`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.build_p1c_clip_cache as clip_cache
from core.p1c_video_sampling import sample_rows_per_video

SHUFFLE_SEED = 20261105


def shuffled_order(clip_index: int, length: int = 16) -> np.ndarray:
    """The fixed frame permutation for one clip; deterministic in the clip's row index and `SHUFFLE_SEED`."""

    return np.random.RandomState(SHUFFLE_SEED + clip_index).permutation(length)


class _ShuffledClipDataset(clip_cache._ClipDataset):  # noqa: SLF001
    def __getitem__(self, k):
        return super().__getitem__(k)[torch.from_numpy(shuffled_order(k))]


def select_spread(clips, per_video: int):
    """Keep `per_video` temporally spread clips of each video, exactly as the probe's `sample_rows_per_video` call does."""

    rows = [{"source_dataset": "ucf_crime_official_video_frames", "source_video_id": c[1], "source_frame_index": c[2]} for c in clips]
    return [clips[i] for i in sample_rows_per_video(rows, per_video)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    parser.add_argument("--frame-order", choices=["natural", "shuffled"], default="natural")
    parser.add_argument("--train-clips-per-video", type=int, default=8)
    mine, rest = parser.parse_known_args(argv)
    if "-h" in rest or "--help" in rest:
        parser.print_help()
        return 0
    if mine.frame_order == "shuffled":
        clip_cache._ClipDataset = _ShuffledClipDataset  # noqa: SLF001
    split = rest[rest.index("--split") + 1]
    if split == "train" and mine.train_clips_per_video > 0:
        tile_all = clip_cache._tile  # noqa: SLF001
        clip_cache._tile = lambda ds: select_spread(tile_all(ds), mine.train_clips_per_video)  # noqa: SLF001
    code = clip_cache.main(rest)
    out = Path(rest[rest.index("--output-dir") + 1])
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frame_order"] = mine.frame_order
    manifest["train_clips_per_video"] = mine.train_clips_per_video if split == "train" else "all"
    if mine.frame_order == "shuffled":
        manifest["shuffle_seed"] = SHUFFLE_SEED
        manifest["verification"]["note"] = ("recomputed in natural order by the shared builder, so max_abs_delta is NOT a "
                                            "reproducibility check for this cache")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
