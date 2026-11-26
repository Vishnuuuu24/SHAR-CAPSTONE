"""Training-set variation for the D-53 attention head: per-epoch frame resampling and class-stratified video subsets.

The registered head trains on one fixed `sample_rows_per_video(seed=0)` draw of 8 frames per video (about 1% of the
cached frames). `make_epoch_resampler` redraws a different 8-frame bag per video every epoch from the same cache.
`stratified_video_subset` keeps a fraction of each class's training videos, for the learning-curve diagnostic.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from core.p1c_head_variants import _pad_sequence, group_pooled_sequences_by_video
from core.p1c_video_sampling import sample_rows_per_video


def make_epoch_resampler(feats: np.ndarray, rows: Sequence[Mapping[str, Any]], *, frames: int, seed: int,
                         keep_videos: set[str] | None = None) -> Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """`resampler(epoch) -> (sequences[V,frames,D] float32, masks[V,frames] bool, targets[V] int64)`.

    Deterministic in (`seed`, `epoch`): each epoch draws every video's `frames` frames with the project's seeded
    temporally-spread sampler, then shuffles the frame order inside each video (the STEP-permuted control, as in
    `group_pooled_sequences_by_video(permute_frame_order=True)`). `keep_videos` restricts to a video subset.
    """

    if keep_videos is not None:
        picked = np.array([i for i, r in enumerate(rows) if r["source_video_id"] in keep_videos])
        feats, rows = feats[picked], [rows[i] for i in picked]

    def resample(epoch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        draw = int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0] % (2**31 - 1))
        keep = sample_rows_per_video(rows, frames, seed=draw)
        seqs = group_pooled_sequences_by_video(np.asarray(feats[keep], dtype=np.float32), [rows[i] for i in keep],
                                               permute_frame_order=True, permute_seed=draw + 1)
        keys = sorted(seqs)
        padded = [_pad_sequence(seqs[k][0], frames) for k in keys]
        return (np.stack([p[0] for p in padded]), np.stack([p[1] for p in padded]),
                np.array([int(seqs[k][1]["target"]) for k in keys], dtype=np.int64))

    return resample


def stratified_video_subset(video_targets: Mapping[str, int], fraction: float, seed: int) -> set[str]:
    """`ceil(fraction * n)` videos of every class (at least one), chosen by a seeded shuffle within the class."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[str]] = {}
    for video, target in sorted(video_targets.items()):
        by_class.setdefault(int(target), []).append(video)
    keep: set[str] = set()
    for target in sorted(by_class):
        vids = by_class[target]
        take = max(1, int(np.ceil(fraction * len(vids))))
        keep |= {vids[i] for i in rng.permutation(len(vids))[:take]}
    return keep
