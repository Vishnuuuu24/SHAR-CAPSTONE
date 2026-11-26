"""Strict source-video temporal-clip planning for the P1C CNN-LSTM control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from data.manifest import ManifestRow


TargetPolicy = Literal["video_inherited_last_frame"]


@dataclass(frozen=True)
class TemporalClip:
    """One fixed-length, single-source-video clip with explicit padded frames."""

    source_dataset: str
    source_video_id: str
    split: str
    label: str
    rows: tuple[ManifestRow, ...]
    length: int
    target_policy: TargetPolicy


def build_temporal_clips(
    rows: Iterable[ManifestRow],
    *,
    clip_length: int,
    stride: int,
    target_policy: TargetPolicy,
) -> tuple[TemporalClip, ...]:
    """Create deterministic contiguous clips without crossing source/split edges.

    The final partial window is retained and padded by repeating its final
    actual frame. ``length`` preserves the unpadded count for LSTM gathering.
    No choice of clip length, stride, or target policy is implicit.
    """

    if isinstance(clip_length, bool) or not isinstance(clip_length, int) or clip_length <= 0:
        raise ValueError("clip_length must be a positive integer")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    if target_policy != "video_inherited_last_frame":
        raise ValueError("target_policy must be video_inherited_last_frame")
    grouped: dict[tuple[str, str, str], list[ManifestRow]] = {}
    for row in rows:
        if row.split not in {"train", "validation"}:
            raise ValueError(f"temporal clips accept only Train/validation rows: {row.filepath}")
        if row.source_frame_index is None:
            raise ValueError(f"temporal row lacks source-frame index: {row.filepath}")
        grouped.setdefault((row.source_dataset, row.source_video_id, row.split), []).append(row)
    if not grouped:
        raise ValueError("cannot build temporal clips from no rows")

    clips: list[TemporalClip] = []
    for (dataset, video, split), group in sorted(grouped.items()):
        ordered = tuple(sorted(group, key=lambda row: (row.source_frame_index, row.filepath, row.file_digest)))
        labels = {row.label for row in ordered}
        if len(labels) != 1:
            raise ValueError(f"source-video group has multiple labels: {(dataset, video, split)}")
        indices = [row.source_frame_index for row in ordered]
        if len(set(indices)) != len(indices):
            raise ValueError(f"source-video group has duplicate frame indices: {(dataset, video, split)}")
        if any(current != previous + 1 for previous, current in zip(indices, indices[1:])):
            raise ValueError(f"source-video group has non-contiguous frame indices: {(dataset, video, split)}")
        for start in range(0, len(ordered), stride):
            actual = ordered[start : start + clip_length]
            if not actual:
                continue
            padded = actual + (actual[-1],) * (clip_length - len(actual))
            clips.append(
                TemporalClip(
                    source_dataset=dataset,
                    source_video_id=video,
                    split=split,
                    label=actual[-1].label,
                    rows=padded,
                    length=len(actual),
                    target_policy=target_policy,
                )
            )
    return tuple(clips)
