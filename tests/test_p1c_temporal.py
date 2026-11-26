from __future__ import annotations

import unittest

from data.manifest import ManifestRow
from data.p1c_temporal import build_temporal_clips


def row(video: str, frame: int, *, split: str = "train", label: str = "Abuse") -> ManifestRow:
    return ManifestRow(
        filepath=f"frames/{video}-{frame}.png",
        source_dataset="fixture",
        source_video_id=video,
        source_frame_index=frame,
        label=label,
        label_scope="video_inherited",
        label_source="fixture",
        annotation_version="fixture-v1",
        inside_official_interval=None,
        split=split,
        file_digest=f"{frame:064x}",
    )


class TemporalClipTests(unittest.TestCase):
    def test_clips_are_contiguous_single_video_and_final_window_is_padded(self) -> None:
        clips = build_temporal_clips(
            [row("video-a", 2), row("video-a", 0), row("video-a", 1), row("video-b", 0, label="Normal")],
            clip_length=2,
            stride=2,
            target_policy="video_inherited_last_frame",
        )
        self.assertEqual([(clip.source_video_id, clip.length) for clip in clips], [("video-a", 2), ("video-a", 1), ("video-b", 1)])
        self.assertEqual([item.source_frame_index for item in clips[0].rows], [0, 1])
        self.assertEqual([item.source_frame_index for item in clips[1].rows], [2, 2])
        self.assertEqual(clips[1].label, "Abuse")

    def test_cross_split_duplicate_and_noncontiguous_groups_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only Train/validation"):
            build_temporal_clips([row("video", 0, split="test")], clip_length=2, stride=1, target_policy="video_inherited_last_frame")
        with self.assertRaisesRegex(ValueError, "duplicate frame"):
            build_temporal_clips([row("video", 0), row("video", 0)], clip_length=2, stride=1, target_policy="video_inherited_last_frame")
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            build_temporal_clips([row("video", 0), row("video", 2)], clip_length=2, stride=1, target_policy="video_inherited_last_frame")

    def test_policy_and_group_label_are_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_policy"):
            build_temporal_clips([row("video", 0)], clip_length=2, stride=1, target_policy="frame_label")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "multiple labels"):
            build_temporal_clips([row("video", 0, label="Abuse"), row("video", 1, label="Normal")], clip_length=2, stride=1, target_policy="video_inherited_last_frame")


if __name__ == "__main__":
    unittest.main()
