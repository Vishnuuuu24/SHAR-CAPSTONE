"""Tests for `core.p1c_event_clips` (clip-level event/background probes).

Synthetic fixtures only, mirroring `tests/test_p1c_temporal_features.py`'s
and `tests/test_p1c_background_mask.py`'s style.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_event_clips import (  # noqa: E402
    Clip,
    _assert_no_leakage,
    _train_test_masks,
    build_event_clips,
    clip_features,
    leave_one_video_out_clips,
)


def _rows(video_frame_counts: list[tuple[str, str, int]]) -> list[dict]:
    """`[(video_id, label, frame_count), ...]` -> cache-shaped rows, stride-5 frames."""

    rows: list[dict] = []
    for video_id, label, count in video_frame_counts:
        for frame in range(count):
            rows.append({
                "source_dataset": "unit_test_dataset",
                "source_video_id": video_id,
                "source_frame_index": frame * 5,
                "target": "0",
                "label": label,
            })
    return rows


class BuildEventClipsFilteringTests(unittest.TestCase):
    def test_only_matching_label_and_annotated_videos_kept(self) -> None:
        rows = _rows([
            ("annotated_shooting", "Shooting", 64),   # eligible
            ("unannotated_shooting", "Shooting", 64), # right label, no interval
            ("annotated_other_label", "Robbery", 64), # has interval, wrong label
        ])
        intervals = {"annotated_shooting": [(0, 319)], "annotated_other_label": [(0, 319)]}
        build = build_event_clips(rows, intervals, label="Shooting", clip_length=16)
        videos_seen = {clip.video_id for clip in build.usable + build.ambiguous}
        self.assertEqual(videos_seen, {"annotated_shooting"})

    def test_never_crosses_video_boundary(self) -> None:
        rows = _rows([("v0", "Shooting", 20), ("v1", "Shooting", 20)])
        intervals = {"v0": [(0, 99)], "v1": [(0, 99)]}  # whole video positive
        build = build_event_clips(rows, intervals, label="Shooting", clip_length=16)
        for clip in build.usable:
            self.assertEqual(len({rows[i]["source_video_id"] for i in clip.row_indices}), 1)


class TilingTests(unittest.TestCase):
    def test_hop_is_non_overlapping(self) -> None:
        rows = _rows([("v0", "Shooting", 32)])
        intervals = {"v0": [(0, 159)]}  # entirely positive, irrelevant here
        build = build_event_clips(rows, intervals, label="Shooting", clip_length=16)
        all_indices = [i for clip in build.usable + build.ambiguous for i in clip.row_indices]
        self.assertEqual(sorted(all_indices), list(range(32)))  # every row used exactly once

    def test_trailing_partial_window_dropped(self) -> None:
        rows = _rows([("v0", "Shooting", 20)])  # one full window of 16, 4 left over
        intervals = {"v0": [(0, 99)]}
        build = build_event_clips(rows, intervals, label="Shooting", clip_length=16)
        n_clips = len(build.usable) + len(build.ambiguous)
        self.assertEqual(n_clips, 1)

    def test_frame_stride_widens_window_footprint(self) -> None:
        rows = _rows([("v0", "Shooting", 64)])
        intervals = {"v0": [(0, 319)]}
        build = build_event_clips(rows, intervals, label="Shooting", clip_length=4, frame_stride=3)
        clip = build.usable[0] if build.usable else build.ambiguous[0]
        # window occupies local rows 0,3,6,9 -> row_indices spaced by 3
        self.assertEqual(clip.row_indices, (0, 3, 6, 9))

    def test_deterministic(self) -> None:
        rows = _rows([("v0", "Shooting", 48)])
        intervals = {"v0": [(10, 40)]}
        first = build_event_clips(rows, intervals, label="Shooting", clip_length=16)
        second = build_event_clips(rows, intervals, label="Shooting", clip_length=16)
        self.assertEqual(first.report(), second.report())


class ClassificationTests(unittest.TestCase):
    def test_fully_inside_span_is_positive(self) -> None:
        rows = _rows([("v0", "Shooting", 16)])  # frames 0..75, all inside a wide span
        intervals = {"v0": [(0, 1000)]}
        build = build_event_clips(rows, intervals, label="Shooting", clip_length=16, negative_margin=0)
        self.assertEqual(len(build.positive), 1)
        self.assertEqual(build.positive[0].event_overlap, 1.0)
        self.assertTrue(build.positive[0].is_positive)

    def test_far_from_any_span_is_negative(self) -> None:
        rows = _rows([("v0", "Shooting", 64)])  # frames 0..315
        intervals = {"v0": [(2000, 2100)]}  # far away, outside the video entirely
        build = build_event_clips(
            rows, intervals, label="Shooting", clip_length=16, negative_margin=4
        )
        self.assertEqual(len(build.negative), len(build.usable))
        self.assertEqual(len(build.positive), 0)
        self.assertEqual(len(build.ambiguous), 0)

    def test_clip_within_margin_of_span_edge_is_ambiguous(self) -> None:
        # v0 has 32 rows (frames 0,5,...,155). Span [80,155] covers local
        # rows 16..31 (second window) fully -> that window is positive.
        # First window (rows 0..15, frames 0..75) is entirely outside the
        # span but its last row (frame 75) is only 1 cached row away from
        # the span start (frame 80) -- inside a margin of >=2, so ambiguous.
        rows = _rows([("v0", "Shooting", 32)])
        intervals = {"v0": [(80, 155)]}
        build = build_event_clips(
            rows, intervals, label="Shooting", clip_length=16, negative_margin=2
        )
        self.assertEqual(len(build.positive), 1)
        self.assertEqual(len(build.ambiguous), 1)
        self.assertEqual(len(build.negative), 0)

    def test_overlap_threshold_boundary(self) -> None:
        # 16-row clip, span covers exactly local rows 8..15 (8/16 = 0.5 overlap).
        rows = _rows([("v0", "Shooting", 16)])
        intervals = {"v0": [(40, 1000)]}  # frame 40 = local row 8
        build = build_event_clips(
            rows, intervals, label="Shooting", clip_length=16,
            positive_overlap_threshold=0.5, negative_margin=0,
        )
        self.assertEqual(len(build.positive), 1)
        self.assertAlmostEqual(build.positive[0].event_overlap, 0.5)

    def test_report_counts_and_per_video_breakdown(self) -> None:
        rows = _rows([("v0", "Shooting", 32)])
        intervals = {"v0": [(80, 155)]}
        build = build_event_clips(rows, intervals, label="Shooting", clip_length=16, negative_margin=2)
        report = build.report()
        self.assertEqual(report["positive"], 1)
        self.assertEqual(report["ambiguous"], 1)
        self.assertEqual(report["negative"], 0)
        self.assertEqual(
            report["by_video"]["v0"], {"positive": 1, "negative": 0, "ambiguous": 1}
        )

    def test_rejects_bad_parameters(self) -> None:
        rows = _rows([("v0", "Shooting", 20)])
        with self.assertRaises(ValueError):
            build_event_clips(rows, {}, label="Shooting", clip_length=0)
        with self.assertRaises(ValueError):
            build_event_clips(rows, {}, label="Shooting", frame_stride=0)
        with self.assertRaises(ValueError):
            build_event_clips(rows, {}, label="Shooting", negative_margin=-1)


class ClipFeaturesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = _rows([("v0", "Shooting", 32), ("v1", "Shooting", 32)])
        self.intervals = {"v0": [(0, 1000)], "v1": [(0, 1000)]}
        self.build = build_event_clips(self.rows, self.intervals, label="Shooting", clip_length=16)
        self.rng = np.random.default_rng(0)
        self.features = self.rng.normal(size=(64, 5)).astype(np.float16)

    def test_widths(self) -> None:
        clips = self.build.usable
        self.assertEqual(clip_features(self.features, self.rows, clips, "single_frame").shape, (len(clips), 5))
        self.assertEqual(clip_features(self.features, self.rows, clips, "dense_mean").shape, (len(clips), 5))
        self.assertEqual(clip_features(self.features, self.rows, clips, "dense_mean_delta").shape, (len(clips), 10))

    def test_unknown_representation_rejected(self) -> None:
        with self.assertRaises(ValueError):
            clip_features(self.features, self.rows, self.build.usable, "median")

    def test_dense_mean_of_constant_per_video_feature_equals_the_constant(self) -> None:
        constant_features = np.zeros((64, 3), dtype=np.float32)
        constant_features[:32] = 1.0
        constant_features[32:] = 2.0
        out = clip_features(constant_features, self.rows, self.build.usable, "dense_mean")
        for clip, row in zip(self.build.usable, out):
            expected = 1.0 if clip.video_id == "v0" else 2.0
            np.testing.assert_allclose(row, np.full(3, expected))

    def test_single_frame_equals_centre_row_exactly(self) -> None:
        out = clip_features(self.features, self.rows, self.build.usable, "single_frame")
        for clip, row in zip(self.build.usable, out):
            centre = clip.row_indices[len(clip.row_indices) // 2]
            np.testing.assert_allclose(row, self.features[centre].astype(np.float32))

    def test_dense_mean_delta_appearance_half_equals_dense_mean(self) -> None:
        dense_mean = clip_features(self.features, self.rows, self.build.usable, "dense_mean")
        delta = clip_features(self.features, self.rows, self.build.usable, "dense_mean_delta")
        np.testing.assert_allclose(delta[:, :5], dense_mean, rtol=1e-3, atol=1e-3)


class LeaveOneVideoOutTests(unittest.TestCase):
    def test_leakage_assertion_fires(self) -> None:
        clips = [
            Clip("v0", (0,), 0, 0, True, 1.0),
            Clip("v1", (1,), 5, 5, False, 0.0),
        ]
        _, test_mask = _train_test_masks(clips, "v0")
        bad_train_mask = np.array([True, True])  # deliberately includes v0
        with self.assertRaises(AssertionError):
            _assert_no_leakage(clips, bad_train_mask, "v0")
        # And the correct mask must not raise.
        good_train_mask, _ = _train_test_masks(clips, "v0")
        _assert_no_leakage(clips, good_train_mask, "v0")

    def test_held_out_video_excluded_from_training(self) -> None:
        clips = [Clip(f"v{i}", (i,), i, i, bool(i % 2), float(i % 2)) for i in range(6)]
        train_mask, test_mask = _train_test_masks(clips, "v2")
        self.assertFalse(any(c.video_id == "v2" for c, keep in zip(clips, train_mask) if keep))
        self.assertTrue(all(c.video_id == "v2" for c, keep in zip(clips, test_mask) if keep))

    def _separable_clips_and_features(self, n_per_video: int = 20, n_videos: int = 4):
        rng = np.random.default_rng(0)
        clips, features = [], []
        for v in range(n_videos):
            for i in range(n_per_video):
                is_positive = i % 2 == 0
                centre = np.array([5.0, 5.0]) if is_positive else np.array([-5.0, -5.0])
                features.append(centre + rng.normal(scale=0.01, size=2))
                clips.append(Clip(f"v{v}", (v * n_per_video + i,), i, i, is_positive, float(is_positive)))
        return clips, np.asarray(features, dtype=np.float32)

    def test_separable_clips_score_near_one(self) -> None:
        clips, x = self._separable_clips_and_features()
        result = leave_one_video_out_clips(clips, x, seed=0)
        self.assertGreater(result["pooled"]["roc_auc"], 0.99)
        self.assertGreater(result["pooled"]["f1"], 0.9)

    def test_single_class_held_out_video_skips_auc(self) -> None:
        clips, x = self._separable_clips_and_features()
        # Overwrite v0's clips to be all-negative -> single class held out.
        clips = [
            Clip(c.video_id, c.row_indices, c.start_source_frame, c.end_source_frame, False, 0.0)
            if c.video_id == "v0" else c
            for c in clips
        ]
        result = leave_one_video_out_clips(clips, x, seed=0)
        self.assertIsNone(result["per_video"]["v0"]["roc_auc"])
        self.assertIsNone(result["per_video"]["v0"]["pr_auc"])
        self.assertEqual(result["per_video"]["v0"]["n_positive"], 0)

    def test_video_level_reports_recall_over_all_videos(self) -> None:
        clips, x = self._separable_clips_and_features()
        result = leave_one_video_out_clips(clips, x, seed=0)
        video_level = result["video_level"]
        self.assertEqual(video_level["n_videos"], 4)
        self.assertIn("recall", video_level)
        self.assertEqual(len(video_level["mean_clip_probability_by_video"]), 4)


if __name__ == "__main__":
    unittest.main()
