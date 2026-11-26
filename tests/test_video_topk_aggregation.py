from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from eval.video_topk_aggregation import aggregate_video_level_topk

CLASSES = ["Abuse", "Normal"]


@dataclass(frozen=True)
class Frame:
    source_dataset: str
    source_video_id: str
    target_index: int


def frame(dataset: str, video: str, target: int) -> Frame:
    return Frame(dataset, video, target)


class GroupingTests(unittest.TestCase):
    def test_groups_by_composite_key_never_video_id_alone(self) -> None:
        records = [frame("a", "v1", 0), frame("b", "v1", 1)]
        probs = np.array([[0.9, 0.1], [0.2, 0.8]])
        result = aggregate_video_level_topk(records, probs, CLASSES, k=1)
        self.assertEqual(result.video_count, 2)
        self.assertEqual(
            {(p.source_dataset, p.source_video_id) for p in result.predictions},
            {("a", "v1"), ("b", "v1")},
        )

    def test_inconsistent_target_within_a_video_raises(self) -> None:
        records = [frame("a", "v1", 0), frame("a", "v1", 1)]
        probs = np.array([[0.9, 0.1], [0.9, 0.1]])
        with self.assertRaisesRegex(ValueError, "disagree on target_index"):
            aggregate_video_level_topk(records, probs, CLASSES, k=1)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "disagree on frame count"):
            aggregate_video_level_topk([frame("a", "v1", 0)], np.zeros((2, 2)), CLASSES, k=1)

    def test_wrong_class_width_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_frames"):
            aggregate_video_level_topk([frame("a", "v1", 0)], np.zeros((1, 3)), CLASSES, k=1)

    def test_negative_probability_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            aggregate_video_level_topk(
                [frame("a", "v1", 0)], np.array([[-0.1, 1.1]]), CLASSES, k=1
            )


class InvalidKTests(unittest.TestCase):
    def test_zero_k_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be"):
            aggregate_video_level_topk([frame("a", "v1", 0)], np.array([[0.9, 0.1]]), CLASSES, k=0)

    def test_negative_k_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be"):
            aggregate_video_level_topk(
                [frame("a", "v1", 0)], np.array([[0.9, 0.1]]), CLASSES, k=-1
            )

    def test_non_integer_k_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be"):
            aggregate_video_level_topk(
                [frame("a", "v1", 0)], np.array([[0.9, 0.1]]), CLASSES, k=2.5
            )


class TopKMeaningTests(unittest.TestCase):
    def test_k1_finds_the_brief_event_a_mean_would_miss(self) -> None:
        # This is the exact scenario the module exists for: an anomaly event
        # that is a small minority of the video's frames. 9 frames weakly
        # favour Normal (0.6); 1 frame, the event, strongly favours Abuse
        # (0.95). A plain mean over all 10 frames still favours Normal, but
        # k=1 (maximum per-class probability) lets that single event frame
        # decide the video, matching the true label Abuse.
        records = [frame("d", "v1", 0)] * 10
        weak_normal = [0.4, 0.6]
        strong_abuse = [0.95, 0.05]
        probs = np.array([weak_normal] * 9 + [strong_abuse])

        k1_result = aggregate_video_level_topk(records, probs, CLASSES, k=1)
        [video_k1] = k1_result.predictions
        self.assertEqual(video_k1.predicted_index, 0)  # Abuse, the true label

        mean_result = aggregate_video_level_topk(records, probs, CLASSES, k=None)
        [video_mean] = mean_result.predictions
        self.assertEqual(video_mean.predicted_index, 1)  # Normal -- the event is outvoted

    def test_k_at_least_frame_count_equals_plain_mean(self) -> None:
        records = [frame("d", "v1", 0)] * 4
        probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        expected_mean = probs.mean(axis=0)

        exact = aggregate_video_level_topk(records, probs, CLASSES, k=4)
        over = aggregate_video_level_topk(records, probs, CLASSES, k=100)
        none_k = aggregate_video_level_topk(records, probs, CLASSES, k=None)

        for result in (exact, over, none_k):
            [video] = result.predictions
            np.testing.assert_allclose(video.topk_probability, expected_mean)
            self.assertEqual(video.k_used, 4)

    def test_fewer_frames_than_k_uses_all_frames_without_error(self) -> None:
        records = [frame("d", "v1", 0)] * 2
        probs = np.array([[0.9, 0.1], [0.3, 0.7]])
        result = aggregate_video_level_topk(records, probs, CLASSES, k=10)
        [video] = result.predictions
        self.assertEqual(video.k_used, 2)
        np.testing.assert_allclose(video.topk_probability, probs.mean(axis=0))

    def test_frame_count_and_k_used_reported(self) -> None:
        records = [frame("d", "v1", 0)] * 5
        probs = np.array([[0.7, 0.3]] * 5)
        [video] = aggregate_video_level_topk(records, probs, CLASSES, k=3).predictions
        self.assertEqual(video.frame_count, 5)
        self.assertEqual(video.k_used, 3)


class TieBreakingTests(unittest.TestCase):
    def test_ties_break_to_lowest_class_index_and_are_counted(self) -> None:
        records = [frame("d", "v1", 0)]
        probs = np.array([[0.5, 0.5]])
        result = aggregate_video_level_topk(records, probs, CLASSES, k=1)
        [video] = result.predictions
        self.assertEqual(video.predicted_index, 0)  # Abuse, the lower index
        self.assertEqual(result.tie_broken_video_count, 1)

    def test_no_tie_when_scores_differ(self) -> None:
        records = [frame("d", "v1", 0)]
        probs = np.array([[0.6, 0.4]])
        result = aggregate_video_level_topk(records, probs, CLASSES, k=1)
        self.assertEqual(result.tie_broken_video_count, 0)


class MetricsTests(unittest.TestCase):
    def test_metrics_match_a_hand_computed_case(self) -> None:
        # video-1 truth Abuse(0): max Abuse prob 0.8 beats max Normal prob 0.4 -> Abuse (correct).
        # video-2 truth Normal(1): max Abuse prob 0.7 beats max Normal prob 0.45 -> Abuse (wrong).
        records = [
            frame("d", "v1", 0),
            frame("d", "v1", 0),
            frame("d", "v2", 1),
            frame("d", "v2", 1),
        ]
        probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.55, 0.45]])
        result = aggregate_video_level_topk(records, probs, CLASSES, k=1)
        self.assertEqual(result.video_count, 2)
        [v1, v2] = result.predictions
        self.assertEqual(v1.predicted_index, 0)
        self.assertEqual(v2.predicted_index, 0)  # wrong: max(0.7) beats max(0.45)
        self.assertAlmostEqual(result.metrics["accuracy"], 0.5)

    def test_predictions_are_sorted_for_stable_reports(self) -> None:
        records = [frame("d", "v2", 0), frame("d", "v1", 0)]
        probs = np.array([[0.9, 0.1], [0.9, 0.1]])
        result = aggregate_video_level_topk(records, probs, CLASSES, k=1)
        self.assertEqual([p.source_video_id for p in result.predictions], ["v1", "v2"])


if __name__ == "__main__":
    unittest.main()
