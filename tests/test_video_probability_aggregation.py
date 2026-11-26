from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from eval.video_probability_aggregation import aggregate_video_level_probabilities

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
        result = aggregate_video_level_probabilities(records, probs, CLASSES)
        self.assertEqual(result.video_count, 2)
        self.assertEqual(
            {(p.source_dataset, p.source_video_id) for p in result.predictions},
            {("a", "v1"), ("b", "v1")},
        )

    def test_inconsistent_target_within_a_video_raises(self) -> None:
        records = [frame("a", "v1", 0), frame("a", "v1", 1)]
        probs = np.array([[0.9, 0.1], [0.9, 0.1]])
        with self.assertRaisesRegex(ValueError, "disagree on target_index"):
            aggregate_video_level_probabilities(records, probs, CLASSES)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "disagree on frame count"):
            aggregate_video_level_probabilities([frame("a", "v1", 0)], np.zeros((2, 2)), CLASSES)

    def test_wrong_class_width_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_frames"):
            aggregate_video_level_probabilities([frame("a", "v1", 0)], np.zeros((1, 3)), CLASSES)

    def test_negative_probability_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            aggregate_video_level_probabilities(
                [frame("a", "v1", 0)], np.array([[-0.1, 1.1]]), CLASSES
            )


class MeanProbabilityTests(unittest.TestCase):
    def test_mean_can_disagree_with_majority_vote(self) -> None:
        # Three frames. Argmax votes: Abuse, Abuse, Normal -> majority vote = Abuse.
        # But the two Abuse frames are barely decided (0.51) and the Normal frame
        # is emphatic (0.99), so the mean vector favours Normal.
        records = [frame("d", "v1", 1)] * 3
        probs = np.array([[0.51, 0.49], [0.51, 0.49], [0.01, 0.99]])
        [video] = aggregate_video_level_probabilities(records, probs, CLASSES).predictions
        self.assertEqual(video.predicted_index, 1)  # Normal, unlike the majority vote
        np.testing.assert_allclose(video.mean_probability, [0.343333, 0.656667], atol=1e-5)

    def test_metrics_match_a_hand_computed_case(self) -> None:
        # video-1 truth Abuse(0), mean -> Abuse (correct).
        # video-2 truth Normal(1), mean -> Abuse (wrong).
        # accuracy 0.5; Abuse P=.5 R=1 F1=2/3; Normal F1=0; macro_f1=1/3.
        records = [frame("d", "v1", 0), frame("d", "v1", 0), frame("d", "v2", 1), frame("d", "v2", 1)]
        probs = np.array([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.55, 0.45]])
        result = aggregate_video_level_probabilities(records, probs, CLASSES)
        self.assertEqual(result.video_count, 2)
        self.assertAlmostEqual(result.metrics["accuracy"], 0.5)
        self.assertAlmostEqual(result.metrics["macro_f1"], 1 / 3)

    def test_frame_count_and_top_margin_reported(self) -> None:
        records = [frame("d", "v1", 0)] * 4
        probs = np.array([[0.7, 0.3]] * 4)
        [video] = aggregate_video_level_probabilities(records, probs, CLASSES).predictions
        self.assertEqual(video.frame_count, 4)
        self.assertAlmostEqual(video.top_margin, 0.4)

    def test_low_margin_videos_counted_against_uniform_default(self) -> None:
        # default threshold = 1/2 = 0.5. v1 margin 0.02 (< 0.5), v2 margin 0.9.
        records = [frame("d", "v1", 0), frame("d", "v2", 0)]
        probs = np.array([[0.51, 0.49], [0.95, 0.05]])
        result = aggregate_video_level_probabilities(records, probs, CLASSES)
        self.assertEqual(result.low_margin_video_count, 1)
        self.assertAlmostEqual(result.low_margin_threshold, 0.5)

    def test_predictions_are_sorted_for_stable_reports(self) -> None:
        records = [frame("d", "v2", 0), frame("d", "v1", 0)]
        probs = np.array([[0.9, 0.1], [0.9, 0.1]])
        result = aggregate_video_level_probabilities(records, probs, CLASSES)
        self.assertEqual([p.source_video_id for p in result.predictions], ["v1", "v2"])


if __name__ == "__main__":
    unittest.main()
