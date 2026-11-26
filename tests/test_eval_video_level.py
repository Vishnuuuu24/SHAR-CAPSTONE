from __future__ import annotations

import unittest
from dataclasses import dataclass

from core.p1c_engine import FramePrediction
from eval.video_level import VIDEO_LEVEL_AGGREGATION_SETTINGS, aggregate_video_level_predictions
from scripts.run_p1c_frame_baseline import _video_level_report

CLASSES = ["Abuse", "Normal"]


@dataclass(frozen=True)
class _FakeEvaluationResult:
    predictions: tuple[FramePrediction, ...]


def frame(dataset: str, video: str, target: int, predicted: int) -> FramePrediction:
    return FramePrediction(
        source_dataset=dataset, source_video_id=video, target_index=target, predicted_index=predicted
    )


class GroupingTests(unittest.TestCase):
    def test_groups_by_composite_key_never_video_id_alone(self) -> None:
        # Two datasets reuse the same video id; a bare source_video_id grouping
        # would wrongly merge them into one "video".
        predictions = [
            frame("dataset-a", "video-1", target=0, predicted=0),
            frame("dataset-b", "video-1", target=1, predicted=1),
        ]
        result = aggregate_video_level_predictions(predictions, CLASSES)
        self.assertEqual(result.video_count, 2)
        keys = {(p.source_dataset, p.source_video_id) for p in result.predictions}
        self.assertEqual(keys, {("dataset-a", "video-1"), ("dataset-b", "video-1")})

    def test_inconsistent_target_index_within_a_video_raises(self) -> None:
        predictions = [
            frame("dataset-a", "video-1", target=0, predicted=0),
            frame("dataset-a", "video-1", target=1, predicted=0),
        ]
        with self.assertRaisesRegex(ValueError, "disagree on target_index"):
            aggregate_video_level_predictions(predictions, CLASSES)

    def test_empty_predictions_raises(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_video_level_predictions([], CLASSES)


class MajorityVoteTests(unittest.TestCase):
    def test_clear_winner(self) -> None:
        predictions = [
            frame("d", "v1", target=0, predicted=0),
            frame("d", "v1", target=0, predicted=0),
            frame("d", "v1", target=0, predicted=1),
        ]
        result = aggregate_video_level_predictions(predictions, CLASSES)
        [video] = result.predictions
        self.assertEqual(video.predicted_index, 0)
        self.assertEqual(video.frame_count, 3)
        self.assertAlmostEqual(video.vote_margin, 2 / 3)
        self.assertFalse(video.tie_broken)
        self.assertEqual(result.tie_broken_video_count, 0)

    def test_tie_breaks_to_lowest_class_index_and_is_counted(self) -> None:
        predictions = [
            frame("d", "v1", target=1, predicted=1),
            frame("d", "v1", target=1, predicted=0),
        ]
        result = aggregate_video_level_predictions(predictions, CLASSES)
        [video] = result.predictions
        self.assertEqual(video.predicted_index, 0)  # lowest of the tied {0, 1}
        self.assertTrue(video.tie_broken)
        self.assertEqual(result.tie_broken_video_count, 1)
        self.assertAlmostEqual(video.vote_margin, 0.5)

    def test_metrics_match_a_hand_computed_small_case(self) -> None:
        # Two videos: video-1 truth=Abuse(0), majority-voted predicted=Abuse(0)
        # (correct); video-2 truth=Normal(1), majority-voted predicted=Abuse(0)
        # (wrong). Hand-computed: accuracy 0.5; Abuse: TP=1,FP=1,FN=0 -> P=.5,
        # R=1, F1=2/3; Normal: TP=0,FP=0,FN=1 -> P=0,R=0,F1=0. macro_f1=1/3.
        predictions = [
            frame("d", "v1", target=0, predicted=0),
            frame("d", "v1", target=0, predicted=0),
            frame("d", "v2", target=1, predicted=0),
            frame("d", "v2", target=1, predicted=1),
            frame("d", "v2", target=1, predicted=0),
        ]
        result = aggregate_video_level_predictions(predictions, CLASSES)
        self.assertEqual(result.video_count, 2)
        self.assertAlmostEqual(result.metrics["accuracy"], 0.5)
        self.assertAlmostEqual(result.metrics["macro_f1"], 1 / 3)
        self.assertAlmostEqual(result.metrics["per_class"]["Abuse"]["f1"], 2 / 3)
        self.assertAlmostEqual(result.metrics["per_class"]["Normal"]["f1"], 0.0)

    def test_frame_count_and_vote_margin_reported_per_video(self) -> None:
        predictions = [
            frame("d", "v1", target=0, predicted=0),
            frame("d", "v1", target=0, predicted=0),
            frame("d", "v1", target=0, predicted=0),
            frame("d", "v1", target=0, predicted=1),
        ]
        [video] = aggregate_video_level_predictions(predictions, CLASSES).predictions
        self.assertEqual(video.frame_count, 4)
        self.assertAlmostEqual(video.vote_margin, 0.75)


class RegisteredSettingsTests(unittest.TestCase):
    def test_registered_settings_are_exactly_none_and_majority_vote(self) -> None:
        self.assertEqual(VIDEO_LEVEL_AGGREGATION_SETTINGS, frozenset({"none", "majority_vote"}))


class RunnerVideoLevelReportTests(unittest.TestCase):
    """Proves the runner wiring: `none` skips aggregation, `majority_vote` computes it."""

    def setUp(self) -> None:
        self.selected = _FakeEvaluationResult(
            predictions=(
                frame("d", "v1", target=0, predicted=0),
                frame("d", "v1", target=0, predicted=0),
            )
        )

    def test_none_setting_skips_aggregation(self) -> None:
        report = _video_level_report({"video_level_aggregation": "none"}, self.selected, CLASSES)
        self.assertEqual(report, {"aggregation": "none"})

    def test_majority_vote_setting_computes_and_labels_the_report(self) -> None:
        report = _video_level_report({"video_level_aggregation": "majority_vote"}, self.selected, CLASSES)
        self.assertEqual(report["aggregation"], "majority_vote")
        self.assertEqual(report["video_count"], 1)
        self.assertEqual(report["tie_broken_video_count"], 0)
        self.assertIn("metrics", report)
        self.assertEqual(len(report["predictions"]), 1)
        self.assertEqual(report["predictions"][0]["predicted_index"], 0)


if __name__ == "__main__":
    unittest.main()
