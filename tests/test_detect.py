"""Live detector (FINAL_PUSH_PLAN Step 4): split table, context features, metrics, and a tiny end-to-end fit."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core import detect_data, detect_metrics, detect_model
from core.detect_data import DetectVideo

DIM = 8


class DetectDataTest(unittest.TestCase):
    def test_category_mapping_is_binary_and_keeps_the_crime_folder(self) -> None:
        self.assertEqual(detect_data.category_of("Abuse/Abuse001_x264.mp4"), (True, "Abuse"))
        self.assertEqual(detect_data.category_of("Training_Normal_Videos_Anomaly/Normal_Videos001_x264.mp4"), (False, "Normal"))
        self.assertEqual(detect_data.category_of("Testing_Normal_Videos_Anomaly/Normal_Videos_003_x264.mp4"), (False, "Normal"))

    def test_frame_plan_takes_every_nth_kept_frame_per_video(self) -> None:
        index = np.zeros(20, dtype=[("source_frame_index", "<i8")])
        index["source_frame_index"] = list(range(0, 50, 5)) + list(range(0, 50, 5))
        videos = [DetectVideo("A/a.mp4", "train", True, "A", 30.0, 0, 10, 50), DetectVideo("N/n.mp4", "test", False, "Normal", 30.0, 10, 10, 50)]
        rows, owner, frames = detect_data.frame_plan(videos, index, every=3)
        self.assertEqual(rows.tolist(), [0, 3, 6, 9, 10, 13, 16, 19])
        self.assertEqual(owner.tolist(), [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(frames.tolist(), [0, 15, 30, 45, 0, 15, 30, 45])

    def test_frame_labels_mark_inclusive_spans(self) -> None:
        labels = detect_data.frame_labels(np.array([0, 15, 30, 45, 60]), [(15, 30)])
        self.assertEqual(labels.tolist(), [0, 1, 1, 0, 0])


class DetectMetricsTest(unittest.TestCase):
    def test_source_frame_takes_the_latest_sampled_score_at_or_before_it(self) -> None:
        dense = detect_metrics.expand_to_source_frames(np.array([0.1, 0.9]), np.array([0, 15]), 20)
        self.assertEqual(dense[:15].tolist(), [np.float32(0.1)] * 15)
        self.assertEqual(dense[15:].tolist(), [np.float32(0.9)] * 5)

    def test_causal_mean_never_uses_the_future_and_centred_mean_does(self) -> None:
        scores = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(detect_metrics.causal_mean(scores, 2), [0.0, 0.0, 0.0, 0.5, 0.5])
        np.testing.assert_allclose(detect_metrics.centred_mean(scores, 3), [0.0, 0.0, 1 / 3, 1 / 3, 0.5])
        np.testing.assert_allclose(detect_metrics.causal_mean(scores, 1), scores)

    def test_alarm_events_count_runs(self) -> None:
        self.assertEqual(detect_metrics.alarm_events(np.array([1, 1, 0, 1, 0, 0, 1], dtype=bool)), 3)
        self.assertEqual(detect_metrics.alarm_events(np.array([0, 0], dtype=bool)), 0)

    def test_report_on_a_perfect_and_a_late_detector(self) -> None:
        frames = np.arange(0, 300, 15)
        event = {"source_frames": frames, "total_frames": 300, "fps": 30.0, "spans": [(60, 149)], "anomaly": True, "category": "Abuse"}
        normal = {"source_frames": frames, "total_frames": 300, "fps": 30.0, "spans": [], "anomaly": False, "category": "Normal"}
        inside = ((frames >= 60) & (frames <= 149)).astype(np.float32)
        perfect = detect_metrics.detection_report([{**event, "scores": inside}, {**normal, "scores": np.zeros(20, dtype=np.float32)}], 0.5)
        self.assertGreater(perfect["frame_auc"], 0.99)
        self.assertEqual((perfect["event_recall"], perfect["median_delay_s"], perfect["false_alarms_per_hour"]), (1.0, 0.0, 0.0))
        late = inside.copy()
        late[frames < 90] = 0.0   # first alarm one second after the event starts
        noisy = np.zeros(20, dtype=np.float32)
        noisy[[2, 3, 10]] = 1.0   # two separate alarm runs on a 10-second normal video
        report = detect_metrics.detection_report([{**event, "scores": late}, {**normal, "scores": noisy}], 0.5)
        self.assertAlmostEqual(report["median_delay_s"], 1.0)
        self.assertAlmostEqual(report["false_alarms_per_hour"], 2 / (10 / 3600))
        self.assertEqual(report["normal_videos_with_an_alarm"], 1.0)


class DetectModelTest(unittest.TestCase):
    def _make(self, root: Path) -> list[DetectVideo]:
        rng = np.random.default_rng(0)
        videos, rows, feats = [], [], []
        for index in range(40):
            anomaly = index % 2 == 0
            split = "train" if index < 32 else "test"
            path = f"{'Abuse' if anomaly else 'Training_Normal_Videos_Anomaly'}/v{index:02d}.mp4"
            offset = rng.normal(0, 2.0, DIM)                      # a per-video offset, larger than the event signal
            x = offset + rng.normal(0, 0.1, (60, DIM))
            if anomaly:
                x[30:40, :2] += 3.0                               # the event: frames 30-39
            videos.append(DetectVideo(path, split, anomaly, "Abuse" if anomaly else "Normal", 30.0, len(rows), 60, 900))
            for frame in range(60):
                rows.append([len(rows), path, split, int(anomaly), videos[-1].category, frame * 15, len(rows)])
            feats.append(x)
        np.save(root / "toy.float16.npy", np.concatenate(feats).astype(np.float16))
        with (root / "rows.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row", "relative_path", "split", "anomaly", "category", "source_frame_index", "blob_row"])
            writer.writerows(rows)
        return videos

    def test_context_features(self) -> None:
        x = np.array([[1.0], [3.0], [5.0]], dtype=np.float32)
        np.testing.assert_allclose(detect_model.add_context(x, "causal")[:, 1], [0.0, 1.0, 2.0])   # running means 1, 2, 3
        np.testing.assert_allclose(detect_model.add_context(x, "offline")[:, 1], [-2.0, 0.0, 2.0])
        self.assertEqual(detect_model.add_context(x, "none").shape, (3, 1))

    def test_topk_score_ignores_padding(self) -> None:
        import torch

        probs = torch.tensor([[0.9, 0.1, 0.8, 0.7], [0.2, 0.6, 0.99, 0.99]])
        mask = torch.tensor([[False, False, False, False], [False, False, True, True]])
        np.testing.assert_allclose(detect_model.topk_video_score(probs, mask).numpy(), [0.9, 0.6], rtol=1e-6)

    def test_store_standardises_with_training_rows_and_models_localise_the_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = self._make(root)
            store = detect_model.load_store(root, ("toy",), videos)
            self.assertEqual(store.dim, DIM)
            self.assertEqual(store.frames(videos[0].relative_path, "causal").shape, (60, 2 * DIM))
            self.assertEqual(store.source_frames_of(videos[1].relative_path)[:3].tolist(), [0, 15, 30])
            fit, held = detect_model.split_heldout(videos, fraction=0.25, seed=0)
            self.assertEqual((len(fit), len(held)), (24, 8))
            self.assertFalse({v.relative_path for v in fit} & {v.relative_path for v in held})
            self.assertTrue(all(v.split == "train" for v in fit + held))
            model, history = detect_model.train_detector(store, fit, held, loss="topk", context="causal", seed=0, epochs=150, patience=150,
                                                         batch_videos=6, learning_rate=2e-3, log=lambda _r: None)
            self.assertGreater(max(h["heldout_video_auc"] for h in history), 0.9)
            target = next(v for v in videos if v.split == "test" and v.anomaly)
            scores = detect_model.score_frames(model, store.frames(target.relative_path, "causal"), "cpu")
            self.assertGreater(scores[30:40].mean(), scores[:30].mean() + 0.2)   # the event stands out within its own video
            ranking, _ = detect_model.train_detector(store, fit, held, loss="ranking", seed=0, epochs=3, patience=3, batch_videos=6, log=lambda _r: None)
            self.assertEqual(detect_model.score_frames(ranking, store.frames(target.relative_path), "cpu").shape, (60,))
            floor = detect_model.video_level_scores(store, fit, [v for v in videos if v.split == "test"])
            self.assertEqual(len(floor), 8)
            with self.assertRaises(ValueError):
                detect_model.train_detector(store, fit, held, loss="ranking", context="causal", log=lambda _r: None)


if __name__ == "__main__":
    unittest.main()
