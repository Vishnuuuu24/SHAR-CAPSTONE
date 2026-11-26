"""Caption-supervised projection (FINAL_PUSH_PLAN Step 6): caption loading, span features, projection, gate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core import caption_projection as cp

DIM = 6
TOPICS = [("a red car drives down the road", 0), ("a man runs across the shop floor", 1), ("smoke and fire rise from the building", 2)]


def _video(rng: np.random.Generator, topic_order: list[int]) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, str]]]:
    """30 frames at 2 fps; each third of the video shows one topic, whose feature dimension is raised."""

    frames = rng.normal(0, 0.2, (30, DIM)).astype(np.float32)
    sentences = []
    for block, topic in enumerate(topic_order):
        frames[block * 10:(block + 1) * 10, TOPICS[topic][1]] += 3.0
        sentences.append((block * 5.0, block * 5.0 + 4.5, TOPICS[topic][0]))
    return frames, np.arange(30) / 2.0, sentences


class CaptionProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.store, self.captions = {}, {}
        for index in range(24):
            order = list(rng.permutation(3))
            frames, times, sentences = _video(rng, order)
            self.store[f"v{index:02d}"] = (frames, times)
            self.captions[f"v{index:02d}"] = sentences
        self.stems = sorted(self.store)
        self.data = cp.build_caption_data(lambda s: self.store[s], self.stems, self.captions)

    def test_uca_files_are_merged_and_a_repeated_video_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = {"duration": 10.0, "timestamps": [[0.0, 4.0], [4.0, 9.0]], "sentences": ["A man walks.", " "]}
            for name, stem in zip(cp.UCA_FILES, ("a", "b", "c")):
                (root / name).write_text(json.dumps({stem: entry}))
            loaded = cp.load_captions(root)
            self.assertEqual(sorted(loaded), ["a", "b", "c"])
            self.assertEqual(loaded["a"], [(0.0, 4.0, "A man walks.")])   # the blank sentence is dropped
            (root / cp.UCA_FILES[2]).write_text(json.dumps({"a": entry}))
            with self.assertRaises(ValueError):
                cp.load_captions(root)

    def test_span_feature_is_the_mean_of_frames_inside_the_span_with_a_nearest_frame_fallback(self) -> None:
        frames = np.arange(10, dtype=np.float32)[:, None] * np.ones((1, 2), dtype=np.float32)
        times = np.arange(10, dtype=np.float64)
        spans = cp.span_features(frames, times, [(2.0, 4.0, "x"), (6.2, 6.4, "y")])
        np.testing.assert_allclose(spans[0], [3.0, 3.0])
        np.testing.assert_allclose(spans[1], [6.0, 6.0])   # no frame inside 6.2-6.4 -> nearest frame to 6.3

    def test_videos_that_only_train_keep_spans_but_no_frames(self) -> None:
        extras = cp.build_caption_data(lambda s: self.store[s], self.stems[:4], self.captions, keep_frames=False)
        self.assertEqual(extras.frames, {})
        self.assertEqual(extras.spans[self.stems[0]].shape, (3, DIM))
        merged = cp.merge(cp.build_caption_data(lambda s: self.store[s], self.stems[4:], self.captions), extras)
        self.assertEqual(len(merged.spans), 24)
        with self.assertRaises(ValueError):
            cp.merge(extras, extras)

    def test_projection_retrieves_the_right_sentence_on_heldout_videos(self) -> None:
        train, heldout = self.stems[:18], self.stems[18:]
        spans, sentences = cp.stack(self.data, train)
        self.assertEqual((spans.shape, len(sentences)), ((54, DIM), 54))
        text_model = cp.fit_text_model(sentences, dim=2)
        projection = cp.fit_projection(spans, text_model.transform(sentences), alphas=(0.1, 1.0))
        gate = cp.retrieval_gate(projection, text_model, self.data, heldout)
        self.assertEqual(gate["sentences"], 18)
        self.assertLess(gate["median_percentile_rank"], 0.4)   # 3 topics x 6 videos: own topic's 6 sentences tie at the top
        self.assertGreater(gate["recall_at_10"], gate["chance_recall_at_10"])
        shuffled = cp.CaptionData(self.data.frames, self.data.times, self.data.spans,
                                  {s: list(reversed(t)) if i % 2 else t for i, (s, t) in enumerate(self.data.sentences.items())})
        matrix = cp.video_matrix(projection, shuffled, heldout)
        self.assertEqual(matrix.shape, (6, DIM + 2 + 2))
        self.assertEqual(cp.video_matrix(projection, self.data, heldout, include_mean_feature=False).shape, (6, 4))


if __name__ == "__main__":
    unittest.main()
