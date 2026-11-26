import unittest

import numpy as np

from core.cache_classifier import fuse, kernel_vote, video_embeddings


class CacheClassifierTest(unittest.TestCase):
    def test_video_embeddings_mean_of_normalised_frames(self):
        feats = np.array([[3.0, 0.0], [0.0, 1.0], [5.0, 0.0]], dtype=np.float16)
        rows = [{"source_video_id": "a", "label": "X"}, {"source_video_id": "a", "label": "X"}, {"source_video_id": "b", "label": "Y"}]
        ids, labels, emb = video_embeddings(feats, rows, chunk=2)
        self.assertEqual((ids, labels), (["a", "b"], ["X", "Y"]))
        np.testing.assert_allclose(emb[0], [2 ** -0.5, 2 ** -0.5], atol=1e-6)
        np.testing.assert_allclose(emb[1], [1.0, 0.0], atol=1e-6)

    def test_label_disagreement_refused(self):
        rows = [{"source_video_id": "a", "label": "X"}, {"source_video_id": "a", "label": "Y"}]
        with self.assertRaises(ValueError):
            video_embeddings(np.ones((2, 2)), rows)

    def test_kernel_vote_prefers_nearest_class_and_sharpens_with_beta(self):
        keys = np.array([[1.0, 0.0], [0.0, 1.0]])
        q = np.array([[0.9, 0.1]]) / np.linalg.norm([0.9, 0.1])
        soft = kernel_vote(q, keys, [0, 1], 2, beta=1.0)
        sharp = kernel_vote(q, keys, [0, 1], 2, beta=20.0)
        self.assertAlmostEqual(soft.sum(), 1.0)
        self.assertGreater(soft[0, 0], 0.5)
        self.assertGreater(sharp[0, 0], soft[0, 0])

    def test_fuse(self):
        np.testing.assert_allclose(fuse(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.25), [0.75, 0.25])


if __name__ == "__main__":
    unittest.main()
