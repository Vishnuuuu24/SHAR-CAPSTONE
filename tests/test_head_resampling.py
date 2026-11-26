import unittest

import numpy as np
import torch
from torch import nn

from core.head_resampling import make_epoch_resampler, stratified_video_subset
from core.p1c_head_variants import _pad_sequence, group_pooled_sequences_by_video, train_sequence_head

FRAMES, DIM = 8, 6


def _cache(videos=6, per_video=40, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    rows, feats = [], []
    for v in range(videos):
        for f in range(per_video):
            rows.append({"source_dataset": "ucf", "source_video_id": f"v{v}", "source_frame_index": f * 5, "target": v % classes,
                         "label": f"c{v % classes}"})
            feats.append(rng.normal(size=DIM) + (v % classes) * 2.0)
    return np.array(feats, dtype=np.float32), rows


class Resampler(unittest.TestCase):
    def test_shapes_targets_and_one_bag_per_video(self):
        feats, rows = _cache()
        seq, mask, tgt = make_epoch_resampler(feats, rows, frames=FRAMES, seed=0)(0)
        self.assertEqual(seq.shape, (6, FRAMES, DIM))
        self.assertEqual(mask.shape, (6, FRAMES))
        # the seeded sampler can drop a duplicate position at a segment boundary, so a bag may hold FRAMES-1 real
        # frames; padding (mask True) is then at the end and the head ignores it, as for any short video
        self.assertTrue((mask.sum(1) <= 1).all())
        self.assertTrue((np.diff(mask.astype(int), axis=1) >= 0).all())
        self.assertEqual(sorted(tgt.tolist()), sorted(v % 3 for v in range(6)))

    def test_deterministic_but_different_each_epoch(self):
        feats, rows = _cache()
        r1, r2 = make_epoch_resampler(feats, rows, frames=FRAMES, seed=3), make_epoch_resampler(feats, rows, frames=FRAMES, seed=3)
        a0, a1, b0 = r1(0)[0], r1(1)[0], r2(0)[0]
        np.testing.assert_array_equal(a0, b0)
        self.assertFalse(np.array_equal(a0, a1))
        self.assertFalse(np.array_equal(a0, make_epoch_resampler(feats, rows, frames=FRAMES, seed=4)(0)[0]))

    def test_frames_come_from_the_video_and_cover_more_than_one_draw(self):
        feats, rows = _cache(videos=1, per_video=200, classes=1)
        r = make_epoch_resampler(feats, rows, frames=FRAMES, seed=0)
        seen = set()
        for e in range(12):
            for vec in r(e)[0][0]:
                match = np.where((feats == vec).all(1))[0]
                self.assertEqual(len(match), 1)          # every sampled row is a real cached frame of that video
                seen.add(int(match[0]))
        self.assertGreater(len(seen), 3 * FRAMES)         # many distinct frames over a few epochs, not one fixed bag

    def test_keep_videos_restricts(self):
        feats, rows = _cache()
        seq, _, tgt = make_epoch_resampler(feats, rows, frames=FRAMES, seed=0, keep_videos={"v0", "v1"})(0)
        self.assertEqual(seq.shape[0], 2)

    def test_stratified_subset_keeps_every_class(self):
        targets = {f"v{i}": i % 4 for i in range(40)}
        keep = stratified_video_subset(targets, 0.3, seed=1)
        self.assertEqual(len(keep), 4 * 3)                 # ceil(0.3 * 10) = 3 per class
        self.assertEqual({targets[v] for v in keep}, {0, 1, 2, 3})
        self.assertEqual(keep, stratified_video_subset(targets, 0.3, seed=1))
        self.assertNotEqual(keep, stratified_video_subset(targets, 0.3, seed=2))
        with self.assertRaises(ValueError):
            stratified_video_subset(targets, 0.0, seed=0)


class _Tiny(nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.fc = nn.Linear(DIM, classes)

    def forward(self, x, key_padding_mask=None):
        keep = (~key_padding_mask).float().unsqueeze(-1)
        return self.fc((x * keep).sum(1) / keep.sum(1).clamp(min=1))


class TrainerHook(unittest.TestCase):
    def _run(self, **kwargs):
        feats, rows = _cache(videos=12, per_video=30)
        keep = np.concatenate([np.arange(v * 30, v * 30 + FRAMES) for v in range(12)])
        train = group_pooled_sequences_by_video(feats[keep], [rows[i] for i in keep], permute_frame_order=True, permute_seed=1)
        val = group_pooled_sequences_by_video(feats, rows, permute_frame_order=True, permute_seed=2)
        torch.manual_seed(0)
        return train_sequence_head(_Tiny(), train, val, ["c0", "c1", "c2"], frames=FRAMES, seed=0, epochs=4, **kwargs), feats, rows

    def test_default_path_unchanged_by_the_new_arguments(self):
        a, _, _ = self._run()
        b, _, _ = self._run(epoch_resampler=None, label_smoothing=0.0)
        self.assertEqual(a.best_validation_macro_f1, b.best_validation_macro_f1)
        self.assertEqual([e["train_mean_loss"] for e in a.epochs], [e["train_mean_loss"] for e in b.epochs])

    def test_resampler_is_called_every_epoch_and_changes_training(self):
        _, feats, rows = self._run()
        calls = []
        base = make_epoch_resampler(feats, rows, frames=FRAMES, seed=0)
        res, _, _ = self._run(epoch_resampler=lambda e: (calls.append(e), base(e))[1])
        self.assertEqual(calls, [0, 1, 2, 3])
        plain, _, _ = self._run()
        self.assertNotEqual([e["train_mean_loss"] for e in res.epochs], [e["train_mean_loss"] for e in plain.epochs])

    def test_label_smoothing_raises_the_loss_floor(self):
        plain, _, _ = self._run()
        smooth, _, _ = self._run(label_smoothing=0.2)
        self.assertGreater(smooth.epochs[-1]["train_mean_loss"], plain.epochs[-1]["train_mean_loss"])


class GateLogic(unittest.TestCase):
    def _rec(self, arm, fold, seed, f1):
        return {"arm": arm, "fold": fold, "seed": seed, "macro_f1": f1, "accuracy": f1, "best_epoch": 3, "train_videos": 476}

    def test_gate_needs_every_fold(self):
        from core.head_experiment import summarise_arms
        recs = []
        for fold, base in enumerate((0.40, 0.45, 0.50, 0.42), start=1):
            for seed in (0, 1):
                recs.append(self._rec("control", fold, seed, base))
                recs.append(self._rec("good", fold, seed, base + 0.05))                      # +0.05 everywhere
                recs.append(self._rec("one_fold", fold, seed, base + (0.20 if fold == 1 else 0.01)))
        s = summarise_arms(recs)
        self.assertTrue(s["good"]["gate_passed"])
        self.assertEqual(s["good"]["folds_passing_gate"], 4)
        self.assertFalse(s["one_fold"]["gate_passed"])                                       # a big gain on one fold is not enough
        self.assertEqual(s["one_fold"]["folds_positive"], 4)
        self.assertAlmostEqual(s["control"]["mean_delta"], 0.0)
        self.assertAlmostEqual(s["good"]["per_fold_delta_vs_control"][2], 0.05)

    def test_arms_registered_and_gated_subset(self):
        from core.head_experiment import ARMS, GATED
        self.assertEqual(ARMS["control"], {})
        self.assertTrue(set(GATED) <= set(ARMS))
        self.assertNotIn("control", GATED)


if __name__ == "__main__":
    unittest.main()
