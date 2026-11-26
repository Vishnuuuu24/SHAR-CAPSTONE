"""Tests for the VideoMAE block-cache + partial fine-tune modules (tiny random models, CPU only)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from transformers import VideoMAEConfig, VideoMAEModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_videomae_cache import (ARRAY_NAME, ROWS_NAME, cache_nbytes, decode_roles, encode_roles, fold_row_indices,
                                     read_rows, select_clips, ulp_gate, union_plan, write_rows)
from core.p1c_videomae_finetune import ClipStore, epoch_batches, lr_multiplier, train_run, video_level_probs
from core.p1c_videomae_split import (CACHE_BLOCKS, TopBlocksClassifier, encode_to_cache_point, param_groups,
                                     trainable_parameter_count)
from core.p1c_videomae_summary import aggregate, gate
from models.videomae_backbone import pool_videomae_tokens

torch.set_num_threads(4)
DIM = 32
CLASSES = ["a", "b", "c"]


def tiny_backbone(seed: int = 0):
    torch.manual_seed(seed)
    config = VideoMAEConfig(num_frames=4, image_size=32, patch_size=16, tubelet_size=2, hidden_size=DIM,
                            num_hidden_layers=12, num_attention_heads=2, intermediate_size=64)
    net = VideoMAEModel(config).eval()
    return SimpleNamespace(network=net, fc_norm=nn.LayerNorm(DIM).eval())


def clip_rows(videos: dict[str, tuple[int, int]]):
    """{video: (target, n_clips)} -> tiled rows, 16-frame windows at stride 5."""
    rows = []
    for v, (t, n) in videos.items():
        for k in range(n):
            rows.append({"source_video_id": v, "start_source_frame": k * 80, "end_source_frame": k * 80 + 75,
                         "target": str(t), "label": CLASSES[t]})
    return rows


class HelperTests(unittest.TestCase):
    def test_select_clips_deterministic_spread_and_short_video(self):
        rows = clip_rows({"v1": (0, 30), "v2": (1, 3)})
        a, b = select_clips(rows), select_clips(rows)
        self.assertEqual(a, b)
        v1 = [r for r in a if r["source_video_id"] == "v1"]
        self.assertEqual(len(v1), 8)
        self.assertEqual(v1[0]["start_source_frame"], 0)
        self.assertEqual(v1[-1]["start_source_frame"], 29 * 80)
        self.assertEqual(len([r for r in a if r["source_video_id"] == "v2"]), 3)

    def test_union_plan_membership_and_conflict(self):
        f1 = {"train": clip_rows({"v1": (0, 20), "v2": (1, 20)}), "validation": clip_rows({"v3": (2, 20)})}
        f2 = {"train": clip_rows({"v3": (2, 20), "v1": (0, 20)}), "validation": clip_rows({"v2": (1, 20)})}
        plan = union_plan({1: f1, 2: f2})
        self.assertEqual(len(plan), 3 * 8)
        roles = {c["source_video_id"]: c["roles"] for c in plan}
        self.assertEqual(roles["v2"], {1: "train", 2: "validation"})
        self.assertEqual(roles["v3"], {1: "validation", 2: "train"})
        bad = {"train": clip_rows({"v3": (2, 12), "v1": (0, 20)}), "validation": clip_rows({"v2": (1, 20)})}
        with self.assertRaises(ValueError):
            union_plan({1: f1, 2: bad})

    def test_roles_rows_roundtrip_and_fold_rows(self):
        plan = union_plan({1: {"train": clip_rows({"v1": (0, 9)}), "validation": clip_rows({"v2": (1, 9)})}})
        self.assertEqual(decode_roles(encode_roles({2: "train", 1: "validation"})), {1: "validation", 2: "train"})
        with tempfile.TemporaryDirectory() as d:
            write_rows(Path(d) / ROWS_NAME, plan)
            rows = read_rows(Path(d) / ROWS_NAME)
        self.assertEqual(fold_row_indices(rows, 1, "train"), list(range(0, 8)))
        self.assertEqual(fold_row_indices(rows, 1, "validation"), list(range(8, 16)))
        self.assertEqual(fold_row_indices(rows, 2, "train"), [])

    def test_registered_cache_byte_total(self):
        self.assertEqual(cache_nbytes(5395), 12_993_576_960)

    def test_lr_schedule(self):
        spe, ep = 10, 8
        m = [lr_multiplier(s, spe, ep) for s in range(spe * ep)]
        self.assertAlmostEqual(m[0], 0.1)
        self.assertAlmostEqual(m[spe - 1], 1.0)
        self.assertAlmostEqual(m[spe], 1.0)
        self.assertTrue(all(a >= b for a, b in zip(m[spe:], m[spe + 1:])))
        self.assertLess(m[-1], 0.01)

    def test_epoch_batches(self):
        a = epoch_batches(37, 16, seed=0, fold=1, epoch=1)
        self.assertEqual(a, epoch_batches(37, 16, seed=0, fold=1, epoch=1))
        self.assertEqual(sorted(i for b in a for i in b), list(range(37)))
        self.assertEqual([len(b) for b in a], [16, 16, 5])
        self.assertNotEqual(a, epoch_batches(37, 16, seed=0, fold=1, epoch=2))
        self.assertNotEqual(a, epoch_batches(37, 16, seed=1, fold=1, epoch=1))

    def test_video_level_mean_and_ulp_gate(self):
        videos, p = video_level_probs(np.array([[1, 0], [0, 1], [0, 1.0]]), ["b", "a", "a"])
        self.assertEqual(videos, ["a", "b"])
        np.testing.assert_allclose(p, [[0, 1], [1, 0]])
        x = np.ones(16, np.float16)
        y = x.copy(); y[0] = np.float16(1.0) + np.float16(0.002)
        self.assertTrue(ulp_gate(x, y)["ok"])
        y[0] = 1.5
        self.assertFalse(ulp_gate(x, y)["ok"])

    def test_gate_and_aggregate(self):
        self.assertTrue(gate({1: .05, 2: .04, 3: .06, 4: .09}, [1, 2, 3, 4])["met"])
        self.assertFalse(gate({1: .05, 2: .039, 3: .06, 4: .09}, [1, 2, 3, 4])["met"])
        self.assertFalse(gate({1: .5}, [1, 2, 3, 4])["met"])
        runs = []
        for arm, base in (("k0", 0.30), ("k2", 0.36), ("k4", 0.40)):
            for fold in (1, 2):
                for seed in (0, 1):
                    f1 = base + 0.01 * fold + 0.001 * seed
                    runs.append({"arm": arm, "fold": fold, "seed": seed, "final_epoch_macro_f1": f1, "final_epoch_accuracy": f1,
                                 "per_class_f1": {c: f1 for c in CLASSES}, "per_class_recall": {c: f1 for c in CLASSES},
                                 "oracle_selected_epoch": {"val_macro_f1": f1 + .01}, "epoch_curve": [{"val_macro_f1": f1}] * 2,
                                 "wall_seconds": 1.0, "trainable_parameters": 5})
        s = aggregate(runs, CLASSES, {1: 0.3, 2: 0.3}, folds=(1, 2))
        self.assertAlmostEqual(s["paired_vs_k0"]["k2"]["mean_delta"], 0.06)
        self.assertTrue(s["paired_vs_k0"]["k2"]["gate"]["met"])
        self.assertAlmostEqual(s["arms"]["k0"]["per_fold_mean"][1], 0.3105)
        self.assertEqual(s["arms"]["k4"]["n_runs"], 4)
        self.assertFalse(s["vs_registered_0.4599"]["k0"]["gate"]["met"])


class ModelPlumbingTests(unittest.TestCase):
    def setUp(self):
        self.backbone = tiny_backbone()
        torch.manual_seed(1)
        self.clips = torch.randn(3, 4, 3, 32, 32)

    def test_cache_split_reproduces_full_forward(self):
        with torch.no_grad():
            full = self.backbone.network(pixel_values=self.clips).last_hidden_state
            cached = encode_to_cache_point(self.backbone, self.clips)
            h = cached
            for layer in list(self.backbone.network.encoder.layer)[CACHE_BLOCKS:]:
                h = layer(h)
            self.assertLess(float((h - full).abs().max()), 1e-5)
            model = TopBlocksClassifier(self.backbone, num_classes=3, trainable_blocks=0, seed=0).eval()
            pooled_full = pool_videomae_tokens(full, self.backbone.fc_norm)
            pooled_split = model.fc_norm(model_tokens(model, cached).mean(1))
            self.assertLess(float((pooled_full - pooled_split).abs().max()), 1e-5)

    def test_only_top_blocks_and_head_get_gradients(self):
        cached = encode_to_cache_point(self.backbone, self.clips).detach()
        for k in (0, 2, 4):
            model = TopBlocksClassifier(self.backbone, num_classes=3, trainable_blocks=k, seed=0)
            model(cached).sum().backward()
            for i, layer in enumerate(model.layers):
                trainable = i >= 4 - k
                for p in layer.parameters():
                    self.assertEqual(p.requires_grad, trainable)
                    self.assertEqual(p.grad is not None and float(p.grad.abs().sum()) > 0, trainable, (k, i))
            self.assertTrue(all(p.grad is not None for p in model.head.parameters()))
            self.assertTrue(all(p.grad is None for p in model.fc_norm.parameters()))
        k2 = TopBlocksClassifier(self.backbone, num_classes=3, trainable_blocks=2, seed=0)
        self.assertEqual(trainable_parameter_count(k2), sum(p.numel() for l in list(k2.layers)[2:] for p in l.parameters()) + DIM * 3 + 3)

    def test_head_init_is_seeded_k31(self):
        w = lambda s: TopBlocksClassifier(self.backbone, num_classes=3, trainable_blocks=2, seed=s).head.weight.detach().clone()
        torch.manual_seed(123); a = w(0)
        torch.manual_seed(999); b = w(0)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, w(1)))

    def test_one_step_reduces_loss(self):
        cached = encode_to_cache_point(self.backbone, self.clips).detach()
        y = torch.tensor([0, 1, 2])
        model = TopBlocksClassifier(self.backbone, num_classes=3, trainable_blocks=2, seed=0)
        opt = torch.optim.AdamW(param_groups(model, block_lr=1e-3, head_lr=1e-2, weight_decay=0.0))
        before = nn.functional.cross_entropy(model(cached), y)
        opt.zero_grad(); before.backward(); opt.step()
        after = nn.functional.cross_entropy(model(cached), y)
        self.assertLess(float(after), float(before))

    def test_grad_checkpoint_matches(self):
        cached = encode_to_cache_point(self.backbone, self.clips).detach()
        grads = []
        for ckpt in (False, True):
            m = TopBlocksClassifier(self.backbone, num_classes=3, trainable_blocks=4, seed=0, grad_checkpoint=ckpt).train()
            m(cached).square().sum().backward()
            grads.append(torch.cat([p.grad.flatten() for p in m.block_parameters()]))
        self.assertLess(float((grads[0] - grads[1]).abs().max()), 1e-4)

    def test_train_run_end_to_end_on_tiny_cache(self):
        plan = union_plan({1: {"train": clip_rows({f"t{i}": (i % 3, 9) for i in range(6)}),
                               "validation": clip_rows({f"x{i}": (i % 3, 9) for i in range(3)})}})
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_rows(root / ROWS_NAME, plan)
            with torch.no_grad():
                hidden = np.stack([encode_to_cache_point(self.backbone, torch.randn(1, 4, 3, 32, 32))[0].numpy() for _ in plan])
            np.save(root / ARRAY_NAME, hidden.astype(np.float16))
            store = ClipStore(root, read_rows(root / ROWS_NAME))
            runs = [train_run(self.backbone, store, fold=1, arm=arm, seed=0, class_names=CLASSES, device=torch.device("cpu"),
                              epochs=2, num_workers=0, log=lambda m: None) for arm in ("k0", "k2")]
        for r in runs:
            self.assertEqual(len(r["epoch_curve"]), 2)
            self.assertEqual(r["n_val_videos"], 3)
            self.assertEqual(r["n_train_clips"], 48)
            self.assertEqual(r["per_video"][0]["true_label"] in CLASSES, True)
            self.assertAlmostEqual(sum(r["per_video"][0]["probs"]), 1.0, places=4)
        self.assertGreater(runs[1]["trainable_parameters"], runs[0]["trainable_parameters"])


def model_tokens(model, cached):
    h = cached.float()
    for layer in model.layers:
        h = layer(h)
    return h


if __name__ == "__main__":
    unittest.main()
