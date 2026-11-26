"""Tests for `core.p1c_domain_adversarial` against a tiny fake DINOv2-shaped
backbone -- never the real ~86M-parameter network or real weights (docs/PROJECT_RULES.md
compute rules: no real caches, no real DINOv2 weights in tests).
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from core.p1c_domain_adversarial import (
    GRL,
    DomainAdversarialModel,
    dann_lambda,
    load_checkpoint,
    save_checkpoint,
    smoke_test,
)

EMBED_DIM = 8
NUM_PREFIX_TOKENS = 3
NUM_PATCHES = 4
NUM_BLOCKS = 4
IMAGE_CHANNELS, IMAGE_H, IMAGE_W = 3, 4, 4
IMAGE_PIXELS = IMAGE_CHANNELS * IMAGE_H * IMAGE_W


class _FakeBlock(nn.Module):
    """Stands in for a timm transformer block: any module operating on the
    last (embed) dimension of a `[N, tokens, D]` tensor."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _FakePatchEmbed(nn.Module):
    def __init__(self, image_pixels: int, num_patches: int, embed_dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.linear = nn.Linear(image_pixels, num_patches * embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        n = images.shape[0]
        flat = images.reshape(n, -1)
        return self.linear(flat).reshape(n, self.num_patches, self.embed_dim)


class _FakeNetwork(nn.Module):
    """Mimics the timm `VisionTransformer` attributes/methods
    `DomainAdversarialModel.forward` calls directly: `patch_embed`,
    `_pos_embed`, `patch_drop`, `norm_pre`, `blocks` (a `Sequential`, sliceable
    the same way timm's is), `norm`, `num_prefix_tokens`, and
    `forward_features` for interface parity with the real backbone.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = EMBED_DIM
        self.num_prefix_tokens = NUM_PREFIX_TOKENS
        self.patch_embed = _FakePatchEmbed(IMAGE_PIXELS, NUM_PATCHES, EMBED_DIM)
        self.prefix_tokens = nn.Parameter(torch.randn(1, NUM_PREFIX_TOKENS, EMBED_DIM) * 0.01)
        self.pos_embed = nn.Parameter(torch.randn(1, NUM_PREFIX_TOKENS + NUM_PATCHES, EMBED_DIM) * 0.01)
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.Sequential(*[_FakeBlock(EMBED_DIM) for _ in range(NUM_BLOCKS)])
        self.norm = nn.LayerNorm(EMBED_DIM)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        prefix = self.prefix_tokens.expand(x.shape[0], -1, -1)
        return torch.cat([prefix, x], dim=1) + self.pos_embed

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        return self.norm(x)


class _FakeBackbone(nn.Module):
    """Mimics `models.dinov2_backbone.DINOv2ViTB14Reg`'s public surface."""

    def __init__(self) -> None:
        super().__init__()
        self.network = _FakeNetwork()
        self.num_prefix_tokens = NUM_PREFIX_TOKENS
        self.out_features = 2 * EMBED_DIM
        self.out_channels = self.out_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network.forward_features(images)


def _make_model(unfreeze_last_blocks: int, n_domains: int = 3, grl_lambda: float = 1.0) -> DomainAdversarialModel:
    return DomainAdversarialModel(
        _FakeBackbone(),
        unfreeze_last_blocks=unfreeze_last_blocks,
        n_domains=n_domains,
        grl_lambda=grl_lambda,
    )


def _make_batch(batch_size: int = 3, frames: int = 16, n_domains: int = 3):
    clips = torch.randn(batch_size, frames, IMAGE_CHANNELS, IMAGE_H, IMAGE_W)
    event_y = torch.randint(0, 2, (batch_size,)).float()
    domain_y = torch.randint(0, n_domains, (batch_size,))
    return clips, event_y, domain_y


class DannLambdaTests(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(dann_lambda(0.0), 0.0)
        self.assertAlmostEqual(dann_lambda(1.0), 0.9999, places=3)

    def test_monotone(self):
        points = [dann_lambda(p) for p in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)]
        self.assertEqual(points, sorted(points))

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            dann_lambda(-0.01)
        with self.assertRaises(ValueError):
            dann_lambda(1.01)


class GRLTests(unittest.TestCase):
    def test_backward_is_exactly_negative_lambda_times_grad(self):
        for lambda_ in (0.0, 0.3, 1.0, 2.5):
            x = torch.randn(5, 4, requires_grad=True)
            grl = GRL(lambda_)
            out = grl(x)
            upstream = torch.randn_like(out)
            out.backward(upstream)
            self.assertTrue(torch.allclose(x.grad, -lambda_ * upstream))

    def test_forward_is_identity(self):
        x = torch.randn(2, 3)
        self.assertTrue(torch.equal(GRL(1.0)(x), x))


class FreezingTests(unittest.TestCase):
    def test_unfreeze_zero_leaves_every_backbone_param_frozen(self):
        model = _make_model(unfreeze_last_blocks=0)
        for parameter in model.backbone.parameters():
            self.assertFalse(parameter.requires_grad)
        self.assertEqual(model.unfrozen_backbone_parameter_names(), set())

    def test_unfreeze_two_trains_exactly_last_two_blocks_and_norm(self):
        model = _make_model(unfreeze_last_blocks=2)
        expected = model.unfrozen_backbone_parameter_names()
        expected_prefixes = {"backbone.network.blocks.2.", "backbone.network.blocks.3.", "backbone.network.norm."}
        self.assertTrue(all(any(name.startswith(p) for p in expected_prefixes) for name in expected))
        self.assertTrue(any(name.startswith("backbone.network.blocks.2.") for name in expected))
        self.assertTrue(any(name.startswith("backbone.network.blocks.3.") for name in expected))
        self.assertTrue(any(name.startswith("backbone.network.norm.") for name in expected))

        actual_trainable_backbone_names = {
            name for name, parameter in model.named_parameters() if name.startswith("backbone.") and parameter.requires_grad
        }
        self.assertEqual(actual_trainable_backbone_names, expected)
        # blocks 0 and 1 (and patch_embed/pos_embed) must stay frozen.
        for name, parameter in model.named_parameters():
            if name.startswith("backbone.network.blocks.0.") or name.startswith("backbone.network.blocks.1."):
                self.assertFalse(parameter.requires_grad)
            if name.startswith("backbone.network.patch_embed") or name in (
                "backbone.network.prefix_tokens",
                "backbone.network.pos_embed",
            ):
                self.assertFalse(parameter.requires_grad)

    def test_backbone_stays_in_eval_after_train(self):
        model = _make_model(unfreeze_last_blocks=2)
        model.train(True)
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.backbone.network.training)
        self.assertTrue(model.event_head.training)
        self.assertTrue(model.domain_head.training)


class SmokeTestTests(unittest.TestCase):
    def test_all_checks_pass_with_unfrozen_blocks_and_domain_head(self):
        model = _make_model(unfreeze_last_blocks=2, n_domains=3, grl_lambda=1.0)
        clips, event_y, domain_y = _make_batch(n_domains=3)
        result = smoke_test(model, clips, event_y, domain_y)

        self.assertTrue(result["grads_reach_unfrozen_blocks"], result["grads_reach_unfrozen_blocks_evidence"])
        self.assertTrue(result["frozen_blocks_have_no_grad"])
        self.assertTrue(result["grl_reverses"])
        self.assertTrue(result["event_head_sees_labels"], result["event_head_sees_labels_evidence"])
        self.assertTrue(result["domain_head_sees_labels"], result["domain_head_sees_labels_evidence"])
        self.assertTrue(result["losses_finite"])
        self.assertGreater(result["event_grad_norm"], 0.0)
        self.assertGreater(result["domain_grad_norm"], 0.0)
        self.assertGreater(result["total_grad_norm"], 0.0)

    def test_frozen_arm_has_no_unfrozen_grads_to_reach(self):
        model = _make_model(unfreeze_last_blocks=0, n_domains=3, grl_lambda=1.0)
        clips, event_y, domain_y = _make_batch(n_domains=3)
        result = smoke_test(model, clips, event_y, domain_y)
        # No unfrozen backbone parameters exist in this arm, so this check is
        # vacuously false by construction, not evidence of a bug -- the
        # frozen-side check is what matters here.
        self.assertFalse(result["grads_reach_unfrozen_blocks"])
        self.assertTrue(result["frozen_blocks_have_no_grad"])
        self.assertEqual(result["event_grad_norm"], 0.0)
        self.assertEqual(result["domain_grad_norm"], 0.0)


class ForwardShapeTests(unittest.TestCase):
    def test_output_shapes(self):
        model = _make_model(unfreeze_last_blocks=2, n_domains=5)
        clips, _, _ = _make_batch(batch_size=2, n_domains=5)
        event_logit, domain_logits, representation = model(clips)
        self.assertEqual(tuple(event_logit.shape), (2,))
        self.assertEqual(tuple(domain_logits.shape), (2, 5))
        self.assertEqual(tuple(representation.shape), (2, 2 * EMBED_DIM))

    def test_rejects_wrong_ndim(self):
        model = _make_model(unfreeze_last_blocks=0)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, IMAGE_H, IMAGE_W))


class CheckpointTests(unittest.TestCase):
    def test_round_trip_reproduces_event_logits(self):
        model = _make_model(unfreeze_last_blocks=2, n_domains=3, grl_lambda=0.42)
        clips, _, _ = _make_batch(batch_size=2, n_domains=3)
        model.eval()
        with torch.no_grad():
            original_logit, _, _ = model(clips)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "checkpoint.pt")
            save_checkpoint(model, path)

            reloaded = copy.deepcopy(model)
            # Perturb the reloaded model's trainable weights so the round
            # trip is a real check, not a no-op comparison.
            with torch.no_grad():
                for parameter in reloaded.parameters():
                    if parameter.requires_grad:
                        parameter.add_(1.0)
            reloaded.grl_lambda = 0.0
            reloaded.eval()

            load_checkpoint(reloaded, path)
            self.assertEqual(reloaded.grl_lambda, 0.42)
            with torch.no_grad():
                reloaded_logit, _, _ = reloaded(clips)
            self.assertTrue(torch.allclose(original_logit, reloaded_logit))

    def test_load_rejects_mismatched_arm(self):
        model = _make_model(unfreeze_last_blocks=2, n_domains=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "checkpoint.pt")
            save_checkpoint(model, path)
            mismatched = _make_model(unfreeze_last_blocks=0, n_domains=3)
            with self.assertRaises(ValueError):
                load_checkpoint(mismatched, path)


if __name__ == "__main__":
    unittest.main()
