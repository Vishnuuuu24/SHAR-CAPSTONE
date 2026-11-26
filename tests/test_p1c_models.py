from __future__ import annotations

import unittest

import torch
from torch import nn

from models.classifiers import (
    CNNLSTMTemporalBaseline,
    FrameClassifier,
    GlobalAverageLinearHead,
    ResNet50C5,
    SameC5MLPHead,
    apply_backbone_trainable_mode,
    count_parameters,
)


class TinyC5(nn.Module):
    out_channels = 16

    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, self.out_channels, kernel_size=3, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class FrameControlTests(unittest.TestCase):
    def test_resnet50_returns_spatial_c5_not_gap_vector(self) -> None:
        backbone = ResNet50C5(weights=None).eval()
        with torch.no_grad():
            c5 = backbone(torch.randn(1, 3, 64, 64))
        self.assertEqual(c5.shape, (1, 2048, 2, 2))

    def test_local_weight_loader_rejects_non_resnet_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact torchvision ResNet50"):
            ResNet50C5.from_local_weights({"not-a-resnet-key": torch.tensor(1)}, weights_name="IMAGENET1K_V2")

    def test_pooled_features_matches_the_pre_refactor_gap_plus_flatten(self) -> None:
        """CRITICAL REGRESSION CHECK (B-2): `ResNet50C5.pooled_features` must be
        `torch.equal` to what the B-1 cache call sites did themselves before the
        pooling-ownership refactor -- byte-for-byte, or the verified ResNet50
        cache at `results/p1c/p1c-frozen-feature-cache-ar-fold1/` is silently
        invalidated. No GPU is available here, so this is the CPU proof; a GPU
        re-verification against the real cache is still owed (see report)."""

        backbone = ResNet50C5(weights=None).eval()
        images = torch.randn(3, 3, 64, 64)
        with torch.no_grad():
            pooled = backbone.pooled_features(images)
            reference = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(backbone(images)), 1)
        self.assertTrue(torch.equal(pooled, reference))

    def test_both_heads_consume_identical_spatial_c5_and_return_14_logits(self) -> None:
        c5 = torch.randn(2, 16, 4, 4, requires_grad=True)
        linear = GlobalAverageLinearHead(16, 14)
        mlp = SameC5MLPHead(16, hidden_dim=8, num_classes=14, dropout=0.0)
        linear_logits = linear(c5)
        mlp_logits = mlp(c5)
        self.assertEqual(linear_logits.shape, (2, 14))
        self.assertEqual(mlp_logits.shape, (2, 14))
        (linear_logits.sum() + mlp_logits.sum()).backward()
        self.assertIsNotNone(c5.grad)
        self.assertTrue(torch.isfinite(c5.grad).all())

    def test_heads_reject_gap_vector_as_spatial_input(self) -> None:
        vector = torch.randn(2, 2048)
        with self.assertRaisesRegex(ValueError, "spatial"):
            GlobalAverageLinearHead(2048, 14)(vector)
        with self.assertRaisesRegex(ValueError, "spatial"):
            SameC5MLPHead(2048, 32, 14, 0.0)(vector)


class BackboneFreezeTests(unittest.TestCase):
    def _model(self, mode: str) -> FrameClassifier:
        backbone = apply_backbone_trainable_mode(ResNet50C5(weights=None), mode)
        return FrameClassifier(backbone, GlobalAverageLinearHead(backbone.out_channels, 14))

    def test_full_mode_returns_the_backbone_unwrapped(self) -> None:
        backbone = ResNet50C5(weights=None)
        self.assertIs(apply_backbone_trainable_mode(backbone, "full"), backbone)

    def test_unregistered_mode_is_a_caller_bug_and_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "trainable mode must be one of"):
            apply_backbone_trainable_mode(ResNet50C5(weights=None), "partial")

    def test_full_mode_trainable_parameter_count_equals_total(self) -> None:
        model = self._model("full")
        counts = count_parameters(model)
        self.assertEqual(counts["trainable_parameters"], counts["total_parameters"])

    def test_frozen_mode_trainable_parameter_count_is_head_only(self) -> None:
        # 2048 (C5 channels) * 14 (classes) weights + 14 biases.
        model = self._model("frozen")
        counts = count_parameters(model)
        self.assertEqual(counts["trainable_parameters"], 2048 * 14 + 14)
        self.assertGreater(counts["total_parameters"], counts["trainable_parameters"])

    def test_frozen_backbone_parameters_have_requires_grad_false(self) -> None:
        model = self._model("frozen")
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.head.parameters()))

    def test_frozen_backbone_stays_in_eval_after_model_train(self) -> None:
        model = self._model("frozen")
        model.train()
        self.assertFalse(model.backbone.training)
        # Every submodule under the wrapped backbone, including BatchNorm.
        self.assertTrue(all(not module.training for module in model.backbone.modules()))
        self.assertTrue(model.head.training)

    def test_frozen_backbone_batch_norm_running_stats_do_not_update(self) -> None:
        model = self._model("frozen")
        model.train()
        bn = next(module for module in model.backbone.modules() if isinstance(module, nn.BatchNorm2d))
        running_mean_before = bn.running_mean.clone()
        images = torch.randn(4, 3, 64, 64)
        for _ in range(3):
            model(images)
        self.assertTrue(torch.equal(bn.running_mean, running_mean_before))

    def test_full_backbone_forward_still_runs_and_matches_reference_shape(self) -> None:
        model = self._model("full")
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(logits.shape, (2, 14))

    def test_frozen_backbone_pooled_features_delegates_to_the_wrapped_backbone(self) -> None:
        """B-2: `core.feature_cache` calls `backbone.pooled_features(images)` on
        whatever `apply_backbone_trainable_mode` returns, so `FrozenBackbone`
        must forward that call rather than only forwarding `forward`."""

        model = self._model("frozen")
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            pooled = model.backbone.pooled_features(images)
            reference = model.backbone.backbone.pooled_features(images)
        self.assertTrue(torch.equal(pooled, reference))
        self.assertEqual(pooled.shape, (2, 2048))


class TemporalControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = CNNLSTMTemporalBaseline(
            TinyC5(),
            projection_dim=8,
            hidden_dim=6,
            num_layers=1,
            num_classes=14,
            dropout=0.0,
        )

    def test_source_grouped_temporal_forward_and_gradient(self) -> None:
        clips = torch.randn(2, 3, 3, 16, 16, requires_grad=True)
        logits = self.model(clips, torch.tensor([3, 2]), ["video-a", "video-b"])
        self.assertEqual(logits.shape, (2, 14))
        logits.sum().backward()
        self.assertIsNotNone(clips.grad)
        self.assertTrue(torch.isfinite(clips.grad).all())

    def test_temporal_model_requires_one_source_id_per_sequence(self) -> None:
        clips = torch.randn(2, 3, 3, 16, 16)
        with self.assertRaisesRegex(ValueError, "source_video_ids"):
            self.model(clips, torch.tensor([3, 3]), ["only-one"])
        with self.assertRaisesRegex(ValueError, "lengths"):
            self.model(clips, torch.tensor([3, 0]), ["a", "b"])

    def test_temporal_model_rejects_non_integral_lengths_before_gather(self) -> None:
        clips = torch.randn(2, 3, 3, 16, 16)
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            self.model(clips, torch.tensor([3.0, 2.0]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
