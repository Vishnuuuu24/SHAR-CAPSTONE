"""Tests for `models.dinov2_backbone` (B-2: the second frozen backbone).

The CLS+mean-patch pooling and slicing logic is tested against a tiny fake
token tensor (`pool_cls_plus_mean_patch`), not the real 86M-parameter timm
network -- constructing that is slow and this module's contract does not
depend on the network's numerics, only on which tokens end up where.  Tests
that must build the real network are marked `_REAL_MODEL_AVAILABLE` and
skipped if the registered local weight file is absent, so nothing here
downloads anything.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.dinov2_backbone import DINOv2ViTB14Reg, pool_cls_plus_mean_patch  # noqa: E402

_WEIGHTS_PATH = REPO_ROOT / ".cache/weights/dinov2_vitb14_reg4-c24ecfb4.safetensors"
_WEIGHTS_SHA256 = "c24ecfb4a1d8ca79193f6b9efcffc461872a09ddac43a0931357e4802931a006"
_REAL_MODEL_AVAILABLE = _WEIGHTS_PATH.is_file()


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class PoolingLayoutTests(unittest.TestCase):
    """`pool_cls_plus_mean_patch` against a fake `[B, tokens, D]` tensor."""

    def test_layout_is_cls_then_mean_of_patches_only(self) -> None:
        # 1 CLS + 4 registers + 3 patches, D=2. CLS = [10, 20]; registers are
        # huge outliers that must NOT leak into the mean; patches are
        # [1,1], [3,3], [5,5] -> mean [3,3].
        tokens = torch.tensor(
            [[
                [10.0, 20.0],       # CLS
                [900.0, 900.0],     # register 1 -- must be excluded
                [900.0, 900.0],     # register 2 -- must be excluded
                [900.0, 900.0],     # register 3 -- must be excluded
                [900.0, 900.0],     # register 4 -- must be excluded
                [1.0, 1.0],          # patch
                [3.0, 3.0],          # patch
                [5.0, 5.0],          # patch
            ]]
        )
        pooled = pool_cls_plus_mean_patch(tokens, num_prefix_tokens=5)
        self.assertEqual(pooled.shape, (1, 4))
        torch.testing.assert_close(pooled, torch.tensor([[10.0, 20.0, 3.0, 3.0]]))

    def test_cls_half_and_mean_patch_half_are_independently_sliceable(self) -> None:
        tokens = torch.randn(2, 1 + 4 + 6, 8)
        pooled = pool_cls_plus_mean_patch(tokens, num_prefix_tokens=5)
        self.assertEqual(pooled.shape, (2, 16))
        torch.testing.assert_close(pooled[:, :8], tokens[:, 0])
        torch.testing.assert_close(pooled[:, 8:], tokens[:, 5:].mean(1))

    def test_rejects_a_non_sequence_tensor(self) -> None:
        with self.assertRaises(ValueError):
            pool_cls_plus_mean_patch(torch.randn(2, 8), num_prefix_tokens=5)

    def test_rejects_num_prefix_tokens_below_one(self) -> None:
        with self.assertRaises(ValueError):
            pool_cls_plus_mean_patch(torch.randn(1, 3, 4), num_prefix_tokens=0)


class OutFeaturesTests(unittest.TestCase):
    """Class-level facts, checkable without constructing the 86M-param network."""

    def test_out_features_is_1536(self) -> None:
        self.assertEqual(DINOv2ViTB14Reg.out_features, 1536)

    def test_out_channels_aliases_out_features_for_frozen_backbone_compat(self) -> None:
        self.assertEqual(DINOv2ViTB14Reg.out_channels, DINOv2ViTB14Reg.out_features)


class FromLocalWeightsRefusalTests(unittest.TestCase):
    """`from_local_weights` must refuse a non-DINOv2 state dict cheaply.

    These still build the real network (there is no way to validate a state
    dict against an architecture without one), so they are the slow tests
    this module has -- but they cover the failure path any CI run needs,
    while the exact-match success path is gated behind the weight file.
    """

    def test_refuses_a_mapping_that_is_not_this_architecture(self) -> None:
        with self.assertRaises(ValueError):
            DINOv2ViTB14Reg.from_local_weights({"not_a_real_key": torch.zeros(1)}, weights_name="LVD142M_REG4")

    def test_refuses_a_non_mapping_state_dict(self) -> None:
        with self.assertRaises(TypeError):
            DINOv2ViTB14Reg.from_local_weights([1, 2, 3], weights_name="LVD142M_REG4")  # type: ignore[arg-type]

    def test_refuses_an_empty_weights_name(self) -> None:
        with self.assertRaises(ValueError):
            DINOv2ViTB14Reg.from_local_weights({}, weights_name="")


@unittest.skipUnless(_REAL_MODEL_AVAILABLE, "registered DINOv2 weight file not present locally")
class RealModelTests(unittest.TestCase):
    """Exercises the actual registered checkpoint; skipped if it is absent.

    Never downloads: the weight file must already exist at the registered
    path (verified against its registered SHA-256 below) before this loads.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from safetensors.torch import load_file

        actual_sha256 = _sha256(_WEIGHTS_PATH)
        if actual_sha256 != _WEIGHTS_SHA256:
            raise AssertionError(
                f"{_WEIGHTS_PATH} sha256 {actual_sha256} does not match the registered "
                f"{_WEIGHTS_SHA256}; refusing to build tests against an unverified file"
            )
        state_dict = load_file(str(_WEIGHTS_PATH))
        cls.backbone = DINOv2ViTB14Reg.from_local_weights(state_dict, weights_name="LVD142M_REG4").eval()

    def test_registered_parameter_count(self) -> None:
        total = sum(parameter.numel() for parameter in self.backbone.parameters())
        self.assertEqual(total, 86_582_016)

    def test_forward_features_shape_at_224x224(self) -> None:
        with torch.no_grad():
            tokens = self.backbone(torch.randn(2, 3, 224, 224))
        self.assertEqual(tokens.shape, (2, 261, 768))

    def test_pooled_features_is_1536d_and_matches_cls_plus_mean_patch(self) -> None:
        images = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            tokens = self.backbone(images)
            pooled = self.backbone.pooled_features(images)
        self.assertEqual(pooled.shape, (2, 1536))
        expected = pool_cls_plus_mean_patch(tokens, self.backbone.num_prefix_tokens)
        torch.testing.assert_close(pooled, expected)

    def test_strict_load_rejects_a_second_construction_with_wrong_keys(self) -> None:
        with self.assertRaises(ValueError):
            DINOv2ViTB14Reg.from_local_weights({"unexpected": torch.zeros(1)}, weights_name="LVD142M_REG4")


if __name__ == "__main__":
    unittest.main()


class BackboneDispatchIsSharedTests(unittest.TestCase):
    """Every entry point that instantiates a backbone must go through one registry.

    `scripts/verify_p1c_feature_cache.py` hardcoded `_local_resnet50` while the
    builder dispatched on `backbone.family`, so verifying a DINOv2 cache loaded ViT
    weights into a ResNet50 and died on the state-dict mismatch. It cost two pod
    runs before a log survived to show it. The rule of two says the dispatch lives
    in one place; these tests assert the call sites actually use it.
    """

    def test_both_registered_families_have_a_loader(self) -> None:
        from scripts.build_p1c_feature_cache import _BACKBONE_LOADERS
        from core.p1c_readiness import BACKBONE_FAMILIES

        self.assertEqual(set(_BACKBONE_LOADERS), set(BACKBONE_FAMILIES))

    def test_an_unregistered_family_is_refused_by_name(self) -> None:
        from scripts.build_p1c_feature_cache import _load_backbone

        with self.assertRaises(ValueError) as caught:
            _load_backbone({"family": "torchvision_resnet18", "weights": {}})
        self.assertIn("torchvision_resnet18", str(caught.exception))

    def test_the_verifier_does_not_hardcode_a_single_backbone(self) -> None:
        source = (REPO_ROOT / "scripts" / "verify_p1c_feature_cache.py").read_text(encoding="utf-8")
        self.assertIn("_load_backbone", source)
        self.assertNotIn(
            "_local_resnet50", source,
            "verify_p1c_feature_cache.py must dispatch on backbone.family, not hardcode ResNet50",
        )
