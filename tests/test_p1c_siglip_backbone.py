"""Tests for `models.siglip_backbone` (SigLIP as the third frozen backbone).

Uses a tiny randomly-initialised `SiglipVisionModel` (no downloads, no weight
file). The real-weights test is skipped if the registered local file is absent.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
from transformers import SiglipVisionConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import siglip_backbone  # noqa: E402
from models.siglip_backbone import SigLIPBaseP16  # noqa: E402

_WEIGHTS_PATH = REPO_ROOT / ".cache/weights/siglip_base_patch16_224_vision-f5d84933.safetensors"
_WEIGHTS_SHA256 = "f5d84933ac67e11fcd249aba72bb2888028fc11507a4b81629c095c881031b77"

# 32x32 image, patch 8 -> 4x4 = 16 patches; hidden 16.
_TINY = SiglipVisionConfig(
    hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, image_size=32, patch_size=8
)


def _tiny() -> SigLIPBaseP16:
    torch.manual_seed(0)
    model = SigLIPBaseP16(config=_TINY)
    model.eval()
    return model


class TinyModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _tiny()
        torch.manual_seed(1)
        self.images = torch.randn(3, 3, 32, 32)

    def test_forward_is_pooler_then_patch_tokens(self) -> None:
        with torch.no_grad():
            tokens = self.model(self.images)
            ref = self.model.network(pixel_values=self.images)
        self.assertEqual(tuple(tokens.shape), (3, 1 + 16, 16))
        self.assertTrue(torch.equal(tokens[:, 0], ref.pooler_output))
        self.assertTrue(torch.equal(tokens[:, 1:], ref.last_hidden_state))

    def test_pooled_is_pooler_concat_mean_patch(self) -> None:
        with torch.no_grad():
            pooled = self.model.pooled_features(self.images)
            ref = self.model.network(pixel_values=self.images)
        self.assertEqual(pooled.shape, (3, 32))
        self.assertEqual(pooled.dtype, torch.float32)
        self.assertTrue(torch.allclose(pooled[:, :16], ref.pooler_output))
        self.assertTrue(torch.allclose(pooled[:, 16:], ref.last_hidden_state.mean(1), atol=1e-6))

    def test_deterministic_in_eval(self) -> None:
        with torch.no_grad():
            a = self.model.pooled_features(self.images)
            b = self.model.pooled_features(self.images)
        self.assertTrue(torch.equal(a, b))

    def test_batch_rows_are_independent(self) -> None:
        with torch.no_grad():
            full = self.model.pooled_features(self.images)
            single = self.model.pooled_features(self.images[1:2])
        self.assertTrue(torch.allclose(full[1], single[0], atol=1e-5))

    def test_non_native_size_is_resized_bilinear_not_normalised(self) -> None:
        big = torch.randn(2, 3, 48, 64)
        with torch.no_grad():
            got = self.model.pooled_features(big)
            expected_in = torch.nn.functional.interpolate(
                big, size=(32, 32), mode="bilinear", align_corners=False, antialias=True
            )
            expected = self.model.pooled_features(expected_in)
        self.assertTrue(torch.equal(got, expected))

    def test_native_size_skips_resize(self) -> None:
        with mock.patch.object(torch.nn.functional, "interpolate", side_effect=AssertionError("resized")):
            with torch.no_grad():
                self.model.pooled_features(self.images)

    def test_wrapper_does_not_normalise(self) -> None:
        # The pipeline owns mean/std = 0.5; scaling the input must change the output.
        with torch.no_grad():
            a = self.model.pooled_features(self.images)
            b = self.model.pooled_features(self.images * 2.0 + 1.0)
        self.assertFalse(torch.allclose(a, b))

    def test_rejects_bad_rank_and_channels(self) -> None:
        with self.assertRaises(ValueError):
            self.model.pooled_features(torch.randn(3, 32, 32))
        with self.assertRaises(ValueError):
            self.model.pooled_features(torch.randn(1, 1, 32, 32))

    def test_registered_constants(self) -> None:
        self.assertEqual(SigLIPBaseP16.num_prefix_tokens, 1)
        self.assertEqual(SigLIPBaseP16.out_features, 1536)
        self.assertEqual(siglip_backbone.NUM_PATCHES, 196)


class FromLocalWeightsTests(unittest.TestCase):
    def test_exact_state_dict_loads_and_reproduces(self) -> None:
        src = _tiny()
        with mock.patch.object(siglip_backbone, "_vision_config", return_value=_TINY), mock.patch.object(
            siglip_backbone, "EMBED_DIM", 16
        ):
            loaded = SigLIPBaseP16.from_local_weights(src.network.state_dict(), weights_name="X")
        loaded.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            self.assertTrue(torch.equal(src.pooled_features(x), loaded.pooled_features(x)))
        self.assertEqual(loaded.weights_name, "X")

    def test_wrong_keys_are_refused(self) -> None:
        with mock.patch.object(siglip_backbone, "_vision_config", return_value=_TINY), mock.patch.object(
            siglip_backbone, "EMBED_DIM", 16
        ):
            bad = {f"vision_model.{k}": v for k, v in _tiny().network.state_dict().items()}  # un-stripped HF keys
            with self.assertRaises(ValueError):
                SigLIPBaseP16.from_local_weights(bad, weights_name="X")
            partial = dict(list(_tiny().network.state_dict().items())[:-1])
            with self.assertRaises(ValueError):
                SigLIPBaseP16.from_local_weights(partial, weights_name="X")

    def test_empty_weights_name_refused(self) -> None:
        with self.assertRaises(ValueError):
            SigLIPBaseP16.from_local_weights({}, weights_name="")


class RegistrationTests(unittest.TestCase):
    def test_family_registered_everywhere_it_must_be(self) -> None:
        from core.feature_cache import FEATURE_CACHE_POOLING_BY_FAMILY
        from core.p1c_readiness import CANONICAL_WEIGHT_ENUMS_BY_FAMILY
        from scripts.build_p1c_feature_cache import _BACKBONE_LOADERS

        family = "transformers_siglip_base_p16_224"
        self.assertIn(family, FEATURE_CACHE_POOLING_BY_FAMILY)
        self.assertIn(family, CANONICAL_WEIGHT_ENUMS_BY_FAMILY)
        self.assertIn(family, _BACKBONE_LOADERS)
        # Distinct pooling id: a DINOv2 cache must never be served to a SigLIP config.
        self.assertNotEqual(
            FEATURE_CACHE_POOLING_BY_FAMILY[family], FEATURE_CACHE_POOLING_BY_FAMILY["timm_dinov2_vitb14_reg4"]
        )


@unittest.skipUnless(_WEIGHTS_PATH.is_file(), "registered SigLIP weights not on disk")
class RealWeightsTests(unittest.TestCase):
    def test_digest_and_load(self) -> None:
        import hashlib

        from safetensors.torch import load_file

        digest = hashlib.sha256()
        with _WEIGHTS_PATH.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
        self.assertEqual(digest.hexdigest(), _WEIGHTS_SHA256)
        model = SigLIPBaseP16.from_local_weights(load_file(str(_WEIGHTS_PATH)), weights_name="W").eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 92_884_224)
        with torch.no_grad():
            out = model.pooled_features(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (1, 1536))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
