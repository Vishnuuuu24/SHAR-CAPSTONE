"""Tests for `models.videomae_backbone` (clip-level VideoMAE-B/16 Kinetics-400 backbone).

The q_bias/v_bias remap and mean+fc_norm pooling are tested against tiny fake
tensors, not the real ~86M-parameter transformers network -- constructing
that is slow and this module's contract does not depend on the network's
numerics, only on which keys land where. The one test that must build the
real network with real weights is marked `_REAL_MODEL_AVAILABLE` and skipped
if the registered local weight file is absent, so nothing here downloads
anything. `HF_HUB_OFFLINE=1` is set before `transformers` is imported so
nothing here can reach the network even by accident.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.videomae_backbone import (  # noqa: E402
    EMBED_DIM,
    NUM_TOKENS,
    VideoMAEBaseK400,
    _remap_checkpoint_state_dict,
    pool_videomae_tokens,
)

_WEIGHTS_PATH = REPO_ROOT / ".cache/weights/videomae-base-k400-065cfde5.safetensors"
_WEIGHTS_SHA256 = "065cfde56f97671da4d196e1a3a6d3cf9304dcad92eb671e3632a6c6cdd04f73"
_REAL_MODEL_AVAILABLE = _WEIGHTS_PATH.is_file()


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fake_layer_state_dict(layer: int, width: int = EMBED_DIM) -> dict[str, torch.Tensor]:
    prefix = f"videomae.encoder.layer.{layer}.attention.attention."
    return {
        prefix + "query.weight": torch.randn(width, width),
        prefix + "q_bias": torch.randn(width),
        prefix + "key.weight": torch.randn(width, width),
        prefix + "value.weight": torch.randn(width, width),
        prefix + "v_bias": torch.randn(width),
    }


def _fake_checkpoint_state_dict() -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for layer in range(2):
        state_dict.update(_fake_layer_state_dict(layer))
    state_dict["fc_norm.weight"] = torch.randn(EMBED_DIM)
    state_dict["fc_norm.bias"] = torch.randn(EMBED_DIM)
    state_dict["classifier.weight"] = torch.randn(400, EMBED_DIM)
    state_dict["classifier.bias"] = torch.randn(400)
    return state_dict


class RemapTests(unittest.TestCase):
    """`_remap_checkpoint_state_dict` on a synthetic state dict."""

    def test_q_bias_and_v_bias_land_in_query_and_value(self) -> None:
        state_dict = _fake_checkpoint_state_dict()
        encoder_state_dict, _ = _remap_checkpoint_state_dict(state_dict)
        for layer in range(2):
            prefix = f"encoder.layer.{layer}.attention.attention."
            torch.testing.assert_close(
                encoder_state_dict[prefix + "query.bias"],
                state_dict[f"videomae.encoder.layer.{layer}.attention.attention.q_bias"],
            )
            torch.testing.assert_close(
                encoder_state_dict[prefix + "value.bias"],
                state_dict[f"videomae.encoder.layer.{layer}.attention.attention.v_bias"],
            )

    def test_key_bias_is_synthesised_zero(self) -> None:
        encoder_state_dict, _ = _remap_checkpoint_state_dict(_fake_checkpoint_state_dict())
        for layer in range(2):
            key_bias = encoder_state_dict[f"encoder.layer.{layer}.attention.attention.key.bias"]
            self.assertTrue(torch.equal(key_bias, torch.zeros_like(key_bias)))

    def test_classifier_keys_are_dropped(self) -> None:
        encoder_state_dict, fc_norm_state_dict = _remap_checkpoint_state_dict(_fake_checkpoint_state_dict())
        for key in encoder_state_dict:
            self.assertNotIn("classifier", key)
        for key in fc_norm_state_dict:
            self.assertNotIn("classifier", key)

    def test_fc_norm_is_kept(self) -> None:
        state_dict = _fake_checkpoint_state_dict()
        _, fc_norm_state_dict = _remap_checkpoint_state_dict(state_dict)
        self.assertEqual(set(fc_norm_state_dict), {"weight", "bias"})
        torch.testing.assert_close(fc_norm_state_dict["weight"], state_dict["fc_norm.weight"])
        torch.testing.assert_close(fc_norm_state_dict["bias"], state_dict["fc_norm.bias"])

    def test_missing_q_bias_raises(self) -> None:
        state_dict = _fake_checkpoint_state_dict()
        del state_dict["videomae.encoder.layer.0.attention.attention.q_bias"]
        with self.assertRaises(ValueError):
            _remap_checkpoint_state_dict(state_dict)

    def test_unknown_non_head_key_raises(self) -> None:
        state_dict = _fake_checkpoint_state_dict()
        state_dict["some_unexpected_top_level_key"] = torch.zeros(1)
        with self.assertRaises(ValueError):
            _remap_checkpoint_state_dict(state_dict)

    def test_state_dict_with_no_layers_raises(self) -> None:
        with self.assertRaises(ValueError):
            _remap_checkpoint_state_dict({"fc_norm.weight": torch.zeros(EMBED_DIM), "fc_norm.bias": torch.zeros(EMBED_DIM)})


class PoolingTests(unittest.TestCase):
    """`pool_videomae_tokens` against a fake `[2, 1568, 768]` tensor."""

    def test_matches_hand_computed_mean_then_fc_norm(self) -> None:
        torch.manual_seed(0)
        tokens = torch.randn(2, NUM_TOKENS, EMBED_DIM)
        fc_norm = torch.nn.LayerNorm(EMBED_DIM)
        pooled = pool_videomae_tokens(tokens, fc_norm)
        expected = fc_norm(tokens.mean(1))
        torch.testing.assert_close(pooled, expected)
        self.assertEqual(pooled.shape, (2, EMBED_DIM))

    def test_rejects_a_non_sequence_tensor(self) -> None:
        fc_norm = torch.nn.LayerNorm(EMBED_DIM)
        with self.assertRaises(ValueError):
            pool_videomae_tokens(torch.randn(2, EMBED_DIM), fc_norm)


class OutFeaturesTests(unittest.TestCase):
    """Class-level facts, checkable without constructing the real network."""

    def test_out_features_is_768(self) -> None:
        self.assertEqual(VideoMAEBaseK400.out_features, 768)

    def test_out_channels_aliases_out_features(self) -> None:
        self.assertEqual(VideoMAEBaseK400.out_channels, VideoMAEBaseK400.out_features)

    def test_num_frames_is_16(self) -> None:
        self.assertEqual(VideoMAEBaseK400.num_frames, 16)


class ForwardShapeValidationTests(unittest.TestCase):
    """`forward` rejects malformed input shape before touching the network."""

    def test_wrong_ndim_raises(self) -> None:
        instance = VideoMAEBaseK400.__new__(VideoMAEBaseK400)
        with self.assertRaises(ValueError):
            VideoMAEBaseK400.forward(instance, torch.randn(2, 16, 224, 224))

    def test_wrong_frame_count_raises(self) -> None:
        instance = VideoMAEBaseK400.__new__(VideoMAEBaseK400)
        with self.assertRaises(ValueError):
            VideoMAEBaseK400.forward(instance, torch.randn(2, 8, 3, 224, 224))

    def test_wrong_channel_count_raises(self) -> None:
        instance = VideoMAEBaseK400.__new__(VideoMAEBaseK400)
        with self.assertRaises(ValueError):
            VideoMAEBaseK400.forward(instance, torch.randn(2, 16, 1, 224, 224))


class FromLocalWeightsRefusalTests(unittest.TestCase):
    """`from_local_weights` refuses non-VideoMAE state dicts cheaply."""

    def test_refuses_a_mapping_that_is_not_this_architecture(self) -> None:
        with self.assertRaises(ValueError):
            VideoMAEBaseK400.from_local_weights({"not_a_real_key": torch.zeros(1)}, weights_name="KINETICS400_FINETUNED")

    def test_refuses_a_non_mapping_state_dict(self) -> None:
        with self.assertRaises(TypeError):
            VideoMAEBaseK400.from_local_weights([1, 2, 3], weights_name="KINETICS400_FINETUNED")  # type: ignore[arg-type]

    def test_refuses_an_empty_weights_name(self) -> None:
        with self.assertRaises(ValueError):
            VideoMAEBaseK400.from_local_weights({}, weights_name="")

    def test_refuses_missing_fc_norm(self) -> None:
        state_dict = _fake_checkpoint_state_dict()
        del state_dict["fc_norm.weight"]
        del state_dict["fc_norm.bias"]
        with self.assertRaises(ValueError):
            VideoMAEBaseK400.from_local_weights(state_dict, weights_name="KINETICS400_FINETUNED")


class ReadinessRegistrationTests(unittest.TestCase):
    """The new family is registered in `core.p1c_readiness` with its own enum."""

    def test_family_registered_with_its_enum(self) -> None:
        from core.p1c_readiness import CANONICAL_WEIGHT_ENUMS_BY_FAMILY

        self.assertIn("transformers_videomae_base_k400", CANONICAL_WEIGHT_ENUMS_BY_FAMILY)
        self.assertEqual(
            CANONICAL_WEIGHT_ENUMS_BY_FAMILY["transformers_videomae_base_k400"],
            {"KINETICS400_FINETUNED"},
        )

    def test_dinov2_enum_is_not_accepted_for_videomae(self) -> None:
        from core.p1c_readiness import CANONICAL_WEIGHT_ENUMS_BY_FAMILY

        self.assertNotIn("LVD142M_REG4", CANONICAL_WEIGHT_ENUMS_BY_FAMILY["transformers_videomae_base_k400"])

    def test_unknown_family_still_rejected(self) -> None:
        from core.p1c_readiness import CANONICAL_WEIGHT_ENUMS_BY_FAMILY

        self.assertNotIn("not_a_registered_family", CANONICAL_WEIGHT_ENUMS_BY_FAMILY)


class FeatureCachePoolingRegistrationTests(unittest.TestCase):
    """The feature cache knows this family's pooling id."""

    def test_pooling_id_is_videomae_mean_fcnorm(self) -> None:
        from core.feature_cache import FEATURE_CACHE_POOLING_BY_FAMILY

        self.assertEqual(
            FEATURE_CACHE_POOLING_BY_FAMILY["transformers_videomae_base_k400"],
            "videomae_mean_fcnorm",
        )


@unittest.skipUnless(_REAL_MODEL_AVAILABLE, "registered VideoMAE weight file not present locally")
class RealModelTests(unittest.TestCase):
    """Exercises the actual registered checkpoint; skipped if it is absent.

    Never downloads: the weight file must already exist at the registered
    path (verified against its registered SHA-256 below) before this loads.
    Kept to a single forward on a 1-clip tensor per the project's compute
    constraints on this backbone's real weights.
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
        cls.backbone = VideoMAEBaseK400.from_local_weights(
            state_dict, weights_name="KINETICS400_FINETUNED"
        ).eval()

    def test_layer0_query_bias_is_nontrivial_and_key_bias_is_exactly_zero(self) -> None:
        layer0 = self.backbone.network.encoder.layer[0].attention.attention
        self.assertGreater(layer0.query.bias.abs().mean().item(), 0.1)
        self.assertEqual(layer0.key.bias.abs().max().item(), 0.0)

    def test_single_forward_on_a_zero_clip_gives_finite_768d_output(self) -> None:
        with torch.no_grad():
            pooled = self.backbone.pooled_features(torch.zeros(1, 16, 3, 224, 224))
        self.assertEqual(pooled.shape, (1, 768))
        self.assertTrue(torch.isfinite(pooled).all())


if __name__ == "__main__":
    unittest.main()
