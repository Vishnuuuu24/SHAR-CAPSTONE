"""P1C: VideoMAE-B/16 fine-tuned on Kinetics-400, a clip-level frozen backbone.

Mirrors `models.dinov2_backbone` in structure, naming and comment density: a
thin wrapper around a pretrained network, loaded only from a caller-verified
local checkpoint, that never downloads weights of its own accord during a
registered run.

## Feature layout (fixed, downstream code depends on it)

`forward(clips)` for `clips` shaped `[B, 16, 3, 224, 224]` (16 RGB frames,
224x224, the project's registered preprocessing) returns the raw token
sequence `[B, 1568, 768]` -- VideoMAE has no CLS token; the 1568 tokens are
8 temporal positions (16 input frames / tubelet_size 2) x 196 spatial patches
(14x14 at patch_size 16). `pooled_features(clips)` returns `fc_norm(mean over
the 1568 tokens)` -> `[B, 768]`, the same readout the Kinetics classification
head applies before its final linear layer (`use_mean_pooling=True` in the
upstream config): mean-pool every token, then LayerNorm.

## Declared trap: the checkpoint's attention bias parameterisation

The upstream VideoMAE checkpoint stores per-layer
`videomae.encoder.layer.{i}.attention.attention.q_bias` and `.v_bias` --
VideoMAE's original paper parameterisation, where only the query and value
projections carry a learned bias and the key projection's bias is fixed at
zero (this halves the bias parameters relative to a standard ViT and was a
deliberate design choice, not an omission). `transformers==5.17.0`'s
`VideoMAEModel`, however, expects a standard per-projection
`{query,key,value}.bias` for each of query, key and value. Calling
`from_pretrained` (or any load that does not perform this exact remap) does
NOT raise -- `strict=False`-style loading fills the three missing/renamed
bias keys with the model's random initialisation and silently proceeds. The
resulting pooled features were measured to have only cosine similarity 0.469
with a correctly loaded model: a real, silent correctness bug, not a
theoretical one. `from_local_weights` below performs the exact remap and
loads `strict=True` specifically so this failure mode is structurally
impossible rather than merely documented.

Pinned local file: `.cache/weights/videomae-base-k400-065cfde5.safetensors`,
sha256 `065cfde56f97671da4d196e1a3a6d3cf9304dcad92eb671e3632a6c6cdd04f73`, from
HF repo `MCG-NJU/videomae-base-finetuned-kinetics` (Kinetics-400 fine-tuned
VideoMAE-B). License per that repo's card: CC-BY-NC-4.0 -- noncommercial,
recorded here because it constrains how any result built on this backbone
may be used or published.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from transformers import VideoMAEConfig, VideoMAEModel

#: The exact HF model identifier this checkpoint was fine-tuned from. Never
#: resolved by any other name during a registered run -- weights are always
#: loaded from the pinned local file, never fetched by this string.
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
EMBED_DIM = 768
NUM_FRAMES = 16
NUM_TOKENS = 1568  # 8 temporal (16 frames / tubelet_size 2) x 196 spatial (14x14 patches)

#: Prefix the checkpoint stores the encoder under; stripped during remap so
#: the resulting keys match `VideoMAEModel`'s own (unprefixed) state dict.
_CHECKPOINT_ENCODER_PREFIX = "videomae."
#: Top-level checkpoint keys that belong to the Kinetics classification head,
#: not the encoder this module wraps -- dropped, never loaded into anything.
_CHECKPOINT_HEAD_KEYS = frozenset({"classifier.weight", "classifier.bias"})
#: Checkpoint key holding the LayerNorm this module keeps as `fc_norm`.
_CHECKPOINT_FC_NORM_KEYS = frozenset({"fc_norm.weight", "fc_norm.bias"})


def _build_network() -> VideoMAEModel:
    config = VideoMAEConfig(
        num_frames=NUM_FRAMES,
        image_size=224,
        patch_size=16,
        tubelet_size=2,
        hidden_size=EMBED_DIM,
        num_hidden_layers=12,
    )
    return VideoMAEModel(config)


def _remap_checkpoint_state_dict(state_dict: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Translate the checkpoint's q_bias/v_bias parameterisation into `VideoMAEModel`'s.

    Returns `(encoder_state_dict, fc_norm_state_dict)`. Raises `ValueError` if
    any expected per-layer bias key is missing, or if any checkpoint key is
    present that this function does not know how to route (a classifier key,
    an `fc_norm` key, or an encoder key) -- an unrecognised key is exactly how
    a checkpoint from the wrong architecture would surface, and silently
    dropping it would defeat the point of `strict=True` below.
    """

    encoder_state_dict: dict[str, torch.Tensor] = {}
    fc_norm_state_dict: dict[str, torch.Tensor] = {}
    layers_with_q_bias: set[str] = set()
    layers_with_v_bias: set[str] = set()

    for key, value in state_dict.items():
        if key in _CHECKPOINT_HEAD_KEYS:
            continue
        if key in _CHECKPOINT_FC_NORM_KEYS:
            fc_norm_state_dict[key.removeprefix("fc_norm.")] = value
            continue
        if not key.startswith(_CHECKPOINT_ENCODER_PREFIX):
            raise ValueError(f"unexpected top-level key in VideoMAE checkpoint: {key!r}")
        stripped = key.removeprefix(_CHECKPOINT_ENCODER_PREFIX)
        if stripped.endswith(".attention.attention.q_bias"):
            layer_prefix = stripped[: -len("q_bias")]
            encoder_state_dict[layer_prefix + "query.bias"] = value
            layers_with_q_bias.add(layer_prefix)
        elif stripped.endswith(".attention.attention.v_bias"):
            layer_prefix = stripped[: -len("v_bias")]
            encoder_state_dict[layer_prefix + "value.bias"] = value
            layers_with_v_bias.add(layer_prefix)
        else:
            encoder_state_dict[stripped] = value

    if not layers_with_q_bias and not layers_with_v_bias:
        raise ValueError(
            "VideoMAE checkpoint carried no q_bias/v_bias keys -- not this architecture's "
            "attention-bias parameterisation"
        )
    if layers_with_q_bias != layers_with_v_bias:
        raise ValueError(
            "VideoMAE checkpoint is missing q_bias or v_bias for at least one layer -- "
            f"q_bias layers {sorted(layers_with_q_bias)} != v_bias layers {sorted(layers_with_v_bias)}"
        )

    # Synthesise the fixed-zero key bias for every layer whose q_bias/v_bias we
    # saw -- the checkpoint never stores it because VideoMAE's own upstream
    # code hardcodes it to zero rather than learning it.
    for layer_prefix in layers_with_q_bias:
        key_bias_name = layer_prefix + "key.bias"
        weight_key = layer_prefix + "key.weight"
        width = encoder_state_dict[weight_key].shape[0] if weight_key in encoder_state_dict else EMBED_DIM
        encoder_state_dict[key_bias_name] = torch.zeros(width)

    return encoder_state_dict, fc_norm_state_dict


def pool_videomae_tokens(tokens: torch.Tensor, fc_norm: nn.LayerNorm) -> torch.Tensor:
    """Mean-pool the token sequence, then LayerNorm -- the Kinetics head's readout.

    Split out of `VideoMAEBaseK400.pooled_features` so the pooling arithmetic
    is testable against a tiny fake token tensor without constructing the
    real ~86M-parameter network. `tokens` is `[B, num_tokens, D]`.
    """

    if tokens.ndim != 3:
        raise ValueError("tokens must be [batch, sequence, embed_dim]")
    return fc_norm(tokens.mean(1))


class VideoMAEBaseK400(nn.Module):
    """VideoMAE-B/16 fine-tuned on Kinetics-400, exposing a pooled 768-D feature.

    `out_features` is 768 (the mean-pooled, LayerNorm'd token feature), the
    input to any head or probe downstream of this backbone. `out_channels`
    aliases it so the generic `FrozenBackbone` wrapper (`models/classifiers.py`),
    which reads `backbone.out_channels` for any wrapped backbone, keeps
    working -- this backbone has no spatial channel dimension to report, so
    the alias is exactly `out_features`, not a real channel count.
    """

    out_features = EMBED_DIM
    out_channels = out_features
    num_frames = NUM_FRAMES

    def __init__(self) -> None:
        super().__init__()
        self.network = _build_network()
        self.fc_norm = nn.LayerNorm(EMBED_DIM)
        self.weights_name: str | None = None

    @classmethod
    def from_local_weights(
        cls,
        state_dict: Mapping[str, Any],
        *,
        weights_name: str,
    ) -> "VideoMAEBaseK400":
        """Build the backbone from a caller-verified local VideoMAE checkpoint.

        The runner validates the checkpoint's digest before it reaches this
        method. Keeping the load local prevents `transformers` from quietly
        downloading a different weight file during a registered experiment,
        and performs the q_bias/v_bias remap documented in the module
        docstring `strict=True` -- see there for why an ordinary load of this
        checkpoint silently corrupts every attention layer instead of
        raising.
        """

        if not isinstance(weights_name, str) or not weights_name:
            raise ValueError("weights_name must identify the registered weight enum")
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a VideoMAE-B/16 Kinetics-400 state mapping")
        try:
            encoder_state_dict, fc_norm_state_dict = _remap_checkpoint_state_dict(state_dict)
        except ValueError as exc:
            raise ValueError(
                "local weights are not an exact MCG-NJU/videomae-base-finetuned-kinetics state dictionary"
            ) from exc
        if set(fc_norm_state_dict) != {"weight", "bias"}:
            raise ValueError(
                "local weights are missing fc_norm.weight/fc_norm.bias -- not an exact "
                "MCG-NJU/videomae-base-finetuned-kinetics state dictionary"
            )
        instance = cls()
        try:
            instance.network.load_state_dict(encoder_state_dict, strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(
                "local weights are not an exact MCG-NJU/videomae-base-finetuned-kinetics state dictionary"
            ) from exc
        instance.fc_norm.load_state_dict(fc_norm_state_dict, strict=True)
        instance.weights_name = weights_name
        return instance

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Return the raw token sequence `[B, 1568, 768]` for `[B, 16, 3, 224, 224]` clips."""

        if clips.ndim != 5 or clips.shape[1] != NUM_FRAMES or clips.shape[2] != 3:
            raise ValueError(f"VideoMAEBaseK400 expects [batch, {NUM_FRAMES}, 3, height, width]")
        tokens = self.network(pixel_values=clips).last_hidden_state
        if tokens.ndim != 3 or tokens.shape[1] != NUM_TOKENS or tokens.shape[-1] != EMBED_DIM:
            raise RuntimeError(f"unexpected VideoMAE forward shape: {tuple(tokens.shape)}")
        return tokens

    def pooled_features(self, clips: torch.Tensor) -> torch.Tensor:
        """Mean over all 1568 tokens, then `fc_norm` -- see module docstring."""

        return pool_videomae_tokens(self.forward(clips), self.fc_norm)
