"""P1C B-2: DINOv2 ViT-B/14-with-registers, the second frozen backbone.

Mirrors `models.classifiers.ResNet50C5` in shape and intent: a thin wrapper
around a pretrained network, loaded only from a caller-verified local
checkpoint, that never downloads weights of its own accord during a
registered run.

## Feature layout (fixed, downstream code depends on it)

`pooled_features(images)` returns a 1536-D vector that is the concatenation,
in this exact order, of:

- `[:768]`  -- the CLS token
- `[768:]`  -- the mean of the 256 PATCH tokens (register tokens excluded)

Register tokens are scratch space the network uses internally and were never
meant to carry content for a downstream reader, so they are dropped from the
mean rather than averaged in with the patches (see the DINOv2-with-registers
paper). `forward_features` returns `[B, 261, 768]`: 1 CLS + `num_prefix_tokens
- 1` = 4 register tokens + 256 patch tokens, so the patch slice starts at
`num_prefix_tokens`, not at a hardcoded `5`.

## Declared deviation: construction omits `img_size=224`

The registered checkpoint (`.cache/weights/dinov2_vitb14_reg4-c24ecfb4.safetensors`,
sha256 `c24ecfb4a1d8ca79193f6b9efcffc461872a09ddac43a0931357e4802931a006`) holds
a `pos_embed` shaped for timm's native 518x518 pretraining grid (37x37 = 1369
position tokens, 86,582,016 total parameters). Constructing the model with
`img_size=224` instead builds a *smaller* `pos_embed` (16x16 = 256 tokens,
85,727,232 total parameters) sized for that resolution -- confirmed by direct
measurement 2026-09-02 -- and a strict `load_state_dict` of the native
checkpoint into that smaller buffer fails on a `pos_embed` shape mismatch
before `from_local_weights` even gets a chance to reject it for another
reason. Reconciling the two shapes would require running the checkpoint
through timm's own `checkpoint_filter_fn` first, which truncates the stored
position grid down to 224's resolution once, at load time.

This module instead constructs at the checkpoint's native resolution
(`dynamic_img_size=True`, no `img_size` override) and lets `dynamic_img_size`
interpolate the *full* 37x37 position grid to whatever the input needs at
every forward pass -- the same on-the-fly interpolation DINOv2's own
upstream code uses to serve multiple resolutions from one checkpoint, and the
only construction that (a) strict-loads the unmodified checkpoint bytes,
(b) reports the registered 86,582,016 parameter count, and (c) still returns
`[B, 261, 768]` for a 224x224 input. The registered `img_size=224` in the
construction call was therefore not carried over literally; everything it was
meant to guarantee (a 224x224-shaped `forward_features` output) still holds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import timm
import torch
from torch import nn

#: The exact timm/HF model identifier this module loads. Never resolved by
#: any other name -- a name a caller cannot say out loud is not a registered
#: architecture.
MODEL_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"
EMBED_DIM = 768


def _build_network() -> nn.Module:
    return timm.create_model(MODEL_NAME, pretrained=False, num_classes=0, dynamic_img_size=True)


def pool_cls_plus_mean_patch(tokens: torch.Tensor, num_prefix_tokens: int) -> torch.Tensor:
    """CLS concatenated with the mean of the patch tokens; register tokens excluded.

    Split out of `DINOv2ViTB14Reg.pooled_features` so the layout -- CLS at
    `[:embed_dim]`, mean-patch at `[embed_dim:]`, and the register tokens
    between them dropped from the mean -- is testable against a tiny fake
    token tensor without constructing the real 86M-parameter network.
    `tokens` is `[B, 1 + registers + patches, D]`; `num_prefix_tokens` is
    `1 + registers` (5 for this architecture).
    """

    if tokens.ndim != 3:
        raise ValueError("tokens must be [batch, sequence, embed_dim]")
    if num_prefix_tokens < 1:
        raise ValueError("num_prefix_tokens must include at least the CLS token")
    cls = tokens[:, 0]
    mean_patch = tokens[:, num_prefix_tokens:].mean(1)
    return torch.cat([cls, mean_patch], dim=1)


class DINOv2ViTB14Reg(nn.Module):
    """DINOv2 ViT-B/14-with-registers, exposing a pooled CLS+mean-patch feature.

    `out_features` is 1536 (768 CLS + 768 mean-patch), the input to any head
    or probe downstream of this backbone. `out_channels` aliases it so the
    generic `FrozenBackbone` wrapper (`models/classifiers.py`), which reads
    `backbone.out_channels` for any wrapped backbone, keeps working -- this
    backbone has no spatial channel dimension to report, so the alias is
    exactly `out_features`, not a real channel count.
    """

    out_features = 2 * EMBED_DIM
    out_channels = out_features

    def __init__(self) -> None:
        super().__init__()
        self.network = _build_network()
        if self.network.num_prefix_tokens != 5 or self.network.embed_dim != EMBED_DIM:
            raise RuntimeError(
                "timm built an unexpected DINOv2 ViT-B/14-reg4 architecture: "
                f"num_prefix_tokens={self.network.num_prefix_tokens}, embed_dim={self.network.embed_dim} "
                "(expected 5 and 768) -- refusing rather than caching features under a silently "
                "different token layout"
            )
        self.num_prefix_tokens = self.network.num_prefix_tokens
        self.weights_name: str | None = None

    @classmethod
    def from_local_weights(
        cls,
        state_dict: Mapping[str, Any],
        *,
        weights_name: str,
    ) -> "DINOv2ViTB14Reg":
        """Build the backbone from a caller-verified local timm/DINOv2 checkpoint.

        The runner validates the checkpoint's digest before it reaches this
        method. Keeping the load local prevents timm from quietly downloading
        a different weight file during a registered experiment. Refuses
        (raises `ValueError`) anything that is not an exact state dict for
        this architecture -- see the module docstring for why `img_size` is
        not passed at construction.
        """

        if not isinstance(weights_name, str) or not weights_name:
            raise ValueError("weights_name must identify the registered weight enum")
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a DINOv2 ViT-B/14-reg4 state mapping")
        instance = cls()
        try:
            instance.network.load_state_dict(dict(state_dict), strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(
                "local weights are not an exact vit_base_patch14_reg4_dinov2.lvd142m state dictionary"
            ) from exc
        instance.weights_name = weights_name
        return instance

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return the raw token sequence `[B, 1 + registers + patches, 768]`."""

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("DINOv2ViTB14Reg expects [batch, 3, height, width]")
        tokens = self.network.forward_features(images)
        if tokens.ndim != 3 or tokens.shape[-1] != EMBED_DIM:
            raise RuntimeError(f"unexpected DINOv2 forward_features shape: {tuple(tokens.shape)}")
        return tokens

    def pooled_features(self, images: torch.Tensor) -> torch.Tensor:
        """CLS concatenated with the mean of the patch tokens -- see module docstring."""

        return pool_cls_plus_mean_patch(self.forward(images), self.num_prefix_tokens)
