"""P1C: SigLIP ViT-B/16 (224), a third frozen image backbone for the DINOv2 pipeline.

Mirrors `models.dinov2_backbone` in interface and intent: a thin wrapper around
a pretrained vision tower, loaded only from a caller-verified local checkpoint,
never downloaded during a registered run. Swapping it in changes the backbone
and nothing else in the cache -> head pipeline.

## Feature layout (fixed, downstream code depends on it)

`forward(images)` returns `[B, 1 + 196, 768]` with the SAME token convention
the DINOv2 wrapper uses, so `scripts/build_p1c_patch_cache.py` and
`pool_cls_plus_mean_patch` work unchanged with `num_prefix_tokens == 1`:

- token 0       -- SigLIP's attention-pooling head output (`pooler_output`):
                   a learned probe query attends over all patch tokens, then
                   LayerNorm + MLP residual. SigLIP has NO CLS token, so this
                   is the vision tower's own global representation and the
                   stand-in for DINOv2's CLS.
- tokens 1..196 -- the patch tokens (`last_hidden_state`, i.e. after the
                   tower's `post_layernorm`; 14x14 grid at patch 16 / 224).

`pooled_features(images)` = `[pooler_output ++ mean(patch tokens)]`, 1536-D,
the same width and order (global first, mean-patch second) as DINOv2's
CLS ++ mean-patch, so heads need no change.

## Input convention

The pipeline hands the backbone frames already resized to the config's
`resize_height x resize_width` and normalised with the config's mean/std; the
SigLIP configs register 224x224, bilinear, mean = std = (0.5, 0.5, 0.5)
(SigLIP's own `preprocessor_config.json`; its processor uses bicubic, bilinear
is kept so only the backbone differs from the DINOv2 pipeline). This wrapper
therefore does not normalise. If an input is NOT 224x224 it is resized here
(bilinear, `align_corners=False`, antialias on) because SigLIP's learned
position table is fixed to the 14x14 grid; at 224 the resize is skipped.

## Weights

Pinned local file `.cache/weights/siglip_base_patch16_224_vision-f5d84933.safetensors`,
sha256 `f5d84933ac67e11fcd249aba72bb2888028fc11507a4b81629c095c881031b77`: the
`vision_model.*` tensors of HF `google/siglip-base-patch16-224` (revision
7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed, upstream `model.safetensors` sha256
2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8, Apache-2.0)
with the `vision_model.` prefix stripped, because transformers 5.x
`SiglipVisionModel` has a flat state dict. Text tower dropped. 92,884,224
parameters, loaded `strict=True`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel

#: The HF model this checkpoint was cut from. Never resolved by this name at run time.
MODEL_NAME = "google/siglip-base-patch16-224"
EMBED_DIM = 768
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2


def _vision_config() -> SiglipVisionConfig:
    """The base-patch16-224 vision config (the hub config.json only sets patch_size=16)."""

    return SiglipVisionConfig(patch_size=PATCH_SIZE, image_size=IMAGE_SIZE)


class SigLIPBaseP16(nn.Module):
    """SigLIP ViT-B/16-224 vision tower exposing `[pooler_output ++ mean(patch)]`.

    `out_features` is 1536; `out_channels` aliases it for `FrozenBackbone`.
    """

    out_features = 2 * EMBED_DIM
    out_channels = out_features
    #: Token 0 of `forward` is the pooled representation; the patches start at 1.
    num_prefix_tokens = 1

    def __init__(self, config: SiglipVisionConfig | None = None) -> None:
        super().__init__()
        self.network = SiglipVisionModel(config if config is not None else _vision_config())
        hidden = self.network.config.hidden_size
        if hidden != EMBED_DIM and config is None:
            raise RuntimeError(f"unexpected SigLIP hidden size {hidden}; expected {EMBED_DIM}")
        self.embed_dim = hidden
        self.image_size = self.network.config.image_size
        self.weights_name: str | None = None

    @classmethod
    def from_local_weights(cls, state_dict: Mapping[str, Any], *, weights_name: str) -> "SigLIPBaseP16":
        """Build from a caller-verified local flat `SiglipVisionModel` state dict.

        The runner validates the checkpoint digest first. Loads `strict=True`:
        anything that is not an exact state dict for this architecture raises
        `ValueError` rather than leaving randomly-initialised tensors behind.
        """

        if not isinstance(weights_name, str) or not weights_name:
            raise ValueError("weights_name must identify the registered weight enum")
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a SigLIP vision-tower state mapping")
        instance = cls()
        try:
            instance.network.load_state_dict(dict(state_dict), strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError("local weights are not an exact google/siglip-base-patch16-224 vision-tower state dict") from exc
        instance.weights_name = weights_name
        return instance

    def _prepare(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("SigLIPBaseP16 expects [batch, 3, height, width]")
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False, antialias=True
            )
        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Token sequence `[B, 1 + patches, D]`: pooler_output first, then patch tokens."""

        out = self.network(pixel_values=self._prepare(images))
        return torch.cat([out.pooler_output.unsqueeze(1), out.last_hidden_state], dim=1)

    def pooled_features(self, images: torch.Tensor) -> torch.Tensor:
        """`[pooler_output ++ mean(patch tokens)]`, `[B, 1536]` -- see module docstring."""

        from models.dinov2_backbone import pool_cls_plus_mean_patch

        return pool_cls_plus_mean_patch(self.forward(images), self.num_prefix_tokens)
