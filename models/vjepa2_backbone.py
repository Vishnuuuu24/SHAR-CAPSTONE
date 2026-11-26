"""P1C: V-JEPA 2 ViT-L (frozen), a clip-level backbone trained with a self-supervised latent-prediction objective.

Wraps the `transformers` V-JEPA 2 *encoder only* (the predictor is never used here), loaded strictly from a
caller-verified local copy of Meta's original checkpoint (`https://dl.fbaipublicfiles.com/vjepa2/vitl.pt`, MIT licence
per the facebookresearch/vjepa2 repository and the converted `facebook/vjepa2-vitl-fpc64-256` model card).

## Feature layout (fixed, downstream code depends on it)

`forward(clips)` for `clips` shaped `[B, 16, 3, H, W]` returns the token sequence `[B, 8 * (H/16) * (W/16), 1024]`
(tubelet 2 -> 8 temporal positions; patch 16). There is no CLS token. `pooled_features(clips)` is the mean over all
tokens of the encoder's final-LayerNorm output -> `[B, 1024]`. CHOICE: mean-pooling is the simplest readout that needs
no trained attentive probe; it is a different readout from the one Meta evaluates with, so a weak result here is a
result about *frozen mean-pooled features*, not about V-JEPA 2 in general.

## Position handling

The encoder has no learned position table: it uses rotary position embeddings whose (time, row, column) coordinates are
decoded from each token's flat index using `config.crop_size` and `config.frames_per_clip`. The config below therefore
sets them to the actual input (224, 16) so the coordinates are right for the project's registered 224 frames; the
rotary frequencies do not depend on the grid size. NOT YET VERIFIED against Meta's own implementation (running its
hubconf.py was not authorised): the load is strict, so no weight can stay at its random initialisation, but the numerical
equivalence of this wrapper to Meta's encoder on a real clip is unchecked (compare K-29: a silently wrong load is the
failure mode this guards against).

## Checkpoint key conversion

`convert_encoder_keys` is adapted from `transformers/models/vjepa2/convert_vjepa2_to_hf.py` (Apache-2.0, Hugging Face
Inc.): it renames the original keys and splits each fused `qkv` into query/key/value. Only the encoder branch is kept.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from transformers import VJEPA2Config
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Encoder

EMBED_DIM = 1024
NUM_FRAMES = 16
IMAGE_SIZE = 224
_PREFIX = "module.backbone."


def encoder_config(image_size: int = IMAGE_SIZE, num_frames: int = NUM_FRAMES) -> VJEPA2Config:
    config = VJEPA2Config(crop_size=image_size, frames_per_clip=num_frames, hidden_size=EMBED_DIM, num_attention_heads=16,
                          num_hidden_layers=24, mlp_ratio=4, pred_hidden_size=384, pred_num_attention_heads=12,
                          pred_num_hidden_layers=12, pred_num_mask_tokens=10)
    # A standalone encoder gets no attention dispatch (eager). SDPA is the fast path; on one random 16x224x224 clip the
    # pooled feature differed from eager by max 7.6e-6 (cosine 1.0).
    config._attn_implementation = "sdpa"
    return config


def convert_encoder_keys(original: Mapping[str, torch.Tensor], emb_dim: int = EMBED_DIM) -> dict[str, torch.Tensor]:
    """Original V-JEPA 2 encoder keys -> `VJEPA2Encoder` keys (no `encoder.` prefix). Raises on any key it cannot place."""

    out: dict[str, torch.Tensor] = {}
    for key, val in original.items():
        key = key.replace(_PREFIX, "")
        if key.startswith("blocks."):
            key = "layer." + key[len("blocks."):]
        elif key.startswith("patch_embed."):
            key = "embeddings.patch_embeddings." + key[len("patch_embed."):]
        elif key.startswith("norm."):
            key = "layernorm." + key[len("norm."):]
        else:
            raise ValueError(f"unexpected encoder key in V-JEPA 2 checkpoint: {key!r}")
        key = key.replace("attn.", "attention.")
        if ".qkv." in key:
            prefix, suffix = key.split(".qkv.")
            parts = (val[:emb_dim], val[emb_dim:2 * emb_dim], val[2 * emb_dim:])
            for name, part in zip(("query", "key", "value"), parts):
                out[f"{prefix}.{name}.{suffix}"] = part
        else:
            out[key] = val
    return out


class VJEPA2LargeEncoder(nn.Module):
    """Frozen V-JEPA 2 ViT-L encoder with a mean-pooled 1024-D feature. `out_channels` aliases `out_features`."""

    out_features = EMBED_DIM
    out_channels = out_features
    num_frames = NUM_FRAMES

    def __init__(self, image_size: int = IMAGE_SIZE) -> None:
        super().__init__()
        self.network = VJEPA2Encoder(encoder_config(image_size))
        self.weights_name: str | None = None

    @classmethod
    def from_local_weights(cls, encoder_state: Mapping[str, Any], *, weights_name: str,
                           image_size: int = IMAGE_SIZE) -> "VJEPA2LargeEncoder":
        """`encoder_state` is the checkpoint's `encoder` mapping. Loads strictly: any missing or extra key raises."""

        if not weights_name:
            raise ValueError("weights_name must identify the registered weight enum")
        converted = convert_encoder_keys(encoder_state)
        instance = cls(image_size)
        try:
            instance.network.load_state_dict(converted, strict=True)
        except RuntimeError as exc:
            raise ValueError("local weights are not an exact V-JEPA 2 ViT-L encoder state dictionary") from exc
        instance.weights_name = weights_name
        return instance

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5 or clips.shape[1] != NUM_FRAMES or clips.shape[2] != 3:
            raise ValueError(f"VJEPA2LargeEncoder expects [batch, {NUM_FRAMES}, 3, height, width]")
        tokens = self.network(pixel_values_videos=clips).last_hidden_state
        expected = (NUM_FRAMES // 2) * (clips.shape[3] // 16) * (clips.shape[4] // 16)
        if tokens.ndim != 3 or tokens.shape[1] != expected or tokens.shape[-1] != EMBED_DIM:
            raise RuntimeError(f"unexpected V-JEPA 2 forward shape: {tuple(tokens.shape)}")
        return tokens

    def pooled_features(self, clips: torch.Tensor) -> torch.Tensor:
        return self.forward(clips).mean(1)
