"""DINOv2 ViT-L/14-with-registers, frozen, with the same pooled feature layout as the registered ViT-B (D-65).

`pooled_features` returns 2048-D: `[:1024]` CLS, `[1024:]` mean of the 256 patch tokens (register tokens excluded),
built by the same `pool_cls_plus_mean_patch` as `models.dinov2_backbone.DINOv2ViTB14Reg`. Weights are Meta's official
`dinov2_vitl14_reg4_pretrain.pth` (original DINOv2 key names), converted by timm's own DINOv2 checkpoint filter and
then strict-loaded; construction at the native 518 grid with `dynamic_img_size`, as for ViT-B (see that module).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import timm
import torch
from torch import nn

from models.dinov2_backbone import pool_cls_plus_mean_patch

MODEL_NAME = "vit_large_patch14_reg4_dinov2.lvd142m"
EMBED_DIM = 1024
WEIGHTS_PATH = ".cache/weights/dinov2_vitl14_reg4_pretrain.pth"
WEIGHTS_SHA256 = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
WEIGHTS_SOURCE = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"


class DINOv2ViTL14Reg(nn.Module):
    out_features = 2 * EMBED_DIM

    def __init__(self) -> None:
        super().__init__()
        self.network = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0, dynamic_img_size=True)
        if self.network.num_prefix_tokens != 5 or self.network.embed_dim != EMBED_DIM:
            raise RuntimeError(f"unexpected DINOv2 ViT-L/14-reg4 layout: prefix={self.network.num_prefix_tokens}, dim={self.network.embed_dim}")
        self.num_prefix_tokens = self.network.num_prefix_tokens

    @classmethod
    def from_official_checkpoint(cls, repo_root: Path) -> "DINOv2ViTL14Reg":
        """Verify the pinned sha256, convert the official keys with timm's filter, strict-load."""

        from timm.models.vision_transformer import checkpoint_filter_fn

        path = Path(repo_root) / WEIGHTS_PATH
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != WEIGHTS_SHA256:
            raise RuntimeError(f"{path} sha256 {digest} != registered {WEIGHTS_SHA256}")
        instance = cls()
        state = torch.load(path, map_location="cpu", weights_only=True)
        instance.network.load_state_dict(checkpoint_filter_fn(state, instance.network), strict=True)
        return instance.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.network.forward_features(images)
        if tokens.ndim != 3 or tokens.shape[-1] != EMBED_DIM:
            raise RuntimeError(f"unexpected forward_features shape {tuple(tokens.shape)}")
        return tokens

    def pooled_features(self, images: torch.Tensor) -> torch.Tensor:
        return pool_cls_plus_mean_patch(self.forward(images), self.num_prefix_tokens)
