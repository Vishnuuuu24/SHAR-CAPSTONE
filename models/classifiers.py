"""P1C mandatory frame and temporal control architectures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def global_average_pool(spatial: torch.Tensor) -> torch.Tensor:
    """Flatten a `[B, C, H, W]` map to `[B, C]` via global average pooling.

    The one definition of "pooled feature" every GAP-consuming backbone and
    head in this project shares (rule of two): `core/feature_cache.py` and
    `core/feature_cache_verify.py` used to each build their own
    `nn.AdaptiveAvgPool2d((1,1))` for this, which only makes sense for a
    spatial CNN output -- `ResNet50C5.pooled_features` below is the one
    caller that still needs it now that a non-spatial backbone
    (`models.dinov2_backbone.DINOv2ViTB14Reg`) exists and pools itself.
    """

    return torch.flatten(nn.functional.adaptive_avg_pool2d(spatial, (1, 1)), 1)


class ResNet50C5(nn.Module):
    """Return the spatial ResNet50 layer4/C5 map before global pooling."""

    out_channels = 2048

    def __init__(self, weights: ResNet50_Weights | None = None):
        super().__init__()
        network = resnet50(weights=weights)
        self.features = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
            network.layer3,
            network.layer4,
        )
        self.weights_name = weights.name if weights is not None else None

    @classmethod
    def from_local_weights(
        cls,
        state_dict: Mapping[str, Any],
        *,
        weights_name: str,
    ) -> "ResNet50C5":
        """Build C5 from a caller-verified local torchvision checkpoint.

        The runner validates the checkpoint's digest before it reaches this
        method.  Keeping the load local prevents torchvision from quietly
        downloading a different weight file during a registered experiment.
        """

        if not isinstance(weights_name, str) or not weights_name:
            raise ValueError("weights_name must identify the registered weight enum")
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a torchvision ResNet50 state mapping")
        network = resnet50(weights=None)
        try:
            network.load_state_dict(dict(state_dict), strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError("local weights are not an exact torchvision ResNet50 state dictionary") from exc
        instance = cls(weights=None)
        instance.features = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
            network.layer3,
            network.layer4,
        )
        instance.weights_name = weights_name
        return instance

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("ResNet50C5 expects [batch, 3, height, width]")
        features = self.features(images)
        if features.ndim != 4 or features.shape[1] != self.out_channels:
            raise RuntimeError(f"unexpected C5 shape: {tuple(features.shape)}")
        return features

    def pooled_features(self, images: torch.Tensor) -> torch.Tensor:
        """Global-average-pooled, flattened C5 -- the vector `core.feature_cache` caches.

        Numerically identical to the pre-refactor call sites' own
        `nn.AdaptiveAvgPool2d((1,1))` + `torch.flatten(..., 1)`
        (`tests/test_classifiers.py` proves the equivalence): both reduce to
        the same `adaptive_avg_pool2d` call, so B-1's existing verified cache
        at `results/p1c/p1c-frozen-feature-cache-ar-fold1/` is unaffected.
        """

        return global_average_pool(self.forward(images))


class GlobalAverageLinearHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        if in_channels <= 0 or num_classes <= 1:
            raise ValueError("in_channels must be positive and num_classes must exceed one")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, c5: torch.Tensor) -> torch.Tensor:
        if c5.ndim != 4:
            raise ValueError("GAP+linear head requires a spatial [B,C,H,W] C5 tensor")
        return self.classifier(torch.flatten(self.pool(c5), 1))


class SameC5MLPHead(nn.Module):
    """C5 -> GAP -> MLP control; it never reshapes a GAP vector as spatial data."""

    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        if min(in_channels, hidden_dim, num_classes) <= 0 or num_classes <= 1:
            raise ValueError("channel/hidden/class dimensions are invalid")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, c5: torch.Tensor) -> torch.Tensor:
        if c5.ndim != 4:
            raise ValueError("same-C5 MLP head requires a spatial [B,C,H,W] C5 tensor")
        return self.mlp(torch.flatten(self.pool(c5), 1))


#: Registered values for `fixed_protocol.backbone.trainable`. Absent resolves
#: to "full" -- the exact current behaviour for every existing config, the
#: same contract an absent `training.throughput` block already carries.
BACKBONE_TRAINABLE_MODES = ("full", "frozen")


class FrozenBackbone(nn.Module):
    """Wrap a backbone so training cannot update it or drift its BatchNorm stats.

    Freezing `requires_grad` alone is not enough. `FrameClassifierEngine.train_epoch`
    calls `model.train(True)` once per epoch (core/p1c_engine.py), and
    `nn.Module.train` recurses into every child regardless of `requires_grad` --
    so a plain frozen backbone would still flip its BatchNorm layers back into
    training mode and keep updating their running mean/var on every forward
    pass. A "frozen" backbone whose BN statistics keep drifting is not
    actually frozen; this wrapper refuses that half-measure by overriding
    `train()` to force `eval()` on the wrapped module no matter what mode the
    surrounding model is asked to switch to.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.out_channels = backbone.out_channels
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenBackbone":
        # Ignore the requested mode: the wrapped backbone (and everything
        # under it) always runs in eval, even when the parent model's
        # .train(True) recurses into this module as a child.
        super().train(False)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def pooled_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone.pooled_features(images)


def apply_backbone_trainable_mode(backbone: nn.Module, mode: str) -> nn.Module:
    """Return `backbone`, optionally wrapped so training cannot touch it.

    `mode="full"` returns `backbone` unchanged. `mode="frozen"` wraps it in
    `FrozenBackbone`. Any other value is a caller bug, not a runtime input --
    `core.p1c_readiness` already refuses a registered value outside
    `BACKBONE_TRAINABLE_MODES` before a runner reaches this function.
    """

    if mode not in BACKBONE_TRAINABLE_MODES:
        raise ValueError(f"backbone trainable mode must be one of {BACKBONE_TRAINABLE_MODES}; got {mode!r}")
    if mode == "frozen":
        return FrozenBackbone(backbone)
    return backbone


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Trainable-vs-total parameter counts for the run manifest and logs.

    Exists so a frozen-backbone run can prove the freeze actually took effect
    (docs/PROJECT_RULES.md: undetected wrong results are not acceptable here) instead of
    asserting it happened with nothing to check it against.
    """

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"trainable_parameters": trainable, "total_parameters": total}


class FrameClassifier(nn.Module):
    def __init__(self, backbone: ResNet50C5, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))


class CNNLSTMTemporalBaseline(nn.Module):
    """Source-video sequence baseline using shared C5 frame features and an LSTM."""

    def __init__(
        self,
        backbone: ResNet50C5,
        projection_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        if min(projection_dim, hidden_dim, num_layers, num_classes) <= 0 or num_classes <= 1:
            raise ValueError("temporal architecture dimensions are invalid")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(backbone.out_channels, projection_dim)
        self.lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        clips: torch.Tensor,
        lengths: torch.Tensor,
        source_video_ids: Sequence[str],
    ) -> torch.Tensor:
        if clips.ndim != 5 or clips.shape[2] != 3:
            raise ValueError("temporal baseline expects [batch,time,3,height,width]")
        batch, time = clips.shape[:2]
        if not isinstance(lengths, torch.Tensor):
            raise TypeError("lengths must be a torch.Tensor")
        if lengths.shape != (batch,):
            raise ValueError("lengths must contain one value per clip")
        if lengths.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("lengths must use an integer dtype")
        if lengths.device != clips.device:
            raise ValueError("lengths and clips must be on the same device")
        if (
            len(source_video_ids) != batch
            or any(not isinstance(value, str) or not value for value in source_video_ids)
        ):
            raise ValueError("source_video_ids must identify every source-grouped clip")
        if torch.any(lengths < 1) or torch.any(lengths > time):
            raise ValueError("clip lengths must be within [1,time]")
        frames = clips.reshape(batch * time, *clips.shape[2:])
        c5 = self.backbone(frames)
        frame_features = torch.flatten(self.pool(c5), 1)
        projected = self.projection(frame_features).reshape(batch, time, -1)
        outputs, _ = self.lstm(projected)
        gather_index = (lengths.to(outputs.device) - 1).view(batch, 1, 1).expand(-1, 1, outputs.shape[-1])
        final = outputs.gather(1, gather_index).squeeze(1)
        return self.classifier(final)
