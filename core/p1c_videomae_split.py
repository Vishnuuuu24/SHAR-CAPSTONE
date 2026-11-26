"""Split a VideoMAE-B encoder at a block boundary so the frozen lower part can be cached.

VideoMAE-B has 12 transformer blocks and (with `use_mean_pooling=True`, as in
the registered Kinetics-400 checkpoint) NO final encoder LayerNorm, so the
pooled readout is exactly `fc_norm(mean over tokens of block-11 output)`.
That makes a clean cut possible: everything up to and including block
`CACHE_BLOCKS - 1` is a pure function of the pixels, so its output (the hidden
state that ENTERS block `CACHE_BLOCKS`) can be stored once, and any number of
top-block fine-tuning runs can start from it.

`encode_to_cache_point` produces that stored tensor; `TopBlocksClassifier`
consumes it. Arms differ in exactly one factor, `trainable_blocks` (k): the top
k of the blocks held by the classifier receive gradients, the blocks below them
(but above the cache point) run under `torch.no_grad`, and `fc_norm` is frozen
in every arm (it is part of the registered readout, not a factor under test).
"""

from __future__ import annotations

import copy
from typing import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from core.p1c_head_training import set_local_seed
from models.videomae_backbone import VideoMAEBaseK400, pool_videomae_tokens

#: Blocks 0..7 are cached; blocks 8..11 run in the trainer.
CACHE_BLOCKS = 8


def encode_to_cache_point(backbone: VideoMAEBaseK400, clips: torch.Tensor, cache_blocks: int = CACHE_BLOCKS) -> torch.Tensor:
    """`[B,16,3,H,W]` clips -> the hidden state entering block `cache_blocks`, `[B,tokens,D]`."""

    network = backbone.network
    layers = network.encoder.layer
    if not 0 < cache_blocks <= len(layers):
        raise ValueError(f"cache_blocks must be in 1..{len(layers)}; got {cache_blocks}")
    hidden = network.embeddings(clips, None)
    for layer in layers[:cache_blocks]:
        hidden = layer(hidden)
    return hidden


class TopBlocksClassifier(nn.Module):
    """Blocks `cache_blocks..end` of a VideoMAE encoder + frozen `fc_norm` + a new linear head."""

    def __init__(
        self,
        backbone: VideoMAEBaseK400,
        *,
        num_classes: int,
        trainable_blocks: int,
        seed: int,
        cache_blocks: int = CACHE_BLOCKS,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        top = list(backbone.network.encoder.layer)[cache_blocks:]
        if not 0 <= trainable_blocks <= len(top):
            raise ValueError(f"trainable_blocks must be in 0..{len(top)}; got {trainable_blocks}")
        self.layers = nn.ModuleList(copy.deepcopy(layer) for layer in top)
        self.fc_norm = copy.deepcopy(backbone.fc_norm)
        self.trainable_blocks = trainable_blocks
        self.grad_checkpoint = grad_checkpoint
        for parameter in self.fc_norm.parameters():
            parameter.requires_grad_(False)
        first_trainable = len(self.layers) - trainable_blocks
        for index, layer in enumerate(self.layers):
            for parameter in layer.parameters():
                parameter.requires_grad_(index >= first_trainable)
        # K-31: the head's initial weights must be a function of the recorded
        # seed, so seed IMMEDIATELY before constructing it.
        set_local_seed(seed)
        self.head = nn.Linear(self.fc_norm.normalized_shape[0], num_classes)

    def block_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.layers.parameters() if p.requires_grad]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """`hidden` is the cached `[B,tokens,D]` state (any float dtype) -> `[B,num_classes]` logits."""

        hidden = hidden.float()
        first_trainable = len(self.layers) - self.trainable_blocks
        for index, layer in enumerate(self.layers):
            if index < first_trainable:
                with torch.no_grad():
                    hidden = layer(hidden)
            elif self.grad_checkpoint and self.training:
                hidden = checkpoint(layer, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden)
        return self.head(pool_videomae_tokens(hidden, self.fc_norm))


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_groups(model: TopBlocksClassifier, *, block_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
    groups = [{"params": list(model.head.parameters()), "lr": head_lr, "weight_decay": weight_decay}]
    blocks = model.block_parameters()
    if blocks:
        groups.append({"params": blocks, "lr": block_lr, "weight_decay": weight_decay})
    return groups


def saved_activation_bytes(model: nn.Module, hidden: torch.Tensor, *, autocast_dtype: torch.dtype | None = torch.bfloat16) -> int:
    """Bytes of tensors autograd saves for one forward on `hidden` (a CPU activation-memory estimate).

    Counts every saved-for-backward tensor once (by storage), which is what a
    GPU run would hold at peak before backward begins. Parameters are excluded.
    """

    seen: dict[int, int] = {}
    params = {p.data_ptr() for p in model.parameters()}

    def pack(tensor: torch.Tensor):
        if tensor.data_ptr() not in params:
            seen[tensor.untyped_storage().data_ptr()] = tensor.untyped_storage().nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        if autocast_dtype is None:
            model(hidden)
        else:
            with torch.autocast("cpu", dtype=autocast_dtype):
                model(hidden)
    return sum(seen.values())
