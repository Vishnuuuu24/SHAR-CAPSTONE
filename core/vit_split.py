"""Split a timm DINOv2 ViT into a frozen trunk and a trainable tail (D-67 partial fine-tuning).

`trunk_tokens(net, images, split)` runs the patch embedding, position embedding and blocks `[0, split)`, i.e. exactly
the first half of timm's `VisionTransformer.forward_features`. `Tail` holds deep copies of blocks `[start, 12)` and the
final norm and ends in the registered `CLS ++ mean(patch)` pooling, so `Tail(trunk_tokens(x))` equals
`pooled_features(x)` of the whole backbone at initialisation. Only the last `trainable_blocks` tail blocks (and the
norm, when any block trains) have `requires_grad`; earlier tail blocks are frozen copies.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Iterable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.dinov2_backbone import pool_cls_plus_mean_patch


def trunk_tokens(net: nn.Module, images: torch.Tensor, split: int) -> torch.Tensor:
    """Hidden state entering block `split` (timm forward_features up to that block)."""

    x = net.patch_embed(images)
    x = net._pos_embed(x)
    x = net.patch_drop(x)
    x = net.norm_pre(x)
    for blk in net.blocks[:split]:
        x = blk(x)
    return x


class Tail(nn.Module):
    """Blocks `[start, n)` + final norm + pooling; the last `trainable_blocks` blocks and the norm are trainable."""

    def __init__(self, net: nn.Module, start: int, trainable_blocks: int, num_prefix_tokens: int, grad_checkpoint: bool = True):
        super().__init__()
        n = len(net.blocks)
        if not 0 <= start <= n or not 0 <= trainable_blocks <= n - start:
            raise ValueError(f"bad split: start={start}, trainable_blocks={trainable_blocks}, blocks={n}")
        self.start, self.trainable_blocks, self.num_prefix_tokens = start, trainable_blocks, num_prefix_tokens
        self.blocks = nn.ModuleList(copy.deepcopy(b) for b in net.blocks[start:])
        self.norm = copy.deepcopy(net.norm)
        self.grad_checkpoint = grad_checkpoint
        first_trainable = len(self.blocks) - trainable_blocks
        for i, blk in enumerate(self.blocks):
            for p in blk.parameters():
                p.requires_grad_(i >= first_trainable)
        for p in self.norm.parameters():
            p.requires_grad_(trainable_blocks > 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.grad_checkpoint and torch.is_grad_enabled() and any(p.requires_grad for p in blk.parameters()):
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return pool_cls_plus_mean_patch(self.norm(x), self.num_prefix_tokens)

    def trainable_names(self) -> set[str]:
        return {name for name, p in self.named_parameters() if p.requires_grad}

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if k.split(".")[0] == "norm" or (k.startswith("blocks.") and int(k.split(".")[1]) >= len(self.blocks) - self.trainable_blocks)}


def state_digest(tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    """sha256 over (name, bytes) pairs in name order: a bit-level fingerprint for freeze audits."""

    h = hashlib.sha256()
    for name, t in sorted(tensors, key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def frozen_digest(net: nn.Module, split: int) -> str:
    """Fingerprint of everything a run must never change: embeddings, prefix tokens and blocks `[0, split)`."""

    keep = []
    for name, t in net.state_dict().items():
        if name.startswith("blocks."):
            if int(name.split(".")[1]) < split:
                keep.append((name, t))
        elif not name.startswith("norm.") and not name.startswith("head") and not name.startswith("fc_norm"):
            keep.append((name, t))
    return state_digest(keep)


class CachedTailHead(nn.Module):
    """A sequence head fed by `Tail` over cached trunk states, addressed by frame index.

    `train_sequence_head` hands this module `[B, L, 1]` float tensors of row indices into `cache` (built by passing
    index arrays through `sequences`, so frame order and permutation are the control's). The tail runs on the cache's
    device; pooled features come back to the head's device (CPU, as for the control) before the head runs.
    """

    def __init__(self, tail: Tail, head: nn.Module, cache: torch.Tensor, chunk_frames: int | None = 128):
        super().__init__()
        self.tail, self.head = tail, head
        self.cache = cache  # not a parameter or buffer: never saved, never moved by .to()
        self.chunk_frames = chunk_frames

    def forward(self, index_sequences: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, length = index_sequences.shape[:2]
        idx = index_sequences[..., 0].round().long().reshape(-1).to(self.cache.device)
        # Frames are independent through the tail (no batch statistics), so running it over chunks of the batch gives
        # the same outputs and, after autograd sums the chunks, the same gradients -- at a fraction of the peak memory
        # (a 512-frame batch through 4 checkpointed ViT-B blocks did not fit the 16 GiB MIG slice).
        parts = idx.split(self.chunk_frames) if self.chunk_frames else (idx,)
        feats = torch.cat([self.tail(self.cache[p].float()) for p in parts])
        head_device = next(self.head.parameters()).device
        return self.head(feats.to(head_device).reshape(b, length, -1), key_padding_mask=key_padding_mask)
