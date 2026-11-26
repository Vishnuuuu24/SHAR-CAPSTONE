"""P1C: partial fine-tuning + source-video domain-adversarial training for Shooting.

The final Shooting experiment asks whether unfreezing the last few DINOv2
transformer blocks, and adding a gradient-reversal (GRL) domain discriminator
on source-video identity, improves cross-video Shooting recognition beyond a
fully frozen backbone. Three arms share this one model class, differing only
in `unfreeze_last_blocks` (0, 2, ...) and whether the domain head/GRL are
exercised at all:

1. frozen backbone (`unfreeze_last_blocks=0`)
2. last 2 transformer blocks unfrozen (`unfreeze_last_blocks=2`)
3. arm 2 + GRL domain discriminator on source-video ID

The domain is **source video**, never "camera" -- `videos.csv` has no camera
ID field, so a video is the only grouping this dataset can actually support.

This module builds the GRL, the schedule for its lambda, the model, its
losses, a smoke test that proves the plumbing (gradients reach the right
parameters, GRL actually reverses, both heads see their labels) on one tiny
batch, and checkpoint helpers for the trainable slice of the model. Split
selection, video-disjointness, and held-out-label hygiene are the runner's
job, not this module's -- see `smoke_test`'s docstring.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.dinov2_backbone import DINOv2ViTB14Reg, pool_cls_plus_mean_patch


class GradientReversalFunction(torch.autograd.Function):
    """The GRL primitive from Ganin & Lempitsky, "Unsupervised Domain
    Adaptation by Backpropagation", ICML 2015: identity on the forward pass,
    multiplies the incoming gradient by `-lambda_` on the backward pass. This
    is what lets one optimizer step simultaneously push the shared
    representation to be good for the event head and bad for the domain
    head, without a second, adversarial optimizer.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GRL(nn.Module):
    """`nn.Module` wrapper around `GradientReversalFunction` with a settable
    `lambda_` so a runner can follow `dann_lambda`'s schedule across training
    without rebuilding the model.
    """

    def __init__(self, lambda_: float = 0.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


def dann_lambda(progress: float) -> float:
    """The DANN paper's own lambda schedule: `2 / (1 + exp(-10*p)) - 1` for
    `p` in `[0, 1]` (`p` is training progress, e.g. `step / total_steps`).

    Used as-is, with no modification, so that no lambda value here is
    invented for this project -- it is the schedule the cited paper
    validated, not a number chosen to fit this dataset. The run logs
    gradient norms (`smoke_test`'s `event_grad_norm` / `domain_grad_norm` /
    `total_grad_norm`, tracked per step by the runner) precisely so that if
    this schedule ever needs adjusting, the adjustment is made on measured
    evidence -- e.g. the domain branch's gradient swamping the event
    branch's -- and not on a guess.
    """

    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"progress must be in [0, 1]; got {progress}")
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def _unfrozen_block_indices(num_blocks: int, unfreeze_last_blocks: int) -> range:
    return range(num_blocks - unfreeze_last_blocks, num_blocks)


class DomainAdversarialModel(nn.Module):
    """Partial fine-tuning of a DINOv2 backbone plus an event head and a
    GRL-gated domain head over source-video identity.

    Freezing: every backbone parameter starts frozen; `requires_grad=True`
    is then set on the last `unfreeze_last_blocks` entries of
    `backbone.network.blocks` and on `backbone.network.norm`. The final norm
    is included with the unfrozen tail (not left frozen) because it sits
    *after* every block including the newly-trainable ones -- an untrained
    norm right before the pooling step would badly rescale a representation
    the unfrozen blocks are otherwise adapting, so it has to move with them.
    `unfreeze_last_blocks=0` freezes the entire backbone; only the heads
    train.

    Eval mode is permanent for the backbone. `train()` is overridden the
    same way `models.classifiers.FrozenBackbone` overrides it: calling
    `.train(True)` on this model flips the heads (and GRL, which has no
    state to flip) into train mode but forces `self.backbone.eval()`
    regardless. DINOv2 has no BatchNorm, so there are no running stats to
    protect, but it does have dropout and droppath inside its blocks --
    including the ones being fine-tuned -- and those must stay off in every
    arm so the three arms differ only in what the task description says
    they differ in (frozen vs. partially-unfrozen vs. +domain-adversarial),
    not in whether stochastic regularization happened to be active.
    """

    def __init__(
        self,
        backbone: DINOv2ViTB14Reg,
        *,
        unfreeze_last_blocks: int,
        n_domains: int,
        grl_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        num_blocks = len(backbone.network.blocks)
        if unfreeze_last_blocks < 0 or unfreeze_last_blocks > num_blocks:
            raise ValueError(f"unfreeze_last_blocks must be in [0, {num_blocks}]; got {unfreeze_last_blocks}")
        if n_domains < 1:
            raise ValueError(f"n_domains must be >= 1; got {n_domains}")

        self.backbone = backbone
        self.unfreeze_last_blocks = unfreeze_last_blocks
        self._num_blocks = num_blocks

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for index in _unfrozen_block_indices(num_blocks, unfreeze_last_blocks):
            for parameter in self.backbone.network.blocks[index].parameters():
                parameter.requires_grad_(True)
        if unfreeze_last_blocks > 0:
            for parameter in self.backbone.network.norm.parameters():
                parameter.requires_grad_(True)
        self.backbone.eval()

        representation_dim = backbone.out_features
        self.event_head = nn.Linear(representation_dim, 1)
        self.grl = GRL(grl_lambda)
        self.domain_head = nn.Sequential(self.grl, nn.Linear(representation_dim, n_domains))

    @property
    def grl_lambda(self) -> float:
        return self.grl.lambda_

    @grl_lambda.setter
    def grl_lambda(self, value: float) -> None:
        self.grl.lambda_ = value

    def train(self, mode: bool = True) -> "DomainAdversarialModel":
        super().train(mode)
        self.backbone.eval()
        return self

    def unfrozen_backbone_parameter_names(self) -> set[str]:
        """Fully-qualified `named_parameters()` names of the backbone
        parameters this model actually trains (empty when
        `unfreeze_last_blocks == 0`). Exists so a test can assert "exactly
        blocks[-k:] plus norm, and nothing else" by name rather than by
        re-deriving the freezing logic a second time.
        """

        names: set[str] = set()
        for index in _unfrozen_block_indices(self._num_blocks, self.unfreeze_last_blocks):
            for name, _ in self.backbone.network.blocks[index].named_parameters():
                names.add(f"backbone.network.blocks.{index}.{name}")
        if self.unfreeze_last_blocks > 0:
            for name, _ in self.backbone.network.norm.named_parameters():
                names.add(f"backbone.network.norm.{name}")
        return names

    def forward(self, clips: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """`clips` is `[B, 16, 3, H, W]`. Returns
        `(event_logit [B], domain_logits [B, n_domains], representation [B, 1536])`.

        Calls the timm `VisionTransformer.forward_features` pieces
        explicitly (read from the installed timm source rather than
        guessed) so the frozen prefix can run under `torch.no_grad()` for
        memory, while only the unfrozen tail builds an autograd graph:
        `patch_embed` -> `_pos_embed` -> `patch_drop` -> `norm_pre` ->
        `blocks[:n-k]` (frozen, no_grad) -> `blocks[n-k:]` (unfrozen, grad)
        -> `norm` (unfrozen, grad, per the class docstring). When
        `unfreeze_last_blocks == 0`, `blocks[n-k:]` is empty and `norm` also
        runs under `no_grad`, so the whole backbone pass is gradient-free
        and only the heads train.
        """

        if clips.ndim != 5 or clips.shape[2] != 3:
            raise ValueError("clips must be [batch, frames, 3, height, width]")
        batch, frames = clips.shape[0], clips.shape[1]
        flat = clips.reshape(batch * frames, *clips.shape[2:])

        network = self.backbone.network
        k = self.unfreeze_last_blocks
        split = self._num_blocks - k

        with torch.no_grad():
            x = network.patch_embed(flat)
            x = network._pos_embed(x)
            x = network.patch_drop(x)
            x = network.norm_pre(x)
            x = network.blocks[:split](x)
            if k == 0:
                x = network.norm(x)

        if k > 0:
            x = network.blocks[split:](x)
            x = network.norm(x)

        tokens = x
        pooled = pool_cls_plus_mean_patch(tokens, self.backbone.num_prefix_tokens)
        pooled = pooled.reshape(batch, frames, -1)
        representation = pooled.mean(dim=1)

        event_logit = self.event_head(representation).squeeze(-1)
        domain_logits = self.domain_head(representation)
        return event_logit, domain_logits, representation


def event_loss(event_logit: Tensor, event_y: Tensor) -> Tensor:
    """`BCEWithLogitsLoss` on the event logit against the binary event label."""

    return F.binary_cross_entropy_with_logits(event_logit, event_y.float())


def domain_loss(domain_logits: Tensor, domain_y: Tensor) -> Tensor:
    """`CrossEntropyLoss` on domain logits against the integer source-video-id label."""

    return F.cross_entropy(domain_logits, domain_y)


def total_loss(event_logit: Tensor, event_y: Tensor, domain_logits: Tensor, domain_y: Tensor) -> Tensor:
    """`event_loss + domain_loss`, unweighted.

    The GRL already multiplies the domain branch's backward-pass gradient
    by `-lambda` before it ever reaches the shared representation, so
    `lambda` must not also appear as a weight here -- applying it twice
    (once in the GRL, once as a loss weight) would silently double- or
    mis-scale the domain branch relative to the paper's schedule.
    """

    return event_loss(event_logit, event_y) + domain_loss(domain_logits, domain_y)


def _grad_norm(parameters: list[Tensor]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += parameter.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def smoke_test(
    model: DomainAdversarialModel,
    clips: Tensor,
    event_y: Tensor,
    domain_y: Tensor,
) -> dict[str, Any]:
    """One forward/backward on a tiny batch, checking the plumbing rather
    than any measured effect.

    Covers items 1-5 of the owner's checklist that are properties of the
    model/loss wiring itself. Items 6-7 (video disjointness across splits,
    no held-out label leaking into training) are properties of how a runner
    builds its splits from `core.p1c_event_clips.Clip`s, not of this model,
    and belong in the runner's own tests instead.
    """

    unfrozen_names = model.unfrozen_backbone_parameter_names()
    named_parameters = dict(model.named_parameters())
    unfrozen_backbone_params = [named_parameters[name] for name in unfrozen_names]
    frozen_backbone_params = [
        parameter
        for name, parameter in named_parameters.items()
        if name.startswith("backbone.") and name not in unfrozen_names
    ]

    model.zero_grad(set_to_none=True)
    event_logit, domain_logits, _ = model(clips)
    loss_correct = event_loss(event_logit, event_y)
    loss_flipped = event_loss(event_logit, 1 - event_y.float())
    d_loss = domain_loss(domain_logits, domain_y)
    (loss_correct + d_loss).backward()

    grads_reach_unfrozen_blocks = bool(unfrozen_backbone_params) and all(
        parameter.grad is not None and torch.any(parameter.grad != 0).item()
        for parameter in unfrozen_backbone_params
    )
    frozen_blocks_have_no_grad = all(parameter.grad is None for parameter in frozen_backbone_params)

    grl_input = torch.randn(4, 8, requires_grad=True)
    grl_probe = GRL(1.0)
    grl_output = grl_probe(grl_input)
    upstream_grad = torch.randn_like(grl_output)
    grl_output.backward(upstream_grad)
    grl_reverses = torch.allclose(grl_input.grad, -upstream_grad)

    event_head_sees_labels = not math.isclose(loss_correct.item(), loss_flipped.item())
    domain_head_sees_labels = tuple(domain_logits.shape) == (
        clips.shape[0],
        model.domain_head[-1].out_features,
    ) and math.isfinite(d_loss.item())
    losses_finite = math.isfinite(loss_correct.item()) and math.isfinite(d_loss.item())

    model.zero_grad(set_to_none=True)
    event_logit_e, _, _ = model(clips)
    event_loss(event_logit_e, event_y).backward()
    event_grad_norm = _grad_norm(unfrozen_backbone_params)

    model.zero_grad(set_to_none=True)
    _, domain_logits_d, _ = model(clips)
    domain_loss(domain_logits_d, domain_y).backward()
    domain_grad_norm = _grad_norm(unfrozen_backbone_params)

    model.zero_grad(set_to_none=True)
    event_logit_t, domain_logits_t, _ = model(clips)
    total_loss(event_logit_t, event_y, domain_logits_t, domain_y).backward()
    total_grad_norm = _grad_norm(unfrozen_backbone_params)

    return {
        "grads_reach_unfrozen_blocks": grads_reach_unfrozen_blocks,
        "grads_reach_unfrozen_blocks_evidence": {
            name: (named_parameters[name].grad is not None) for name in sorted(unfrozen_names)
        },
        "frozen_blocks_have_no_grad": frozen_blocks_have_no_grad,
        "grl_reverses": grl_reverses,
        "grl_reverses_evidence": {
            "input_grad": grl_input.grad.tolist(),
            "negated_upstream": (-upstream_grad).tolist(),
        },
        "event_head_sees_labels": event_head_sees_labels,
        "event_head_sees_labels_evidence": {
            "loss_correct": loss_correct.item(),
            "loss_flipped": loss_flipped.item(),
        },
        "domain_head_sees_labels": domain_head_sees_labels,
        "domain_head_sees_labels_evidence": {
            "domain_logits_shape": tuple(domain_logits.shape),
            "domain_loss": d_loss.item(),
        },
        "losses_finite": losses_finite,
        "event_grad_norm": event_grad_norm,
        "domain_grad_norm": domain_grad_norm,
        "total_grad_norm": total_grad_norm,
    }


def save_checkpoint(model: DomainAdversarialModel, path: str) -> None:
    """Save only what this arm actually trains: the unfrozen backbone tail
    (if any), both heads (including the GRL, which has no tensors of its
    own), and `grl_lambda` -- not the full ~86M-parameter backbone, which
    the runner reloads from the registered checkpoint instead.
    """

    trainable_state = {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad}
    torch.save(
        {
            "trainable_state_dict": trainable_state,
            "grl_lambda": model.grl_lambda,
            "unfreeze_last_blocks": model.unfreeze_last_blocks,
        },
        path,
    )


def load_checkpoint(model: DomainAdversarialModel, path: str) -> None:
    """Load a `save_checkpoint` file into `model` in place.

    `model` must already be constructed with the same `unfreeze_last_blocks`
    and `n_domains` as the checkpoint (the runner's job to get right); this
    only restores the trainable tensors and `grl_lambda` into it. Refuses a
    checkpoint whose saved keys do not match this model's currently
    trainable parameters exactly -- a mismatch here means the arm
    configuration silently changed between save and load, which must not
    pass quietly.
    """

    checkpoint = torch.load(path, map_location="cpu")
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    saved_names = set(checkpoint["trainable_state_dict"].keys())
    if saved_names != trainable_names:
        raise ValueError(
            "checkpoint's trainable parameter names do not match this model's; "
            f"only-in-checkpoint={saved_names - trainable_names}, only-in-model={trainable_names - saved_names}"
        )
    model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    model.grl_lambda = checkpoint["grl_lambda"]
