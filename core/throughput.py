"""Resolve a registered throughput policy into executable CUDA-side speedups.

Mirrors `core.precision`: a config names a set of throughput knobs, this module
turns that into concrete `nn.Module`/`Tensor` transforms, or refuses to run.
Nothing here turns an optimization on by itself -- an absent `training.throughput`
block resolves to every knob off, so every existing config keeps its exact
current behaviour with no edits, and a config that registers the block but
misspells or half-fills it is refused rather than silently taking the parts it
understood.

## Why these three and only these three

Each knob changes which CUDA kernels run, never the arithmetic they compute
(modulo ordinary floating-point non-associativity, the same caveat autocast
already carries). That is the boundary that keeps this module out of
`core.precision`'s territory: precision changes numerics on purpose, this
module tries not to.

`channels_last`
    Stores activations as NHWC instead of NCHW. cuDNN's convolution kernels for
    Tensor-Core hardware are written against NHWC; running NCHW through them
    forces an implicit transpose per layer. Setting this on the model alone
    does nothing -- an NCHW-contiguous input tensor gets transposed right back
    on the first layer -- so `prepare_input` is not optional cosmetics, it is
    the other half of the same knob. See `ThroughputPolicy.prepare_input`.

`compile`
    `torch.compile`'s graph capture and kernel fusion. `"none"` disables it
    (kept as the literal string, never `None`, so a manifest field always
    holds an unambiguous value). The other three modes are torch's own names
    and are passed straight through as `mode=`; this module does not
    reinterpret them.

`fused_optimizer`
    AdamW's fused CUDA kernel collapses its per-parameter elementwise ops into
    one launch instead of one launch per op per parameter. It is algebraically
    the same update, just fewer kernel launches.

## Refuses rather than degrades

Enabling any of these on a non-CUDA device raises, the same as mixed precision
does in `core.precision`: `channels_last` is a no-op on CPU that would make a
config's registration a lie, `torch.compile` support outside CUDA is not what
this project has validated, and a fused optimizer kernel simply does not exist
off CUDA. The all-off policy is legal everywhere, including CPU tests.

## Interaction this module does not resolve

`torch.compile` under `cuda_strict`'s `torch.use_deterministic_algorithms(True)`
can refuse to compile some fused kernels and fall back to slower deterministic
ones, or in rarer cases refuse a graph outright. That is `core.device`'s
determinism policy and torch's compiler negotiating, not a throughput-policy
concern -- this module does not catch or reinterpret it. A run that hits it
will see the error torch raises when the two collide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

COMPILE_MODES = ("none", "default", "reduce-overhead", "max-autotune")

_REQUIRED_KEYS = frozenset({"channels_last", "compile", "fused_optimizer"})


@dataclass(frozen=True)
class ThroughputPolicy:
    """An executable throughput decision, resolved from a registered block."""

    channels_last: bool
    compile_mode: str
    fused_optimizer: bool
    device_type: str

    @classmethod
    def disabled(cls, device: torch.device | str) -> "ThroughputPolicy":
        """The all-off policy: what an absent `training.throughput` resolves to."""
        return cls(
            channels_last=False,
            compile_mode="none",
            fused_optimizer=False,
            device_type=torch.device(device).type,
        )

    @property
    def memory_format(self) -> torch.memory_format:
        return torch.channels_last if self.channels_last else torch.contiguous_format

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Apply memory-format and compilation to a model already on its device.

        Order matters within this method too: memory format is set before
        compilation, so `torch.compile` traces the layout the model will
        actually run under rather than retracing after the fact.
        """
        if self.channels_last:
            model = model.to(memory_format=self.memory_format)
        if self.compile_mode != "none":
            model = torch.compile(model, mode=self.compile_mode)
        return model

    def prepare_input(self, images: Tensor) -> Tensor:
        """Lay out a batch to match the model's memory format.

        Essential, not cosmetic: `channels_last` on the model alone gives
        nothing if the input tensor is still NCHW-contiguous, because torch
        would just transpose it back to feed the first layer.
        """
        return images.contiguous(memory_format=self.memory_format)

    def optimizer_kwargs(self) -> dict[str, Any]:
        """Extra kwargs for the caller's optimizer constructor."""
        return {"fused": True} if self.fused_optimizer else {}

    def report(self) -> dict[str, Any]:
        """Provenance for the run manifest."""
        return {
            "channels_last": self.channels_last,
            "compile_mode": self.compile_mode,
            "fused_optimizer": self.fused_optimizer,
            "device_type": self.device_type,
        }


def resolve_throughput_policy(spec: object, device: torch.device | str) -> ThroughputPolicy:
    """Return the policy a registered `training.throughput` block names, or refuse.

    Refuses rather than degrading: a typo'd key, a half-registered block, or a
    knob enabled on hardware that cannot honour it must never be silently
    reinterpreted as "off" or "on anyway".
    """

    device_type = torch.device(device).type
    if spec is None:
        return ThroughputPolicy.disabled(device_type)
    if not isinstance(spec, dict):
        raise TypeError(f"training.throughput must be a mapping or absent; got {type(spec).__name__}")

    unknown = set(spec) - _REQUIRED_KEYS
    if unknown:
        raise ValueError(f"training.throughput has unknown keys: {sorted(unknown)}")
    missing = _REQUIRED_KEYS - set(spec)
    if missing:
        raise ValueError(f"training.throughput is missing required keys: {sorted(missing)}")

    channels_last = spec["channels_last"]
    if not isinstance(channels_last, bool):
        raise TypeError("training.throughput.channels_last must be a bool")
    fused_optimizer = spec["fused_optimizer"]
    if not isinstance(fused_optimizer, bool):
        raise TypeError("training.throughput.fused_optimizer must be a bool")
    compile_mode = spec["compile"]
    if compile_mode not in COMPILE_MODES:
        raise ValueError(f"training.throughput.compile must be one of {sorted(COMPILE_MODES)}; got {compile_mode!r}")

    if (channels_last or fused_optimizer or compile_mode != "none") and device_type != "cuda":
        raise RuntimeError(
            "training.throughput enables a CUDA-only optimization "
            f"(channels_last={channels_last}, compile={compile_mode!r}, fused_optimizer={fused_optimizer}) "
            f"but got device_type={device_type!r}. Refusing silent fallback."
        )

    return ThroughputPolicy(
        channels_last=channels_last,
        compile_mode=str(compile_mode),
        fused_optimizer=fused_optimizer,
        device_type=device_type,
    )
