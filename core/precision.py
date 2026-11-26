"""Resolve a registered training precision into an executable policy.

Mirrors `core.device`: a config names a precision, this module turns that name
into the autocast context and gradient scaler the engine must actually use, or
refuses to run. Nothing here picks a precision on its own -- an unregistered
name is an error, never a default.

This module exists because the precision field was previously *validated and
then ignored*: `core.p1c_readiness` accepted `amp_float16`, no code path ever
read it, and a config could register mixed precision while the run trained in
float32. A registered value that does not reach the hardware is worse than no
field at all, because the provenance record then asserts something untrue.

## What the three policies mean

`float32`
    No autocast, no scaler. The reference numerics, and what every P1C result
    to date was measured in.

`amp_bfloat16`
    Autocast to bfloat16. bfloat16 keeps float32's 8-bit exponent and spends
    the difference on mantissa, so its dynamic range is float32's and gradient
    underflow is not a concern -- **no loss scaling is needed or used**.
    Requires Ampere or newer; on this project's H200 MIG slice it is native.

`amp_float16`
    Autocast to float16. float16 has a 5-bit exponent, so small gradients
    underflow to zero and training silently degrades without loss scaling.
    A `GradScaler` is therefore **mandatory**, not optional, and this module
    will not hand back a float16 policy without one.

## What autocast does and does not cover

Autocast governs the forward pass and the loss. Backward runs in whatever
dtypes the forward recorded, so it is deliberately outside the context. Loss
functions that are numerically unsafe in reduced precision -- cross entropy
among them -- are autocast back to float32 by torch's own op table, so the
loss value stays float32 in every policy here.

## Skipped steps are a measurement, not an error

Under float16 the scaler deliberately *skips* an optimizer step whenever the
gradients overflow, then lowers its scale. That is the mechanism working, not
a fault. But a run that skips many steps has quietly trained less than its
epoch count claims, so the engine counts them and reports them rather than
letting them vanish (docs/PROJECT_RULES.md rule 4).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager

import torch

FLOAT32 = "float32"
AMP_BFLOAT16 = "amp_bfloat16"
AMP_FLOAT16 = "amp_float16"

PRECISIONS = (FLOAT32, AMP_BFLOAT16, AMP_FLOAT16)

_AUTOCAST_DTYPE: dict[str, torch.dtype | None] = {
    FLOAT32: None,
    AMP_BFLOAT16: torch.bfloat16,
    AMP_FLOAT16: torch.float16,
}


@dataclass(frozen=True)
class PrecisionPolicy:
    """An executable precision decision, resolved from a registered name."""

    name: str
    device_type: str
    autocast_dtype: torch.dtype | None
    uses_grad_scaler: bool

    @property
    def is_mixed(self) -> bool:
        return self.autocast_dtype is not None

    def autocast(self) -> ContextManager[Any]:
        """The context the forward pass and loss must run inside."""
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype)

    def make_grad_scaler(self) -> torch.amp.GradScaler | None:
        """A scaler for float16, or None where loss scaling is not needed.

        Returned per-engine rather than stored on the policy: a scaler carries
        mutable state (the current scale, and how many steps have been clean),
        so sharing one across two engines would couple their training.
        """
        if not self.uses_grad_scaler:
            return None
        return torch.amp.GradScaler(self.device_type)

    def report(self) -> dict[str, Any]:
        """Provenance for the run manifest."""
        return {
            "precision": self.name,
            "autocast": self.is_mixed,
            "autocast_dtype": str(self.autocast_dtype) if self.autocast_dtype else None,
            "device_type": self.device_type,
            "grad_scaler": self.uses_grad_scaler,
        }


def resolve_precision_policy(precision: object, device: torch.device | str) -> PrecisionPolicy:
    """Return the policy a registered precision names, or refuse to run.

    Refuses rather than degrading: a mixed-precision run that silently fell
    back to float32 would be slower than its registration implies and would
    report numerics it did not use.
    """

    if precision not in PRECISIONS:
        raise ValueError(
            f"training.precision must be one of {sorted(PRECISIONS)}; got {precision!r}"
        )
    name = str(precision)
    device_type = torch.device(device).type

    if name == FLOAT32:
        return PrecisionPolicy(name, device_type, None, uses_grad_scaler=False)

    # Everything below is mixed precision and needs real hardware support.
    if device_type != "cuda":
        raise RuntimeError(
            f"precision {name} requires a CUDA device; got device_type={device_type!r}. "
            "Refusing silent fallback to float32."
        )
    if name == AMP_BFLOAT16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            f"precision {name} requires bfloat16 support; torch reports this CUDA device "
            "does not provide it. Refusing silent fallback."
        )

    return PrecisionPolicy(
        name,
        device_type,
        _AUTOCAST_DTYPE[name],
        # float16 underflows without loss scaling; bfloat16 has float32's
        # exponent range and must NOT be scaled.
        uses_grad_scaler=(name == AMP_FLOAT16),
    )
