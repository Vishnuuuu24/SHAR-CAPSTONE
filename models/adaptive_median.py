"""Adaptive (switching) median filter on GPU tensors (Hwang & Haddad 1995; Gonzalez & Woods sec. 5.3.3).

A pixel is replaced only where it looks like an impulse: the window grows from 3x3 until its median is not an
extreme (level A); then the centre pixel is kept unless it is itself the window's min or max (level B). Past
`s_max` the median is output. Applied per channel with replicate borders.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _window_stats(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, c, h, w = x.shape
    p = k // 2
    patches = F.unfold(F.pad(x.reshape(n * c, 1, h, w), (p, p, p, p), mode="replicate"), k)
    shape = (n, c, h, w)
    return (patches.min(1).values.reshape(shape), patches.median(1).values.reshape(shape),
            patches.max(1).values.reshape(shape))


def adaptive_median(x: torch.Tensor, s_max: int) -> torch.Tensor:
    """[N,C,H,W] -> same shape; s_max odd >= 3."""

    if s_max < 3 or s_max % 2 == 0:
        raise ValueError("s_max must be an odd integer >= 3")
    out = x.clone()
    undecided = torch.ones_like(x, dtype=torch.bool)
    for k in range(3, s_max + 1, 2):
        zmin, zmed, zmax = _window_stats(x, k)
        level_a = (zmed > zmin) & (zmed < zmax)
        decide = undecided & level_a
        keep = (x > zmin) & (x < zmax)
        out = torch.where(decide, torch.where(keep, x, zmed), out)
        undecided &= ~level_a
    return torch.where(undecided, zmed, out)


class AdaptiveMedianRestorer:
    def __init__(self, s_max: int, chunk: int) -> None:
        self.s_max, self.chunk = s_max, chunk

    @torch.no_grad()
    def restore(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([adaptive_median(x[i:i + self.chunk], self.s_max) for i in range(0, x.shape[0], self.chunk)])
