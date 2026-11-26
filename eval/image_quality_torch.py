"""Batched torch PSNR/SSIM equal to `eval.image_quality` (skimage) for the frozen P2A settings.

Used where restored frames already live on the GPU (P2B). SSIM supports only the registered configuration
(Gaussian window, sigma 1.5 -> 11 taps, population covariance); skimage filters with 'reflect' and then crops
(win_size - 1) // 2 pixels per side, which is exactly a 'valid' convolution, so no padding mode is involved.
Computed in float64; tests pin agreement with the skimage reference.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def psnr_batch(clean: torch.Tensor, candidate: torch.Tensor, *, data_range: float) -> list[float]:
    """[N,3,H,W] float32 pair -> per-image PSNR (inf where identical)."""

    mse = ((clean - candidate) ** 2).double().flatten(1).mean(1)
    return [math.inf if m == 0 else 10.0 * math.log10(data_range ** 2 / m) for m in mse.tolist()]


def _gaussian_taps(sigma: float, truncate: float = 3.5) -> torch.Tensor:
    r = int(truncate * sigma + 0.5)
    x = torch.arange(-r, r + 1, dtype=torch.float64)
    w = torch.exp(-0.5 * (x / sigma) ** 2)
    return w / w.sum()


def ssim_batch(clean: torch.Tensor, candidate: torch.Tensor, *, data_range: float, channel_axis: int, win_size: int,
               gaussian_weights: bool, sigma: float, use_sample_covariance: bool, k1: float, k2: float) -> list[float]:
    """[N,3,H,W] pair -> per-image SSIM, mean over channels, as skimage.metrics.structural_similarity."""

    if not gaussian_weights or use_sample_covariance or channel_axis not in (-1, 2):
        raise ValueError("only the registered SSIM configuration is implemented")
    taps = _gaussian_taps(sigma).to(clean.device)
    if taps.numel() != win_size:
        raise ValueError("win_size disagrees with the Gaussian window implied by sigma")
    n, c, h, w = clean.shape
    x = clean.double().reshape(n * c, 1, h, w)
    y = candidate.double().reshape(n * c, 1, h, w)
    kh, kw = taps.view(1, 1, -1, 1), taps.view(1, 1, 1, -1)

    def filt(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.conv2d(t, kh), kw)

    ux, uy = filt(x), filt(y)
    vx, vy, vxy = filt(x * x) - ux * ux, filt(y * y) - uy * uy, filt(x * y) - ux * uy
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    return s.reshape(n, c, -1).mean(2).mean(1).tolist()
