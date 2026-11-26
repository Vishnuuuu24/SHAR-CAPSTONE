"""Generic convolutional denoising VAE for RESULT 1 (P2B) and its deterministic-autoencoder ablation.

Trained noisy -> clean. The variational variant optimises the ELBO with a calibrated Gaussian decoder
(Rybkin et al. 2021: decoder variance = batch MSE, so no hand-set KL weight); the ablation removes the
KL term and the sampling (z = mu, MSE loss). Inference is deterministic in both: z = mu.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _ResBlock(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ConvDenoisingVAE(nn.Module):
    """Fully convolutional; input and output are [N,3,H,W] in [0,1] with H, W divisible by 2**downsamplings."""

    def __init__(self, *, base_channels: int, latent_channels: int, downsamplings: int, norm_groups: int,
                 variational: bool) -> None:
        super().__init__()
        self.variational = variational
        widths = [base_channels] + [base_channels * 2] * downsamplings
        enc: list[nn.Module] = [nn.Conv2d(3, widths[0], 3, padding=1), _ResBlock(widths[0], norm_groups)]
        for a, b in zip(widths, widths[1:]):
            enc += [nn.Conv2d(a, b, 4, stride=2, padding=1), _ResBlock(b, norm_groups)]
        enc += [nn.GroupNorm(norm_groups, widths[-1]), nn.SiLU()]
        self.encoder = nn.Sequential(*enc)
        self.to_latent = nn.Conv2d(widths[-1], latent_channels * (2 if variational else 1), 1)
        dec: list[nn.Module] = [nn.Conv2d(latent_channels, widths[-1], 1), _ResBlock(widths[-1], norm_groups)]
        for a, b in zip(widths[::-1], widths[::-1][1:]):
            dec += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(a, b, 3, padding=1), _ResBlock(b, norm_groups)]
        dec += [nn.GroupNorm(norm_groups, widths[0]), nn.SiLU(), nn.Conv2d(widths[0], 3, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.to_latent(self.encoder(x * 2.0 - 1.0))
        return h.chunk(2, dim=1) if self.variational else (h, None)

    def forward(self, x: torch.Tensor, *, sample: bool = True):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.variational and sample else mu
        return torch.sigmoid(self.decoder(z)), mu, logvar

    @torch.no_grad()
    def restore(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, sample=False)[0]


def _softclip(x: torch.Tensor, minimum: float) -> torch.Tensor:
    return minimum + F.softplus(x - minimum)


def denoiser_loss(recon: torch.Tensor, clean: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor | None,
                  *, log_sigma_min: float | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-pixel-normalised training loss and its parts. logvar None -> the autoencoder ablation (MSE only)."""

    sq = (recon - clean) ** 2
    mse = sq.mean()
    if logvar is None:
        return mse, {"mse": float(mse.detach())}
    log_sigma = _softclip(0.5 * torch.log(mse), log_sigma_min)
    nll = (sq / (2.0 * torch.exp(2.0 * log_sigma)) + log_sigma + 0.5 * math.log(2.0 * math.pi)).flatten(1).sum(1).mean()
    kl = (0.5 * (mu ** 2 + torch.exp(logvar) - 1.0 - logvar)).flatten(1).sum(1).mean()
    pixels = clean[0].numel()
    return (nll + kl) / pixels, {"mse": float(mse.detach()), "nll": float(nll.detach()), "kl": float(kl.detach()),
                                 "log_sigma": float(log_sigma.detach())}


def build_denoiser(model_cfg: dict, variant_cfg: dict) -> ConvDenoisingVAE:
    return ConvDenoisingVAE(base_channels=int(model_cfg["base_channels"]), latent_channels=int(model_cfg["latent_channels"]),
                            downsamplings=int(model_cfg["downsamplings"]), norm_groups=int(model_cfg["norm_groups"]),
                            variational=bool(variant_cfg["variational"]))
