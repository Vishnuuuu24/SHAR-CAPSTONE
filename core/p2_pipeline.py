"""RESULT 1 (P2A/P2C) per-frame pipeline: frozen degradation -> frozen restoration -> PSNR/SSIM -> classifier input."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from data.degradations import DegradationSpec, SampleKey, apply_degradation
from eval.image_quality import psnr, ssim
from models.restoration import restore_image


@dataclass(frozen=True)
class Condition:
    index: int
    degradation: DegradationSpec
    restoration: str

    @property
    def degradation_id(self) -> str:
        p = dict(self.degradation.parameters)
        level = p.get("sigma_8bit", p.get("density", p.get("variance", "")))
        if self.degradation.family == "low_light":
            level = f"g{p['gamma']}_s{p['gaussian_sigma_8bit']}"
        return f"{self.degradation.family}{'' if level == '' else '_' + str(level)}"

    @property
    def name(self) -> str:
        return f"{self.degradation_id}__{self.restoration}"


def conditions(conventions) -> list[Condition]:
    """Every registered degradation x every registered classical restoration, in config order."""

    out = []
    for spec in conventions.degradation_specs:
        for method in conventions.restoration_parameters:
            out.append(Condition(len(out), spec, method))
    return out


def decode_rgb_float(jpeg: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / np.float32(255.0)


def sample_key(filepath: str, frame_index: int, file_digest: str, global_seed: int) -> SampleKey:
    return SampleKey(relative_path=f"{filepath}/frame{frame_index:06d}", file_digest=file_digest, global_seed=global_seed)


def process_frames(clean: Sequence[np.ndarray], keys: Sequence[SampleKey], condition: Condition, conventions) -> tuple[np.ndarray, list[float], list[float]]:
    """(restored frames as uint8 [n,H,W,3] for the classifier, PSNR list, SSIM list). Metrics on float32 before quantization."""

    restored_u8, ps, ss = [], [], []
    args = conventions.restoration_arguments(condition.restoration)
    for img, key in zip(clean, keys):
        degraded = apply_degradation(img, spec=condition.degradation, sample=key)
        restored = restore_image(degraded, **args)
        p = psnr(img, restored, **conventions.psnr_options)
        ps.append(p if math.isfinite(p) else float("inf"))
        ss.append(ssim(img, restored, **conventions.ssim_options))
        restored_u8.append(np.rint(restored * np.float32(255.0)).astype(np.uint8))
    return np.stack(restored_u8), ps, ss


def summarise(table: dict[str, dict[str, Any]], clean_name: str) -> dict[str, dict[str, Any]]:
    """Add paired deltas to per-condition scores: vs clean input, and vs the same degradation left unrestored."""

    out = {}
    for name, row in table.items():
        deg = name.split("__")[0]
        noisy = table.get(f"{deg}__identity")
        r = dict(row)
        for metric in ("macro_f1", "accuracy"):
            r[f"{metric}_delta_vs_clean"] = row[metric] - table[clean_name][metric]
            if noisy is not None:
                r[f"{metric}_delta_vs_unrestored"] = row[metric] - noisy[metric]
                gap = table[clean_name][metric] - noisy[metric]
                r[f"{metric}_recovered_fraction"] = (row[metric] - noisy[metric]) / gap if abs(gap) > 1e-9 else None
        out[name] = r
    return out
