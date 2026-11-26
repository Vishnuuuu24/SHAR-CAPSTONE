"""P2B denoiser training: frame pool, on-the-fly noisy/clean crops, and the training loop (D-60)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from core.p1c_video_sampling import sample_rows_per_video
from core.p2_pipeline import decode_rgb_float
from core.validation_frames import CLOSURE, MANIFEST
from data.degradations import DegradationSpec, SampleKey, apply_degradation

TRAIN_MANIFEST = "results/p1b/p1b-video-ar-fold{fold}/train_manifest.csv"


def training_frame_pool(repo_root: Path, folds: Sequence[int], classes: Sequence[str], frames_per_video: int) -> list[dict]:
    """Evenly spaced frames from AR train-manifest videos that are in no fold's validation set.

    Returns [{video, frame, filepath, file_digest, jpeg}] in a stable order. Labels are not returned: the
    denoiser never sees them, and the validation videos it is later judged on are excluded entirely.
    """

    from data.p1c_loader import FramePreprocessing
    from data.video_dataset import P1CVideoFrameDataset

    root = Path(repo_root)
    held_out = set()
    for fold in folds:
        with (root / MANIFEST.format(fold=fold)).open(newline="", encoding="utf-8") as h:
            held_out |= {r["source_video_id"] for r in csv.DictReader(h)}
    pre = FramePreprocessing.from_config({"resize_height": 224, "resize_width": 224, "interpolation": "bilinear",
                                          "color_mode": "RGB", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    pool, seen = [], set()
    for fold in folds:
        path = root / TRAIN_MANIFEST.format(fold=fold)
        with path.open(newline="", encoding="utf-8") as h:
            rows = list(csv.DictReader(h))
        dataset = P1CVideoFrameDataset(path, closure_path=root / CLOSURE, preprocessing=pre, class_names=list(classes),
                                       verify_digest=False, split="train")
        by_video = defaultdict(list)
        for i in sample_rows_per_video(rows, frames_per_video, seed=None):
            by_video[rows[i]["source_video_id"]].append(i)
        for vid in sorted(by_video):
            if vid in held_out or vid in seen:
                continue
            seen.add(vid)
            for i in by_video[vid]:
                pool.append({"video": vid, "frame": int(rows[i]["source_frame_index"]), "filepath": rows[i]["filepath"],
                             "file_digest": rows[i]["file_digest"], "jpeg": dataset.raw_jpeg(i)})
    return pool


class NoisyCropDataset(torch.utils.data.Dataset):
    """Item i is a deterministic function of (seed, i): frame, crop window, degradation and its noise.

    So a run is reproducible whatever the DataLoader worker count, and no corrupted frame is cached.
    """

    def __init__(self, pool: Sequence[dict], specs: Sequence[DegradationSpec], *, crop: int, seed: int,
                 noise_global_seed: int, length: int) -> None:
        self.pool, self.specs, self.crop, self.seed = pool, list(specs), crop, seed
        self.noise_global_seed, self.length = noise_global_seed, length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, i])
        item = self.pool[int(rng.integers(len(self.pool)))]
        img = decode_rgb_float(item["jpeg"])
        y = int(rng.integers(img.shape[0] - self.crop + 1))
        x = int(rng.integers(img.shape[1] - self.crop + 1))
        clean = np.ascontiguousarray(img[y:y + self.crop, x:x + self.crop])
        spec = self.specs[int(rng.integers(len(self.specs)))]
        key = SampleKey(relative_path=f"{item['filepath']}/frame{item['frame']:06d}/draw{i}",
                        file_digest=item["file_digest"], global_seed=self.noise_global_seed)
        noisy = apply_degradation(clean, spec=spec, sample=key)
        return torch.from_numpy(noisy).permute(2, 0, 1), torch.from_numpy(clean).permute(2, 0, 1)


def train_denoiser(model: torch.nn.Module, loader, *, steps: int, learning_rate: float, betas: Sequence[float],
                   loss_fn: Callable, device: torch.device, log_every: int = 500,
                   on_log: Callable[[dict], None] | None = None) -> list[dict]:
    """Adam on loss_fn(recon, clean, mu, logvar) for exactly `steps` batches; refuses a non-finite loss."""

    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=tuple(betas))
    history, running = [], defaultdict(float)
    for step, (noisy, clean) in enumerate(loader, 1):
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        recon, mu, logvar = model(noisy)
        loss, parts = loss_fn(recon, clean, mu, logvar)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {parts}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running["loss"] += float(loss.detach())
        for k, v in parts.items():
            running[k] += v
        if step % log_every == 0 or step == steps:
            n = log_every if step % log_every == 0 else step % log_every
            row = {"step": step, **{k: v / n for k, v in running.items()}}
            history.append(row)
            running.clear()
            if on_log:
                on_log(row)
        if step == steps:
            break
    model.eval()
    return history
