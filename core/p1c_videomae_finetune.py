"""Partial fine-tuning of VideoMAE-B's top blocks from the cached block-8 hidden state.

Arms differ in ONE factor, the number of trainable top blocks k in {0, 2, 4}
(`core.p1c_videomae_split.TopBlocksClassifier`). Everything else is the
registered recipe below, fixed a priori: EXACTLY `EPOCHS` epochs, the reported
metric is the FINAL epoch, and the per-epoch validation curve is recorded only
so an oracle-selected epoch can be shown and labelled as not a headline.

Units: each cached clip carries its video's label; validation prediction is
the mean clip softmax over a video's clips (video-level), scored with
`eval.metrics.classification_metrics`.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from core.p1c_videomae_cache import ARRAY_NAME, fold_row_indices
from core.p1c_videomae_split import TopBlocksClassifier, param_groups, trainable_parameter_count
from eval.metrics import classification_metrics

ARMS = {"k0": 0, "k2": 2, "k4": 4}
HYPERPARAMETERS = {
    "epochs": 8, "warmup_epochs": 1, "batch_size": 16, "block_lr": 1e-5, "head_lr": 1e-3, "weight_decay": 0.05,
    "optimizer": "AdamW", "schedule": "per-step linear warmup then cosine to 0", "grad_clip": 1.0,
    "precision": "bf16 autocast", "loss": "cross_entropy", "augmentation": "none", "fc_norm": "frozen in every arm",
    "selection": "none -- final epoch is the reported metric",
}


def lr_multiplier(step: int, steps_per_epoch: int, epochs: int, warmup_epochs: int = 1) -> float:
    """LR factor for 0-based optimiser `step`: linear warmup over `warmup_epochs`, then cosine to 0."""

    warmup, total = warmup_epochs * steps_per_epoch, epochs * steps_per_epoch
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def epoch_batches(n_items: int, batch_size: int, seed: int, fold: int, epoch: int) -> list[list[int]]:
    """Shuffled batches of positions `0..n_items-1`; a pure function of (seed, fold, epoch), arm-independent."""

    order = np.random.default_rng([seed, fold, epoch]).permutation(n_items)
    return [order[i:i + batch_size].tolist() for i in range(0, n_items, batch_size)]


def video_level_probs(clip_probs: np.ndarray, clip_videos: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Mean clip softmax per video; videos returned sorted."""

    videos = sorted(set(clip_videos))
    index = {v: i for i, v in enumerate(videos)}
    sums = np.zeros((len(videos), clip_probs.shape[1]))
    counts = np.zeros(len(videos))
    for p, v in zip(clip_probs, clip_videos):
        sums[index[v]] += p
        counts[index[v]] += 1
    return videos, sums / counts[:, None]


def score_videos(videos: Sequence[str], probs: np.ndarray, truth: Mapping[str, int], class_names: Sequence[str]) -> dict[str, Any]:
    pred = probs.argmax(1)
    metrics = classification_metrics([class_names[truth[v]] for v in videos], [class_names[i] for i in pred], list(class_names))
    return metrics


class ClipStore:
    """Row-addressed reader of the memmapped hidden-state cache (opened lazily so DataLoader workers each get their own map)."""

    def __init__(self, root: Path, rows: Sequence[Mapping[str, Any]]):
        self.path, self.rows, self._map = Path(root) / ARRAY_NAME, rows, None

    def get(self, row: int) -> torch.Tensor:
        if self._map is None:
            self._map = np.load(self.path, mmap_mode="r")
        return torch.from_numpy(np.array(self._map[row]))


class _Clips(Dataset):
    def __init__(self, store: ClipStore, row_ids: Sequence[int]):
        self.store, self.row_ids = store, list(row_ids)

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, i):
        row = self.row_ids[i]
        return self.store.get(row), self.store.rows[row]["target"]


def _loader(store, row_ids, batches, num_workers, pin):
    return DataLoader(_Clips(store, row_ids), batch_sampler=batches, num_workers=num_workers, pin_memory=pin)


@torch.no_grad()
def evaluate(model, store, val_rows, class_names, device, batch_size, num_workers):
    model.eval()
    batches = [list(range(i, min(i + batch_size, len(val_rows)))) for i in range(0, len(val_rows), batch_size)]
    probs = []
    for hidden, _y in _loader(store, val_rows, batches, num_workers, device.type == "cuda"):
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(hidden.to(device, non_blocking=True))
        probs.append(F.softmax(logits.float(), 1).cpu().numpy())
    clip_videos = [store.rows[r]["source_video_id"] for r in val_rows]
    videos, vprobs = video_level_probs(np.concatenate(probs), clip_videos)
    truth = {store.rows[r]["source_video_id"]: store.rows[r]["target"] for r in val_rows}
    return videos, vprobs, score_videos(videos, vprobs, truth, class_names)


def train_run(backbone, store: ClipStore, *, fold: int, arm: str, seed: int, class_names: Sequence[str], device: torch.device,
              epochs: int = HYPERPARAMETERS["epochs"], num_workers: int = 4, grad_checkpoint: bool = False,
              max_train: int | None = None, max_val: int | None = None, log=print) -> dict[str, Any]:
    hp = HYPERPARAMETERS
    train_rows = fold_row_indices(store.rows, fold, "train")[:max_train]
    val_rows = fold_row_indices(store.rows, fold, "validation")[:max_val]
    overlap = {store.rows[r]["source_video_id"] for r in train_rows} & {store.rows[r]["source_video_id"] for r in val_rows}
    if overlap:
        raise RuntimeError(f"train/validation share videos in fold {fold}: {sorted(overlap)[:3]}")
    model = TopBlocksClassifier(backbone, num_classes=len(class_names), trainable_blocks=ARMS[arm], seed=seed,
                                grad_checkpoint=grad_checkpoint).to(device)
    optimizer = AdamW(param_groups(model, block_lr=hp["block_lr"], head_lr=hp["head_lr"], weight_decay=hp["weight_decay"]))
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]
    steps_per_epoch = math.ceil(len(train_rows) / hp["batch_size"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    curve, step, started = [], 0, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        batches = epoch_batches(len(train_rows), hp["batch_size"], seed, fold, epoch)
        total, count = 0.0, 0
        for hidden, y in _loader(store, train_rows, batches, num_workers, device.type == "cuda"):
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * lr_multiplier(step, steps_per_epoch, epochs, hp["warmup_epochs"])
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(hidden.to(device, non_blocking=True))
            loss = F.cross_entropy(logits.float(), y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at fold {fold} {arm} seed {seed} epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, hp["grad_clip"])
            optimizer.step()
            total, count, step = total + float(loss) * len(y), count + len(y), step + 1
        videos, vprobs, metrics = evaluate(model, store, val_rows, class_names, device, hp["batch_size"], num_workers)
        curve.append({"epoch": epoch, "train_loss": total / count, "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"]})
        log(f"  fold{fold} {arm} s{seed} epoch {epoch}/{epochs} loss {total / count:.4f} val macro-F1 {metrics['macro_f1']:.4f} ({time.time() - started:.0f}s)")
    best = max(curve, key=lambda c: c["val_macro_f1"])
    names = list(class_names)
    video_target = {store.rows[r]["source_video_id"]: store.rows[r]["target"] for r in val_rows}
    return {
        "fold": fold, "arm": arm, "trainable_blocks": ARMS[arm], "seed": seed, "epochs": epochs,
        "final_epoch_macro_f1": metrics["macro_f1"], "final_epoch_accuracy": metrics["accuracy"],
        "per_class_f1": {c: metrics["per_class"][c]["f1"] for c in names},
        "per_class_recall": {c: metrics["per_class"][c]["recall"] for c in names},
        "per_class_support": {c: metrics["per_class"][c]["support"] for c in names},
        "confusion_matrix": metrics["confusion_matrix"], "epoch_curve": curve,
        "oracle_selected_epoch": {"epoch": best["epoch"], "val_macro_f1": best["val_macro_f1"],
                                  "note": "oracle-selected on validation; NOT a headline"},
        "wall_seconds": time.time() - started, "trainable_parameters": trainable_parameter_count(model),
        "n_train_clips": len(train_rows), "n_val_clips": len(val_rows), "n_val_videos": len(videos),
        "per_video": [{"source_video_id": v, "true_label": names[video_target[v]],
                       "pred_label": names[int(p.argmax())], "probs": [float(x) for x in p]} for v, p in zip(videos, vprobs)],
    }
