"""D-65 round 1: frozen DINOv2 ViT-L vs ViT-B under the prospective protocol.

Arms (all on the same frames, seeds, optimiser, budget and inner monitor set):
  B_ref   the registered D-53 step_permuted head on ViT-B features (reference row, no projection)
  B_proj  ProjectedStepHead(1536 -> 1536) on ViT-B features (matched control)
  L_proj  ProjectedStepHead(2048 -> 1536) on ViT-L features (candidate)
Epoch selection uses an inner monitor set of `MONITOR_PER_CLASS` training videos per class (fixed per fold); the
108-video validation views are scored once with the selected epoch. Registered contrast: L_proj vs B_proj.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ARMS = ("B_ref", "B_proj", "L_proj")
CONTRAST = ("L_proj", "B_proj")
MONITOR_PER_CLASS = 4
L_CACHE = "results/p1c/p1c-dinov2l-cache"


def inner_monitor_videos(video_targets: Mapping[str, int], per_class: int, seed: int) -> set[str]:
    """`per_class` training videos of every class, seeded, for epoch selection only."""

    rng = np.random.default_rng(seed)
    by: dict[int, list[str]] = {}
    for v, t in sorted(video_targets.items()):
        by.setdefault(int(t), []).append(v)
    out: set[str] = set()
    for t in sorted(by):
        if len(by[t]) <= per_class:
            raise ValueError(f"class {t} has only {len(by[t])} training videos; cannot hold out {per_class}")
        out |= {by[t][i] for i in rng.permutation(len(by[t]))[:per_class]}
    return out


def load_l_cache(root: Path = Path(L_CACHE)) -> tuple[np.ndarray, list[tuple[str, int]]]:
    with (root / "rows.csv").open(newline="") as h:
        keys = [(r["source_video_id"], int(r["source_frame_index"])) for r in csv.DictReader(h)]
    return np.load(root / "features.float16.npy", mmap_mode="r"), keys


def run_fold(fold: int, seeds: Sequence[int], arms: Sequence[str] = ARMS, *, epochs: int | None = None) -> tuple[list[dict], list[dict]]:
    import torch

    from core.backbone_cache import align
    from core.confusion_groups import macro_f1
    from core.p1c_head_variants import train_sequence_head
    from core.p1c_heads import ProjectedStepHead
    from core.p1c_oracle_interval import predict_video_probs
    from scripts.run_p1c_xd_extra_probe import FRAMES, train_head
    from scripts.run_subset7_search import load_fold_features, sequences

    fd = load_fold_features(fold, ("train", "validation"))
    classes = fd["class_names"]
    data = {"B": fd}
    if any(a.startswith("L") for a in arms):
        lf, lkeys = load_l_cache()
        data["L"] = {s: (np.asarray(align(lf, lkeys, fd[s][1]), dtype=np.float32), fd[s][1]) for s in ("train", "validation")}
    train_targets = {k[1]: int(v[1]["target"]) for k, v in sequences(fd, "train", 0).items()}
    monitor = inner_monitor_videos(train_targets, MONITOR_PER_CLASS, seed=fold)
    records, probs = [], []
    for seed in seeds:
        for arm in arms:
            src = data["L" if arm.startswith("L") else "B"]
            all_train = sequences(src, "train", seed)
            train_seq = {k: v for k, v in all_train.items() if k[1] not in monitor}
            monitor_seq = {k: v for k, v in all_train.items() if k[1] in monitor}
            val_seq = sequences(src, "validation", seed)
            kwargs = {} if epochs is None else {"epochs": epochs}
            if arm == "B_ref":
                result, head = train_head(seed, train_seq, monitor_seq, classes, epochs)
            else:
                torch.manual_seed(seed)  # same StepHead init for every projected arm (see ProjectedStepHead)
                head = ProjectedStepHead(src["train"][0].shape[1], frames=FRAMES, num_classes=len(classes), projection_seed=10_000 + seed)
                result = train_sequence_head(head, train_seq, monitor_seq, classes, frames=FRAMES, seed=seed, **kwargs)
                head.eval()
            keys = sorted(val_seq)
            p = np.stack([predict_video_probs(head, val_seq[k][0], FRAMES) for k in keys])
            truth = np.array([int(val_seq[k][1]["target"]) for k in keys])
            records.append({"arm": arm, "fold": fold, "seed": seed, "macro_f1": macro_f1(truth, p.argmax(1), len(classes)),
                            "accuracy": float((p.argmax(1) == truth).mean()), "best_epoch": int(result.best_epoch),
                            "monitor_macro_f1": float(result.best_validation_macro_f1), "epochs_run": len(result.epochs),
                            "train_videos": len(train_seq), "monitor_videos": len(monitor_seq),
                            "trainable_parameters": int(sum(q.numel() for q in head.parameters() if q.requires_grad))})
            for k, row in zip(keys, p):
                probs.append({"arm": arm, "fold": fold, "seed": seed, "video": k[1], "true": classes[int(val_seq[k][1]["target"])],
                              **{f"p_{c}": float(x) for c, x in zip(classes, row)}})
    return records, probs
