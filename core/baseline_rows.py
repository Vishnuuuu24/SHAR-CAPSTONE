"""D-69: the plan's three registered baseline rows (PRD S1, PHASE_PLAN P1C) under the D-65 protocol, on cached features.

Rows (all heads see the same per-video 8-frame seed-0 draws and the same inner monitor as D-66/D-68):
  R_lin    frozen ResNet50 C5 -> GAP -> linear, per frame; video = mean of frame logits      (GlobalAverageLinearHead)
  R_mlp    frozen ResNet50 C5 -> GAP -> MLP(512, ReLU, dropout 0.2), per frame; mean logits  (SameC5MLPHead)
  R_lstm   frozen ResNet50 C5 -> GAP -> Linear(512) -> LSTM(512, 1 layer) -> last valid step  (CNNLSTMTemporalBaseline)
  D_lstm   the same LSTM head on frozen DINOv2-B features (separates head from backbone; descriptive)
  D_step   the registered D-53 step_permuted head on DINOv2-B (reference; reproduces D-66 B_ref / D-68 B_frozen)

The ResNet backbone is frozen (D-42), so GAP of the cached C5 features is exactly the registered models' frame
feature; the heads below are the registered heads' layers applied to that cached vector. Temporal rows keep frames in
ascending source order (no permutation); validation videos are scored on every cached frame in consecutive 8-frame
chunks with mean-logit combination, the head-training rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

ROWS = ("R_lin", "R_mlp", "R_lstm", "D_lstm", "D_step")
REFERENCE = "D_step"
RESNET_CACHE = "results/p1c/p1c-frozen-feature-cache-ar-fold{fold}"
PROJECTION_DIM, HIDDEN_DIM, LSTM_LAYERS = 512, 512, 1
MLP_HIDDEN, MLP_DROPOUT = 512, 0.2


class FrameMeanHead(nn.Module):
    """Per-frame classifier, video logits = mean over the valid frames (padding ignored)."""

    def __init__(self, frame_classifier: nn.Module):
        super().__init__()
        self.frame_classifier = frame_classifier

    def forward(self, seq: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.frame_classifier(seq)
        if key_padding_mask is None:
            return logits.mean(1)
        keep = (~key_padding_mask).to(logits.dtype).unsqueeze(-1)
        return (logits * keep).sum(1) / keep.sum(1).clamp_min(1.0)


class LSTMSequenceHead(nn.Module):
    """`CNNLSTMTemporalBaseline` after its backbone: projection -> LSTM -> output at the last valid step -> linear."""

    def __init__(self, in_dim: int, num_classes: int, projection_dim: int = PROJECTION_DIM, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = LSTM_LAYERS, dropout: float = 0.0):
        super().__init__()
        self.projection = nn.Linear(in_dim, projection_dim)
        self.lstm = nn.LSTM(projection_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, seq: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t = seq.shape[:2]
        if key_padding_mask is None:
            lengths = torch.full((b,), t, dtype=torch.long)
        else:
            valid = ~key_padding_mask
            lengths = valid.sum(1)
            # padding must be trailing, as `_pad_sequence` makes it, or "last valid step" would be wrong
            if bool((valid.long().diff(dim=1) > 0).any()) or bool((lengths < 1).any()):
                raise ValueError("padding must be trailing and every sequence must have a valid frame")
        outputs, _ = self.lstm(self.projection(seq))
        idx = (lengths.to(outputs.device) - 1).view(b, 1, 1).expand(-1, 1, outputs.shape[-1])
        return self.classifier(outputs.gather(1, idx).squeeze(1))


def build_head(row: str, in_dim: int, num_classes: int, frames: int) -> nn.Module:
    if row == "R_lin":
        return FrameMeanHead(nn.Linear(in_dim, num_classes))
    if row == "R_mlp":
        return FrameMeanHead(nn.Sequential(nn.Linear(in_dim, MLP_HIDDEN), nn.ReLU(inplace=False), nn.Dropout(MLP_DROPOUT),
                                           nn.Linear(MLP_HIDDEN, num_classes)))
    if row in ("R_lstm", "D_lstm"):
        return LSTMSequenceHead(in_dim, num_classes)
    raise ValueError(row)


def ordered_sequences(feats: np.ndarray, rows: list, split: str, seed: int, temporal: bool) -> dict:
    """Per-video sequences: ascending frame order for temporal rows, the registered permutation otherwise."""

    from core.p1c_head_variants import group_pooled_sequences_by_video
    from scripts.run_subset7_search import PERMUTE_BASE

    return group_pooled_sequences_by_video(feats, rows, permute_frame_order=not temporal, permute_seed=PERMUTE_BASE[split] + seed)


def load_resnet(fold: int, rows_by_split: dict, repo_root: Path) -> tuple[dict, dict]:
    """ResNet GAP features aligned to the DINOv2 fold rows (same frames), plus each split's cache sha256."""

    import csv
    import json

    from core.backbone_cache import align

    out, sha = {}, {}
    for split, rows in rows_by_split.items():
        root = repo_root / RESNET_CACHE.format(fold=fold) / split
        with (root / "rows.csv").open(newline="") as h:
            keys = [(r["source_video_id"], int(r["source_frame_index"])) for r in csv.DictReader(h)]
        out[split] = np.asarray(align(np.load(root / "features.float16.npy", mmap_mode="r"), keys, rows), dtype=np.float32)
        sha[split] = json.loads((root / "manifest.json").read_text())["cache_sha256"]
    return out, sha


def run_fold(fold: int, seeds: Sequence[int], rows: Sequence[str] = ROWS, *, epochs: int | None = None,
             repo_root: Path = Path(".")) -> tuple[list[dict], list[dict], dict]:
    from core.backbone_experiment import MONITOR_PER_CLASS, inner_monitor_videos
    from core.confusion_groups import macro_f1
    from core.p1c_head_variants import train_sequence_head
    from core.p1c_oracle_interval import predict_video_probs
    from scripts.run_p1c_xd_extra_probe import FRAMES, train_head
    from scripts.run_subset7_search import load_fold_features, sequences

    fd = load_fold_features(fold, ("train", "validation"))
    classes = fd["class_names"]
    resnet, resnet_sha = load_resnet(fold, {s: fd[s][1] for s in ("train", "validation")}, repo_root)
    feats = {"R": resnet, "D": {s: fd[s][0] for s in ("train", "validation")}}
    train_targets = {k[1]: int(v[1]["target"]) for k, v in sequences(fd, "train", 0).items()}
    monitor = inner_monitor_videos(train_targets, MONITOR_PER_CLASS, seed=fold)
    kw = {} if epochs is None else {"epochs": epochs}
    records, probs = [], []
    for seed in seeds:
        for row in rows:
            temporal = row.endswith("lstm")
            src = feats[row[0]]
            all_train = ordered_sequences(src["train"], fd["train"][1], "train", seed, temporal)
            tr = {k: v for k, v in all_train.items() if k[1] not in monitor}
            mo = {k: v for k, v in all_train.items() if k[1] in monitor}
            val = ordered_sequences(src["validation"], fd["validation"][1], "validation", seed, temporal)
            if row == REFERENCE:
                result, head = train_head(seed, tr, mo, classes, epochs)
            else:
                torch.manual_seed(seed)
                head = build_head(row, src["train"].shape[1], len(classes), FRAMES)
                result = train_sequence_head(head, tr, mo, classes, frames=FRAMES, seed=seed, **kw)
                head.eval()
            keys = sorted(val)
            p = np.stack([predict_video_probs(head, val[k][0], FRAMES) for k in keys])
            truth = np.array([int(val[k][1]["target"]) for k in keys])
            records.append({"arm": row, "fold": fold, "seed": seed, "macro_f1": macro_f1(truth, p.argmax(1), len(classes)),
                            "accuracy": float((p.argmax(1) == truth).mean()), "best_epoch": int(result.best_epoch),
                            "monitor_macro_f1": float(result.best_validation_macro_f1), "epochs_run": len(result.epochs),
                            "train_loss_at_best": float(result.epochs[result.best_epoch]["train_mean_loss"]),
                            "train_videos": len(tr), "monitor_videos": len(mo), "temporal_order": temporal,
                            "trainable_parameters": int(sum(q.numel() for q in head.parameters() if q.requires_grad))})
            probs.extend({"arm": row, "fold": fold, "seed": seed, "video": k[1], "true": classes[int(val[k][1]["target"])],
                          **{f"p_{c}": float(x) for c, x in zip(classes, r)}} for k, r in zip(keys, p))
    return records, probs, {"fold": fold, "resnet_cache_sha256": resnet_sha, "dinov2_cache_sha256": fd["cache_sha256"]}
