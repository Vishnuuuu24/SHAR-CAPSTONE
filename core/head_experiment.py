"""Training-data variants of the D-53 head, one factor at a time against the registered control (D-63).

Arms (all share the control's seeded head initialisation, optimiser, validation set and best-epoch rule):
  control        D-53 exactly: one fixed 8-frame draw per training video.
  resample       a fresh 8-frame bag per training video every epoch, from the same cached frames (same epoch budget).
  resample_long  the same, with a 100-epoch ceiling and patience 10 (resampling slows overfitting).
  label_smooth   control with label smoothing 0.1.
  lc25/lc50/lc75 control trained on that fraction of each class's training videos (diagnostic, not gated).
  extra34        control + 34 extra UCF videos for each of Burglary, RoadAccidents, Robbery, Stealing (D-64).
  extra_all      control + all 300 extra videos (50 / 100 / 100 / 50).
  extra_all_w    extra_all with inverse-frequency class weights (the extra videos shift the class priors).
The extra videos are outside the 700-video AR protocol (core/extra_ucf_videos.py): a named protocol deviation.
Gated arms must beat the control by `GATE_DELTA` macro-F1 on every fold.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.confusion_groups import GATE_DELTA
from core.head_resampling import make_epoch_resampler, stratified_video_subset

ARMS: dict[str, dict[str, Any]] = {
    "control": {},
    "resample": {"resample": True},
    "resample_long": {"resample": True, "epochs": 100, "patience": 10},
    "label_smooth": {"label_smoothing": 0.1},
    "lc25": {"fraction": 0.25},
    "lc50": {"fraction": 0.50},
    "lc75": {"fraction": 0.75},
    "extra34": {"extra_per_class": 34},
    "extra_all": {"extra_per_class": 10_000},
    "extra_all_w": {"extra_per_class": 10_000, "weighted": True},
}
GATED = ("resample", "resample_long", "label_smooth", "extra34", "extra_all", "extra_all_w")
EXTRA_CACHE = "results/p1c/p1c-extra-ucf-cache"


def load_extra_cache(root: Path = Path(EXTRA_CACHE)) -> tuple[np.ndarray, list[dict]]:
    with (root / "rows.csv").open(newline="") as h:
        rows = [{**r, "target": int(r["target"]), "source_frame_index": int(r["source_frame_index"])} for r in csv.DictReader(h)]
    return np.load(root / "features.float16.npy").astype(np.float32), rows


def extra_sequences(feats: np.ndarray, rows: Sequence[dict], per_class: int, seed: int) -> dict:
    """Sequences (same construction as the fold's training sequences) for a seeded, class-balanced subset of extra videos."""

    from core.extra_ucf_videos import balanced_extra_subset
    from core.p1c_head_variants import group_pooled_sequences_by_video

    keep = balanced_extra_subset({r["source_video_id"]: r["label"] for r in rows}, per_class, seed)
    idx = [i for i, r in enumerate(rows) if r["source_video_id"] in keep]
    return group_pooled_sequences_by_video(feats[idx], [rows[i] for i in idx], permute_frame_order=True, permute_seed=1000 + seed)


def run_fold(fold: int, seeds: Sequence[int], arms: Sequence[str], *, epochs_override: int | None = None) -> tuple[list[dict], list[dict]]:
    """(run records, per-video probability rows) for every arm x seed on one fold."""

    from core.p1c_oracle_interval import predict_video_probs
    from scripts.run_p1c_xd_extra_probe import FRAMES, train_head
    from scripts.run_subset7_search import load_fold_features, sequences

    fd = load_fold_features(fold, ("train", "validation"), keep_all_train=True)
    classes = fd["class_names"]
    all_feats, all_rows = fd["train_all"]
    extra = load_extra_cache() if any("extra_per_class" in ARMS[a] for a in arms) else None
    records, probs = [], []
    for seed in seeds:
        val_seq = sequences(fd, "validation", seed)
        for arm in arms:
            spec = ARMS[arm]
            train_seq = sequences(fd, "train", seed)
            keep = None
            if "fraction" in spec:
                keep = stratified_video_subset({k[1]: int(v[1]["target"]) for k, v in train_seq.items()}, spec["fraction"], seed)
                train_seq = {k: v for k, v in train_seq.items() if k[1] in keep}
            kwargs: dict[str, Any] = {}
            if spec.get("resample"):
                kwargs["epoch_resampler"] = make_epoch_resampler(all_feats, all_rows, frames=FRAMES, seed=seed, keep_videos=keep)
            if "extra_per_class" in spec:
                extra_seq = extra_sequences(extra[0], extra[1], spec["extra_per_class"], seed)
                if set(extra_seq) & set(train_seq):
                    raise RuntimeError("extra videos overlap the fold's training videos")
                train_seq = {**train_seq, **extra_seq}
            if spec.get("weighted"):
                from core.combined_probe import inverse_frequency_weights
                kwargs["class_weights"] = inverse_frequency_weights([int(v[1]["target"]) for v in train_seq.values()], len(classes))
            if "label_smoothing" in spec:
                kwargs["label_smoothing"] = spec["label_smoothing"]
            if "patience" in spec:
                kwargs["patience"] = spec["patience"]
            epochs = epochs_override if epochs_override is not None else spec.get("epochs")
            result, head = train_head(seed, train_seq, val_seq, classes, epochs, **kwargs)
            keys = sorted(val_seq)
            p = np.stack([predict_video_probs(head, val_seq[k][0], FRAMES) for k in keys])
            truth = np.array([int(val_seq[k][1]["target"]) for k in keys])
            from core.confusion_groups import macro_f1
            records.append({"arm": arm, "fold": fold, "seed": seed, "macro_f1": macro_f1(truth, p.argmax(1), len(classes)),
                            "trainer_macro_f1": float(result.best_validation_macro_f1), "accuracy": float((p.argmax(1) == truth).mean()),
                            "best_epoch": int(result.best_epoch), "epochs_run": len(result.epochs), "train_videos": len(train_seq),
                            "val_curve": [float(e["validation_macro_f1"]) for e in result.epochs], "train_seconds": float(result.train_seconds)})
            for k, row in zip(keys, p):
                probs.append({"arm": arm, "fold": fold, "seed": seed, "video": k[1], "true": classes[int(val_seq[k][1]["target"])],
                              **{f"p_{c}": float(x) for c, x in zip(classes, row)}})
    return records, probs


def summarise_arms(records: Sequence[Mapping[str, Any]], control: str = "control") -> dict[str, dict]:
    """Per arm: fold means over seeds, paired per-fold delta vs control, and the all-folds gate verdict."""

    folds = sorted({r["fold"] for r in records})

    def fold_mean(arm: str, key: str = "macro_f1") -> np.ndarray:
        return np.array([np.mean([r[key] for r in records if r["arm"] == arm and r["fold"] == f]) for f in folds])

    base = fold_mean(control)
    out = {}
    for arm in sorted({r["arm"] for r in records}):
        m = fold_mean(arm)
        d = m - base
        out[arm] = {"macro_f1": float(m.mean()), "accuracy": float(fold_mean(arm, "accuracy").mean()),
                    "per_fold_macro_f1": [float(x) for x in m], "per_fold_delta_vs_control": [float(x) for x in d],
                    "mean_delta": float(d.mean()), "folds_positive": int((d > 0).sum()),
                    "folds_passing_gate": int((d >= GATE_DELTA).sum()), "gate_passed": bool((d >= GATE_DELTA).all()),
                    "median_best_epoch": float(np.median([r["best_epoch"] for r in records if r["arm"] == arm])),
                    "train_videos": float(np.mean([r["train_videos"] for r in records if r["arm"] == arm]))}
    return out
