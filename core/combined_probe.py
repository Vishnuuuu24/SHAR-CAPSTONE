"""Row selection, pooling and reporting for the SHAR-Combined arm comparison (ucf_only vs all_sources).

The head, its hyperparameters and the video-level rule are the headline ``step_permuted`` path
(``scripts.run_p1c_xd_extra_probe._train_eval``); only the data selection, the class weighting and the
per-source reporting live here. Rows come exclusively from ``core.combined_split.load_trainable``, so
the sealed test split, untrainable rows and unmapped (empty ``unified_label``) rows can never be reached.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.combined_dataset import CUVA, UCF, XD
from core.combined_features import FRAMES
from core.combined_split import assert_test_sealed, load_trainable
from core.p1c_readiness import sha256_file
from core.p1c_xd_probe import GATE_DELTA, gate_check
from eval.metrics import classification_metrics

CLASS_NAMES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "Normal",
               "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
ARMS = {"ucf_only": (UCF,), "all_sources": (UCF, XD, CUVA)}
PARTS = {"ucf": UCF, "xd": XD, "cuva": CUVA}
HIGHLIGHT = ("Shooting", "Assault", "Fighting", "Abuse", "Arrest")
VideoKey = tuple[str, str]


def load_cache(cache_root: Path, manifest: Path) -> tuple[np.ndarray, dict[VideoKey, int], dict[str, Any]]:
    """``(features memmap, {(source, video id): row index} for ok rows, cache manifest)``; refuses a stale cache."""

    cache_manifest = json.loads((cache_root / "manifest.json").read_text())
    if cache_manifest["combined_manifest_sha256"] != sha256_file(manifest):
        raise ValueError("cache was built from a different SHAR-Combined manifest.csv than the one being probed")
    features = np.load(cache_root / "features.float16.npy", mmap_mode="r")
    with (cache_root / "rows.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != features.shape[0]:
        raise ValueError(f"rows.csv has {len(rows)} rows but the array has {features.shape[0]}")
    index = {(r["source_dataset"], r["source_video_id"]): int(r["index"]) for r in rows if r["ok"] == "1"}
    return features, index, cache_manifest


def fold_rows(manifest: Path, fold: int, arm: str, index: Mapping[VideoKey, int]) -> tuple[list[dict], list[dict], dict[str, int]]:
    """``(train, validation, counts)`` for one fold and arm.

    Validation is every trainable ``train_pool`` row with ``fold == k`` (all sources, identical for both arms).
    Training is ``fold != k``, restricted to the arm's sources. Rows whose cache extraction failed are dropped
    and counted. Test rows are unreachable: ``load_trainable`` filters them and ``assert_test_sealed`` re-checks.
    """

    train, val = load_trainable(manifest, fold)
    train = [r for r in train if r["source_dataset"] in ARMS[arm]]
    assert_test_sealed(train + val)
    counts = {"train_missing_from_cache": sum((r["source_dataset"], r["source_video_id"]) not in index for r in train),
              "val_missing_from_cache": sum((r["source_dataset"], r["source_video_id"]) not in index for r in val)}
    keep = lambda rs: [r for r in rs if (r["source_dataset"], r["source_video_id"]) in index]  # noqa: E731
    return keep(train), keep(val), counts


def pool_videos(features: np.ndarray, indices: Sequence[int], chunk: int = 64) -> np.ndarray:
    """``[n, FRAMES, 1536]`` float32: CLS ++ mean(patch tokens) per frame -- the pooling the headline cache stores."""

    out = np.empty((len(indices), FRAMES, 2 * features.shape[-1]), dtype=np.float32)
    order = np.asarray(indices)
    for s in range(0, len(order), chunk):
        block = np.asarray(features[np.sort(order[s:s + chunk])], dtype=np.float32)  # sorted read, restored below
        perm = np.argsort(np.argsort(order[s:s + chunk]))
        block = block[perm]
        out[s:s + len(block)] = np.concatenate([block[:, :, 0], block[:, :, 1:].mean(axis=2)], axis=-1)
    return out


def frame_rows_and_pooled(rows: Sequence[dict], pooled_by_video: Mapping[VideoKey, np.ndarray]) -> tuple[np.ndarray, list[dict]]:
    """Flatten videos to the per-frame rows ``group_pooled_sequences_by_video`` expects (8 per video)."""

    frame_rows, blocks = [], []
    for r in rows:
        key = (r["source_dataset"], r["source_video_id"])
        blocks.append(pooled_by_video[key])
        for j in range(FRAMES):
            frame_rows.append({"source_dataset": key[0], "source_video_id": key[1], "source_frame_index": j,
                               "target": CLASS_NAMES.index(r["unified_label"]), "label": r["unified_label"]})
    return np.concatenate(blocks, axis=0), frame_rows


def inverse_frequency_weights(targets: Sequence[int], num_classes: int) -> list[float]:
    """``n / (K * n_c)`` per class (mean weight over samples is 1); a class absent from training gets 1.0."""

    counts = np.bincount(np.asarray(targets, dtype=int), minlength=num_classes).astype(float)
    return [float(len(targets) / (num_classes * c)) if c > 0 else 1.0 for c in counts]


def part_metrics(truth: Sequence[int], pred: Sequence[int], class_names: Sequence[str] = CLASS_NAMES) -> dict[str, Any]:
    """Video-level metrics for one source part.

    ``macro_f1`` averages F1 over the classes that have support in this part (for UCF that is all 14);
    ``macro_f1_all_classes`` averages over every label and is reported only for reference.
    """

    m = classification_metrics([class_names[t] for t in truth], [class_names[p] for p in pred], class_names)
    present = [c for c in class_names if m["per_class"][c]["support"] > 0]
    return {"n_videos": len(truth), "macro_f1": float(np.mean([m["per_class"][c]["f1"] for c in present])),
            "macro_f1_all_classes": m["macro_f1"], "accuracy": m["accuracy"], "classes_present": present,
            "per_class_f1": {c: m["per_class"][c]["f1"] for c in class_names},
            "per_class_support": {c: m["per_class"][c]["support"] for c in class_names}}


def evaluate_parts(preds: Mapping[VideoKey, np.ndarray], val_rows: Sequence[dict]) -> dict[str, dict[str, Any]]:
    """``{part: part_metrics}`` over the validation videos of each source; parts with no videos are omitted."""

    out = {}
    for part, source in PARTS.items():
        rows = [r for r in val_rows if r["source_dataset"] == source]
        if not rows:
            continue
        truth = [CLASS_NAMES.index(r["unified_label"]) for r in rows]
        pred = [int(np.argmax(preds[(source, r["source_video_id"])])) for r in rows]
        out[part] = part_metrics(truth, pred)
    return out


def _mean_by_fold(runs: Sequence[Mapping[str, Any]], arm: str, part: str, getter) -> dict[int, float]:
    by_fold: dict[int, list[float]] = defaultdict(list)
    for r in runs:
        if r["arm"] == arm and part in r["parts"]:
            by_fold[int(r["fold"])].append(getter(r["parts"][part]))
    return {f: float(np.mean(v)) for f, v in sorted(by_fold.items())}


def summarize(runs: Sequence[Mapping[str, Any]], class_names: Sequence[str] = CLASS_NAMES, delta: float = GATE_DELTA) -> dict[str, Any]:
    """Per-arm, per-part macro-F1 (seed mean -> fold mean), UCF per-class F1, and the paired all_sources - ucf_only gate.

    ``runs`` are ``{arm, fold, seed, parts: {part: part_metrics}}``. The gate is the pre-registered one:
    UCF-part fold-mean (over seeds) macro-F1 gain >= ``delta`` on every fold.
    """

    arms = sorted({r["arm"] for r in runs})
    out: dict[str, Any] = {"arms": {}, "gate_threshold": delta}
    for arm in arms:
        entry: dict[str, Any] = {"parts": {}}
        for part in PARTS:
            fm = _mean_by_fold(runs, arm, part, lambda m: m["macro_f1"])
            if not fm:
                continue
            seed_vals = [r["parts"][part]["macro_f1"] for r in runs if r["arm"] == arm and part in r["parts"]]
            entry["parts"][part] = {"fold_mean_macro_f1": fm, "macro_f1_mean_over_folds": float(np.mean(list(fm.values()))),
                                    "macro_f1_sd_over_runs": float(np.std(seed_vals, ddof=1)) if len(seed_vals) > 1 else None,
                                    "n_runs": len(seed_vals)}
        per_class = {c: _mean_by_fold(runs, arm, "ucf", lambda m, c=c: m["per_class_f1"][c]) for c in class_names}
        entry["ucf_per_class_f1"] = {c: {"fold_mean": v, "mean_over_folds": float(np.mean(list(v.values())))} for c, v in per_class.items()}
        out["arms"][arm] = entry
    if {"ucf_only", "all_sources"} <= set(arms):
        base, test = out["arms"]["ucf_only"], out["arms"]["all_sources"]
        cmp: dict[str, Any] = {"gate_on_ucf_part": gate_check(base["parts"]["ucf"]["fold_mean_macro_f1"],
                                                              test["parts"]["ucf"]["fold_mean_macro_f1"], delta=delta)}
        cmp["ucf_mean_delta_over_folds"] = (test["parts"]["ucf"]["macro_f1_mean_over_folds"]
                                            - base["parts"]["ucf"]["macro_f1_mean_over_folds"])
        for part in ("xd", "cuva"):  # descriptive only; the arm trained on these sources, so no gate
            if part in base["parts"] and part in test["parts"]:
                cmp[f"{part}_per_fold_delta"] = {f: test["parts"][part]["fold_mean_macro_f1"][f] - base["parts"][part]["fold_mean_macro_f1"][f]
                                                 for f in base["parts"][part]["fold_mean_macro_f1"] if f in test["parts"][part]["fold_mean_macro_f1"]}
        cmp["ucf_per_class_f1_delta"] = {c: test["ucf_per_class_f1"][c]["mean_over_folds"] - base["ucf_per_class_f1"][c]["mean_over_folds"]
                                         for c in class_names}
        cmp["highlight_per_class_f1_delta"] = {c: cmp["ucf_per_class_f1_delta"][c] for c in HIGHLIGHT}
        out["comparison_all_sources_minus_ucf_only"] = cmp
    return out
