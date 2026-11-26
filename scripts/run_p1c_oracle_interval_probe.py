#!/usr/bin/env python3
"""Oracle-interval upper bound (diagnostic only): does giving `step_permuted` frames from INSIDE the true event help?

Evaluated on the interval-covered validation subset only; an inherited-label score, never comparable to the 14-class headline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_background_mask import load_intervals  # noqa: E402
from core.p1c_head_variants import group_pooled_sequences_by_video, load_pooled_features, train_sequence_head  # noqa: E402
from core.p1c_oracle_interval import (  # noqa: E402
    arm_row_positions, glance_spans, load_glances, macro_recall, mcnemar_exact, predict_video_probs, sample_evenly,
    select_event_positions,
)
from eval.metrics import classification_metrics  # noqa: E402
from scripts.run_p1c_head_comparison import _build_head, _load_fold, _load_patch_cache, _patch_indices_for  # noqa: E402

ANNOTATION = "Datasets/UCF Crime Official/annotations/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt"
HEADLINE_DIR = "results/p1c/p1c-head-comparison-stepctl"
PAIRS = [("uniform", "oracle_event"), ("uniform", "anti_event"), ("uniform8", "oracle_event")]
ARMS = ("uniform", "uniform8", "oracle_event", "anti_event")
FRAMES = 8
VIDEOS_CSV = "results/p1c/p1c-video-frames/videos.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_rows(rows) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        groups[row["source_video_id"]].append(position)
    return {video: np.asarray(positions) for video, positions in groups.items()}


def _foldcache_rows(fold: int) -> list[dict]:
    path = Path(f"results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}/train/rows.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _foldcache_pooled(fold: int, rows: list[dict], chunk: int = 2000) -> np.ndarray:
    """Pooled 1536-D vectors for `rows` gathered from the fold cache memmap, chunked."""

    feats = np.load(f"results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}/train/features.float16.npy", mmap_mode="r")
    idx = np.asarray([int(r["row_index"]) for r in rows], dtype=np.int64)
    out = np.empty((len(idx), feats.shape[1]), dtype=np.float32)
    for s in range(0, len(idx), chunk):
        out[s : s + chunk] = np.asarray(feats[idx[s : s + chunk]], dtype=np.float32)
    return out


def _train_event_rows(full_rows: list[dict], train_rows, intervals):
    """Replace covered train videos' uniform 8 rows with 8 in-event rows (midpoints of 8 equal segments, repeats if <8)."""

    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in full_rows:
        by_video[row["source_video_id"]].append(row)
    covered = {row["source_video_id"] for row in train_rows if row["source_video_id"] in intervals}
    kept = [row for row in train_rows if row["source_video_id"] not in covered]
    short, per_class = 0, defaultdict(int)
    for video in sorted(covered):
        rows = by_video[video]
        pool = select_event_positions(np.asarray([int(r["source_frame_index"]) for r in rows]), intervals[video], inside=True)
        if len(pool) == 0:
            raise RuntimeError(f"{video}: no cached frame inside the official interval")
        short += len(pool) < FRAMES
        per_class[rows[0]["label"]] += 1
        kept.extend(rows[i] for i in sample_evenly(pool, FRAMES, 0))
    return kept, {"covered_train_videos": len(covered), "covered_train_videos_with_fewer_than_8_event_frames": short,
                  "covered_train_videos_per_class": dict(sorted(per_class.items()))}


def _predict_run(head, val_pooled, val_rows, intervals, arms, seed):
    """`{(video, arm): (probs, target, n_pool)}` for one trained head; uniform arm = the headline validation path."""

    head.eval()
    out = {}
    if "uniform" in arms:
        sequences = group_pooled_sequences_by_video(val_pooled, val_rows, permute_frame_order=True, permute_seed=2000 + seed)
        for (_dataset, video), (sequence, row) in sequences.items():
            out[(video, "uniform")] = (predict_video_probs(head, sequence, FRAMES), int(row["target"]), len(sequence))
    for video, positions in _group_rows(val_rows).items():
        spans = intervals.get(video)
        frame_idx = np.asarray([int(val_rows[p]["source_frame_index"]) for p in positions])
        target = int(val_rows[positions[0]]["target"])
        for arm in arms:
            if arm == "uniform" or (arm != "uniform8" and spans is None):
                continue
            sel = arm_row_positions(frame_idx, spans, arm, FRAMES, 2000 + seed)
            if sel is None:
                continue
            n_pool = int(select_event_positions(frame_idx, spans, arm == "oracle_event").size) if spans else len(positions)
            out[(video, arm)] = (predict_video_probs(head, val_pooled[positions[sel]], FRAMES), target, n_pool)
    return out


def _headline_f1(fold: int, seed: int) -> float:
    path = Path(HEADLINE_DIR) / f"fold{fold}" / f"step_permuted-seed-{seed}.json"
    return json.loads(path.read_text())["probe_result"]["video_level_metrics"]["macro_f1"]


def _aggregate(records, class_names, arm_pairs):
    """records: list of dicts (fold, seed, video, arm, probs, target). Covered videos only."""

    arms = sorted({r["arm"] for r in records})
    summary = {"arms": {}, "per_class": {}, "paired": {}}
    avg = defaultdict(list)
    for r in records:
        avg[(r["fold"], r["video"], r["arm"])].append(r)
    for arm in arms:
        recs = [r for r in records if r["arm"] == arm]
        truth = [r["target"] for r in recs]
        pred = [int(np.argmax(r["probs"])) for r in recs]
        sa = [(v[0]["target"], int(np.argmax(np.mean([x["probs"] for x in v], axis=0)))) for k, v in avg.items() if k[2] == arm]
        summary["arms"][arm] = {
            "n_runs_x_videos": len(recs), "n_videos": len(sa),
            "run_pooled_accuracy": float(np.mean(np.asarray(truth) == np.asarray(pred))),
            "run_pooled_macro_recall": macro_recall(truth, pred),
            "seed_mean_prob_accuracy": float(np.mean([t == p for t, p in sa])),
            "seed_mean_prob_macro_recall": macro_recall([t for t, _ in sa], [p for _, p in sa]),
        }
        per_class = defaultdict(list)
        for t, p in sa:
            per_class[class_names[t]].append(t == p)
        summary["per_class"][arm] = {c: {"n_videos": len(v), "seed_mean_prob_accuracy": float(np.mean(v))} for c, v in sorted(per_class.items())}
    for a, b in arm_pairs:
        keys = sorted(k[:2] for k in avg if k[2] == a and (k[0], k[1], b) in avg)
        ok = lambda k, arm: int(np.argmax(np.mean([x["probs"] for x in avg[(k[0], k[1], arm)]], axis=0))) == avg[(k[0], k[1], arm)][0]["target"]
        only_a = sum(ok(k, a) and not ok(k, b) for k in keys)
        only_b = sum(ok(k, b) and not ok(k, a) for k in keys)
        summary["paired"][f"{a}_vs_{b}"] = {
            "n_videos": len(keys), f"only_{a}_correct": only_a, f"only_{b}_correct": only_b,
            "both_correct": sum(ok(k, a) and ok(k, b) for k in keys), "both_wrong": sum(not ok(k, a) and not ok(k, b) for k in keys),
            "mcnemar_exact_p_two_sided": mcnemar_exact(only_a, only_b), "unit": "seed-mean-probability prediction per (fold, video)",
        }
    return summary


def _git_rev() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()


def _cross_mcnemar(records_a, records_b, arm):
    def correct(records):
        agg = defaultdict(list)
        for r in records:
            if r["arm"] == arm:
                agg[(r["fold"], r["video"])].append(r)
        return {k: int(np.argmax(np.mean([x["probs"] for x in v], axis=0))) == v[0]["target"] for k, v in agg.items()}

    ca, cb = correct(records_a), correct(records_b)
    keys = sorted(set(ca) & set(cb))
    only_a, only_b = sum(ca[k] and not cb[k] for k in keys), sum(cb[k] and not ca[k] for k in keys)
    return {"eval_arm": arm, "n_videos": len(keys), "only_first_correct": only_a, "only_second_correct": only_b,
            "mcnemar_exact_p_two_sided": mcnemar_exact(only_a, only_b)}


def _paired_gate(overall, train_arms, gate=0.04) -> dict:
    """Per-fold (mean over seeds) all-video macro-F1 delta, second arm minus first; the project's +0.04-on-every-fold gate."""

    folds = sorted({r["fold"] for r in overall})
    mean = {a: {f: float(np.mean([r["all_videos_macro_f1"] for r in overall if r["train_arm"] == a and r["fold"] == f])) for f in folds}
            for a in train_arms}
    delta = {f: mean[train_arms[1]][f] - mean[train_arms[0]][f] for f in folds}
    return {"per_fold_mean": mean, "per_fold_delta": delta, "mean_delta": float(np.mean(list(delta.values()))),
            "folds_meeting_gate": int(sum(d >= gate for d in delta.values())), "passes_gate": all(d >= gate for d in delta.values())}


def _main_train_oracle(args, load_args, intervals, train_intervals=None, out_name="train-oracle", extra_config=None) -> int:
    """Train on fold-cache features (all ~447 frames/video available); control = headline sampling, oracle = in-event for covered train videos."""

    cls_mmap, features_mmap, key_to_index = _load_patch_cache(args.patch_cache_root)
    train_intervals = intervals if train_intervals is None else train_intervals
    out_dir = args.output / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_arms = ("train_uniform_foldcache", "train_oracle")
    records = {a: [] for a in train_arms}
    overall, details = [], {}
    for fold in args.folds:
        fold_data = _load_fold(fold, load_args)
        class_names, val_rows = fold_data["class_names"], fold_data["validation_rows"]
        full_rows = _foldcache_rows(fold)
        by_key = {(r["source_video_id"], int(r["source_frame_index"])): r for r in full_rows}
        control_rows = [by_key[(r["source_video_id"], int(r["source_frame_index"]))] for r in fold_data["train_rows"]]
        oracle_rows, details[fold] = _train_event_rows(full_rows, control_rows, train_intervals)
        val_pooled = load_pooled_features(cls_mmap, features_mmap, _patch_indices_for(val_rows, key_to_index))
        for train_arm, rows in zip(train_arms, (control_rows, oracle_rows)):
            pooled = _foldcache_pooled(fold, rows)
            for seed in args.seeds:
                torch.manual_seed(seed)
                head = _build_head("step_permuted", len(class_names), FRAMES)
                train_seq = group_pooled_sequences_by_video(pooled, rows, permute_frame_order=True, permute_seed=1000 + seed)
                val_seq = group_pooled_sequences_by_video(val_pooled, val_rows, permute_frame_order=True, permute_seed=2000 + seed)
                result = train_sequence_head(head, train_seq, val_seq, class_names, frames=FRAMES, seed=seed)
                preds = _predict_run(head, val_pooled, val_rows, intervals, ARMS, seed)
                uni = [(t, int(np.argmax(p))) for (v, a), (p, t, _) in preds.items() if a == "uniform"]
                f1 = classification_metrics([class_names[t] for t, _ in uni], [class_names[p] for _, p in uni], class_names)["macro_f1"]
                row = {"train_arm": train_arm, "fold": fold, "seed": seed, "all_videos_macro_f1": f1,
                       "train_run_macro_f1": result.probe_result.video_level_metrics["macro_f1"]}
                overall.append(row)
                print(json.dumps(row), flush=True)
                for (video, arm), (probs, target, n_pool) in preds.items():
                    if video in intervals:
                        records[train_arm].append({"fold": fold, "seed": seed, "video": video, "arm": arm, "probs": probs, "target": target, "n_pool": n_pool})

    summary = {"mode": "train_oracle", "metric_names": "interval-covered subset accuracy / macro-recall (inherited video label); all-video macro-F1 is the 14-class inherited-label score",
               "per_train_arm": {a: _aggregate(records[a], class_names, PAIRS) for a in train_arms}, "all_video_macro_f1_runs": overall,
               "all_video_macro_f1_mean_sd": {a: [float(np.mean([r["all_videos_macro_f1"] for r in overall if r["train_arm"] == a])),
                                                  float(np.std([r["all_videos_macro_f1"] for r in overall if r["train_arm"] == a], ddof=1))] for a in train_arms},
               "train_uniform_vs_train_oracle": {arm: _cross_mcnemar(records[train_arms[0]], records[train_arms[1]], arm) for arm in ARMS},
               "covered_train_details_per_fold": details, "paired_gate": _paired_gate(overall, train_arms)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (out_dir / "per_video.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["train_arm", "fold", "seed", "video", "label", "eval_arm", "pred", "correct", "p_true", "n_candidate_frames"])
        for a in train_arms:
            for r in records[a]:
                pred = int(np.argmax(r["probs"]))
                writer.writerow([a, r["fold"], r["seed"], r["video"], class_names[r["target"]], r["arm"], class_names[pred], int(pred == r["target"]), f"{r['probs'][r['target']]:.6f}", r["n_pool"]])
    manifest = json.loads((args.patch_cache_root / "manifest.json").read_text())
    config = {
        "mode": "train_oracle", "folds": args.folds, "seeds": args.seeds, "git_revision": _git_rev(), "frames_per_video": FRAMES,
        "validation_features": "patch cache (same as mode 1)", "patch_cache_features_sha256": manifest["features_sha256"],
        "train_features": "fold cache results/p1c/p1c-dinov2-feature-cache-ar-fold{N}/train/features.float16.npy (row_index memmap gather)",
        "fold_cache_manifest_sha256": {str(f): _sha256(Path(f"results/p1c/p1c-dinov2-feature-cache-ar-fold{f}/train/manifest.json")) for f in args.folds},
        "annotation_sha256": _sha256(args.annotation), "head_init": "torch.manual_seed(seed) before head construction",
        "train_uniform_foldcache": "sample_rows_per_video(train_rows, 8, seed=0) as headline",
        "train_oracle": "covered train videos: 8 midpoints of equal segments of in-event cached frames (repeats if <8); others as headline",
        "sampler": "see mode-1 config", "seed_averaging": "mean softmax probability over seeds per (fold, video)",
        **(extra_config or {}),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_video_macro_f1_mean_sd": summary["all_video_macro_f1_mean_sd"], "paired_gate": summary["paired_gate"]}, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-oracle", action="store_true", help="draw covered train videos' 8 frames from inside the event")
    parser.add_argument("--patch-cache-root", type=Path, default=Path("results/p1c/p1c-patch-cache"))
    parser.add_argument("--output", type=Path, default=Path("results/p1c/p1c-oracle-interval-probe"))
    parser.add_argument("--annotation", type=Path, default=Path(ANNOTATION))
    parser.add_argument("--glance", type=Path, default=None, help="GlanceVAD click CSV: add +/- window spans to the TRAINING intervals (with --train-oracle)")
    parser.add_argument("--glance-window-s", type=float, default=2.0, help="half-width of each click's span, seconds (fixed before the run)")
    args = parser.parse_args(argv)
    load_args = SimpleNamespace(
        frames=FRAMES, sampling_seed=0, extract_closure_path=Path("results/p1c/p1c-video-frames/extract_closure.json"),
    )
    intervals = load_intervals(args.annotation)
    if args.glance is not None:
        if not args.train_oracle:
            parser.error("--glance requires --train-oracle")
        with Path(VIDEOS_CSV).open(newline="", encoding="utf-8") as handle:
            fps = {r["file_name"].rsplit(".", 1)[0]: float(r["reported_fps"]) for r in csv.DictReader(handle)}
        glances = load_glances(args.glance)
        spans = glance_spans(glances, fps, args.glance_window_s)
        both = set(spans) & set(intervals)
        train_intervals = {**spans, **intervals}
        extra = {"glance_csv": str(args.glance), "glance_csv_sha256": _sha256(args.glance), "glance_window_s": args.glance_window_s,
                 "glance_videos": len(glances), "glance_clicks": sum(map(len, glances.values())),
                 "videos_with_both_glance_and_official_interval": len(both), "official_interval_precedence": True,
                 "train_oracle": "covered train videos (official interval OR glance window, official wins): 8 midpoints of equal segments of in-span cached frames; others as headline",
                 "evaluation_intervals": "official test-split intervals only (validation diagnostics unchanged)"}
        return _main_train_oracle(args, load_args, intervals, train_intervals, "train-glance", extra)
    if args.train_oracle:
        return _main_train_oracle(args, load_args, intervals)
    cls_mmap, features_mmap, key_to_index = _load_patch_cache(args.patch_cache_root)
    mode, arms, suffix = "frozen_head", ARMS, ""
    args.output.mkdir(parents=True, exist_ok=True)

    records, sanity, extra = [], [], {}
    for fold in args.folds:
        fold_data = _load_fold(fold, load_args)
        class_names = fold_data["class_names"]
        val_rows = fold_data["validation_rows"]
        train_rows = fold_data["train_rows"]
        train_pooled = load_pooled_features(cls_mmap, features_mmap, _patch_indices_for(train_rows, key_to_index))
        val_pooled = load_pooled_features(cls_mmap, features_mmap, _patch_indices_for(val_rows, key_to_index))
        for seed in args.seeds:
            torch.manual_seed(seed)  # the headline script leaves head init unseeded (not reproducible)
            head = _build_head("step_permuted", len(class_names), FRAMES)
            train_seq = group_pooled_sequences_by_video(train_pooled, train_rows, permute_frame_order=True, permute_seed=1000 + seed)
            val_seq = group_pooled_sequences_by_video(val_pooled, val_rows, permute_frame_order=True, permute_seed=2000 + seed)
            result = train_sequence_head(head, train_seq, val_seq, class_names, frames=FRAMES, seed=seed)
            preds = _predict_run(head, val_pooled, val_rows, intervals, arms, seed)
            uni = [(t, int(np.argmax(p))) for (v, a), (p, t, _) in preds.items() if a == "uniform"]
            f1 = classification_metrics([class_names[t] for t, _ in uni], [class_names[p] for _, p in uni], class_names)["macro_f1"]
            check = {"fold": fold, "seed": seed, "uniform_all_videos_macro_f1": f1, "train_run_macro_f1": result.probe_result.video_level_metrics["macro_f1"]}
            check["saved_headline_macro_f1"] = _headline_f1(fold, seed)
            check["abs_diff_vs_saved"] = abs(f1 - check["saved_headline_macro_f1"])
            check["uniform_matches_own_training_run_1e-6"] = abs(f1 - check["train_run_macro_f1"]) < 1e-6
            sanity.append(check)
            print(json.dumps(check), flush=True)
            for (video, arm), (probs, target, n_pool) in preds.items():
                if video in intervals:
                    records.append({"fold": fold, "seed": seed, "video": video, "arm": arm, "probs": probs, "target": target, "n_pool": n_pool})

    summary = _aggregate(records, class_names, PAIRS)
    summary["mode"] = mode
    summary["metric_names"] = "interval-covered subset accuracy / macro-recall (inherited video label); NOT comparable to the 14-class macro-F1 headline"
    summary["sanity_uniform_vs_saved_headline"] = sanity
    summary["train_oracle_details"] = extra
    short = {a: sum(1 for r in records if r["arm"] == a and r["n_pool"] < FRAMES) for a in ("oracle_event", "anti_event")}
    summary["runs_x_videos_with_fewer_than_8_candidates_sampled_with_repeats"] = short
    (args.output / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.output / f"per_video{suffix}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "fold", "seed", "video", "label", "arm", "pred", "correct", "p_true", "n_candidate_frames"])
        for r in records:
            pred = int(np.argmax(r["probs"]))
            writer.writerow([mode, r["fold"], r["seed"], r["video"], class_names[r["target"]], r["arm"], class_names[pred], int(pred == r["target"]), f"{r['probs'][r['target']]:.6f}", r["n_pool"]])
    manifest = json.loads((args.patch_cache_root / "manifest.json").read_text())
    config = {
        "mode": mode, "folds": args.folds, "seeds": args.seeds, "git_revision": _git_rev(), "frames_per_video": FRAMES,
        "patch_cache_cls_sha256": manifest["cls_sha256"], "patch_cache_features_sha256": manifest["features_sha256"],
        "annotation_sha256": _sha256(args.annotation), "head": "step_permuted (train permute seed 1000+seed, val permute seed 2000+seed, sampling_seed 0)",
        "sampler": "event/anti/uniform8: k=8 midpoints of k equal segments over the candidate frames sorted by source_frame_index "
                   "(repeats if fewer than 8), then permuted with default_rng(2000+seed); uniform: headline path (all cached frames, permuted, 8-frame chunks, mean logits)",
        "event_definition": "cached frame with start <= source_frame_index <= end for any official span",
        "head_init": "torch.manual_seed(seed) before head construction (deviation: headline script leaves init unseeded)",
        "seed_averaging": "mean softmax probability over seeds per (fold, video)",
    }
    (args.output / f"config{suffix}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("arms", "paired")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
