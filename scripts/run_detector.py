#!/usr/bin/env python3
"""Live detector on the official UCF-Crime Anomaly-Detection split (FINAL_PUSH_PLAN Step 4b-4c, D-71).

A separate task and a separate table from the 14-class classifier (rule 3). Trained with video-level
labels only; scored against the official frame intervals of the 290 test videos (rule 1: event-level).

Declared before any run:
  rows   D0 video-level score given to every frame (floor: no localisation)
         D1 Sultani et al. 2018 ranking loss on 32 segment means
         D2 top-k multiple-instance loss, per-frame MLP
         D3 D2 + causal context (deviation from the running mean of past frames) + causal 5-frame score mean  <- the live row
         D4 D2 + offline context (deviation from the whole-video mean) + centred 9-frame score mean          <- what "live" costs
  seeds  0, 1, 2 for every trained row; mean and SD reported
  backbone  chosen among dinov2b / siglip / dinov2b+siglip by the held-out TRAINING-video AUC of row D2, seed 0;
            the other two are reported for row D3 only, as an ablation
  held-out  10% of training videos (seed 0): epoch choice and the alarm threshold
  alarm threshold  99th percentile of the row's scores on held-out NORMAL training frames (1% frame false-positive rate)
The 290 test videos are read once, for these rows.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FEATURE_SETS = {"dinov2b": ("dinov2b",), "siglip": ("siglip",), "dinov2b+siglip": ("dinov2b", "siglip")}
ROWS = {  # name -> (loss, context, smoothing, window)
    "D1_ranking": ("ranking", "none", "none", 1),
    "D2_topk": ("topk", "none", "none", 1),
    "D3_topk_causal": ("topk", "causal", "causal", 5),
    "D4_topk_offline": ("topk", "offline", "centred", 9),
}
THRESHOLD_PERCENTILE = 99.0


def _smooth(scores: np.ndarray, kind: str, window: int) -> np.ndarray:
    from core.detect_metrics import causal_mean, centred_mean

    return {"none": lambda s, w: np.asarray(s, dtype=np.float32), "causal": causal_mean, "centred": centred_mean}[kind](scores, window)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=Path("results/detect/features"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--smoke", action="store_true", help="2 epochs, seed 0, dinov2b only; every row is exercised on held-out TRAINING videos; no test read, nothing written")
    ap.add_argument("--code-revision", default="unrecorded")
    args = ap.parse_args(argv)

    import torch

    from core import detect_data, detect_model
    from core.detect_metrics import detection_report

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    started = datetime.now(timezone.utc)
    args.output.mkdir(parents=True, exist_ok=True)
    videos = detect_data.load_split(REPO_ROOT)
    fit, held = detect_model.split_heldout(videos, fraction=0.1, seed=0)
    held_normal = [v for v in held if not v.anomaly]
    test = [v for v in videos if v.split == "test"]
    intervals = detect_data.load_test_intervals(REPO_ROOT)
    missing = [v.stem for v in test if v.stem not in intervals]
    if missing:
        raise SystemExit(f"{len(missing)} test videos have no official annotation line, e.g. {missing[:3]}")
    epochs = 2 if args.smoke else args.epochs
    seeds = [0] if args.smoke else args.seeds
    say = lambda record: print(json.dumps(record), flush=True)  # noqa: E731

    def store_for(name: str):
        store = detect_model.load_store(args.features, FEATURE_SETS[name], videos)
        store.arrays = [np.ascontiguousarray(a) for a in store.arrays]   # into RAM: 2.8 GiB per backbone
        return store

    # ---- backbone choice on held-out training videos (row D2, seed 0); the test split is not involved
    choice = {}
    for name in (["dinov2b"] if args.smoke else list(FEATURE_SETS)):
        store = store_for(name)
        _model, history = detect_model.train_detector(store, fit, held, loss="topk", context="none", seed=0, device=device, epochs=epochs, log=say)
        choice[name] = max(h["heldout_video_auc"] for h in history)
        del store
    chosen = max(choice, key=choice.get)
    say({"backbone_choice_heldout_video_auc": choice, "chosen": chosen})
    if args.smoke:
        # Code-path check only: held-out TRAINING videos stand in for test, with a made-up span (the middle third).
        score_videos = held
        spans_of = lambda v: [(v.decoded_frames // 3, 2 * v.decoded_frames // 3)] if v.anomaly else []  # noqa: E731
    else:
        score_videos = test
        spans_of = lambda v: intervals[v.stem]  # noqa: E731

    def evaluate(store, loss: str, context: str, smoothing: str, window: int, seed: int) -> tuple[dict, dict]:
        model, history = detect_model.train_detector(store, fit, held, loss=loss, context=context, seed=seed, device=device, epochs=epochs, log=say)
        frame_scores = lambda v: _smooth(detect_model.score_frames(model, store.frames(v.relative_path, context), device), smoothing, window)  # noqa: E731
        threshold = float(np.percentile(np.concatenate([frame_scores(v) for v in held_normal]), THRESHOLD_PERCENTILE))
        records = [{"scores": frame_scores(v), "source_frames": store.source_frames_of(v.relative_path), "total_frames": v.decoded_frames,
                    "fps": v.fps, "spans": spans_of(v), "anomaly": v.anomaly, "category": v.category, "stem": v.stem} for v in score_videos]
        report = detection_report(records, threshold)
        report["heldout_video_auc"] = max(h["heldout_video_auc"] for h in history)
        return report, {r["stem"]: r["scores"] for r in records}

    def aggregate(reports: list[dict]) -> dict:
        keys = ("frame_auc", "frame_ap", "anomaly_video_frame_auc", "false_alarms_per_hour", "normal_videos_with_an_alarm",
                "event_recall", "median_delay_s", "heldout_video_auc")
        out = {}
        for k in keys:
            values = [r[k] for r in reports if r[k] is not None and not np.isnan(r[k])]   # delay is None when no event was detected
            out[k] = {"mean": float(np.mean(values)), "sd": float(np.std(values))} if values else {"mean": None, "sd": None}
        out["event_recall_by_category"] = {c: float(np.mean([r["event_recall_by_category"][c] for r in reports]))
                                           for c in reports[0]["event_recall_by_category"]}
        out["per_seed"] = reports
        return out

    table, score_dump = {}, {}
    store = store_for(chosen)
    # D0: one probability per video (logistic regression on the mean feature), no localisation
    video_scores = detect_model.video_level_scores(store, fit, list({v.relative_path: v for v in held_normal + score_videos}.values()))
    flat = lambda v: np.full(store.spans[v.relative_path][1] - store.spans[v.relative_path][0], video_scores[v.relative_path], dtype=np.float32)  # noqa: E731
    threshold = float(np.percentile(np.concatenate([flat(v) for v in held_normal]), THRESHOLD_PERCENTILE))
    d0 = detection_report([{"scores": flat(v), "source_frames": store.source_frames_of(v.relative_path), "total_frames": v.decoded_frames,
                            "fps": v.fps, "spans": spans_of(v), "anomaly": v.anomaly, "category": v.category} for v in score_videos], threshold)
    d0["heldout_video_auc"] = float("nan")
    table["D0_video_level"] = aggregate([d0])
    say({"row": "D0_video_level", "frame_auc": d0["frame_auc"]})
    for row, (loss, context, smoothing, window) in ROWS.items():
        reports = []
        for seed in seeds:
            report, scores = evaluate(store, loss, context, smoothing, window, seed)
            reports.append({"seed": seed, **report})
            if seed == seeds[0]:
                score_dump[row] = scores
        table[row] = aggregate(reports)
        say({"row": row, "frame_auc": table[row]["frame_auc"], "false_alarms_per_hour": table[row]["false_alarms_per_hour"],
             "event_recall": table[row]["event_recall"], "median_delay_s": table[row]["median_delay_s"]})
    del store
    if args.smoke:
        say({"smoke": "ok", "test_read": False, "rows_exercised": list(table)})
        return 0

    ablation = {chosen: table["D3_topk_causal"]}
    for name in FEATURE_SETS:
        if name != chosen:
            store = store_for(name)
            ablation[name] = aggregate([{"seed": s, **evaluate(store, *ROWS["D3_topk_causal"], s)[0]} for s in args.seeds])
            say({"ablation_D3": name, "frame_auc": ablation[name]["frame_auc"]})
            del store

    for row, scores in score_dump.items():
        np.savez_compressed(args.output / f"test_scores_{row}_seed{args.seeds[0]}.npz", **scores)
    (args.output / "summary.json").write_text(json.dumps({
        "provenance": {
            "plan": "docs/FINAL_PUSH_PLAN_2026-11-11.md Step 4", "decision": "D-71", "code_revision": args.code_revision,
            "host": socket.gethostname(), "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
            "task": "UCF-Crime official Anomaly-Detection split; video-level training labels; official test intervals",
            "feature_manifest": json.loads((args.features / "manifest.json").read_text()),
            "videos": {"fit": len(fit), "heldout": len(held), "heldout_normal": len(held_normal), "test": len(test)},
            "seeds": args.seeds, "heldout_seed": 0, "epochs_ceiling": epochs, "threshold_percentile": THRESHOLD_PERCENTILE,
            "rows": {"D0_video_level": "logistic regression on the video mean feature", **{k: list(v) for k, v in ROWS.items()}},
            "test_read": "once, for the declared rows and the D3 backbone ablation",
        },
        "backbone_choice_heldout_video_auc": choice, "chosen_backbone": chosen, "table": table, "ablation_D3_by_backbone": ablation,
    }, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
