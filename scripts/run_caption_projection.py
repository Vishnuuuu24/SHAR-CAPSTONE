#!/usr/bin/env python3
"""One shot at caption supervision for the 14-class classifier (FINAL_PUSH_PLAN Step 6).

Step 6a (gate, run first): on each fold, fit the span-feature -> caption-vector map on four fifths of the
training pool plus the captioned videos outside the 700-video protocol, and check on the held-out fifth
whether a span retrieves its own sentence. Declared rule: the gate passes if the mean over folds of the
median percentile rank is <= 0.35 (0.5 is chance). If it fails, the classifier experiment is not run.

Step 6b (only if the gate passes): inner cross-validation of
  cap_proj   [mean feature ++ mean and max over time of the projected frames]   <- the registered candidate
against the control D_mean (the same mean feature alone). Context rows: cap_only, cap_proj_noextra.
Pass rule: paired 95% interval excludes zero AND gain >= +0.02. No second attempt.

Features are the detector cache (one frame per 15 source frames, every UCF video). Test videos of a fold
are never scored, and no caption of a held-out video is ever used. CPU only.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

POOL_CACHE = "results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}"
AR_SPLITS = "Datasets/UCF Crime Official/raw/UCF_Crimes/Action_Regnition_splits"
UCA = "Datasets/UCA"
GATE_THRESHOLD = 0.35
CANDIDATE = [("cap_proj", "D_mean")]
CONTEXT = [("cap_only", "D_mean"), ("cap_proj_noextra", "D_mean")]


def _protocol_stems() -> set[str]:
    stems = set()
    for path in (REPO_ROOT / AR_SPLITS).glob("t*_00*.txt"):
        stems |= {line.strip().rsplit("/", 1)[-1].rsplit(".", 1)[0] for line in path.read_text().splitlines() if line.strip()}
    return stems


def _frames_of(features: Path, backbones: tuple[str, ...]):
    from core import detect_data, detect_model

    videos = detect_data.load_split(REPO_ROOT)
    by_stem = {v.stem: v for v in videos}
    store = detect_model.load_store(features, backbones, videos, standardise=False)

    def frames_of(stem: str) -> tuple[np.ndarray, np.ndarray]:
        video = by_stem[stem]
        return store.frames(video.relative_path), store.source_frames_of(video.relative_path) / video.fps

    return frames_of, by_stem


def _worker(job: tuple) -> dict:
    fold, features, backbones, extras_path, parts, repeats, split_seed, threads, gate_only = job
    import torch

    from core import caption_projection as cp
    from core import inner_cv

    torch.set_num_threads(threads)
    pool = inner_cv.load_pool(REPO_ROOT / POOL_CACHE.format(fold=fold), fold)
    frames_of, _by_stem = _frames_of(Path(features), tuple(backbones))
    captions = cp.load_captions(REPO_ROOT / UCA)
    extras: cp.CaptionData = pickle.loads(Path(extras_path).read_bytes())
    extra_stems = sorted(extras.spans)
    data = cp.merge(cp.build_caption_data(frames_of, pool.video_ids, captions), extras)
    out = {"fold": fold, "videos": len(pool.video_ids), "cache_sha256": pool.cache_sha256, "arms": {}, "tables": {},
           "captioned_pool_videos": sum(bool(data.sentences[v]) for v in pool.video_ids), "extra_videos": len(extra_stems)}
    if gate_only:
        gate = []
        for repeat, part, train_index, heldout_index in inner_cv.inner_splits(pool.targets, parts=parts, repeats=1, seed=split_seed):
            spans, sentences = cp.stack(data, [pool.video_ids[i] for i in train_index] + extra_stems)
            text_model = cp.fit_text_model(sentences, seed=part)
            projection = cp.fit_projection(spans, text_model.transform(sentences))
            gate.append({"fold": fold, "part": part, "alpha": float(projection[-1].alpha_), "training_sentences": len(sentences),
                         **cp.retrieval_gate(projection, text_model, data, [pool.video_ids[i] for i in heldout_index])})
            print(json.dumps(gate[-1]), flush=True)
        out["gate"] = gate
        return out
    splits = inner_cv.inner_splits(pool.targets, parts=parts, repeats=repeats, seed=split_seed)
    mean_matrix = np.stack([data.frames[v].mean(axis=0) for v in pool.video_ids])
    gate_log: list = []
    arms = {
        "D_mean": inner_cv.meanpool_logistic_arm(matrix=lambda _pool: mean_matrix),
        "cap_proj": cp.caption_arm(data, extra_stems, gate_log=gate_log),
        "cap_only": cp.caption_arm(data, extra_stems, include_mean_feature=False),
        "cap_proj_noextra": cp.caption_arm(data, []),
    }
    for name, arm in arms.items():
        started = time.perf_counter()
        tables = inner_cv.run_arm(pool, arm, splits)
        out["arms"][name] = {**inner_cv.cell_scores(pool, tables), "seconds": round(time.perf_counter() - started, 1)}
        out["tables"][name] = tables
        print(json.dumps({"fold": fold, "arm": name, "macro_f1": [round(x, 4) for x in out["arms"][name]["macro_f1"]],
                          "seconds": out["arms"][name]["seconds"]}), flush=True)
    out["gate"] = gate_log
    out["pool"] = pool.light()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=Path("results/detect/features"))
    ap.add_argument("--backbones", nargs="+", default=["dinov2b", "siglip"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--parts", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--code-revision", default="unrecorded")
    args = ap.parse_args(argv)

    from core import caption_projection as cp
    from core import inner_cv_report as report

    args.output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    captions = cp.load_captions(REPO_ROOT / UCA)
    frames_of, by_stem = _frames_of(args.features, tuple(args.backbones))
    protocol = _protocol_stems()
    extra_stems = sorted(s for s in captions if s in by_stem and s not in protocol and captions[s])
    extras_path = args.output / "extra_caption_spans.pkl"
    if not extras_path.exists():
        extras_path.write_bytes(pickle.dumps(cp.build_caption_data(frames_of, extra_stems, captions, keep_frames=False)))
    print(json.dumps({"protocol_videos": len(protocol), "extra_captioned_videos": len(extra_stems),
                      "extra_sentences": sum(len(captions[s]) for s in extra_stems)}), flush=True)

    def run(gate_only: bool) -> list[dict]:
        jobs = [(f, str(args.features), list(args.backbones), str(extras_path), args.parts, args.repeats, args.split_seed, args.threads, gate_only)
                for f in args.folds]
        with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            return sorted(pool.map(_worker, jobs), key=lambda r: r["fold"])

    provenance = {
        "plan": "docs/FINAL_PUSH_PLAN_2026-11-11.md Step 6", "code_revision": args.code_revision, "host": socket.gethostname(),
        "started_utc": started.isoformat(), "label_view": "inherited video-level labels (D-40), not event-only", "test_videos_scored": False,
        "features": {"dir": str(args.features), "backbones": args.backbones, "manifest": json.loads((args.features / "manifest.json").read_text())},
        "captions": "UCA (UCF-Crime Annotation), time-stamped sentences; training videos only",
        "text_model": f"TF-IDF 1-2 grams, min_df 2 -> TruncatedSVD({cp.TEXT_DIM}) -> L2 normalise, fitted on training sentences of each split",
        "projection": "StandardScaler + RidgeCV(alphas 1e2..1e5), fitted on training spans of each split",
        "config": {"folds": args.folds, "parts": args.parts, "repeats": args.repeats, "split_seed": args.split_seed, "bootstrap_draws": args.draws,
                   "gate_rule": f"mean over folds of the median percentile rank <= {GATE_THRESHOLD} (chance 0.5)",
                   "pass_rule": f"paired 95% interval excludes 0 and gain >= {report.MIN_GAIN}"},
        "extra_captioned_videos": len(extra_stems),
    }
    gate_results = run(gate_only=True)
    gate_rows = [g for r in gate_results for g in r["gate"]]
    per_fold = {str(r["fold"]): float(np.mean([g["median_percentile_rank"] for g in r["gate"]])) for r in gate_results}
    gate = {"median_percentile_rank_per_fold": per_fold, "mean": float(np.mean(list(per_fold.values()))),
            "recall_at_10": float(np.mean([g["recall_at_10"] for g in gate_rows])),
            "chance_recall_at_10": float(np.mean([g["chance_recall_at_10"] for g in gate_rows])), "threshold": GATE_THRESHOLD}
    gate["passed"] = bool(gate["mean"] <= GATE_THRESHOLD)
    print(json.dumps({"gate": gate}), flush=True)
    (args.output / "gate.json").write_text(json.dumps({"provenance": provenance, "gate": gate, "rows": gate_rows}, indent=1) + "\n")
    if not gate["passed"]:
        print(json.dumps({"step_6b": "not run: the gate failed"}), flush=True)
        return 0

    results = run(gate_only=False)
    arm_names = list(results[0]["arms"])
    probs_path = report.write_probabilities(args.output, results, arm_names)
    arms = report.arm_summaries(probs_path, results, arm_names, args.draws)
    contrasts = report.paired_contrasts(probs_path, CANDIDATE + CONTEXT, arm_names, args.draws)
    provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["cache_sha256"] = {str(r["fold"]): r["cache_sha256"] for r in results}
    report.write_outputs(args.output, provenance, results, arms, contrasts,
                         extra={"gate": gate, "split_gate_rows": [g for r in results for g in r["gate"]],
                                "registered_candidate": list(CANDIDATE[0])})
    return 0


if __name__ == "__main__":
    sys.exit(main())
