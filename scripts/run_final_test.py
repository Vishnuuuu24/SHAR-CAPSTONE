#!/usr/bin/env python3
"""The one declared final read of the Action-Recognition test folds (D-73; FINAL_PUSH_PLAN Step 7).

Final 14-class pipeline = the stack that passed the inner-cross-validation rule (D-74, stack E3): the
equal-weight average of three logistic-regression arms on frozen per-video features,
  L_study     DINOv2 ViT-L/14-reg4, mean over the study frames
  B+S_study   DINOv2 ViT-B/14-reg4 ++ SigLIP ViT-B/16, mean over the study frames
  B+VJ        DINOv2 ViT-B/14-reg4 (mean over all cached frames) ++ V-JEPA 2 ViT-L clip features
each trained on ALL 532 training videos of the fold (train + validation; C chosen by nested cross-validation
inside those videos, so no validation set is needed). `--members` can name other registered arms, including
`attention` (the D-53 head, fixed 8 epochs, final-epoch weights).

Scored once on the inherited-label `noisy_proxy_test` view (D-40) of each official fold: macro-F1 and
accuracy (same aggregation as D-57: seeds averaged inside a fold, then folds averaged), top-3 accuracy and
the D-72 family scores. Refuses to run if its output already exists. The first read (D-57: 0.4138 / 41.8%)
is reported beside this one, never replaced by it.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

POOL_CACHE = "results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}"
SIGLIP_POOLED = "results/p1c/inner-cv-candidates/siglip_pooled.float16.npy"
FINAL_MEMBERS = ("L_study", "B+S_study", "B+VJ")     # stack E3
LOGISTIC_MEMBERS = {"B_all": ("B_all",), "L_study": ("L_study",), "B+S_study": ("B_study", "S_study"), "B+VJ": ("B_all", "VJ")}
ATTENTION_EPOCHS = 8
MONITOR_VIDEOS = 14   # training videos scored each epoch only because the trainer needs a monitor; nothing is selected
TEST_VIEW = "noisy_proxy_test (inherited video-level labels; not event-only, D-40)"


def score_fold(pool, seeds, member_arms: dict, *, attention: bool = False) -> tuple[list[dict], int]:
    """Train on the pool's train + validation videos, predict its test videos; returns `(runs, training videos)`.

    `member_arms` are `core.inner_cv` arms; `attention=True` adds the D-53 head as one more member.
    """

    from core import inner_cv
    from core.p1c_head_variants import CHECKPOINT_LAST_EPOCH
    from core.p1c_oracle_interval import predict_video_probs

    train_videos = pool.videos_of_split("train") + pool.videos_of_split("validation")
    test_videos = sorted(pool.videos_of_split("test"))
    if set(train_videos) & set(test_videos):
        raise RuntimeError(f"fold {pool.fold}: test videos overlap the training pool")
    if not member_arms and not attention:
        raise ValueError("the pipeline has no members")
    position = {v: i for i, v in enumerate(pool.video_ids)}
    train_index = np.array([position[v] for v in train_videos])
    test_index = np.array([position[v] for v in test_videos])
    truth = pool.targets[test_index]
    runs = []
    for seed in seeds:
        parts = {}
        if attention:
            head, _monitor, _result = inner_cv.fit_attention_head(
                pool, train_videos, train_videos[:MONITOR_VIDEOS], seed, epochs=ATTENTION_EPOCHS, checkpoint_rule=CHECKPOINT_LAST_EPOCH,
                bootstrap_iterations=100)
            sequences = inner_cv.permuted_sequences(pool, test_videos, [pool.frames_of(v) for v in test_videos], 2000 + seed)
            parts["attention"] = np.stack([predict_video_probs(head, sequences[(inner_cv.SOURCE_DATASET, v)][0], inner_cv.FRAMES) for v in test_videos])
        for name, arm in member_arms.items():
            parts[name] = arm(pool, train_index, test_index, seed)
        runs.append({"fold": pool.fold, "seed": seed, "videos": test_videos, "truth": truth.tolist(),
                     "probs": np.mean(list(parts.values()), axis=0).tolist(),
                     "member_macro_f1": {n: _macro(truth, p, len(pool.class_names)) for n, p in parts.items()}})
        print(json.dumps({"fold": pool.fold, "seed": seed, "test_videos": len(test_videos), "member_macro_f1": runs[-1]["member_macro_f1"]}), flush=True)
    return runs, len(train_videos)


def _worker(job: tuple) -> dict:
    fold, seeds, threads, members = job
    import torch

    from core import inner_cv

    torch.set_num_threads(threads)
    pool = inner_cv.load_pool(REPO_ROOT / POOL_CACHE.format(fold=fold), fold, splits=("train", "validation", "test"))
    runs, train_count = score_fold(pool, seeds, _member_arms(members, fold, pool), attention="attention" in members)
    return {"fold": fold, "class_names": pool.class_names, "cache_sha256": pool.cache_sha256, "train_videos": train_count, "runs": runs}


def _macro(truth: np.ndarray, probs: np.ndarray, classes: int) -> float:
    from core.confusion_groups import macro_f1

    return macro_f1(truth, probs.argmax(axis=1), classes)


def _member_arms(members, fold: int, pool) -> dict:
    """The logistic arms of the stack, each on its registered feature blocks. Only D-74 survivors belong in the final pipeline."""

    from core import inner_cv
    from core.inner_cv_features import feature_blocks

    unknown = [m for m in members if m != "attention" and m not in LOGISTIC_MEMBERS]
    if unknown:
        raise ValueError(f"unknown stack members {unknown}; known: {['attention', *sorted(LOGISTIC_MEMBERS)]}")
    names = [m for m in members if m != "attention"]
    wanted = {block for m in names for block in LOGISTIC_MEMBERS[m]}
    if not wanted:
        return {}
    # V-JEPA 2 clip features of this fold's test videos come from the other folds' caches (a feature store only).
    blocks = feature_blocks(fold, REPO_ROOT, pool, REPO_ROOT / SIGLIP_POOLED, wanted, vjepa_fallback_folds=[f for f in (1, 2, 3, 4) if f != fold])

    def arm_for(member: str):
        matrix = np.concatenate([blocks[b] for b in LOGISTIC_MEMBERS[member]], axis=1)
        return inner_cv.meanpool_logistic_arm(matrix=lambda _pool: matrix)

    return {m: arm_for(m) for m in names}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=Path("results/p1c/final-14class-test"))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--members", nargs="+", default=list(FINAL_MEMBERS), help="stack members (default: E3, the D-74 survivor)")
    ap.add_argument("--workers", type=int, default=2, help="4 workers OOM-killed the 16 GiB pod on 2026-10-28")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rehearse", action="store_true", help="build pools and feature blocks only; trains and scores nothing")
    ap.add_argument("--code-revision", default="unrecorded")
    args = ap.parse_args(argv)

    from core import family
    from core.confusion_groups import confusion, macro_f1, per_class_f1

    if args.rehearse:
        # Builds every fold's pool and feature blocks exactly as the real run will, then stops: no model, no score.
        from core import inner_cv
        from core.inner_cv_features import feature_blocks

        wanted = {block for m in args.members if m != "attention" for block in LOGISTIC_MEMBERS[m]}
        for fold in args.folds:
            pool = inner_cv.load_pool(REPO_ROOT / POOL_CACHE.format(fold=fold), fold, splits=("train", "validation", "test"))
            blocks = feature_blocks(fold, REPO_ROOT, pool, REPO_ROOT / SIGLIP_POOLED, wanted, vjepa_fallback_folds=[f for f in (1, 2, 3, 4) if f != fold])
            print(json.dumps({"rehearsal_fold": fold, "videos": len(pool.video_ids), "test_videos": len(pool.videos_of_split("test")),
                              "blocks": {n: list(b.shape) for n, b in blocks.items()},
                              "all_finite": bool(all(np.isfinite(b).all() for b in blocks.values()))}), flush=True)
        return 0

    out = args.output / "test.json"
    if out.exists():
        raise SystemExit(f"{out} exists: the final test read (D-73) has already been spent")
    args.output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    with mp.get_context("spawn").Pool(min(args.workers, len(args.folds))) as pool:
        results = sorted(pool.map(_worker, [(f, args.seeds, args.threads, args.members) for f in args.folds]), key=lambda r: r["fold"])
    classes = results[0]["class_names"]
    n = len(classes)
    per_fold, records, pooled = {}, [], np.zeros((n, n), dtype=np.int64)
    for result in results:
        cells = []
        for run in result["runs"]:
            truth, probs = np.asarray(run["truth"]), np.asarray(run["probs"])
            predicted = probs.argmax(axis=1)
            fam = family.family_scores(truth, predicted, classes)
            cells.append({"macro_f1": macro_f1(truth, predicted, n), "accuracy": float((truth == predicted).mean()),
                          "top3_accuracy": family.top_k_accuracy(truth, probs, 3), "family_accuracy": fam["accuracy"], "family_macro_f1": fam["macro_f1"],
                          "per_class_f1": per_class_f1(truth, predicted, n).tolist(), "per_family_f1": fam["per_family_f1"]})
            pooled += confusion(truth, predicted, n)
            records.append({"fold": result["fold"], "seed": run["seed"], **{k: cells[-1][k] for k in ("macro_f1", "accuracy", "top3_accuracy", "family_accuracy", "family_macro_f1")},
                            "member_macro_f1": run["member_macro_f1"], "test_videos": len(run["videos"])})
        per_fold[str(result["fold"])] = {k: float(np.mean([c[k] for c in cells])) for k in ("macro_f1", "accuracy", "top3_accuracy", "family_accuracy", "family_macro_f1")}
        per_fold[str(result["fold"])]["per_class_f1"] = np.mean([c["per_class_f1"] for c in cells], axis=0).tolist()
    summary = {k: float(np.mean([v[k] for v in per_fold.values()])) for k in ("macro_f1", "accuracy", "top3_accuracy", "family_accuracy", "family_macro_f1")}
    with (args.output / "probs.csv").open("w") as handle:
        handle.write(",".join(["arm", "fold", "seed", "video", "true", *[f"p_{c}" for c in classes]]) + "\n")
        for result in results:
            for run in result["runs"]:
                for video, truth, probs in zip(run["videos"], run["truth"], run["probs"]):
                    handle.write(",".join(["final", str(result["fold"]), str(run["seed"]), video, classes[truth], *[f"{p:.6f}" for p in probs]]) + "\n")
    out.write_text(json.dumps({
        "test_view": TEST_VIEW, "decision": "D-73",
        "pipeline": f"equal-weight probability average of {args.members}; every member trained on all 532 fold training videos; "
                    "logistic members: StandardScaler + LogisticRegression, C by nested CV on the training videos",
        "members": {m: (list(LOGISTIC_MEMBERS[m]) if m in LOGISTIC_MEMBERS else f"D-53 attention head, {ATTENTION_EPOCHS} epochs fixed") for m in args.members},
        "summary": summary, "per_fold": per_fold,
        "per_class_f1": dict(zip(classes, np.mean([v["per_class_f1"] for v in per_fold.values()], axis=0).tolist())),
        "pooled_confusion": pooled.tolist(), "class_names": classes, "families": family.FAMILY_OF, "runs": records,
        "first_read_D57": {"macro_f1": 0.4138, "accuracy": 0.418, "artifact": "results/p1c/headline-14class-test/test.json"},
        "provenance": {"code_revision": args.code_revision, "host": socket.gethostname(), "started_utc": started.isoformat(),
                       "finished_utc": datetime.now(timezone.utc).isoformat(), "folds": args.folds, "seeds": args.seeds,
                       "cache_sha256": {str(r["fold"]): r["cache_sha256"] for r in results},
                       "train_videos_per_fold": results[0]["train_videos"]},
    }, indent=1) + "\n")
    print(json.dumps({"summary": summary, "per_fold": {f: {k: round(v[k], 4) for k in ("macro_f1", "accuracy")} for f, v in per_fold.items()}}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
