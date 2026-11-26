#!/usr/bin/env python3
"""Seven-class subset search (option C, 2026-10-24).

  search  train a 7-class step_permuted head for every candidate subset (novelty >= --min-novelty) on each fold x seed;
          best epoch picked on that subset's validation videos (the baseline convention). Validation only. Resumable.
  select  apply the pre-registered rule (core.subset7.winner) -> overall winner and the best 7/7-novelty subset.
  test    retrain ONLY the selected subsets and score them once on the inherited-label `noisy_proxy_test` view (D-40).
          Refuses to run if test results already exist: the test set is spent once.

Features: the registered DINOv2 fold caches (cls_plus_mean_patch, 1536-D) for train/validation/test, all from one builder.
CPU training; the pod still claims the MIG slice (Kyverno).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.confusion_groups import macro_f1, per_class_f1  # noqa: E402
from core.feature_cache import load_feature_cache  # noqa: E402
from core.p1c_head_variants import group_pooled_sequences_by_video  # noqa: E402
from core.p1c_video_sampling import sample_rows_per_video  # noqa: E402
from core.specialist import GroupMap, remap_sequences  # noqa: E402
from core.subset7 import FAMILY, candidate_subsets, novelty, subset_id, winner  # noqa: E402
from core import subset7  # noqa: E402

CONFIG_TEMPLATE = "configs/experiments/p1c/P1C-VIDEO-DINOV2-B2-AR-FOLD{fold}.yaml"
CACHE_ROOT_TEMPLATE = "results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}"
CLOSURE = "results/p1c/p1c-video-frames/extract_closure.json"
PERMUTE_BASE = {"train": 1000, "validation": 2000, "test": 3000}
TEST_NAME = "noisy_proxy_test (inherited video-level labels; not event-only, D-40)"


def load_fold_features(fold: int, splits, keep_all_train: bool = False) -> dict:
    from scripts.build_p1c_feature_cache import _load_kwargs_for_config
    from scripts.run_p1c_frame_baseline import _load_config
    from scripts.run_p1c_xd_extra_probe import FRAMES

    _c, protocol, _r = _load_config(Path(CONFIG_TEMPLATE.format(fold=fold)))
    protocol["data"]["video_frames"]["extract_closure_path"] = str(REPO_ROOT / CLOSURE)
    trainable = protocol["backbone"].get("trainable", "full")
    out = {"class_names": list(protocol["model"]["class_names"]), "cache_sha256": {}}
    if out["class_names"] != sorted(FAMILY):
        raise RuntimeError(f"fold {fold} class_names differ from core.subset7.FAMILY order: {out['class_names']}")
    for split in splits:
        feats, rows, man = load_feature_cache(Path(CACHE_ROOT_TEMPLATE.format(fold=fold)) / split,
                                              **_load_kwargs_for_config(protocol, split, trainable))
        if split == "train":
            if keep_all_train:
                out["train_all"] = (feats, rows)  # every cached frame, as stored (float16), for per-epoch resampling
            keep = sample_rows_per_video(rows, FRAMES, seed=0)
            feats, rows = np.asarray(feats[keep], dtype=np.float32), [rows[i] for i in keep]
        else:
            feats = np.asarray(feats, dtype=np.float32)
        out[split] = (feats, rows)
        out["cache_sha256"][split] = man.cache_sha256
    return out


def sequences(fd: dict, split: str, seed: int) -> dict:
    feats, rows = fd[split]
    return group_pooled_sequences_by_video(feats, rows, permute_frame_order=True, permute_seed=PERMUTE_BASE[split] + seed)


def score(gm: GroupMap, seqs: dict, preds: dict) -> tuple[float, np.ndarray, np.ndarray]:
    keys = sorted(seqs)
    truth = np.array([seqs[k][1]["target"] for k in keys])
    pred = np.array([int(np.argmax(preds[k])) for k in keys])
    return macro_f1(truth, pred, len(gm.names)), truth, pred


def _search_worker(job: tuple) -> str:
    fold, seeds, subsets, out_path, threads, epochs = job
    import torch
    from scripts.run_p1c_xd_extra_probe import _train_eval  # sets 4 threads on import; override after
    torch.set_num_threads(threads)

    out_path = Path(out_path)
    done = set()
    if out_path.exists():
        done = {(r["subset"], r["seed"]) for r in map(json.loads, out_path.read_text().splitlines()) if r}
    fd = load_fold_features(fold, ("train", "validation"))
    classes = fd["class_names"]
    with out_path.open("a") as h:
        for seed in seeds:
            tr, va = sequences(fd, "train", seed), sequences(fd, "validation", seed)
            for subset in subsets:
                sid = subset_id(subset)
                if (sid, seed) in done:
                    continue
                gm = GroupMap(classes, list(subset))
                trs, vas = remap_sequences(tr, gm), remap_sequences(va, gm)
                t0 = time.perf_counter()
                result, preds = _train_eval(seed, trs, vas, gm.names, epochs)
                f1, _t, _p = score(gm, vas, preds)
                rec = {"subset": sid, "fold": fold, "seed": seed, "macro_f1": f1, "best_epoch": result.best_epoch,
                       "train_videos": len(trs), "val_videos": len(vas), "seconds": round(time.perf_counter() - t0, 2)}
                h.write(json.dumps(rec) + "\n")
                h.flush()
                print(json.dumps(rec), flush=True)
    return str(out_path)


def _test_worker(job: tuple) -> list[dict]:
    fold, seeds, subsets, threads, epochs = job
    import torch
    from scripts.run_p1c_xd_extra_probe import _train_eval  # sets 4 threads on import; override after
    torch.set_num_threads(threads)

    fd = load_fold_features(fold, ("train", "validation", "test"))
    classes, recs = fd["class_names"], []
    for seed in seeds:
        tr, va, te = (sequences(fd, s, seed) for s in ("train", "validation", "test"))
        overlap = {k[1] for k in te} & ({k[1] for k in tr} | {k[1] for k in va})
        if overlap:
            raise RuntimeError(f"fold {fold}: test videos also in train/validation: {sorted(overlap)[:5]}")
        for subset in subsets:
            gm = GroupMap(classes, list(subset))
            trs, vas, tes = remap_sequences(tr, gm), remap_sequences(va, gm), remap_sequences(te, gm)
            result, preds = _train_eval(seed, trs, {**vas, **tes}, gm.names, epochs, monitor_seq=vas)
            vf1, _vt, _vp = score(gm, vas, {k: preds[k] for k in vas})
            tf1, tt, tp = score(gm, tes, {k: preds[k] for k in tes})
            recs.append({"subset": subset_id(subset), "fold": fold, "seed": seed, "val_macro_f1": vf1, "test_macro_f1": tf1,
                         "test_per_class_f1": dict(zip(gm.names, per_class_f1(tt, tp, len(gm.names)).tolist())),
                         "best_epoch": result.best_epoch, "test_videos": len(tes), "cache_sha256": fd["cache_sha256"]})
            print(json.dumps({k: v for k, v in recs[-1].items() if k not in ("test_per_class_f1", "cache_sha256")}), flush=True)
    return recs


def _provenance(args) -> dict:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    return {"git_revision": rev, "folds": args.folds, "seeds": args.seeds, "min_novelty": args.min_novelty,
            "epochs_override": args.epochs, "family_map": FAMILY, "device": "cpu",
            "subset7_sha256": hashlib.sha256(Path(subset7.__file__).read_bytes()).hexdigest()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("search", "select", "test"))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--min-novelty", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per fold worker (one worker per fold)")
    ap.add_argument("--epochs", type=int, default=None, help="epoch ceiling override (smoke only)")
    ap.add_argument("--limit", type=int, default=None, help="first N candidates only (smoke only)")
    ap.add_argument("--output", type=Path, default=Path("results/subset7-search"))
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    classes = sorted(FAMILY)  # workers assert this equals each fold config's class_names (config load runs readiness hashing)
    cands = candidate_subsets(classes, args.min_novelty)[: args.limit]
    ctx = mp.get_context("spawn")
    if args.stage == "search":
        (args.output / "search").mkdir(exist_ok=True)
        (args.output / "search_config.json").write_text(json.dumps(_provenance(args) | {"candidates": len(cands)}, indent=2) + "\n")
        jobs = [(f, args.seeds, cands, str(args.output / "search" / f"fold{f}.jsonl"), args.threads, args.epochs) for f in args.folds]
        with ctx.Pool(len(jobs)) as pool:
            pool.map(_search_worker, jobs)
        return 0
    if args.stage == "select":
        recs = [json.loads(line) for f in args.folds for line in (args.output / "search" / f"fold{f}.jsonl").read_text().splitlines() if line]
        table = subset7.aggregate(recs, args.folds, args.seeds)
        missing = {subset_id(c) for c in cands} - set(table)
        if missing:
            raise SystemExit(f"search incomplete: {len(missing)} candidate subsets have no results")
        picks = {"overall": winner(table), "novelty_7": winner(table, min_novelty=7)}
        ranked = sorted(table.items(), key=lambda kv: -kv[1]["mean"])
        (args.output / "selection.json").write_text(json.dumps({
            "rule": "highest mean validation macro-F1 over folds x seeds; ties -> higher novelty -> lexicographic (core.subset7.winner)",
            "winners": picks, "provenance": _provenance(args),
            "ranked": [{"subset": sid, **row} for sid, row in ranked]}, indent=1) + "\n")
        for name, sid in picks.items():
            print(name, sid, json.dumps(table[sid]))
        return 0
    out = args.output / "test.json"
    if out.exists():
        raise SystemExit(f"{out} exists: the test set has already been spent on this selection")
    sel = json.loads((args.output / "selection.json").read_text())
    subsets = sorted({tuple(sid.split("+")) for sid in sel["winners"].values()})
    jobs = [(f, args.seeds, subsets, args.threads, args.epochs) for f in args.folds]
    with ctx.Pool(len(jobs)) as pool:
        recs = [r for part in pool.map(_test_worker, jobs) for r in part]
    summary = {}
    for sid in {subset_id(s) for s in subsets}:
        rows = [r for r in recs if r["subset"] == sid]
        per_fold = {f: {k: float(np.mean([r[k] for r in rows if r["fold"] == f])) for k in ("val_macro_f1", "test_macro_f1")}
                    for f in args.folds}
        summary[sid] = {"novelty": novelty(sid.split("+")), "per_fold": per_fold,
                        "test_mean": float(np.mean([v["test_macro_f1"] for v in per_fold.values()])),
                        "val_mean_retrained": float(np.mean([v["val_macro_f1"] for v in per_fold.values()])),
                        "search_val_mean": next(r["mean"] for r in sel["ranked"] if r["subset"] == sid),
                        "test_per_class_f1": {c: float(np.mean([r["test_per_class_f1"][c] for r in rows])) for c in sid.split("+")}}
    out.write_text(json.dumps({"test_view": TEST_NAME, "winners": sel["winners"], "summary": summary,
                               "runs": recs, "provenance": _provenance(args)}, indent=1) + "\n")
    print(json.dumps({sid: {"test_mean": round(v["test_mean"], 4), "search_val_mean": round(v["search_val_mean"], 4)}
                      for sid, v in summary.items()}, indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())
