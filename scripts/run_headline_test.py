#!/usr/bin/env python3
"""One-time 14-class test of the final pipeline (D-53: frozen DINOv2 cls+mean-patch, 8 frames/video, step_permuted head).

Per fold x seed: train on the fold's training videos, best epoch on validation (headline convention), then score the
inherited-label `noisy_proxy_test` view (D-40) once. Refuses to run if results already exist.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.confusion_groups import confusion, per_class_f1  # noqa: E402
from core.specialist import GroupMap, remap_sequences  # noqa: E402
from scripts.run_subset7_search import TEST_NAME, load_fold_features, score, sequences  # noqa: E402


def _worker(job) -> list[dict]:
    fold, seeds, threads = job
    import torch
    from scripts.run_p1c_xd_extra_probe import _train_eval
    torch.set_num_threads(threads)

    fd = load_fold_features(fold, ("train", "validation", "test"))
    classes, out = fd["class_names"], []
    gm = GroupMap(classes, classes)
    for seed in seeds:
        tr, va, te = (remap_sequences(sequences(fd, s, seed), gm) for s in ("train", "validation", "test"))
        if {k[1] for k in te} & ({k[1] for k in tr} | {k[1] for k in va}):
            raise RuntimeError(f"fold {fold}: test videos overlap train/validation")
        result, preds = _train_eval(seed, tr, {**va, **te}, classes, None, monitor_seq=va)
        rec = {"fold": fold, "seed": seed, "best_epoch": result.best_epoch, "cache_sha256": fd["cache_sha256"]}
        for name, seqs in (("val", va), ("test", te)):
            f1, t, p = score(gm, seqs, {k: preds[k] for k in seqs})
            rec |= {f"{name}_macro_f1": f1, f"{name}_accuracy": float((t == p).mean()), f"{name}_videos": len(seqs)}
            if name == "test":
                rec |= {"test_per_class_f1": dict(zip(classes, per_class_f1(t, p, len(classes)).tolist())),
                        "test_confusion": confusion(t, p, len(classes)).tolist()}
        out.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if not k.startswith("test_per") and k not in ("test_confusion", "cache_sha256")}), flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2, help="parallel fold workers; 4 OOM-killed the 16 GiB pod (2026-10-28)")
    ap.add_argument("--output", type=Path, default=Path("results/p1c/headline-14class-test"))
    args = ap.parse_args(argv)
    out = args.output / "test.json"
    if out.exists():
        raise SystemExit(f"{out} exists: the 14-class test has already been spent")
    args.output.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(min(args.workers, len(args.folds))) as pool:
        runs = [r for part in pool.map(_worker, [(f, args.seeds, args.threads) for f in args.folds]) for r in part]
    folds = sorted({r["fold"] for r in runs})
    per_fold = {f: {k: float(np.mean([r[k] for r in runs if r["fold"] == f])) for k in ("val_macro_f1", "val_accuracy", "test_macro_f1", "test_accuracy")} for f in folds}
    summary = {k: float(np.mean([v[k] for v in per_fold.values()])) for k in ("val_macro_f1", "val_accuracy", "test_macro_f1", "test_accuracy")}
    classes = list(runs[0]["test_per_class_f1"])
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    out.write_text(json.dumps({"test_view": TEST_NAME, "pipeline": "D-53 (fold-cache features)", "summary": summary, "per_fold": per_fold,
                               "test_per_class_f1": {c: float(np.mean([r["test_per_class_f1"][c] for r in runs])) for c in classes},
                               "runs": runs, "git_revision": rev, "folds": args.folds, "seeds": args.seeds}, indent=1) + "\n")
    print(json.dumps({"summary": summary, "per_fold": per_fold}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
