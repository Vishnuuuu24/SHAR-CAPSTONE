#!/usr/bin/env python3
"""Partial fine-tune of VideoMAE-B's top blocks from the cached block-8 hidden state.

Thin CLI: arguments, orchestration, file output. Training lives in
`core.p1c_videomae_finetune`, the encoder split in `core.p1c_videomae_split`,
aggregation and the gate in `core.p1c_videomae_summary`. Resumable: a
`per_run/<arm>-fold<f>-s<seed>.json` that exists is reused, not re-run.

Never reads Test: the cache is addressed only through each fold's own
train/validation membership.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_videomae_cache import MANIFEST_NAME, ROWS_NAME, git_revision, read_rows, sha256_file
from core.p1c_videomae_finetune import ARMS, HYPERPARAMETERS, ClipStore, train_run
from core.p1c_videomae_summary import aggregate, load_registered_baseline, write_per_video_csv
from scripts.build_p1c_feature_cache import _load_backbone

CONFIG = "configs/experiments/p1c/P1C-VIDEO-VIDEOMAE-AR-FOLD1.yaml"
BASELINE = "results/p1c/p1c-head-comparison-stepctl/summary.json"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path, required=True, help="block-8 cache (stage to node-local emptyDir for the run)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--arms", choices=sorted(ARMS), nargs="+", default=sorted(ARMS))
    p.add_argument("--epochs", type=int, default=HYPERPARAMETERS["epochs"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--smoke", action="store_true", help="fold 1, seed 0, 2 epochs, 32 train / 16 val clips; separate output dir")
    p.add_argument("--force", action="store_true", help="re-run runs whose per_run json exists")
    args = p.parse_args(argv)

    limits = {}
    if args.smoke:
        args.folds, args.seeds, args.epochs = args.folds[:1], args.seeds[:1], min(args.epochs, 2)
        limits = {"max_train": 32, "max_val": 16}
    out = args.output_dir or Path("results/p1c/p1c-videomae-finetune" + ("-smoke" if args.smoke else ""))
    (out / "per_run").mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(args.cache_dir / MANIFEST_NAME))
    # Only the backbone/class registration is needed; the frame-baseline readiness walk (blob digests) is not.
    proto = yaml.safe_load((REPO_ROOT / CONFIG).read_text())["fixed_protocol"]
    weights = proto["backbone"]["weights"]
    if sha256_file(REPO_ROOT / weights["path"]) != weights["sha256"]:
        raise RuntimeError("VideoMAE checkpoint digest does not match the config's registered sha256")
    if manifest["backbone_weight_sha256"] != proto["backbone"]["weights"]["sha256"]:
        raise RuntimeError("cache was built from a different backbone checkpoint than the config registers")
    if sha256_file(args.cache_dir / ROWS_NAME) != manifest["rows_sha256"]:
        raise RuntimeError("rows.csv does not match the manifest")
    class_names = proto["model"]["class_names"]
    rows = read_rows(args.cache_dir / ROWS_NAME)
    store = ClipStore(args.cache_dir, rows)
    device = torch.device(args.device)
    backbone = _load_backbone(proto["backbone"]).eval()
    (out / "config.json").write_text(json.dumps({
        "seeds": args.seeds, "folds": args.folds, "arms": {a: ARMS[a] for a in args.arms}, "hyperparameters": HYPERPARAMETERS,
        "epochs_run": args.epochs, "smoke": args.smoke, "limits": limits, "grad_checkpoint": args.grad_checkpoint,
        "git": git_revision(REPO_ROOT), "cache_manifest_sha256": sha256_file(args.cache_dir / MANIFEST_NAME),
        "cache_array_sha256": manifest["cache_sha256"], "backbone_weight_sha256": manifest["backbone_weight_sha256"],
        "labels": "inherited video-level UCF labels; validation only, Test never read (D-40)",
    }, indent=2, sort_keys=True) + "\n")

    for fold in args.folds:
        for seed in args.seeds:
            for arm in args.arms:
                path = out / "per_run" / f"{arm}-fold{fold}-s{seed}.json"
                if path.exists() and not args.force:
                    print(f"skip {path.name} (exists)", flush=True)
                    continue
                run = train_run(backbone, store, fold=fold, arm=arm, seed=seed, class_names=class_names, device=device,
                                epochs=args.epochs, num_workers=args.num_workers, grad_checkpoint=args.grad_checkpoint,
                                log=lambda m: print(m, flush=True), **limits)
                path.write_text(json.dumps(run, indent=1) + "\n")
                print(f"done {path.name}: final macro-F1 {run['final_epoch_macro_f1']:.4f}, {run['wall_seconds']:.0f}s", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    runs = [json.load(open(f)) for f in sorted((out / "per_run").glob("*.json"))]
    baseline_path = REPO_ROOT / BASELINE
    baseline = load_registered_baseline(baseline_path) if baseline_path.exists() else None
    summary = aggregate(runs, class_names, baseline, folds=args.folds if args.smoke else (1, 2, 3, 4))
    summary["n_runs_total"] = len(runs)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    write_per_video_csv(out / "per_video.csv", runs, class_names)
    for arm, a in summary["arms"].items():
        print(f"{arm}: final-epoch macro-F1 {a['final_epoch_macro_f1_mean']:.4f} (sd over runs {a['sd_over_runs']:.4f}, "
              f"sd across fold means {a['sd_across_fold_means']:.4f}) folds {a['per_fold_mean']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
