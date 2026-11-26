#!/usr/bin/env python3
"""RESULT 1, learned stage: P2B denoisers (and, with --comparator, the pinned pretrained restorer) on the same frames, noise and classifier as the classical run (D-59, D-60).

The same 108 AR validation videos x 32 frames and the same per-frame noise as run_p2_classical.py; every
registered degradation x {identity, each trained denoiser} -> PSNR/SSIM on the GPU against the clean frame
(float, before quantization; eval.image_quality_torch, pinned to the skimage reference by tests) -> uint8 ->
the registered DINOv2 preprocessing -> pooled features -> the D-53 head retrained per fold x seed.
The identity rows reproduce the classical run's identity rows and are checked against its features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p2_downstream import classify, clean_feature_check, condition_table  # noqa: E402
from core.p2_pipeline import Condition, decode_rgb_float, sample_key, summarise  # noqa: E402
from core.p2a_conventions import load_frozen_p2a_conventions  # noqa: E402
from core.subset7 import FAMILY  # noqa: E402
from core.validation_frames import CLOSURE, select_validation_frames  # noqa: E402
from data.degradations import apply_degradation  # noqa: E402

CONVENTIONS = REPO_ROOT / "configs/p2a_conventions.yaml"
CONFIG = REPO_ROOT / "configs/p2b_denoising_vae.yaml"
COMPARATOR = REPO_ROOT / "configs/p2b_pretrained_comparator.yaml"
EXTRA = REPO_ROOT / "configs/p2b_extra_restorers.yaml"
CLASSICAL = REPO_ROOT / "results/p2/p2-classical"
DIM = 1536


def load_denoisers(model_dir: Path, device):
    import torch
    from models.denoising_vae import build_denoiser

    cfg = yaml.safe_load(CONFIG.read_text())
    manifest = json.loads((model_dir / "manifest.json").read_text())
    out = {}
    for name, m in sorted(manifest["models"].items()):
        path = model_dir / m["checkpoint"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != m["sha256"]:
            raise RuntimeError(f"{path} does not match its recorded sha256")
        model = build_denoiser(cfg["model"], cfg["variants"][m["variant"]])
        model.load_state_dict(torch.load(path, map_location="cpu"))
        out[name] = model.to(device).eval()
    return out, manifest


def extract(args, conv, conds, restorers, denoisers, videos, order, out: Path) -> dict:
    import torch
    from eval.image_quality_torch import psnr_batch, ssim_batch
    from scripts.build_combined_patch_cache import _load_backbone_and_preprocessing
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    from core.p1c_roi import preprocess_crop

    device = torch.device(args.device)
    backbone, pre, weight_sha = _load_backbone_and_preprocessing(device)
    feats = np.lib.format.open_memmap(out / "features.float16.npy", mode="w+", dtype=np.float16,
                                      shape=(len(conds), len(order), args.frames, DIM))
    by_key = {(c.degradation.transform_id, c.restoration): c for c in conds}
    threads = ThreadPoolExecutor(args.threads)

    def preprocess(u8: np.ndarray):
        # Same per-frame function as build_p1c_patch_cache._preprocess_batch (PIL resize releases the GIL).
        return torch.stack(list(threads.map(lambda fr: preprocess_crop(Image.fromarray(fr, mode="RGB"), pre), u8)))

    quality, t0 = [], time.time()
    for vi, v in enumerate(order):
        meta = videos[v]
        clean = [decode_rgb_float(meta["dataset"].raw_jpeg(i)) for i in meta["rows"]]
        keys = [sample_key(meta["filepath"], f, meta["file_digest"], 0) for f in meta["frames"]]
        clean_t = torch.from_numpy(np.stack(clean)).permute(0, 3, 1, 2).to(device)
        for spec in conv.degradation_specs:
            degraded = np.stack([apply_degradation(img, spec=spec, sample=k) for img, k in zip(clean, keys)])
            deg_t = torch.from_numpy(degraded).permute(0, 3, 1, 2).to(device)
            for r in restorers:
                c = by_key[(spec.transform_id, r)]
                with torch.no_grad():
                    restored = deg_t if r == "identity" else denoisers[r].restore(deg_t)
                    ps = psnr_batch(clean_t, restored, **conv.psnr_options)
                    ss = ssim_batch(clean_t, restored, **conv.ssim_options)
                    u8 = torch.round(restored * 255.0).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                    f = backbone.pooled_features(preprocess(u8).to(device))
                feats[c.index, vi] = f.to(torch.float16).cpu().numpy()
                finite = [p for p in ps if math.isfinite(p)]
                quality.append({"condition": c.name, "video": v, "psnr_mean": float(np.mean(finite)) if finite else float("inf"),
                                "psnr_infinite_frames": len(ps) - len(finite), "ssim_mean": float(np.mean(ss))})
        if (vi + 1) % 10 == 0 or vi + 1 == len(order):
            print(json.dumps({"videos_done": vi + 1, "of": len(order), "elapsed_s": round(time.time() - t0)}), flush=True)
    feats.flush()
    with (out / "quality.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(quality[0]))
        w.writeheader()
        w.writerows(quality)
    return {"backbone_weight_sha256": weight_sha, "extract_seconds": round(time.time() - t0)}


def classical_identity_check(conds, order, feats) -> dict | None:
    """Identity-restoration rows must reproduce the classical run's features for the same degradation."""

    path = CLASSICAL / "features.float16.npy"
    if not path.exists():
        return None
    ref = np.load(path, mmap_mode="r")
    ref_names = [r["condition"] for r in csv.DictReader((CLASSICAL / "quality.csv").open())]
    ref_index = {n: i for i, n in enumerate(dict.fromkeys(ref_names))}
    ref_order = sorted(dict.fromkeys(r["video"] for r in csv.DictReader((CLASSICAL / "quality.csv").open())))
    pos = {v: i for i, v in enumerate(ref_order)}
    worst = 0.0
    for c in conds:
        if c.restoration != "identity":
            continue
        a = np.asarray(feats[c.index], dtype=np.float32)
        b = np.asarray(ref[ref_index[c.name]][[pos[v] for v in order]], dtype=np.float32)
        worst = max(worst, float(np.abs(a - b).max()))
    return {"max_abs_diff_vs_classical_identity_rows": worst}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, default=Path("results/p2/p2b-denoisers"), help="trained-denoiser dir; 'none' to skip")
    ap.add_argument("--comparator", action="store_true", help="add the pinned pretrained comparator (configs/p2b_pretrained_comparator.yaml)")
    ap.add_argument("--extra", action="store_true", help="add the D-62 restorers (configs/p2b_extra_restorers.yaml)")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-videos", type=int, default=None, help="smoke only")
    ap.add_argument("--epochs", type=int, default=None, help="smoke only")
    ap.add_argument("--output", type=Path, default=Path("results/p2/p2b-learned"))
    args = ap.parse_args(argv)
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    conv = load_frozen_p2a_conventions(CONVENTIONS)
    denoisers, model_manifest = ({}, None) if str(args.models) == "none" else load_denoisers(args.models, torch.device(args.device))
    comparator = None
    if args.comparator:
        from models.restoration_comparator import load_comparator
        name, restorer, comparator = load_comparator(COMPARATOR, REPO_ROOT, torch.device(args.device))
        denoisers[name] = restorer
    extra = None
    if args.extra:
        from models.restoration_comparator import load_restorers
        loaded = load_restorers(EXTRA, REPO_ROOT, torch.device(args.device))
        denoisers.update({n: r for n, r, _ in loaded})
        extra = {n: prov for n, _, prov in loaded}
    restorers = ["identity", *denoisers]
    conds = [Condition(i, spec, r) for i, (spec, r) in enumerate((s, r) for s in conv.degradation_specs for r in restorers)]
    classes = sorted(FAMILY)
    videos = select_validation_frames(REPO_ROOT, args.folds, classes, args.frames)
    order = sorted(videos)[: args.limit_videos]
    info = extract(args, conv, conds, restorers, denoisers, videos, order, args.output)
    feats = np.load(args.output / "features.float16.npy", mmap_mode="r")
    check = clean_feature_check(REPO_ROOT, order, videos, feats, args.folds)
    reproduce = classical_identity_check(conds, order, feats)
    print(json.dumps({"clean_feature_check": check, "classical_identity_check": reproduce}), flush=True)
    runs = classify(conds, videos, order, feats, classes, folds=args.folds, seeds=args.seeds, threads=args.threads, epochs=args.epochs)
    quality = list(csv.DictReader((args.output / "quality.csv").open()))
    summary = summarise(condition_table(conds, quality, runs, args.folds), conds[0].name)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (args.output / "classification_runs.json").write_text(json.dumps(runs, indent=1) + "\n")
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    (args.output / "config.json").write_text(json.dumps({
        "git_revision": rev, "conventions_sha256": conv.source_sha256, "denoiser_config_sha256": model_manifest and model_manifest["config_sha256"],
        "denoisers": model_manifest and {k: v["sha256"] for k, v in model_manifest["models"].items()}, "comparator": comparator, "extra_restorers": extra, "conditions": len(conds),
        "videos": len(order), "frames_per_video": args.frames, "folds": args.folds, "seeds": args.seeds,
        "frame_rule": "same as results/p2/p2-classical (sample_rows_per_video seed=None, native 320x240 JPEG from the D-34 blob)",
        "noise_seed": "SampleKey(relative_path=<filepath>/frame<index>, file_digest, global_seed=0), as the classical run",
        "quality_metrics": "eval.image_quality_torch (float64, equal to the skimage reference) on float [0,1] before quantization",
        "classifier": "D-53 step_permuted head retrained per fold x seed on fold-cache train features (D-59)",
        "frame_blob_sha256": json.loads((REPO_ROOT / CLOSURE).read_text())["blob_sha256"], "clean_feature_check": check,
        "classical_identity_check": reproduce, "limit_videos": args.limit_videos, "epochs_override": args.epochs,
        "test_split": "untouched", **info}, indent=1) + "\n")
    for name, r in summary.items():
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("psnr_db", "ssim", "macro_f1", "macro_f1_delta_vs_clean", "macro_f1_delta_vs_unrestored")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
