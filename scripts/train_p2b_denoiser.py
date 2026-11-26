#!/usr/bin/env python3
"""RESULT 1, P2B: train the generic denoising VAE and its autoencoder ablation (configs/p2b_denoising_vae.yaml, D-60).

One model per variant x seed, noisy -> clean on 128x128 crops of native 320x240 UCF training frames, noise drawn
on the fly from the 10 frozen P2A degradations. Writes <variant>_seed<s>.pt, a per-model history, and a manifest
with checkpoint digests. The 108 validation videos the models are later judged on are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p2a_conventions import load_frozen_p2a_conventions  # noqa: E402
from core.p2b_training import NoisyCropDataset, train_denoiser, training_frame_pool  # noqa: E402
from core.subset7 import FAMILY  # noqa: E402
from models.denoising_vae import build_denoiser, denoiser_loss  # noqa: E402

CONVENTIONS = REPO_ROOT / "configs/p2a_conventions.yaml"
CONFIG = REPO_ROOT / "configs/p2b_denoising_vae.yaml"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["dvae", "dae"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default: the config's seeds")
    ap.add_argument("--steps", type=int, default=None, help="smoke only; default: the config's steps")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=Path("results/p2/p2b-denoisers"))
    args = ap.parse_args(argv)
    import torch

    cfg_bytes = CONFIG.read_bytes()
    cfg = yaml.safe_load(cfg_bytes)
    tr = cfg["training"]
    steps = args.steps or int(tr["steps"])
    seeds = args.seeds if args.seeds is not None else list(tr["seeds"])
    conv = load_frozen_p2a_conventions(CONVENTIONS)
    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pool = training_frame_pool(REPO_ROOT, args.folds, sorted(FAMILY), int(tr["frames_per_video"]))
    print(json.dumps({"pool_frames": len(pool), "pool_videos": len({p["video"] for p in pool}), "load_s": round(time.time() - t0)}), flush=True)
    device = torch.device(args.device)
    models = {}
    for variant in args.variants:
        vcfg = cfg["variants"][variant]
        loss_fn = (lambda r, c, m, lv, _m=vcfg.get("log_sigma_min"): denoiser_loss(r, c, m, lv, log_sigma_min=_m))
        for seed in seeds:
            name = f"{variant}_seed{seed}"
            torch.manual_seed(seed)
            model = build_denoiser(cfg["model"], vcfg)
            data = NoisyCropDataset(pool, conv.degradation_specs, crop=int(tr["crop"]), seed=seed,
                                    noise_global_seed=int(tr["noise_global_seed_offset"]) + seed, length=steps * int(tr["batch_size"]))
            loader = torch.utils.data.DataLoader(data, batch_size=int(tr["batch_size"]), shuffle=False, num_workers=args.workers,
                                                 pin_memory=True, persistent_workers=False, prefetch_factor=4 if args.workers else None)
            torch.manual_seed(seed)
            t1 = time.time()
            history = train_denoiser(model, loader, steps=steps, learning_rate=float(tr["learning_rate"]), betas=tr["adam_betas"],
                                     loss_fn=loss_fn, device=device,
                                     on_log=lambda row, _n=name, _t=t1: print(json.dumps({"model": _n, **{k: round(v, 6) for k, v in row.items()}, "elapsed_s": round(time.time() - _t)}), flush=True))
            ckpt = args.output / f"{name}.pt"
            torch.save(model.state_dict(), ckpt)
            (args.output / f"{name}.history.json").write_text(json.dumps(history, indent=1) + "\n")
            models[name] = {"variant": variant, "seed": seed, "checkpoint": ckpt.name,
                            "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(), "train_seconds": round(time.time() - t1),
                            "final": history[-1], "parameters": sum(p.numel() for p in model.parameters())}
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    (args.output / "manifest.json").write_text(json.dumps({
        "git_revision": rev, "config": str(CONFIG.relative_to(REPO_ROOT)), "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
        "conventions_sha256": conv.source_sha256, "steps": steps, "seeds": seeds, "folds": args.folds,
        "pool_frames": len(pool), "pool_videos": len({p["video"] for p in pool}),
        "pool_digest": hashlib.sha256("\n".join(f"{p['video']}/{p['frame']}" for p in pool).encode()).hexdigest(),
        "models": models}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
