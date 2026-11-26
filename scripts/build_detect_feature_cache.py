#!/usr/bin/env python3
"""Frozen frame features for the live detector over the official Anomaly-Detection split (FINAL_PUSH_PLAN Step 4a).

One pass over the D-34 frame blob feeds two frozen backbones from the same decoded, resized frame:
DINOv2 ViT-B/14-reg4 (the project's registered backbone) and SigLIP ViT-B/16 (a CLIP-family backbone, as
used by the published frozen-feature detectors). Both store `[CLS-or-pooler ++ mean(patch)]`, 1536-d float16.

Before the pass, both paths are checked against caches that already exist (cosine >= 0.9999): DINOv2-B
against the fold-1 feature cache, SigLIP against the study-frame patch cache. The pass streams into
memory-mapped arrays and records progress, so `--resume` continues after an interrupted pod.

`--limit-frames N` is the smoke/calibration mode: checks, N frames, throughput and a projected ETA.
GPU job. Give the owner the projected ETA before the full pass (docs/PROJECT_RULES.md: anything over ~30 minutes).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SIGLIP_CONFIG = "configs/experiments/p1c/P1C-VIDEO-SIGLIP-AR-FOLD1.yaml"
B_REFERENCE = "results/p1c/p1c-dinov2-feature-cache-ar-fold1/train"
SIGLIP_REFERENCE = "results/p1c/p1c-siglip-patch-cache"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DIM = 1536


def _sha256_array(path: Path, chunk: int = 1 << 26) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float32), b.astype(np.float32)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=Path("results/detect/features"))
    ap.add_argument("--every", type=int, default=3, help="keep every N-th blob frame (3 -> one per 15 source frames)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--check-frames", type=int, default=128)
    ap.add_argument("--limit-frames", type=int, default=None, help="smoke only: checks plus this many frames, nothing durable written")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--code-revision", default="unrecorded")
    args = ap.parse_args(argv)

    import torch
    import yaml

    from core import detect_data
    from core.backbone_cache import blob_rows_for
    from core.frame_batches import FrameBatcher
    from core.validation_frames import CLOSURE
    from scripts.build_combined_patch_cache import _load_backbone_and_preprocessing
    from scripts.build_p1c_feature_cache import _local_siglip

    started = datetime.now(timezone.utc)
    device = torch.device(args.device)
    frame_index = np.load(REPO_ROOT / detect_data.FRAME_INDEX, mmap_mode="r")
    videos = detect_data.load_split(REPO_ROOT)
    blob_rows, owner, source_frames = detect_data.frame_plan(videos, frame_index, args.every)
    total = len(blob_rows)
    print(json.dumps({"videos": len(videos), "frames_planned": total, "every": args.every}), flush=True)

    batcher = FrameBatcher(args.threads)
    dinov2, _pre, dinov2_sha = _load_backbone_and_preprocessing(device)
    siglip_weights = yaml.safe_load((REPO_ROOT / SIGLIP_CONFIG).read_text())["fixed_protocol"]["backbone"]["weights"]
    siglip = _local_siglip(siglip_weights).to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def encode(x) -> tuple[np.ndarray, np.ndarray]:
        x = x.to(device, non_blocking=True)                         # registered preprocessing: ImageNet statistics
        with torch.inference_mode():
            b = dinov2.pooled_features(x)
            s = siglip.pooled_features((x * std + mean - 0.5) / 0.5)   # same pixels, SigLIP's (0.5, 0.5) statistics
        return b.to(torch.float16).cpu().numpy(), s.to(torch.float16).cpu().numpy()

    def forward(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return encode(batcher(rows))

    # ---- path checks against caches that already exist
    with (REPO_ROOT / B_REFERENCE / "rows.csv").open() as handle:
        ref_rows = list(csv.DictReader(handle))
    pick = np.linspace(0, len(ref_rows) - 1, args.check_frames).astype(int)
    pick = np.array([i for i in pick if not ref_rows[i]["label"].startswith("Normal")])   # Normal stems are not unique (K-11)
    keys = [(ref_rows[i]["source_video_id"], int(ref_rows[i]["source_frame_index"])) for i in pick]
    got_b, _ = forward(blob_rows_for(keys, REPO_ROOT / detect_data.VIDEOS_CSV, frame_index))
    ref_b = np.load(REPO_ROOT / B_REFERENCE / "features.float16.npy", mmap_mode="r")[pick]
    with (REPO_ROOT / SIGLIP_REFERENCE / "rows.csv").open() as handle:
        s_rows = list(csv.DictReader(handle))
    s_pick = np.linspace(0, len(s_rows) - 1, args.check_frames).astype(int)
    s_pick = np.array([i for i in s_pick if not s_rows[i]["label"].startswith("Normal")])
    s_keys = [(s_rows[i]["source_video_id"], int(s_rows[i]["source_frame_index"])) for i in s_pick]
    _, got_s = forward(blob_rows_for(s_keys, REPO_ROOT / detect_data.VIDEOS_CSV, frame_index))
    ref_s = np.concatenate([
        np.asarray(np.load(REPO_ROOT / SIGLIP_REFERENCE / "cls.float16.npy", mmap_mode="r")[s_pick], dtype=np.float32),
        np.stack([np.asarray(np.load(REPO_ROOT / SIGLIP_REFERENCE / "features.float16.npy", mmap_mode="r")[i], dtype=np.float32).mean(axis=0) for i in s_pick]),
    ], axis=1)
    checks = {"dinov2b_vs_fold1_cache": {"frames": int(len(pick)), "min_cosine": float(_cosine(got_b, ref_b).min())},
              "siglip_vs_patch_cache": {"frames": int(len(s_pick)), "min_cosine": float(_cosine(got_s, ref_s).min())}}
    print(json.dumps({"path_checks": checks}), flush=True)
    if min(c["min_cosine"] for c in checks.values()) < 0.9999:
        raise SystemExit(f"a backbone path does not reproduce its existing cache: {checks}")

    # ---- the pass
    smoke = args.limit_frames is not None
    count = min(args.limit_frames, total) if smoke else total
    out_dir = args.output
    arrays, done = {}, 0
    if not smoke:
        out_dir.mkdir(parents=True, exist_ok=True)
        progress_path = out_dir / "progress.json"
        if args.resume and progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress["frames_planned"] != total or progress["every"] != args.every:
                raise SystemExit("cannot resume: the frame plan differs from the interrupted run")
            done = int(progress["frames_done"])
        for name in ("dinov2b", "siglip"):
            path = out_dir / f"{name}.float16.npy"
            arrays[name] = (np.load(path, mmap_mode="r+") if done else np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(total, DIM)))
    torch.cuda.reset_peak_memory_stats(device)
    starts = list(range(done, count, args.batch))
    loader = ThreadPoolExecutor(1)                                   # decode the next batch while the GPU works on this one
    t0, last_report = time.time(), time.time()
    pending = loader.submit(batcher, blob_rows[starts[0]:starts[0] + args.batch]) if starts else None
    for position, start in enumerate(starts):
        images = pending.result()
        if position + 1 < len(starts):
            following = starts[position + 1]
            pending = loader.submit(batcher, blob_rows[following:following + args.batch])
        b, s = encode(images)
        if not smoke:
            arrays["dinov2b"][start:start + len(b)] = b
            arrays["siglip"][start:start + len(s)] = s
            if position % 200 == 199 or start + args.batch >= count:
                for array in arrays.values():
                    array.flush()
                progress_path.write_text(json.dumps({"frames_done": min(start + args.batch, count), "frames_planned": total, "every": args.every}))
        if time.time() - last_report > 120 or start + args.batch >= count:
            last_report = time.time()
            processed = min(start + args.batch, count) - done
            rate = processed / (time.time() - t0)
            print(json.dumps({"frames_done": min(start + args.batch, count), "of": count, "frames_per_s": round(rate, 1),
                              "eta_min": round((count - done - processed) / rate / 60, 1)}), flush=True)
    loader.shutdown()
    seconds = time.time() - t0
    stats = {"frames": count - done, "seconds": round(seconds, 1), "frames_per_s": round((count - done) / max(seconds, 1e-9), 1),
             "gpu_peak_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
             "projected_full_minutes": round(total / ((count - done) / max(seconds, 1e-9)) / 60, 1)}
    print(json.dumps({"smoke" if smoke else "extraction": stats}), flush=True)
    if smoke:
        return 0

    with (out_dir / "rows.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "relative_path", "split", "anomaly", "category", "source_frame_index", "blob_row"])
        for row in range(total):
            video = videos[int(owner[row])]
            writer.writerow([row, video.relative_path, video.split, int(video.anomaly), video.category, int(source_frames[row]), int(blob_rows[row])])
    for array in arrays.values():
        array.flush()
    del arrays
    (out_dir / "manifest.json").write_text(json.dumps({
        "plan": "docs/FINAL_PUSH_PLAN_2026-11-11.md Step 4a", "decision": "D-71", "code_revision": args.code_revision,
        "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "task": "UCF-Crime official Anomaly-Detection split (separate task from the 14-class classifier, rule 3)",
        "videos": {s: sum(v.split == s for v in videos) for s in ("train", "test")}, "frames": total, "every": args.every,
        "sampling": f"every {args.every}-th blob frame = one frame per {5 * args.every} source frames",
        "feature": "[CLS-or-pooler ++ mean(patch)], 1536-d float16, registered 224 bilinear preprocessing",
        "frame_blob_sha256": json.loads((REPO_ROOT / CLOSURE).read_text())["blob_sha256"],
        "backbones": {"dinov2b": {"family": "timm_dinov2_vitb14_reg4", "weights_sha256": dinov2_sha,
                                  "features_sha256": _sha256_array(out_dir / "dinov2b.float16.npy")},
                      "siglip": {"family": "transformers_siglip_base_p16_224", "weights_sha256": siglip_weights["sha256"],
                                 "features_sha256": _sha256_array(out_dir / "siglip.float16.npy")}},
        "path_checks": checks, **stats}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
