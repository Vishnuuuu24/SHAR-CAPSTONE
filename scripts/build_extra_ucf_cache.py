#!/usr/bin/env python3
"""Pooled DINOv2 features (the registered 1536-d CLS + mean-patch) for 8 frames of each extra UCF video (D-64).

Same backbone, preprocessing and 8-frame rule as the fold caches (`sample_rows_per_video(seed=0)`); frames are read
from the D-34 blob. Writes results/p1c/p1c-extra-ucf-cache/{features.float16.npy, rows.csv, manifest.json}.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.extra_ucf_videos import VIDEOS_CSV, blob_rows, protocol_video_ids, select_extra_videos  # noqa: E402
from core.p1c_video_sampling import sample_rows_per_video  # noqa: E402
from core.subset7 import FAMILY  # noqa: E402
from core.validation_frames import CLOSURE  # noqa: E402

FRAMES = 8
DIM = 1536


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-videos", type=int, default=None, help="smoke only")
    ap.add_argument("--output", type=Path, default=Path("results/p1c/p1c-extra-ucf-cache"))
    args = ap.parse_args(argv)
    import torch
    from data.video_blob import VideoBlobReader
    from scripts.build_combined_patch_cache import _load_backbone_and_preprocessing
    from scripts.build_p1c_patch_cache import _preprocess_batch
    from core.p2_pipeline import decode_rgb_float

    args.output.mkdir(parents=True, exist_ok=True)
    classes = sorted(FAMILY)
    with (REPO_ROOT / VIDEOS_CSV).open(newline="") as h:
        extra = select_extra_videos(csv.DictReader(h), protocol_video_ids(REPO_ROOT))[: args.limit_videos]
    reader = VideoBlobReader(REPO_ROOT / CLOSURE, verify_digest=False)
    frame_index = np.load(REPO_ROOT / "results/p1c/p1c-video-frames/frame_index.npy", mmap_mode="r")
    device = torch.device(args.device)
    backbone, pre, weight_sha = _load_backbone_and_preprocessing(device)

    out_rows, feats, t0 = [], [], time.time()
    for n, v in enumerate(extra, 1):
        rows = blob_rows(v, frame_index)
        keep = sample_rows_per_video(rows, FRAMES, seed=0)
        u8 = np.stack([np.rint(decode_rgb_float(reader.payload(rows[i]["blob_row"])) * 255.0).astype(np.uint8) for i in keep])
        with torch.no_grad():
            f = backbone.pooled_features(_preprocess_batch(u8, pre).to(device)).to(torch.float16).cpu().numpy()
        feats.append(f)
        out_rows += [{"source_dataset": "ucf", "source_video_id": v["video"], "source_frame_index": rows[i]["source_frame_index"],
                      "label": v["label"], "target": classes.index(v["label"]), "blob_row": rows[i]["blob_row"]} for i in keep]
        if n % 50 == 0 or n == len(extra):
            print(json.dumps({"videos_done": n, "of": len(extra), "elapsed_s": round(time.time() - t0)}), flush=True)
    array = np.concatenate(feats)
    np.save(args.output / "features.float16.npy", array)
    with (args.output / "rows.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    (args.output / "manifest.json").write_text(json.dumps({
        "git_revision": rev, "role": "extra training videos outside the 700-video AR protocol; video-level label = source directory (D-40)",
        "videos": len(extra), "per_class": {c: sum(1 for v in extra if v["label"] == c) for c in sorted({v["label"] for v in extra})},
        "frames_per_video": FRAMES, "frame_rule": "sample_rows_per_video(seed=0) over the video's kept stride-5 frames", "shape": list(array.shape),
        "features_sha256": hashlib.sha256(array.tobytes()).hexdigest(), "backbone_weight_sha256": weight_sha,
        "frame_blob_sha256": json.loads((REPO_ROOT / CLOSURE).read_text())["blob_sha256"],
        "video_list_sha256": hashlib.sha256("\n".join(v["video"] for v in extra).encode()).hexdigest(), "limit_videos": args.limit_videos}, indent=1) + "\n")
    print(json.dumps({"videos": len(extra), "features": list(array.shape)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
