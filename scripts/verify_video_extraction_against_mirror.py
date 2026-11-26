#!/usr/bin/env python3
"""Spot-check a D-34/D-35 video extraction against the Kaggle mirror oracle.

Background. `scripts/extract_p1c_video_frames.py` decodes the official
UCF-Crime MP4 corpus directly; the Kaggle mirror under
`Datasets/UCF Crime Dataset Kaggle/` is retained specifically as an
independent, previously-produced sample of the same underlying frames (D-34)
to sanity-check that a from-scratch extraction landed on the right pixels at
the right indices. A mirror filename such as `Arrest043_x264_1230.png` names
its source video (`Arrest043_x264.mp4`) and the *source* frame index (1230,
in the original video's own decode order) it was sampled at.

This script samples N (video, mirror frame index) pairs, finds the matching
row in an extraction's `frame_index.npy` by (video basename, source frame
index) -- deliberately not by directory name, since the mirror's single
`NormalVideos` folder does not correspond 1:1 with the official corpus's
three Normal-ish directories and this script must not invent that mapping
either -- downscales the extracted JPEG frame to 64x64 to match the mirror's
fixed resolution, and reports mean absolute difference and PSNR against the
mirror PNG.

What this script deliberately does NOT do: assert bit-identity or even a
tight similarity threshold. The mirror's own producer used an unknown resize
filter and did its own (unknown) encode pipeline; this extractor uses
Pillow's bilinear downscale and a fresh JPEG q95 round-trip. Two independent,
lossy, non-identical pipelines sampling the same real video frame are
expected to differ by an ordinary resampling amount, not to match exactly.
The exit-code gate is therefore set low (a permissive PSNR floor, default 20
dB): a genuine index-mapping bug (matching the wrong frame) or a channel-order
bug (BGR/RGB swap) produces near-random or systematically wrong pixels far
below ordinary resampling noise, which is what this floor is tuned to catch
-- it is not a claim that 20 dB is a good similarity score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.extract_p1c_video_frames import read_frame_record  # noqa: E402

DEFAULT_MIRROR_ROOT = REPO_ROOT / "Datasets" / "UCF Crime Dataset Kaggle"
DEFAULT_CLOSURE_PATH = REPO_ROOT / "results" / "p1c" / "p1c-video-frames" / "extract_closure.json"
DEFAULT_SAMPLES = 20
DEFAULT_PSNR_FLOOR_DB = 20.0
MIRROR_MOSAIC_SHAPE = (64, 64)
_MIRROR_NAME_RE = re.compile(r"^(?P<video_base>.+)_(?P<frame_index>\d+)\.png$")


def _load_videos_csv(videos_csv_path: Path) -> list[dict[str, str]]:
    with videos_csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _index_mirror_frames_for_bases(mirror_root: Path, video_bases: set[str]) -> dict[str, list[tuple[Path, int]]]:
    """Map each requested video basename to its (mirror_path, frame_index) pairs.

    Walks every class/split folder once. This does not assume or use any
    directory-to-label correspondence between the mirror and the official
    corpus -- only the video's own basename, embedded in the mirror
    filename, is used to associate a mirror PNG with an official video.
    """

    found: dict[str, list[tuple[Path, int]]] = {base: [] for base in video_bases}
    for split_dir in ("Train", "Test"):
        split_root = mirror_root / split_dir
        if not split_root.is_dir():
            continue
        for class_dir in sorted(split_root.iterdir()):
            if not class_dir.is_dir():
                continue
            for png_path in class_dir.iterdir():
                match = _MIRROR_NAME_RE.match(png_path.name)
                if not match:
                    continue
                video_base = match.group("video_base")
                if video_base in found:
                    found[video_base].append((png_path, int(match.group("frame_index"))))
    return found


def _mean_absolute_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((255.0 ** 2) / mse)


def verify(
    *,
    closure_path: Path,
    mirror_root: Path,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    closure_path = closure_path.resolve()
    if not closure_path.is_file():
        raise FileNotFoundError(f"extraction closure not found: {closure_path}")
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    output_dir = closure_path.parent
    blob_path = output_dir / closure["blob_path"]
    frame_index_path = output_dir / closure["frame_index_path"]
    videos_csv_path = output_dir / closure["videos_csv_path"]

    videos = _load_videos_csv(videos_csv_path)
    video_base_by_id = {int(row["video_id"]): Path(row["file_name"]).stem for row in videos}
    if not video_base_by_id:
        raise ValueError(f"{videos_csv_path} has no rows to verify against")

    frame_index = np.load(frame_index_path, allow_pickle=False)
    mirror_frames_by_base = _index_mirror_frames_for_bases(mirror_root, set(video_base_by_id.values()))

    candidates: list[tuple[int, str, int, Path]] = []  # (video_id, video_base, source_frame_index, mirror_path)
    for video_id, video_base in video_base_by_id.items():
        mirror_hits = mirror_frames_by_base.get(video_base, [])
        if not mirror_hits:
            continue
        rows_for_video = frame_index[frame_index["video_id"] == video_id]
        extracted_indices = set(int(value) for value in rows_for_video["source_frame_index"])
        for mirror_path, mirror_frame_index in mirror_hits:
            if mirror_frame_index in extracted_indices:
                candidates.append((video_id, video_base, mirror_frame_index, mirror_path))

    if not candidates:
        raise ValueError(
            "no (video, frame_index) overlap found between this extraction and the mirror; "
            "nothing to verify -- check that the extraction's stride includes the mirror's "
            "every-10th-frame sampling and that the extracted videos exist in the mirror"
        )

    rng = random.Random(seed)
    sample_count = min(samples, len(candidates))
    sampled = rng.sample(candidates, sample_count)

    rows: list[dict[str, Any]] = []
    for video_id, video_base, mirror_frame_index, mirror_path in sampled:
        row_match = frame_index[
            (frame_index["video_id"] == video_id) & (frame_index["source_frame_index"] == mirror_frame_index)
        ]
        if len(row_match) != 1:
            raise RuntimeError(
                f"internal inconsistency: expected exactly one frame_index row for "
                f"video_id={video_id} source_frame_index={mirror_frame_index}, found {len(row_match)}"
            )
        record = row_match[0]
        extracted = read_frame_record(
            blob_path, byte_offset=int(record["byte_offset"]), byte_length=int(record["byte_length"])
        )
        extracted_64 = np.asarray(
            Image.fromarray(extracted, mode="RGB").resize(MIRROR_MOSAIC_SHAPE[::-1], resample=Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        with Image.open(mirror_path) as opened:
            mirror_array = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        if mirror_array.shape != (*MIRROR_MOSAIC_SHAPE, 3):
            raise ValueError(f"mirror image is not 64x64x3: {mirror_path} has shape {mirror_array.shape}")
        rows.append(
            {
                "video_base": video_base,
                "source_frame_index": mirror_frame_index,
                "mirror_path": str(mirror_path),
                "mean_absolute_difference": _mean_absolute_difference(extracted_64, mirror_array),
                "psnr_db": _psnr(extracted_64, mirror_array),
            }
        )

    mean_psnr = float(np.mean([row["psnr_db"] for row in rows if math.isfinite(row["psnr_db"])]))
    mean_mad = float(np.mean([row["mean_absolute_difference"] for row in rows]))
    return {
        "closure_path": str(closure_path),
        "mirror_root": str(mirror_root),
        "requested_samples": samples,
        "candidate_pool_size": len(candidates),
        "sampled": rows,
        "mean_psnr_db": mean_psnr,
        "mean_mean_absolute_difference": mean_mad,
    }


def _print_table(result: dict[str, Any]) -> None:
    print(f"candidate (video, frame) pairs available: {result['candidate_pool_size']}")
    print(f"{'video_base':<28}{'frame_idx':>10}{'MAD':>10}{'PSNR_dB':>10}")
    for row in result["sampled"]:
        print(
            f"{row['video_base']:<28}{row['source_frame_index']:>10}"
            f"{row['mean_absolute_difference']:>10.3f}{row['psnr_db']:>10.2f}"
        )
    print("-" * 58)
    print(f"{'mean':<28}{'':>10}{result['mean_mean_absolute_difference']:>10.3f}{result['mean_psnr_db']:>10.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE_PATH)
    parser.add_argument("--mirror-root", type=Path, default=DEFAULT_MIRROR_ROOT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--psnr-floor-db", type=float, default=DEFAULT_PSNR_FLOOR_DB)
    args = parser.parse_args()

    result = verify(
        closure_path=args.closure,
        mirror_root=args.mirror_root.resolve(),
        samples=args.samples,
        seed=args.seed,
    )
    _print_table(result)
    if result["mean_psnr_db"] < args.psnr_floor_db:
        print(
            f"FAIL: mean PSNR {result['mean_psnr_db']:.2f} dB is below the permissive floor "
            f"{args.psnr_floor_db:.2f} dB -- this indicates an index-mapping or channel-order "
            "bug, not ordinary resampling difference",
            file=sys.stderr,
        )
        return 1
    print(f"OK: mean PSNR {result['mean_psnr_db']:.2f} dB >= floor {args.psnr_floor_db:.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
