#!/usr/bin/env python3
"""RAFT optical-flow descriptor cache for the flow arm (GPU, pod-only).

Every temporal remedy on record so far (`docs/REMEDIES_TRIED.md`) operated in
DINOv2's FEATURE space or borrowed Kinetics motion features -- explicit
pixel-level optical flow has never been computed. This script writes it: for
the SAME 82,679-frame study set `scripts/build_p1c_patch_cache.py` already
covers (`core.p1c_roi.plan_roi_frames` over the four DINOv2 fold caches,
`frames_per_video=8`, `sampling_seed=0`), it computes one RAFT flow field per
plan frame against the next blob row of the same video, pools it with
`core.p1c_flow.flow_descriptor`, and streams the result to disk. Mirrors
`build_p1c_patch_cache.py`'s plan/decode/stream/finalize shape closely (same
plan, same `RawFramePlanDataset`-style decode, same `open_memmap` streaming,
same `finalize_streamed_array`/`write_manifest`, same `--finalize-only`
recovery path -- read K-30 before touching finalisation).

The model, pinned
------------------
`torchvision.models.optical_flow.raft_small(weights=Raft_Small_Weights.C_T_V2)`,
0.99M params. The checkpoint is loaded from the local copy
(`.cache/weights/raft-small-C_T_V2-01064c6d.pth`) rather than the
`weights=` download path, so this never needs network access from inside a
pod; its sha256 is checked against `RAFT_WEIGHTS_SHA256` before use.
`Raft_Small_Weights.C_T_V2.transforms()` maps uint8 -> [-1, 1]; flow is
computed at NATIVE 240x320 resolution (both dimensions divide by 8, RAFT's
required stride, so no resize is needed or applied). RAFT returns a list of
iterative refinements; only the LAST element (`out[-1]`, `[B, 2, 240, 320]`)
is used.

The frame pairs -- the one honest compromise
-----------------------------------------------
The extraction blob holds every 5th source frame, so two CONSECUTIVE blob
rows of one video are 5 source frames apart (~0.167 s at 30 fps). Truly
adjacent frames would mean decoding all 1,950 MP4s again, which this project
cannot afford before the review. So each plan frame is paired with the NEXT
blob row of the SAME video (`pair_source_frame_gap: 5` in the manifest, plus
an explicit statement that this is 0.167 s-spaced flow, never claimed as
frame-adjacent). A video's LAST row has no "next" row: it is paired
BACKWARDS with the previous row instead, and the resulting flow field is
negated (multiplied by -1) so its sign still means "the direction this frame
is moving in time", not "the direction the previous frame moved into this
one" -- recorded as `paired_backwards: 1` for that row. A video with only one
row in this dataset has no partner at all: it gets an all-zero descriptor and
`flow_unavailable: 1`, never a fabricated flow value.

Output: `results/p1c/p1c-flow-cache/features.float16.npy` `[82679, 162]`,
`rows.csv`, `manifest.json`.

    OMP_NUM_THREADS=2 taskset -c 0-1 ./.venv/bin/python scripts/build_p1c_flow_cache.py \\
        --device cuda:0 --num-workers 6 --batch-size 32

Not runnable/tested in this session: no GPU, no pods (see docs/PROJECT_RULES.md and the
brief this script implements against). Written to mirror the patch-cache
script's proven shape as closely as this problem allows; treat as unverified
until it is actually launched and its self-checks (row/array count agreement,
non-empty-tail check) pass on a pod.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_extraction_io import finalize_streamed_array, write_manifest
from core.p1c_flow import (
    DEFAULT_GRID,
    DEFAULT_ORIENTATION_BINS,
    DESCRIPTOR_LENGTH,
    descriptor_field_names,
    flow_descriptor_batch,
)
from core.p1c_readiness import sha256_file
from core.p1c_roi import plan_roi_frames
from scripts.run_p1c_frame_baseline import _load_config, _make_datasets

CONFIGS = [f"configs/experiments/p1c/P1C-VIDEO-DINOV2-B2-AR-FOLD{f}.yaml" for f in (1, 2, 3, 4)]
DEFAULT_CACHE_ROOTS = [Path(f"results/p1c/p1c-dinov2-feature-cache-ar-fold{f}") for f in (1, 2, 3, 4)]
FRAMES_PER_VIDEO = 8
SAMPLING_SEED = 0
#: Blob rows are every 5th source frame -- two consecutive blob rows of one
#: video are this many source frames, ~0.167 s at 30 fps, apart.
PAIR_SOURCE_FRAME_GAP = 5
ROWS_CSV_FIELDS = (
    "source_video_id", "source_frame_index", "pair_frame_index", "paired_backwards", "flow_unavailable",
)
RAFT_WEIGHTS_PATH = REPO_ROOT / ".cache/weights/raft-small-C_T_V2-01064c6d.pth"
RAFT_WEIGHTS_SHA256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
#: Written after the GPU pass writes its last row, before finalisation --
#: K-30's lesson applied here too: `--finalize-only` refuses to run without it,
#: rather than trusting a possibly-truncated array.
_EXTRACTION_MARKER = "extraction_complete.marker"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=Path("results/p1c/p1c-flow-cache"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--extract-closure-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug: cap the plan to the first N frames")
    parser.add_argument(
        "--finalize-only", action="store_true",
        help="skip the GPU pass and only hash/verify/manifest an already-written output root "
             "(recovery path for K-30: the extraction succeeded and finalisation was OOM-killed)",
    )
    return parser.parse_args(argv)


def _video_frame_order(dataset: Any) -> dict[str, list[tuple[int, int]]]:
    """`video_id -> [(source_frame_index, dataset_index), ...]` sorted ascending by frame index.

    Every row this dataset holds for a video, not only the plan's sampled
    subset -- resolving "the next blob row of the same video" needs the
    video's FULL row set, exactly as `core.p1c_roi.plan_roi_frames` reads
    metadata directly rather than decoding pixels to answer this question.
    """

    videos = dataset._videos  # noqa: SLF001 -- metadata only, one-off extraction
    ref, frames = dataset._video_ref, dataset._frame_index  # noqa: SLF001
    order: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index in range(len(dataset)):
        video = videos[int(ref[index])]["source_video_id"]
        order[video].append((int(frames[index]), index))
    for video in order:
        order[video].sort(key=lambda item: item[0])
    return order


def _resolve_pairs(
    dataset: Any, remaining: set[tuple[str, int]], plan_index: dict[tuple[str, int], int],
) -> list[dict[str, Any]]:
    """Resolved pair records for every plan frame this dataset can answer for.

    Each record carries dataset-local indices (`source_dataset_index`,
    `pair_dataset_index` or `None`) plus the bookkeeping fields
    (`paired_backwards`, `flow_unavailable`) `rows.csv` needs. Removes each
    resolved `(video, frame)` from `remaining`, mirroring
    `core.p1c_roi.select_plan_rows`'s single-resolution-per-pair contract.
    """

    order_by_video = _video_frame_order(dataset)
    targets = dataset._target  # noqa: SLF001
    resolved: list[dict[str, Any]] = []
    for video, order in order_by_video.items():
        for position, (frame_index, dataset_index) in enumerate(order):
            key = (video, frame_index)
            if key not in remaining:
                continue
            if len(order) == 1:
                record = {
                    "source_video_id": video,
                    "source_frame_index": frame_index,
                    "pair_frame_index": frame_index,
                    "source_dataset_index": dataset_index,
                    "pair_dataset_index": None,
                    "paired_backwards": 0,
                    "flow_unavailable": 1,
                }
            elif position == len(order) - 1:
                prev_frame_index, prev_dataset_index = order[position - 1]
                record = {
                    "source_video_id": video,
                    "source_frame_index": frame_index,
                    "pair_frame_index": prev_frame_index,
                    "source_dataset_index": dataset_index,
                    "pair_dataset_index": prev_dataset_index,
                    "paired_backwards": 1,
                    "flow_unavailable": 0,
                }
            else:
                next_frame_index, next_dataset_index = order[position + 1]
                record = {
                    "source_video_id": video,
                    "source_frame_index": frame_index,
                    "pair_frame_index": next_frame_index,
                    "source_dataset_index": dataset_index,
                    "pair_dataset_index": next_dataset_index,
                    "paired_backwards": 0,
                    "flow_unavailable": 0,
                }
            record["target"] = int(targets[dataset_index])
            record["global_index"] = plan_index[key]
            resolved.append(record)
            remaining.discard(key)
    return resolved


def _decode_frame(dataset: Any, dataset_index: int) -> np.ndarray:
    """One raw RGB uint8 `[H, W, 3]` frame, decoded straight from the blob (no resize/normalise)."""

    import io

    from PIL import Image

    blob_row = int(dataset._blob_rows[dataset_index])  # noqa: SLF001
    payload = dataset._reader.payload(blob_row)  # noqa: SLF001
    with Image.open(io.BytesIO(payload)) as opened:
        return np.asarray(opened.convert("RGB"), dtype=np.uint8)


def _load_raft(device: torch.device):
    """RAFT-small (C_T_V2) loaded from the local checkpoint, plus its preprocessing transform."""

    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

    weight_sha = sha256_file(RAFT_WEIGHTS_PATH)
    if weight_sha != RAFT_WEIGHTS_SHA256:
        raise RuntimeError(
            f"{RAFT_WEIGHTS_PATH} sha256 {weight_sha} does not match the pinned "
            f"{RAFT_WEIGHTS_SHA256} -- refusing to run flow with an unverified checkpoint"
        )
    model = raft_small(weights=None)
    state_dict = torch.load(RAFT_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    transforms = Raft_Small_Weights.C_T_V2.transforms()
    return model, transforms, weight_sha


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.finalize_only:
        return _finalize_only(args)
    device = torch.device(args.device)

    plan = plan_roi_frames(DEFAULT_CACHE_ROOTS, frames_per_video=FRAMES_PER_VIDEO, sampling_seed=SAMPLING_SEED)
    if args.limit:
        plan = plan[: args.limit]
    total = len(plan)
    print(f"plan: {total} frames", flush=True)
    remaining: set[tuple[str, int]] = set(plan)
    plan_index = {pair: i for i, pair in enumerate(plan)}

    args.output_root.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(
        args.output_root / "features.float16.npy", mode="w+", dtype=np.float16, shape=(total, DESCRIPTOR_LENGTH),
    )
    rows: list[dict[str, Any]] = [None] * total  # type: ignore[list-item]

    per_dataset = []
    for config_path in CONFIGS:
        _config, protocol, _run_class = _load_config(Path(config_path))
        if args.extract_closure_path is not None:
            protocol["data"]["video_frames"]["extract_closure_path"] = str(args.extract_closure_path)
        train, validation = _make_datasets(protocol, seed=0)
        per_dataset.extend([train, validation])

    # One pass per dataset, exactly as `build_p1c_patch_cache.py` resolves its
    # plan before decoding: each resolved record keeps its owning dataset
    # alongside it, since a plan (video, frame) pair and its "next blob row"
    # partner both live in whichever fold-split dataset the video belongs to.
    resolved_by_dataset: list[tuple[Any, dict[str, Any]]] = []
    for dataset in per_dataset:
        if not remaining:
            break
        for record in _resolve_pairs(dataset, remaining, plan_index):
            resolved_by_dataset.append((dataset, record))
    if remaining:
        raise RuntimeError(f"{len(remaining)} planned frames not found in any fold manifest")

    raft_model, transforms, weight_sha = _load_raft(device)
    backwards_count = 0
    unavailable_count = 0

    with torch.no_grad():
        batch: list[tuple[Any, dict[str, Any]]] = []

        def _flush(batch_items: list[tuple[Any, dict[str, Any]]]) -> None:
            nonlocal backwards_count, unavailable_count
            active = [(dataset, record) for dataset, record in batch_items if not record["flow_unavailable"]]
            for dataset, record in [(d, r) for d, r in batch_items if r["flow_unavailable"]]:
                features[record["global_index"]] = np.zeros(DESCRIPTOR_LENGTH, dtype=np.float16)
                rows[record["global_index"]] = {name: record[name] for name in ROWS_CSV_FIELDS}
                unavailable_count += 1
            if not active:
                return
            image1 = np.stack([_decode_frame(dataset, record["source_dataset_index"]) for dataset, record in active])
            image2 = np.stack([_decode_frame(dataset, record["pair_dataset_index"]) for dataset, record in active])
            tensor1 = torch.from_numpy(image1).permute(0, 3, 1, 2).contiguous()
            tensor2 = torch.from_numpy(image2).permute(0, 3, 1, 2).contiguous()
            tensor1, tensor2 = transforms(tensor1, tensor2)
            flows = raft_model(tensor1.to(device), tensor2.to(device))[-1]
            flow_np = flows.to(torch.float32).cpu().numpy()
            for i, (_dataset, record) in enumerate(active):
                field = flow_np[i]
                if record["paired_backwards"]:
                    field = -field
                    backwards_count += 1
                descriptor = flow_descriptor_batch(field[np.newaxis, ...])[0]
                features[record["global_index"]] = descriptor.astype(np.float16)
                rows[record["global_index"]] = {name: record[name] for name in ROWS_CSV_FIELDS}

        for dataset, record in resolved_by_dataset:
            batch.append((dataset, record))
            if len(batch) >= args.batch_size:
                _flush(batch)
                batch = []
        if batch:
            _flush(batch)

    assert all(row is not None for row in rows), "not every plan row was written"

    with (args.output_root / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROWS_CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    (args.output_root / _EXTRACTION_MARKER).write_text("ok\n", encoding="utf-8")
    shape, features_sha = finalize_streamed_array(args.output_root, features)
    write_manifest(args.output_root, _manifest_body(
        shape=shape, features_sha=features_sha, weight_sha=weight_sha, row_count=total,
        backwards_count=backwards_count, unavailable_count=unavailable_count,
    ))
    print(f"wrote {args.output_root}: {total} rows, {backwards_count} backwards-paired, {unavailable_count} unavailable", flush=True)
    return 0


def _manifest_body(
    *, shape: list[int], features_sha: str, weight_sha: str, row_count: int, backwards_count: int, unavailable_count: int,
) -> dict[str, Any]:
    return {
        "model": "torchvision.models.optical_flow.raft_small",
        "raft_weights": "Raft_Small_Weights.C_T_V2",
        "raft_weights_path": str(RAFT_WEIGHTS_PATH),
        "raft_weights_sha256": weight_sha,
        "torchvision_version": __import__("torchvision").__version__,
        "resolution": "native_240x320_no_resize",
        "pair_source_frame_gap": PAIR_SOURCE_FRAME_GAP,
        "pair_spacing_statement": (
            "Consecutive blob rows of one video are 5 source frames apart (~0.167 s at 30 fps). "
            "This cache is 0.167 s-spaced flow, NOT frame-adjacent flow -- re-extracting truly "
            "adjacent frames would mean decoding all 1,950 source MP4s again."
        ),
        "descriptor_grid": list(DEFAULT_GRID),
        "descriptor_orientation_bins": DEFAULT_ORIENTATION_BINS,
        "descriptor_length": DESCRIPTOR_LENGTH,
        "descriptor_field_names": descriptor_field_names(),
        "frames_per_video": FRAMES_PER_VIDEO,
        "sampling_seed": SAMPLING_SEED,
        "row_count": row_count,
        "shape": shape,
        "dtype": "float16",
        "features_sha256": features_sha,
        "backwards_paired_rows": backwards_count,
        "flow_unavailable_rows": unavailable_count,
    }


def _finalize_only(args: argparse.Namespace) -> int:
    """Hash, verify and manifest an output root whose arrays are already complete (K-30 recovery)."""

    root = args.output_root
    marker = root / _EXTRACTION_MARKER
    if not marker.exists():
        raise RuntimeError(
            f"{marker} is missing -- the GPU pass never signalled completion; "
            "refusing to finalize a possibly-truncated array"
        )
    with (root / "rows.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    features = np.load(root / "features.float16.npy", mmap_mode="r")
    if len(rows) != features.shape[0]:
        raise RuntimeError(f"incomplete root: {len(rows)} rows, {features.shape[0]} feature rows")

    features_sha = sha256_file(root / "features.float16.npy")
    raft_sha = sha256_file(RAFT_WEIGHTS_PATH)
    backwards_count = sum(1 for row in rows if int(row["paired_backwards"]))
    unavailable_count = sum(1 for row in rows if int(row["flow_unavailable"]))
    write_manifest(root, _manifest_body(
        shape=list(features.shape), features_sha=features_sha, weight_sha=raft_sha, row_count=len(rows),
        backwards_count=backwards_count, unavailable_count=unavailable_count,
    ) | {"finalized_separately": "K-30-style recovery: GPU pass completed, finalisation run via --finalize-only."})
    print(f"finalize-only: wrote manifest for {root}, {len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
