#!/usr/bin/env python3
"""Build the ROI-study feature cache: person-crop DINOv2 features (GPU, pod-only).

Single-factor ablation for the ROI study (see the ROI brief): the registered
RESULT 2 pipeline -- DINOv2 ViT-B/14-reg pooled CLS++mean-patch 1536-D, the
same frames `core.p1c_roi.plan_roi_frames` selects (8 train frames/video,
`sample_rows_per_video(rows, 8, seed=0)`, plus every validation row, unioned
and deduplicated across the four AR folds) -- with ONE change: the DINOv2
input is a person-ROI crop instead of the full frame.

Conservative ROI policy (owner-decided, do not change here):

- detector: YOLO11m-pose, person class only, `imgsz=640, conf=0.25, iou=0.7,
  classes=[0]`, on the raw 320x240 RGB frame.
- ROI = the UNION box of every person detection at or above `--conf`,
  expanded by `--margin-fraction` of the union box's own width/height on
  each side, clamped to the frame (`core.p1c_roi.union_person_box`).
- no detection above threshold -> full-frame fallback, `roi_fallback=1`.
- crop -> resize to the registered 224x224 bilinear -> registered mean/std ->
  DINOv2 `pooled_features` (`core.p1c_roi.preprocess_crop`, the identical
  code path `data.video_dataset.P1CVideoFrameDataset.__getitem__` uses for a
  full frame -- this is what lets a fallback row be checked against the
  already-registered feature cache, see `fallback_check.json` below).

Frames are resolved to pixels by building each fold's registered
`P1CVideoFrameDataset` (via `scripts.run_p1c_frame_baseline._make_datasets`,
same as `scripts/extract_p1c_shooting_tokens.py`) and matching planned
`(video, frame)` pairs against its metadata -- a video can appear in more
than one fold's manifest (K-fold splits), so a pair is resolved from
whichever fold/split dataset contains it, and each pair is only decoded once
(`core.p1c_roi.select_plan_rows` removes a pair from the still-wanted set as soon as it
is found).

Streamed the same way `extract_p1c_shooting_tokens.py` streams: one
pre-allocated fp16 memmap sized from the plan up front, one row written at a
time, never accumulated in RAM (16 GiB pod, K-15).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_extraction_io import finalize_streamed_array, write_manifest  # noqa: E402
from core.p1c_readiness import sha256_file  # noqa: E402
from core.p1c_roi import (  # noqa: E402
    RawFramePlanDataset,
    RoiRecord,
    crop_to_box,
    plan_roi_frames,
    preprocess_crop,
    select_plan_rows,
    union_person_box,
    write_roi_records_csv,
)
from data.p1c_loader import FramePreprocessing  # noqa: E402
from scripts.build_p1c_feature_cache import _load_backbone  # noqa: E402
from scripts.run_p1c_frame_baseline import _load_config, _make_datasets  # noqa: E402

CONFIGS = [f"configs/experiments/p1c/P1C-VIDEO-DINOV2-B2-AR-FOLD{fold}.yaml" for fold in (1, 2, 3, 4)]
DEFAULT_CACHE_ROOTS = [
    Path(f"results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}") for fold in (1, 2, 3, 4)
]
DEFAULT_DETECTOR_WEIGHTS = Path(".cache/weights/yolo11m-pose-29b17eaf.pt")
DEFAULT_DETECTOR_SHA256 = "29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2"
DEFAULT_DETECTOR_LICENCE = "AGPL-3.0"
FRAMES_PER_VIDEO = 8
SAMPLING_SEED = 0

# Frame-granular metadata resolution (`select_plan_rows`) and the raw-decode
# `Dataset` wrapper (`RawFramePlanDataset`) live in `core.p1c_roi` -- rule of
# two -- because `scripts/check_p1c_roi_detector.py` needs exactly the same
# two pieces to resolve and decode its own (much smaller) frame subset.


def _fallback_check(records: list[RoiRecord], features: np.ndarray, cache_roots: list[Path]) -> dict[str, Any]:
    """Max/median |delta| between a fallback row's ROI feature and the registered cache's.

    A fallback row's crop IS the full frame, run through the identical
    `preprocess_crop` code path the registered cache used -- so it should
    reproduce that cache's vector up to GPU nondeterminism (K-28, median
    ~2e-3 abs). Looked up across all 8 fold/split roots because a video can
    be train in one fold and validation in another.
    """

    lookup: dict[tuple[str, int], tuple[Path, int]] = {}
    for root in cache_roots:
        for split in ("train", "validation"):
            rows_path = root / split / "rows.csv"
            if not rows_path.exists():
                continue
            import csv

            with rows_path.open() as handle:
                for i, row in enumerate(csv.DictReader(handle)):
                    lookup.setdefault(
                        (row["source_video_id"], int(row["source_frame_index"])), (root / split, i)
                    )

    arrays: dict[Path, np.ndarray] = {}
    deltas: list[float] = []
    checked = 0
    for record, feature in zip(records, features):
        if record.roi_fallback != 1:
            continue
        hit = lookup.get((record.source_video_id, record.source_frame_index))
        if hit is None:
            continue
        root, row_index = hit
        if root not in arrays:
            arrays[root] = np.load(root / "features.float16.npy", mmap_mode="r")
        cached = np.asarray(arrays[root][row_index], dtype=np.float32)
        deltas.append(float(np.abs(feature.astype(np.float32) - cached).max()))
        checked += 1
    return {
        "rows_checked": checked,
        "max_abs_delta": max(deltas) if deltas else None,
        "median_abs_delta": statistics.median(deltas) if deltas else None,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=Path("results/p1c/p1c-roi-cache"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--extract-closure-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug: cap the plan to the first N frames")
    parser.add_argument("--detector-weights", type=Path, default=DEFAULT_DETECTOR_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--margin-fraction", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    device = torch.device(args.device)

    # Pass 1 -- plan from cache metadata only, no pixel decode yet.
    plan = plan_roi_frames(DEFAULT_CACHE_ROOTS, frames_per_video=FRAMES_PER_VIDEO, sampling_seed=SAMPLING_SEED)
    if args.limit:
        plan = plan[: args.limit]
    total = len(plan)
    print(f"plan: {total} frames", flush=True)
    remaining: set[tuple[str, int]] = set(plan)
    plan_index = {pair: i for i, pair in enumerate(plan)}

    args.output_root.mkdir(parents=True, exist_ok=True)
    records: list[RoiRecord | None] = [None] * total
    detections: list[dict[str, Any] | None] = [None] * total
    features_out = np.lib.format.open_memmap(
        args.output_root / "features.float16.npy", mode="w+", dtype=np.float16, shape=(total, 1536)
    )

    # Pass 2 -- resolve plan pairs to pixels fold by fold, dedup across folds
    # (K-fold splits mean a video can appear in more than one fold's
    # manifest; a pair is decoded from whichever dataset finds it first and
    # then removed from `remaining`).
    protocol = preprocessing = class_names = backbone_block = None
    per_config: list[tuple[str, Any, Any]] = []
    for config_path in CONFIGS:
        _config, this_protocol, _run_class = _load_config(Path(config_path))
        if args.extract_closure_path is not None:
            this_protocol["data"]["video_frames"]["extract_closure_path"] = str(args.extract_closure_path)
        train, validation = _make_datasets(this_protocol, seed=0)
        per_config.append((config_path, train, validation))
        if protocol is None:
            protocol = this_protocol
            preprocessing = FramePreprocessing.from_config(protocol["preprocessing"])
            class_names = protocol["model"]["class_names"]
            backbone_block = protocol["backbone"]
    assert protocol is not None

    detector_weights = args.detector_weights
    detector_sha256 = sha256_file(detector_weights)
    if detector_sha256 != DEFAULT_DETECTOR_SHA256 and detector_weights == DEFAULT_DETECTOR_WEIGHTS:
        raise ValueError(
            f"{detector_weights} sha256 {detector_sha256} does not match the registered "
            f"{DEFAULT_DETECTOR_SHA256}"
        )

    import ultralytics  # local: this is the one script permitted to import it

    pose_model = ultralytics.YOLO(str(detector_weights))
    backbone = _load_backbone(backbone_block).to(device).eval()

    written = 0
    with torch.no_grad():
        for config_path, train, validation in per_config:
            for dataset in (train, validation):
                selected_raw = select_plan_rows(dataset, remaining)
                if not selected_raw:
                    continue
                selected = [
                    (index, video, frame, target, plan_index[(video, frame)])
                    for index, video, frame, target in selected_raw
                ]
                raw_dataset = RawFramePlanDataset(dataset, selected)
                loader = DataLoader(
                    raw_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                )
                for batch in loader:
                    pixel_batch = batch["pixels"].numpy()
                    global_indices = batch["global_index"].tolist()
                    videos_batch = batch["source_video_id"]
                    frames_batch = batch["source_frame_index"].tolist()
                    targets_batch = batch["target"].tolist()

                    # ultralytics wants contiguous BGR uint8 arrays; a negative-stride view is not.
                    bgr_list = [np.ascontiguousarray(frame[..., ::-1]) for frame in pixel_batch]
                    results = pose_model.predict(
                        bgr_list, imgsz=640, conf=args.conf, iou=0.7, classes=[0], verbose=False,
                        device=args.device, half=(device.type == "cuda"),
                    )

                    crops = []
                    for k, result in enumerate(results):
                        frame = pixel_batch[k]
                        height, width = frame.shape[:2]
                        if result.boxes is not None and len(result.boxes):
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                        else:
                            boxes = np.zeros((0, 4))
                            confidences = np.zeros((0,))
                        if result.keypoints is not None and len(result.keypoints):
                            keypoints = result.keypoints.data.cpu().numpy()
                        else:
                            keypoints = np.zeros((0, 17, 3))

                        box = union_person_box(
                            boxes, confidences, conf_threshold=args.conf,
                            frame_width=width, frame_height=height,
                            margin_fraction=args.margin_fraction,
                        )
                        image = Image.fromarray(frame, mode="RGB")
                        kept = confidences >= args.conf
                        person_count = int(np.sum(kept))
                        max_conf = float(confidences[kept].max()) if person_count else 0.0

                        if box is None:
                            crop_image = image
                            box_out = (-1, -1, -1, -1)
                            area_fraction = 0.0
                            fallback = 1
                        else:
                            crop_image = crop_to_box(image, box)
                            box_out = box
                            area_fraction = ((box[2] - box[0]) * (box[3] - box[1])) / (width * height)
                            fallback = 0

                        crops.append(preprocess_crop(crop_image, preprocessing))

                        global_index = global_indices[k]
                        video = videos_batch[k]
                        frame_index = frames_batch[k]
                        target = targets_batch[k]
                        records[global_index] = RoiRecord(
                            source_video_id=video, source_frame_index=frame_index, target=target,
                            label=class_names[target], person_count=person_count,
                            box_x0=box_out[0], box_y0=box_out[1], box_x1=box_out[2], box_y1=box_out[3],
                            box_area_fraction=area_fraction, roi_fallback=fallback, max_person_conf=max_conf,
                        )
                        detections[global_index] = {
                            "source_video_id": video,
                            "source_frame_index": frame_index,
                            "persons": [
                                {
                                    "box": [round(float(v), 1) for v in boxes[j]],
                                    "conf": round(float(confidences[j]), 3),
                                    "keypoints": [
                                        [round(float(v), 1) for v in point] for point in keypoints[j]
                                    ] if j < len(keypoints) else [],
                                }
                                for j in range(len(boxes))
                                if confidences[j] >= args.conf
                            ],
                        }

                    batch_tensor = torch.stack(crops).to(device)
                    features = backbone.pooled_features(batch_tensor).to(torch.float16).cpu().numpy()
                    for k, global_index in enumerate(global_indices):
                        features_out[global_index] = features[k]
                    written += len(global_indices)
                    if written % 1024 < len(global_indices):
                        print(f"  {written}/{total} frames", flush=True)

    if remaining:
        raise RuntimeError(
            f"{len(remaining)} planned (video, frame) pairs were not found in any of the four "
            f"AR fold manifests: {sorted(remaining)[:5]}..."
        )
    assert all(record is not None for record in records), "not every plan row was written"

    write_roi_records_csv(args.output_root / "rows.csv", records)  # type: ignore[arg-type]
    with (args.output_root / "detections.jsonl").open("w", encoding="utf-8") as handle:
        for detection in detections:
            handle.write(json.dumps(detection) + "\n")

    shape, features_sha256 = finalize_streamed_array(args.output_root, features_out)
    fallback_rows = sum(1 for record in records if record.roi_fallback == 1)  # type: ignore[union-attr]
    fallback_check = _fallback_check(records, np.load(args.output_root / "features.float16.npy", mmap_mode="r"), DEFAULT_CACHE_ROOTS)  # type: ignore[arg-type]
    (args.output_root / "fallback_check.json").write_text(
        json.dumps(fallback_check, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    write_manifest(args.output_root, {
        "detector_weights": str(detector_weights),
        "detector_sha256": detector_sha256,
        "detector_licence": DEFAULT_DETECTOR_LICENCE,
        "ultralytics_version": ultralytics.__version__,
        "policy": {
            "conf": args.conf, "iou": 0.7, "imgsz": 640, "classes": [0],
            "margin_fraction": args.margin_fraction,
        },
        "backbone_family": backbone_block["family"],
        "backbone_weight_sha256": backbone_block["weights"]["sha256"],
        "preprocessing": json.loads(json.dumps({
            "resize_height": preprocessing.resize_height, "resize_width": preprocessing.resize_width,
            "interpolation": preprocessing.interpolation, "color_mode": preprocessing.color_mode,
            "mean": list(preprocessing.mean), "std": list(preprocessing.std),
        })),
        "row_count": total,
        "fallback_rows": fallback_rows,
        "fallback_fraction": fallback_rows / total if total else None,
        "shape": shape,
        "dtype": "float16",
        "features_sha256": features_sha256,
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "batch_size": args.batch_size,
    })
    print(f"wrote {args.output_root}: {total} rows, {fallback_rows} fallback ({fallback_rows / total:.4f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
