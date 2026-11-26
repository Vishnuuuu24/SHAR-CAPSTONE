#!/usr/bin/env python3
"""Pre-flight sanity check for the ROI detector (GPU, pod-only; run BEFORE build_p1c_roi_cache).

Two independent checks over the same ROI plan `core.p1c_roi.plan_roi_frames`
produces (never the detector's own output as evaluation ground truth for
itself -- docs/PROJECT_RULES.md rule 2):

(a) Person detector sanity. 200 frames from the plan, stratified 14-15 per
    class (deterministic, `numpy.random.default_rng(0)`), run through the
    registered YOLO11m-pose detector. Writes one annotated contact sheet per
    class (drawn boxes, PIL only) and a `summary.json` of per-class detection
    rate, person-count histogram, median box height, and median union-box
    area fraction -- so a human can look at the sheets and the numbers before
    a multi-hour cache build runs on a bad detector configuration.

(b) Weapon-vocabulary false-positive / hit-rate check. The pinned
    YOLO-World v2 s checkpoint already has `set_classes` baked in (names
    `{0:'person',1:'gun',2:'pistol',3:'rifle',4:'knife'}`) -- `set_classes` is
    NOT called again here, which would need CLIP and is unnecessary. For
    every plan video with label Shooting AND an official temporal interval:
    at most 150 frames inside the interval and 150 outside, deterministic
    (evenly spaced -- the same `core.p1c_video_sampling.sample_rows_per_video`
    deterministic-branch algorithm, reused directly rather than
    reimplemented). Detection rate for gun/pistol/rifle at conf>=0.10 and
    conf>=0.25, split inside vs outside, per video and pooled. A 500-frame
    non-Shooting control sample (same deterministic RNG) gives a
    false-positive rate. Writes `weapon_check.json` (which records the exact
    per-video/control caps used) and a contact sheet of the top-24
    highest-confidence weapon detections.

## Two-pass, bounded-memory design (K-15)

The full ROI plan is ~82,679 frames; decoding all of them into a RAM dict
(~230 KB/frame raw RGB) would be ~19 GB in a 16 GiB pod -- and hours of
single-threaded decode, since none of that pixel data is actually needed for
either check.

- **Pass 1 (metadata only, no JPEG decode).** `core.p1c_roi.select_plan_rows`
  walks each of the four fold datasets' arrays to resolve every plan pair to
  `(dataset, dataset_index, target, label)`. This is what lets both checks'
  frame *selections* below be made from labels/intervals alone.
- **Selection (metadata only).** The ~200 stratified person-check frames, the
  capped per-video Shooting inside/outside frames, and the 500-frame control
  sample are all chosen from pass 1's resolved metadata -- a few thousand
  frames total, never the whole plan.
- **Pass 2 (decode only the selection).** `core.p1c_roi.RawFramePlanDataset`
  + a `DataLoader(num_workers=6)`, grouped by which fold dataset each
  selected frame resolved to in pass 1 -- the same decode path
  `scripts/build_p1c_roi_cache.py` uses (rule of two). Only the selected
  frames' pixels are ever held in memory.
- **Batched detection.** Both detectors are called on lists of frames
  (`_BATCH_SIZE`), not one frame per `predict()` call.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_background_mask import load_intervals  # noqa: E402
from core.p1c_roi import RawFramePlanDataset, plan_roi_frames, select_plan_rows, union_person_box  # noqa: E402
from core.p1c_video_sampling import sample_rows_per_video  # noqa: E402
from scripts.build_p1c_roi_cache import CONFIGS, DEFAULT_CACHE_ROOTS  # noqa: E402
from scripts.diagnose_p1c_event_separability import ANNOTATION  # noqa: E402
from scripts.run_p1c_frame_baseline import _load_config, _make_datasets  # noqa: E402

DEFAULT_PERSON_WEIGHTS = Path(".cache/weights/yolo11m-pose-29b17eaf.pt")
DEFAULT_WEAPON_WEIGHTS = Path(".cache/weights/yolov8s-worldv2-weapons-vocab.pt")
WEAPON_NAMES = {0: "person", 1: "gun", 2: "pistol", 3: "rifle", 4: "knife"}
WEAPON_CLASS_IDS = (1, 2, 3, 4)
PERSON_CHECK_FRAMES = 200
NON_SHOOTING_CONTROL_FRAMES = 500
SHOOTING_PER_VIDEO_CAP = 150
DETECTOR_BATCH_SIZE = 32
DECODE_NUM_WORKERS = 6


def _resolve_plan_metadata(
    plan_pairs: list[tuple[str, int]], *, extract_closure_path: Path | None
) -> dict[tuple[str, int], tuple[Any, int, int, str]]:
    """Pass 1: `(video, frame) -> (dataset, dataset_index, target, label)` for every plan pair.

    No JPEG is decoded here -- `core.p1c_roi.select_plan_rows` reads only the
    dataset's metadata arrays. The returned `dataset` objects are kept (they
    are single-digit-MiB metadata, not the pixel blob -- see
    `data/video_dataset.py`) so pass 2 can decode straight from them without
    re-resolving.
    """

    remaining = set(plan_pairs)
    resolved: dict[tuple[str, int], tuple[Any, int, int, str]] = {}
    for config_path in CONFIGS:
        if not remaining:
            break
        _config, protocol, _run_class = _load_config(Path(config_path))
        if extract_closure_path is not None:
            protocol["data"]["video_frames"]["extract_closure_path"] = str(extract_closure_path)
        class_names = protocol["model"]["class_names"]
        train, validation = _make_datasets(protocol, seed=0)
        for dataset in (train, validation):
            if not remaining:
                break
            for dataset_index, video, frame, target in select_plan_rows(dataset, remaining):
                resolved[(video, frame)] = (dataset, dataset_index, target, class_names[target])
    if remaining:
        raise RuntimeError(f"{len(remaining)} requested frames were not found in any fold manifest")
    return resolved


def _decode_selected_frames(
    selected: list[tuple[str, int]],
    resolved: dict[tuple[str, int], tuple[Any, int, int, str]],
) -> dict[tuple[str, int], np.ndarray]:
    """Pass 2: decode ONLY `selected` frames' pixels, grouped by their resolved fold dataset.

    Uses the same `RawFramePlanDataset` + multi-worker `DataLoader` decode
    path `scripts/build_p1c_roi_cache.py` uses (rule of two), so this never
    touches the ~82,679-frame full plan's pixels -- only the (at most a few
    thousand) frames a check actually needs.
    """

    by_dataset: dict[int, tuple[Any, list[tuple[int, str, int, int, int]]]] = {}
    for global_index, pair in enumerate(selected):
        dataset, dataset_index, target, _label = resolved[pair]
        video, frame = pair
        key = id(dataset)
        if key not in by_dataset:
            by_dataset[key] = (dataset, [])
        by_dataset[key][1].append((dataset_index, video, frame, target, global_index))

    pixels: dict[tuple[str, int], np.ndarray] = {}
    for dataset, entries in by_dataset.values():
        loader = DataLoader(
            RawFramePlanDataset(dataset, entries), batch_size=DETECTOR_BATCH_SIZE, shuffle=False,
            num_workers=DECODE_NUM_WORKERS,
        )
        for batch in loader:
            for pixel_array, video, frame in zip(
                batch["pixels"].numpy(), batch["source_video_id"], batch["source_frame_index"].tolist(),
            ):
                pixels[(video, frame)] = pixel_array
    return pixels


def _stratified_sample(
    plan_pairs: list[tuple[str, int]], labels: dict[tuple[str, int], str], *, per_class_target: int, rng: np.random.Generator
) -> list[tuple[str, int]]:
    """~`per_class_target` frames total, spread 14-15 per class, deterministic under `rng`."""

    by_label: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for pair in plan_pairs:
        by_label[labels[pair]].append(pair)
    class_names = sorted(by_label)
    base = per_class_target // len(class_names)
    extra = per_class_target - base * len(class_names)
    sampled: list[tuple[str, int]] = []
    for i, label in enumerate(class_names):
        count = base + (1 if i < extra else 0)
        pool = by_label[label]
        count = min(count, len(pool))
        indices = rng.choice(len(pool), size=count, replace=False)
        sampled.extend(pool[j] for j in sorted(indices.tolist()))
    return sampled


def _deterministic_evenly_spaced_subset(pairs: list[tuple[str, int]], cap: int) -> list[tuple[str, int]]:
    """Up to `cap` frames from one video's already-frame-ordered `pairs`, evenly spaced.

    Reuses `core.p1c_video_sampling.sample_rows_per_video`'s deterministic
    (`seed=None`) branch directly rather than reimplementing segment-based
    coverage a second time (rule of two): `pairs` is treated as one video's
    full row list, and the function's own "keep everything if already <=
    budget" behaviour means this is a no-op when `len(pairs) <= cap`.
    """

    if not pairs:
        return []
    rows = [{"source_dataset": "roi_detector_check", "source_video_id": pairs[0][0]} for _ in pairs]
    keep = sample_rows_per_video(rows, cap, seed=None)
    return [pairs[i] for i in keep]


def _select_weapon_check_frames(
    plan_pairs: list[tuple[str, int]],
    labels: dict[tuple[str, int], str],
    intervals: dict[str, list[tuple[int, int]]],
    *,
    per_video_cap: int,
    control_frames: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], dict[str, Any]]:
    """`(selected_shooting_pairs, control_pairs, selection_report)` -- metadata only.

    `selected_shooting_pairs` is capped to `per_video_cap` inside-interval and
    `per_video_cap` outside-interval frames per annotated Shooting video
    (`_deterministic_evenly_spaced_subset`); `control_pairs` is a
    deterministic `rng`-drawn non-Shooting sample.
    """

    by_video: dict[str, dict[str, list[tuple[str, int]]]] = defaultdict(lambda: {"inside": [], "outside": []})
    for pair in plan_pairs:
        video, frame = pair
        if labels[pair] != "Shooting" or video not in intervals:
            continue
        spans = intervals[video]
        location = "inside" if any(start <= frame <= end for start, end in spans) else "outside"
        by_video[video][location].append(pair)

    selected: list[tuple[str, int]] = []
    per_video_report: dict[str, Any] = {}
    for video, buckets in sorted(by_video.items()):
        inside_selected = _deterministic_evenly_spaced_subset(buckets["inside"], per_video_cap)
        outside_selected = _deterministic_evenly_spaced_subset(buckets["outside"], per_video_cap)
        selected.extend(inside_selected)
        selected.extend(outside_selected)
        per_video_report[video] = {
            "inside_available": len(buckets["inside"]), "inside_selected": len(inside_selected),
            "outside_available": len(buckets["outside"]), "outside_selected": len(outside_selected),
        }

    non_shooting_pairs = [pair for pair in plan_pairs if labels[pair] != "Shooting"]
    control_indices = rng.choice(
        len(non_shooting_pairs), size=min(control_frames, len(non_shooting_pairs)), replace=False,
    )
    control_pairs = [non_shooting_pairs[i] for i in sorted(control_indices.tolist())]

    selection_report = {
        "per_video_inside_cap": per_video_cap,
        "per_video_outside_cap": per_video_cap,
        "non_shooting_control_cap": control_frames,
        "videos_checked": len(by_video),
        "shooting_frames_selected": len(selected),
        "non_shooting_control_selected": len(control_pairs),
        "per_video": per_video_report,
    }
    return selected, control_pairs, selection_report


def _draw_boxes(image: Image.Image, boxes: np.ndarray, confidences: np.ndarray, *, color: str = "red") -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for box, conf in zip(boxes, confidences):
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0 + 2, max(0, y0 - 12)), f"{conf:.2f}", fill=color)
    return canvas


def _contact_sheet(tiles: list[Image.Image], *, columns: int = 5, rows: int = 3) -> Image.Image:
    if not tiles:
        return Image.new("RGB", (320, 240), color=(0, 0, 0))
    width, height = tiles[0].size
    sheet = Image.new("RGB", (width * columns, height * rows), color=(0, 0, 0))
    for i, tile in enumerate(tiles[: columns * rows]):
        row, col = divmod(i, columns)
        sheet.paste(tile, (col * width, row * height))
    return sheet


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _batched_predict(model: Any, frames: list[np.ndarray], *, device: str, **kwargs: Any) -> list[Any]:
    """`model.predict` over `frames` in `DETECTOR_BATCH_SIZE` chunks; returns one result per frame."""

    results: list[Any] = []
    for chunk in _chunked(frames, DETECTOR_BATCH_SIZE):
        bgr_list = [np.ascontiguousarray(frame[..., ::-1]) for frame in chunk]
        results.extend(model.predict(bgr_list, device=device, verbose=False, **kwargs))
    return results


def _run_person_check(
    sample: list[tuple[str, int]],
    labels: dict[tuple[str, int], str],
    pixels: dict[tuple[str, int], np.ndarray],
    *,
    weights: Path,
    conf: float,
    margin_fraction: float,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Detect over the ALREADY-SELECTED (metadata-only, `main`-computed) `sample`.

    Takes `sample` as given rather than re-deriving it here, for the same
    reason `_run_weapon_check` does: one selection pass in `main`, decoded
    exactly once in pass 2 -- re-selecting here would be dead work and a
    second place the decoded set and the detected set could drift apart.
    """

    import ultralytics

    model = ultralytics.YOLO(str(weights))
    frames = [pixels[pair] for pair in sample]
    detection_results = _batched_predict(
        model, frames, device=device, imgsz=640, conf=conf, iou=0.7, classes=[0],
    )

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sheets_dir = output_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    for pair, frame, result in zip(sample, frames, detection_results):
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None and len(result.boxes) else np.zeros((0, 4))
        confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None and len(result.boxes) else np.zeros((0,))
        height, width = frame.shape[:2]
        union = union_person_box(
            boxes, confidences, conf_threshold=conf, frame_width=width, frame_height=height,
            margin_fraction=margin_fraction,
        )
        label = labels[pair]
        by_label[label].append({
            "pair": pair,
            "person_count": int(np.sum(confidences >= conf)),
            "box_heights": [float(b[3] - b[1]) for b in boxes[confidences >= conf]],
            "union_area_fraction": (
                ((union[2] - union[0]) * (union[3] - union[1])) / (width * height) if union else None
            ),
            "tile": _draw_boxes(Image.fromarray(frame, mode="RGB"), boxes[confidences >= conf], confidences[confidences >= conf]),
        })

    summary: dict[str, Any] = {}
    for label, rows in sorted(by_label.items()):
        person_counts = [row["person_count"] for row in rows]
        heights = [h for row in rows for h in row["box_heights"]]
        areas = [row["union_area_fraction"] for row in rows if row["union_area_fraction"] is not None]
        summary[label] = {
            "frames_checked": len(rows),
            "detection_rate": sum(1 for c in person_counts if c > 0) / len(rows) if rows else None,
            "person_count_histogram": dict(sorted(Counter(person_counts).items())),
            "median_box_height_px": statistics.median(heights) if heights else None,
            "median_union_box_area_fraction": statistics.median(areas) if areas else None,
        }
        sheet = _contact_sheet([row["tile"] for row in rows])
        sheet.save(sheets_dir / f"{label}.jpg", quality=90)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _run_weapon_check(
    shooting_pairs: list[tuple[str, int]],
    control_pairs: list[tuple[str, int]],
    selection_report: dict[str, Any],
    pixels: dict[tuple[str, int], np.ndarray],
    intervals: dict[str, list[tuple[int, int]]],
    *,
    weights: Path,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Detect over the ALREADY-SELECTED (metadata-only, `main`-computed) frame sets.

    Takes `shooting_pairs`/`control_pairs`/`selection_report` as given rather
    than re-deriving them here: `main` makes exactly one selection pass and
    decodes exactly that selection (pass 2) -- re-selecting inside this
    function, even under the same seed, would be dead work and a second
    place the two selections could silently drift apart.
    """

    import ultralytics

    model = ultralytics.YOLO(str(weights))  # set_classes already baked into this checkpoint; do NOT call it again

    def _detect_all(pairs: list[tuple[str, int]]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        frames = [pixels[pair] for pair in pairs]
        results = _batched_predict(model, frames, device=device, imgsz=640, conf=0.10, iou=0.7)
        detections = []
        for result in results:
            if result.boxes is None or not len(result.boxes):
                detections.append((np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int64)))
            else:
                detections.append((
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(np.int64),
                ))
        return detections

    per_video: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "inside": {"0.10": 0, "0.25": 0, "frames": 0},
        "outside": {"0.10": 0, "0.25": 0, "frames": 0},
    })
    pooled = {
        "inside": {"0.10": 0, "0.25": 0, "frames": 0},
        "outside": {"0.10": 0, "0.25": 0, "frames": 0},
    }
    top_detections: list[dict[str, Any]] = []

    shooting_detections = _detect_all(shooting_pairs)
    for (video, frame_index), (boxes, confidences, classes) in zip(shooting_pairs, shooting_detections):
        weapon_mask = np.isin(classes, WEAPON_CLASS_IDS)
        weapon_confidences = confidences[weapon_mask]
        weapon_boxes = boxes[weapon_mask]
        weapon_classes = classes[weapon_mask]
        spans = intervals[video]
        location = "inside" if any(start <= frame_index <= end for start, end in spans) else "outside"

        per_video[video][location]["frames"] += 1
        pooled[location]["frames"] += 1
        for threshold_key, threshold in (("0.10", 0.10), ("0.25", 0.25)):
            if np.any(weapon_confidences >= threshold):
                per_video[video][location][threshold_key] += 1
                pooled[location][threshold_key] += 1

        for box, confidence, class_id in zip(weapon_boxes, weapon_confidences, weapon_classes):
            top_detections.append({
                "video": video, "frame": frame_index, "conf": float(confidence),
                "class": WEAPON_NAMES[int(class_id)], "box": [float(v) for v in box],
            })

    false_positive_frames = 0
    control_detections = _detect_all(control_pairs)
    for _boxes, confidences, classes in control_detections:
        if np.any(np.isin(classes, WEAPON_CLASS_IDS) & (confidences >= 0.25)):
            false_positive_frames += 1

    def _rates(counts: dict[str, Any]) -> dict[str, Any]:
        frames = counts["frames"]
        return {
            "frames": frames,
            "detection_rate_conf_0.10": counts["0.10"] / frames if frames else None,
            "detection_rate_conf_0.25": counts["0.25"] / frames if frames else None,
        }

    result = {
        "selection": selection_report,
        "pooled": {location: _rates(counts) for location, counts in pooled.items()},
        "per_video": {video: {loc: _rates(counts) for loc, counts in v.items()} for video, v in per_video.items()},
        "non_shooting_control": {
            "frames": len(control_pairs),
            "false_positive_rate_conf_0.25": (
                false_positive_frames / len(control_pairs) if control_pairs else None
            ),
        },
    }
    (output_dir / "weapon_check.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    top_detections.sort(key=lambda d: d["conf"], reverse=True)
    tiles = []
    for detection in top_detections[:24]:
        frame = pixels[(detection["video"], detection["frame"])]
        tile = _draw_boxes(
            Image.fromarray(frame, mode="RGB"),
            np.asarray([detection["box"]]), np.asarray([detection["conf"]]), color="yellow",
        )
        tiles.append(tile)
    _contact_sheet(tiles, columns=6, rows=4).save(output_dir / "weapon_top24.jpg", quality=90)
    return result


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=Path("results/p1c/p1c-roi-detector-check"))
    parser.add_argument("--person-weights", type=Path, default=DEFAULT_PERSON_WEIGHTS)
    parser.add_argument("--weapon-weights", type=Path, default=DEFAULT_WEAPON_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--margin-fraction", type=float, default=0.10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--extract-closure-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug: cap the plan before sampling")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)

    plan = plan_roi_frames(DEFAULT_CACHE_ROOTS, frames_per_video=8, sampling_seed=0)
    if args.limit:
        plan = plan[: args.limit]
    print(f"plan: {len(plan)} frames", flush=True)

    intervals = load_intervals(ANNOTATION)

    # Pass 1: metadata only, whole plan, no pixel decode.
    resolved = _resolve_plan_metadata(plan, extract_closure_path=args.extract_closure_path)
    labels = {pair: resolved[pair][3] for pair in plan}

    # Selection: metadata only, a few thousand frames total -- computed exactly
    # once here, then handed unchanged to both decode (pass 2) and detection,
    # so the frames decoded and the frames scored can never drift apart.
    person_rng = np.random.default_rng(0)
    person_sample = _stratified_sample(plan, labels, per_class_target=PERSON_CHECK_FRAMES, rng=person_rng)
    weapon_rng = np.random.default_rng(0)
    shooting_pairs, control_pairs, selection_report = _select_weapon_check_frames(
        plan, labels, intervals, per_video_cap=SHOOTING_PER_VIDEO_CAP,
        control_frames=NON_SHOOTING_CONTROL_FRAMES, rng=weapon_rng,
    )
    decode_set = sorted(set(person_sample) | set(shooting_pairs) | set(control_pairs))
    print(f"decoding {len(decode_set)} of {len(plan)} plan frames (bounded selection, K-15)", flush=True)

    # Pass 2: decode only the selection.
    pixels = _decode_selected_frames(decode_set, resolved)

    person_summary = _run_person_check(
        person_sample, labels, pixels, weights=args.person_weights, conf=args.conf,
        margin_fraction=args.margin_fraction, device=args.device, output_dir=args.output_root,
    )
    print(f"person check: {len(person_summary)} classes summarised", flush=True)

    weapon_result = _run_weapon_check(
        shooting_pairs, control_pairs, selection_report, pixels, intervals,
        weights=args.weapon_weights, device=args.device, output_dir=args.output_root,
    )
    print(
        f"weapon check: pooled inside {weapon_result['pooled']['inside']}, "
        f"outside {weapon_result['pooled']['outside']}, "
        f"control FP rate {weapon_result['non_shooting_control']['false_positive_rate_conf_0.25']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
