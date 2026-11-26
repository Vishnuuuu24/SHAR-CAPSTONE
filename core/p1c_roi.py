"""ROI study: pure, testable geometry and preprocessing for a person-crop ROI.

This module owns exactly the parts of the ROI ablation that do not need a
detector or a live blob: turning raw detections into one union box, cropping
a PIL image to that box, describing a plan frame with `RoiRecord`, choosing
which frames the study runs over, and reproducing the registered dataset's
resize/normalise path for a crop. `scripts/build_p1c_roi_cache.py` is the
only place that imports `ultralytics` or touches a GPU; this module never
does, so it is importable and testable anywhere, including this CPU session.

## Conservative ROI policy (owner-decided; see the ROI brief)

- detector: YOLO11m-pose, person class only, `imgsz=640, conf=0.25, iou=0.7`.
- ROI = the UNION box of every person detection above `conf_threshold`,
  expanded by `margin_fraction` of the union box's own width/height on each
  side, clamped to the frame.
- No detection above threshold -> no box (`None`); the caller falls back to
  the full frame and records `roi_fallback=1`.
- Crop -> resize to the registered resolution, bilinear -> registered
  mean/std -- the SAME code path `data.video_dataset.P1CVideoFrameDataset`
  uses for a full frame, via `NONE_AUGMENTATION.apply` followed by the same
  float32/255, mean/std arithmetic. A fallback row must therefore reproduce
  the registered feature cache's value for that frame up to GPU tolerance --
  this module does not check that itself (no backbone here), but is the
  reason `preprocess_crop` calls the identical two functions the dataset
  calls rather than re-implementing resize/normalise independently.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from core.p1c_video_sampling import sample_rows_per_video
from data.augmentation import NONE_AUGMENTATION
from data.p1c_loader import FramePreprocessing, _INTERPOLATION

#: `RoiRecord` field order, and the header `rows.csv` is written/read with.
ROI_RECORD_FIELDS = (
    "source_video_id",
    "source_frame_index",
    "target",
    "label",
    "person_count",
    "box_x0",
    "box_y0",
    "box_x1",
    "box_y1",
    "box_area_fraction",
    "roi_fallback",
    "max_person_conf",
)


@dataclass(frozen=True)
class RoiRecord:
    """One planned frame's ROI outcome -- one row of `rows.csv`.

    `box_x0`/`box_y0`/`box_x1`/`box_y1` are `-1` and `box_area_fraction` is
    `0.0` when `roi_fallback == 1` (no detection above threshold): there is
    no box to report, not a degenerate zero-area one.
    """

    source_video_id: str
    source_frame_index: int
    target: int
    label: str
    person_count: int
    box_x0: int
    box_y0: int
    box_x1: int
    box_y1: int
    box_area_fraction: float
    roi_fallback: int
    max_person_conf: float

    def __post_init__(self) -> None:
        if self.roi_fallback not in (0, 1):
            raise ValueError(f"roi_fallback must be 0 or 1; got {self.roi_fallback}")
        if self.person_count < 0:
            raise ValueError(f"person_count must be >= 0; got {self.person_count}")
        if self.roi_fallback == 1:
            if (self.box_x0, self.box_y0, self.box_x1, self.box_y1) != (-1, -1, -1, -1):
                raise ValueError("a fallback row must record box_x0=box_y0=box_x1=box_y1=-1")

    def to_csv_row(self) -> dict[str, str]:
        return {
            "source_video_id": self.source_video_id,
            "source_frame_index": str(self.source_frame_index),
            "target": str(self.target),
            "label": self.label,
            "person_count": str(self.person_count),
            "box_x0": str(self.box_x0),
            "box_y0": str(self.box_y0),
            "box_x1": str(self.box_x1),
            "box_y1": str(self.box_y1),
            "box_area_fraction": repr(self.box_area_fraction),
            "roi_fallback": str(self.roi_fallback),
            "max_person_conf": repr(self.max_person_conf),
        }

    @classmethod
    def from_csv_row(cls, row: Mapping[str, str]) -> "RoiRecord":
        return cls(
            source_video_id=row["source_video_id"],
            source_frame_index=int(row["source_frame_index"]),
            target=int(row["target"]),
            label=row["label"],
            person_count=int(row["person_count"]),
            box_x0=int(row["box_x0"]),
            box_y0=int(row["box_y0"]),
            box_x1=int(row["box_x1"]),
            box_y1=int(row["box_y1"]),
            box_area_fraction=float(row["box_area_fraction"]),
            roi_fallback=int(row["roi_fallback"]),
            max_person_conf=float(row["max_person_conf"]),
        )


def write_roi_records_csv(path: Path, records: Sequence[RoiRecord]) -> None:
    """Write `records` to `path` in `ROI_RECORD_FIELDS` order."""

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROI_RECORD_FIELDS))
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row())


def read_roi_records_csv(path: Path) -> list[RoiRecord]:
    """Inverse of `write_roi_records_csv`."""

    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [RoiRecord.from_csv_row(row) for row in csv.DictReader(handle)]


def union_person_box(
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    *,
    conf_threshold: float,
    frame_width: int,
    frame_height: int,
    margin_fraction: float = 0.10,
) -> tuple[int, int, int, int] | None:
    """The union box of every detection at or above `conf_threshold`, expanded and clamped.

    `boxes_xyxy` is `[N, 4]` (x0, y0, x1, y1) in frame pixel coordinates;
    `confidences` is `[N]`. Returns `None` when no detection clears the
    threshold -- the caller's signal to fall back to the full frame.

    The margin is a fraction of the UNION box's own width/height (not each
    individual detection's), added on every side, then the whole box is
    clamped to `[0, frame_width] x [0, frame_height]`.
    """

    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    if boxes_xyxy.shape[0] != confidences.shape[0]:
        raise ValueError(
            f"boxes_xyxy has {boxes_xyxy.shape[0]} rows but confidences has {confidences.shape[0]}"
        )
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"frame_width/frame_height must be positive; got {frame_width}x{frame_height}")
    if margin_fraction < 0:
        raise ValueError(f"margin_fraction must be >= 0; got {margin_fraction}")

    keep = confidences >= conf_threshold
    if not np.any(keep):
        return None

    kept_boxes = boxes_xyxy[keep]
    x0 = float(np.min(kept_boxes[:, 0]))
    y0 = float(np.min(kept_boxes[:, 1]))
    x1 = float(np.max(kept_boxes[:, 2]))
    y1 = float(np.max(kept_boxes[:, 3]))

    width = x1 - x0
    height = y1 - y0
    margin_x = margin_fraction * width
    margin_y = margin_fraction * height
    x0 -= margin_x
    x1 += margin_x
    y0 -= margin_y
    y1 += margin_y

    x0 = int(round(max(0.0, min(x0, float(frame_width)))))
    y0 = int(round(max(0.0, min(y0, float(frame_height)))))
    x1 = int(round(max(0.0, min(x1, float(frame_width)))))
    y1 = int(round(max(0.0, min(y1, float(frame_height)))))

    # Degenerate after rounding/clamping (e.g. a single-pixel detection at a
    # frame edge) -- widen by one pixel toward the frame interior rather than
    # hand a caller a zero-area crop.
    if x1 <= x0:
        x1 = min(frame_width, x0 + 1)
        x0 = max(0, x1 - 1)
    if y1 <= y0:
        y1 = min(frame_height, y0 + 1)
        y0 = max(0, y1 - 1)

    return (x0, y0, x1, y1)


def crop_to_box(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Crop `image` to `box` (x0, y0, x1, y1), pixel coordinates."""

    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"box must have positive width and height; got {box}")
    return image.crop((x0, y0, x1, y1))


def preprocess_crop(image: Image.Image, preprocessing: FramePreprocessing) -> torch.Tensor:
    """The registered dataset's resize+normalise path, applied to any PIL image.

    Numerically the same code as `data.video_dataset.P1CVideoFrameDataset.__getitem__`:
    `NONE_AUGMENTATION.apply` (a deterministic bilinear/registered-interpolation
    resize -- `seed` is irrelevant for the `none` policy but is threaded through
    for signature parity) followed by the same float32/255, mean/std arithmetic.
    A fallback (full-frame) row run through this function must reproduce the
    registered cache's feature for that frame up to GPU tolerance -- see the
    module docstring and K-28.
    """

    if image.mode != preprocessing.color_mode:
        image = image.convert(preprocessing.color_mode)
    resized = NONE_AUGMENTATION.apply(
        image,
        resize_width=preprocessing.resize_width,
        resize_height=preprocessing.resize_height,
        resample=_INTERPOLATION[preprocessing.interpolation],
        seed=0,
    )
    pixels = np.asarray(resized, dtype=np.float32)
    expected = (preprocessing.resize_height, preprocessing.resize_width, 3)
    if pixels.shape != expected:
        raise ValueError(f"preprocessed crop has shape {pixels.shape}, expected {expected}")
    tensor = torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).div_(255.0)
    mean = torch.tensor(preprocessing.mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(preprocessing.std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def select_plan_rows(dataset: Any, remaining: set[tuple[str, int]]) -> list[tuple[int, str, int, int]]:
    """`(dataset index, video, frame, target)` for plan pairs still in `remaining`.

    Frame-granular metadata resolution, shared (rule of two) by
    `scripts/build_p1c_roi_cache.py` and `scripts/check_p1c_roi_detector.py`:
    both need to find WHERE (which fold/split dataset) a planned `(video,
    frame)` pair lives, without decoding a single JPEG to do it. Reads a
    `P1CVideoFrameDataset`'s metadata arrays directly
    (`_videos`/`_video_ref`/`_frame_index`/`_target`), and removes each pair
    from `remaining` as soon as it is found, so a pair present in more than
    one fold's manifest (K-fold splits) is resolved exactly once regardless
    of which dataset is checked first.
    """

    videos = dataset._videos  # noqa: SLF001 -- one-off extraction, metadata only
    ref, frames, targets = dataset._video_ref, dataset._frame_index, dataset._target  # noqa: SLF001
    out: list[tuple[int, str, int, int]] = []
    for index in range(len(dataset)):
        video = videos[int(ref[index])]["source_video_id"]
        frame = int(frames[index])
        key = (video, frame)
        if key in remaining:
            out.append((index, video, frame, int(targets[index])))
            remaining.discard(key)
    return out


class RawFramePlanDataset(Dataset):
    """Raw decoded RGB uint8 frames for a selected subset of one fold-split dataset.

    Deliberately bypasses `P1CVideoFrameDataset.__getitem__` (which resizes
    and normalises for the registered full-frame pipeline): the ROI study
    needs the undecoded pixel array so the detector sees the same raw frame
    the registered pipeline would have resized, and so `preprocess_crop` --
    not this dataset -- owns resize/normalise for whatever box is chosen.

    Shared (rule of two) by `scripts/build_p1c_roi_cache.py` and
    `scripts/check_p1c_roi_detector.py`: both wrap a resolved
    `(dataset_index, video, frame, target, global_index)` selection over one
    `P1CVideoFrameDataset` and decode only those rows through a multi-worker
    `DataLoader`, never the dataset's full manifest.
    """

    def __init__(self, dataset: Any, selected: list[tuple[int, str, int, int, int]]) -> None:
        self._dataset = dataset
        self._selected = selected  # (dataset_index, video, frame, target, global_index)

    def __len__(self) -> int:
        return len(self._selected)

    def __getitem__(self, i: int) -> dict[str, Any]:
        dataset_index, video, frame, target, global_index = self._selected[i]
        blob_row = int(self._dataset._blob_rows[dataset_index])  # noqa: SLF001
        payload = self._dataset._reader.payload(blob_row)  # noqa: SLF001
        with Image.open(io.BytesIO(payload)) as opened:
            image = opened.convert("RGB")
            pixels = np.asarray(image, dtype=np.uint8)
        return {
            "pixels": pixels,
            "source_video_id": video,
            "source_frame_index": frame,
            "target": target,
            "global_index": global_index,
        }


def _read_rows_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plan_roi_frames(
    cache_roots: Sequence[Path],
    *,
    frames_per_video: int = 8,
    sampling_seed: int = 0,
) -> list[tuple[str, int]]:
    """`(source_video_id, source_frame_index)` pairs the ROI study needs pixels for.

    For each root in `cache_roots` (a fold's feature-cache directory holding
    `train/rows.csv` and `validation/rows.csv`): TRAIN rows are thinned with
    `core.p1c_video_sampling.sample_rows_per_video(rows, frames_per_video,
    seed=sampling_seed)` -- the exact call the head-training recipe this
    study reproduces makes -- and every VALIDATION row is kept, full and
    untouched, matching `scripts/run_p1c_sampled_head.py`'s own split
    treatment. The two sets are unioned per fold, then deduplicated by
    `(source_video_id, source_frame_index)` ACROSS folds (the same frame can
    appear as validation in one fold and train in another), and the result
    is returned sorted.
    """

    if frames_per_video < 1:
        raise ValueError(f"frames_per_video must be >= 1; got {frames_per_video}")

    wanted: set[tuple[str, int]] = set()
    for root in cache_roots:
        root = Path(root)
        train_rows = _read_rows_csv(root / "train" / "rows.csv")
        validation_rows = _read_rows_csv(root / "validation" / "rows.csv")

        keep = sample_rows_per_video(train_rows, frames_per_video, seed=sampling_seed)
        for index in keep:
            row = train_rows[int(index)]
            wanted.add((row["source_video_id"], int(row["source_frame_index"])))
        for row in validation_rows:
            wanted.add((row["source_video_id"], int(row["source_frame_index"])))

    return sorted(wanted)
