"""Pixel-content evidence for the Kaggle-frame-index-equals-source-index claim.

``data/ucf_intervals.py`` assumes the Kaggle mirror's ``<video>_<frame>.png``
numbers are unscaled original-video frame indices, with no every-10th-frame
conversion.  That assumption is foundational to every interval-membership
decision in the official-interval-derived event-only protocol.  Before this
module, the only corroboration in the repository was three faculty-preview
examples that checked numeric-key *membership* in the manifest, never pixel
*content* against the decoded original video.  This module compares actual
pixel content at a systematic, deterministic sample of
``(source_video_id, source_frame_index)`` pairs; ``scripts/verify_frame_alignment.py``
drives it and writes an immutable evidence report.

The Kaggle mirror's exact resize/crop policy from source resolution down to
64x64 is not documented anywhere in this repository, and the two resolutions
are not the same aspect ratio, so a single assumed resize policy would itself
be an invented default.  Both plausible policies are therefore explicit,
named, required arguments rather than a silent choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from eval.image_quality import psnr, ssim


ResizePolicy = Literal["stretch_to_target", "center_crop_then_resize"]
RESIZE_POLICIES: tuple[ResizePolicy, ...] = ("stretch_to_target", "center_crop_then_resize")
INTERPOLATIONS = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


@dataclass(frozen=True)
class FrameAlignmentResult:
    source_video_id: str
    source_frame_index: int
    kaggle_relative_path: str
    resize_policy: str
    interpolation: str
    source_resolution: tuple[int, int]
    kaggle_resolution: tuple[int, int]
    ssim: float
    psnr: float


def read_kaggle_png(path: Path) -> np.ndarray:
    """Decode one Kaggle mirror PNG into canonical float32 HWC RGB in [0,1]."""

    if not path.is_file():
        raise FileNotFoundError(f"Kaggle frame is missing: {path}")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"OpenCV could not decode Kaggle frame: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.astype(np.float32) / np.float32(255.0))


def read_source_video_frame(path: Path, source_frame_index: int) -> np.ndarray:
    """Decode one exact frame from the official source video as float32 RGB [0,1]."""

    if not path.is_file():
        raise FileNotFoundError(f"official source video is missing: {path}")
    if isinstance(source_frame_index, bool) or not isinstance(source_frame_index, int) or source_frame_index < 0:
        raise ValueError("source_frame_index must be a non-negative integer")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV cannot open official source video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0 and source_frame_index >= frame_count:
            raise ValueError(
                f"frame index {source_frame_index} is outside {path.name}'s {frame_count}-frame range"
            )
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
        ok, bgr = capture.read()
    finally:
        capture.release()
    if not ok or bgr is None:
        raise RuntimeError(f"OpenCV could not read frame {source_frame_index} from {path}")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"decoded frame is not a three-channel image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.astype(np.float32) / np.float32(255.0))


def resize_to_match(
    source: np.ndarray,
    target_shape: tuple[int, int],
    *,
    policy: ResizePolicy,
    interpolation: str,
) -> np.ndarray:
    """Resize ``source`` onto ``target_shape`` (height, width) under one named policy."""

    if policy not in RESIZE_POLICIES:
        raise ValueError(f"policy must be one of {RESIZE_POLICIES}")
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {sorted(INTERPOLATIONS)}")
    target_height, target_width = target_shape
    if target_height < 1 or target_width < 1:
        raise ValueError("target_shape must be positive")
    flag = INTERPOLATIONS[interpolation]
    if policy == "stretch_to_target":
        resized = cv2.resize(source, (target_width, target_height), interpolation=flag)
    else:
        source_height, source_width = source.shape[:2]
        target_aspect = target_width / target_height
        source_aspect = source_width / source_height
        if source_aspect > target_aspect:
            crop_width = max(1, int(round(source_height * target_aspect)))
            left = (source_width - crop_width) // 2
            cropped = source[:, left : left + crop_width]
        else:
            crop_height = max(1, int(round(source_width / target_aspect)))
            top = (source_height - crop_height) // 2
            cropped = source[top : top + crop_height, :]
        resized = cv2.resize(np.ascontiguousarray(cropped), (target_width, target_height), interpolation=flag)
    return np.ascontiguousarray(np.clip(resized, 0.0, 1.0).astype(np.float32))


def compare_frame(
    kaggle_png_path: Path,
    source_video_path: Path,
    *,
    source_video_id: str,
    source_frame_index: int,
    resize_policy: ResizePolicy,
    interpolation: str,
) -> FrameAlignmentResult:
    """Compare one Kaggle PNG against the decoded original-video frame it claims to be."""

    kaggle = read_kaggle_png(kaggle_png_path)
    source = read_source_video_frame(source_video_path, source_frame_index)
    resized_source = resize_to_match(
        source, (kaggle.shape[0], kaggle.shape[1]), policy=resize_policy, interpolation=interpolation
    )
    win_size = min(7, kaggle.shape[0], kaggle.shape[1])
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)
    return FrameAlignmentResult(
        source_video_id=source_video_id,
        source_frame_index=source_frame_index,
        kaggle_relative_path=str(kaggle_png_path),
        resize_policy=resize_policy,
        interpolation=interpolation,
        source_resolution=(int(source.shape[1]), int(source.shape[0])),
        kaggle_resolution=(int(kaggle.shape[1]), int(kaggle.shape[0])),
        ssim=ssim(
            kaggle,
            resized_source,
            data_range=1.0,
            channel_axis=-1,
            win_size=win_size,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            k1=0.01,
            k2=0.03,
        ),
        psnr=psnr(kaggle, resized_source, data_range=1.0),
    )


def best_policy_result(
    kaggle_png_path: Path,
    source_video_path: Path,
    *,
    source_video_id: str,
    source_frame_index: int,
    interpolation: str,
) -> FrameAlignmentResult:
    """Return whichever registered resize policy scores the higher SSIM.

    The Kaggle mirror's own resize/crop convention is undocumented, so no
    single policy can be assumed correct in advance; trying both and keeping
    the better match is itself the evidence about which (if either) the
    mirror actually used.
    """

    results = [
        compare_frame(
            kaggle_png_path,
            source_video_path,
            source_video_id=source_video_id,
            source_frame_index=source_frame_index,
            resize_policy=policy,
            interpolation=interpolation,
        )
        for policy in RESIZE_POLICIES
    ]
    return max(results, key=lambda result: result.ssim)
