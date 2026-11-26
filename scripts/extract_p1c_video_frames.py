#!/usr/bin/env python3
"""Extract frames directly from the official UCF-Crime MP4 corpus (D-34/D-35).

Background. D-33/`scripts/pack_p1c_frames.py` packs the *Kaggle* 64x64 PNG
mirror. D-34 replaces that pixel source for future P1C/P2A work: this script
decodes the official 320x240 MP4 corpus once, at native resolution, into a
single JPEG-payload packed blob, and leaves training resolution as a
separate, later, loader-side decision (downscaling happens in a dataset
class, not here). D-35 fixes the temporal stride at 5 (~6 fps at the
corpus's nominal 30 fps) as a deliberately over-provisioned hedge: stride is
expensive to widen later (it requires re-decoding the full 97.7 GiB corpus)
while resolution is cheap to narrow later (pure loader configuration), so
this tool over-provisions the expensive-to-revise parameter and leaves the
cheap one alone.

This script is a *sibling*, not a replacement, of `scripts/pack_p1c_frames.py`
and `data.p1c_loader.P1CPackedFrameDataset`. Those remain the paused
Kaggle-sourced path and are untouched by this file. This script's output is a
different artifact family (variable-length JPEG records, not fixed
`i * 12288` raw offsets) read by a different, not-yet-written loader.

Scope and non-scope (read this before pointing `--directories` at anything):
this script extracts whatever MP4 directories it is told to, and records
each video's immediate parent directory name verbatim as `source_directory`
provenance in `videos.csv`. It does NOT decide, assert, or hardcode which
directories are the SHAR `Normal` class, and it does NOT assign SHAR labels
at all. That mapping -- including whether `Training_Normal_Videos_Anomaly`,
`Testing_Normal_Videos_Anomaly`, and `z_Normal_Videos_event` are one class,
several, or partially out of scope -- is exactly blocker B-001's surviving
"official-Normal-source verification" obligation and blocker B-002's open
corpus-terms/completeness intake, and is a later, separately-gated decision.

JPEG re-encoding is a second lossy step on top of already-lossy H.264, and
that is a deliberate, accepted trade (D-34): above 64x64 the accelerator,
not the loader, is the binding compute cost, so spending decode-time CPU to
buy a ~10x smaller-than-raw payload is free in the budget that matters. JPEG
quality 95 is the registered setting; it is not re-derived here.

Decoder constraints (measured on this host, not chosen for convenience):
only OpenCV (cv2 4.13.0) is available -- no system ffmpeg/ffprobe, no PyAV,
no decord. Frames are read with sequential `cv2.VideoCapture.read()` calls
and kept every `stride`-th decoded frame; `CAP_PROP_POS_FRAMES` seeking is
deliberately never used because it is unreliable on H.264 and would defeat
the sequential-read performance that makes a full-corpus decode cheap
(measured 3,667 frames/s per core for 320x240 H.264 demux+decode).
`CAP_PROP_FRAME_COUNT`/`CAP_PROP_FPS` container metadata is recorded as
*reported* only -- it is frequently wrong for H.264 -- and is never used to
decide how many frames to read; `decoded_frame_count` (frames actually read
to EOF) is the ground truth.

Color order: cv2 decodes frames as BGR. This script converts every kept
frame to RGB with `cv2.cvtColor(..., COLOR_BGR2RGB)` before JPEG-encoding it
with Pillow (`PIL.Image.fromarray(frame, mode="RGB")`), not with
`cv2.imencode`. This is not a stylistic choice: `cv2.imencode`/`cv2.imwrite`
always treat their input array as BGR (the universal OpenCV I/O convention)
and internally swap it to true RGB before the JPEG DCT/YCbCr stage, so
feeding it an already-RGB array would silently swap red and blue in the
bytes actually written to disk. Pillow performs no such implicit swap: an
array passed with `mode="RGB"` is taken at face value. `frame_index.npy` /
`extract_closure.json` therefore both promise `color_order: "RGB"` truthfully.

Fail-closed geometry. Every video establishes its own frame shape from its
first *kept* frame. If a later kept frame in the same video differs from
that video's own established shape, the run aborts unconditionally -- a
video changing resolution mid-stream is stream corruption, not a corpus
heterogeneity question, and there is no flag to override it. Separately, if
a video's established shape differs from the corpus-wide expectation
(320x240, i.e. `expected_image_shape` rows=240, cols=320, channels=3), the
default is to abort naming the file; `--allow-heterogeneous-geometry`
downgrades that to a per-video record (`decode_status=heterogeneous_geometry`
in `videos.csv`, and a `heterogeneous_geometry_videos` list in the closure)
and continues.

Resumability is at video granularity, not frame granularity, because video
sizes range from a 14 MiB median to an 8 GiB max -- per-frame checkpointing
would be the wrong unit and per-video checkpointing is cheap relative to one
video's own decode time. `extract_state.json` records which videos (by
position in one deterministic, sorted discovery order) are already fully
written; a resumed invocation skips those videos without reopening them and
appends to the existing blob/CSV/intermediate-index artifacts. Resuming with
a different source root, directory set, stride, JPEG quality, expected
shape, or heterogeneous-geometry policy than the state file was started with
is refused (fail-closed) rather than silently reinterpreted.

`extract_closure.json` is written exactly once, only after every discovered
video (subject to `--limit-videos`, see below) is fully processed, and
refuses to silently overwrite an existing closure -- use a fresh
`--output-dir` for a different extraction.

Two independent "do less this call" knobs exist and must not be confused:
`--limit-videos` shrinks the *discovered corpus itself* to its first N
videos in sorted order, so a small `--limit-videos` run is a genuinely
complete, closeable extraction over a subset (this is what a smoke test
should use). `--max-new-videos` caps how many *not-yet-done* videos a single
invocation will process before returning, without shrinking the discovered
corpus or writing a closure if work remains; it exists for resuming a large
run incrementally (and for deterministically simulating an interruption in
tests) and is not in the brief that specified this script -- it is this
implementation's own addition, documented here as such.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.progress import BatchProgress, log_phase  # noqa: E402
from data.inventory import sha256_file  # noqa: E402

EXTRACTOR_CODE_VERSION = "p1c-video-extract-v1"
DEFAULT_STRIDE = 5
DEFAULT_JPEG_QUALITY = 95
DEFAULT_EXPECTED_HEIGHT = 240
DEFAULT_EXPECTED_WIDTH = 320
DEFAULT_SOURCE_ROOT = REPO_ROOT / "Datasets" / "UCF Crime Official" / "raw" / "UCF_Crimes" / "Videos"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "p1c" / "p1c-video-frames"
VIDEO_CSV_FIELDS = [
    "video_id",
    "source_directory",
    "file_name",
    "relative_path",
    "file_bytes",
    "file_sha256",
    "reported_frame_count",
    "reported_fps",
    "decoded_frame_count",
    "kept_frame_count",
    "frame_height",
    "frame_width",
    "first_row_index",
    "decode_status",
]
_STATE_CONFIG_FIELDS = {
    "source_root",
    "directories",
    "stride",
    "jpeg_quality",
    "expected_height",
    "expected_width",
    "allow_heterogeneous_geometry",
    "video_list_sha256",
    "video_count",
}
_FRAME_INDEX_DTYPE = np.dtype(
    [
        ("byte_offset", "<i8"),
        ("byte_length", "<i8"),
        ("video_id", "<i8"),
        ("source_frame_index", "<i8"),
    ]
)
_FFMPEG_BUILD_LINE_PREFIX = "FFMPEG:"


@dataclass(frozen=True)
class VideoRef:
    source_directory: str
    file_name: str
    absolute_path: Path

    @property
    def relative_path(self) -> str:
        return f"{self.source_directory}/{self.file_name}"


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def cv2_ffmpeg_build_line() -> str:
    """Return the single `FFMPEG:` line from `cv2.getBuildInformation()`.

    This records *whether and how* the cv2 binary this run used was built
    against FFmpeg (e.g. "FFMPEG:  YES"), not a system ffmpeg binary version
    -- there is no system ffmpeg/ffprobe on this host, and cv2's video
    decoding uses its own linked build, not a subprocess.
    """

    for line in cv2.getBuildInformation().splitlines():
        stripped = line.strip()
        if stripped.startswith(_FFMPEG_BUILD_LINE_PREFIX):
            return stripped
    raise RuntimeError("cv2.getBuildInformation() has no FFMPEG: line; cannot record decoder provenance")


def encode_jpeg_rgb(frame_rgb: np.ndarray, *, quality: int) -> bytes:
    """Encode an already-RGB-ordered uint8 array as a JPEG payload."""

    if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError(f"encode_jpeg_rgb requires an HxWx3 uint8 array, got shape={frame_rgb.shape} dtype={frame_rgb.dtype}")
    buffer = io.BytesIO()
    Image.fromarray(frame_rgb, mode="RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def decode_jpeg_bytes(payload: bytes) -> np.ndarray:
    """Decode a JPEG payload back to an RGB uint8 array."""

    with Image.open(io.BytesIO(payload)) as opened:
        return np.asarray(opened.convert("RGB"), dtype=np.uint8)


def read_frame_record(blob_path: Path, *, byte_offset: int, byte_length: int) -> np.ndarray:
    """Read and decode one JPEG record directly out of the packed blob by offset."""

    with blob_path.open("rb") as handle:
        handle.seek(byte_offset)
        payload = handle.read(byte_length)
    if len(payload) != byte_length:
        raise ValueError(f"short read from {blob_path} at offset {byte_offset}: expected {byte_length} bytes, got {len(payload)}")
    return decode_jpeg_bytes(payload)


def discover_videos(source_root: Path, directories: Sequence[str] | None) -> list[VideoRef]:
    """Discover `.mp4` files under `source_root` in one deterministic order.

    Directories are always sorted alphabetically regardless of the order
    `directories` was given in, so discovery order -- and therefore
    `video_id` assignment -- depends only on the *set* of directories
    requested, not the order they were listed on the command line. Within a
    directory, files are sorted by name.
    """

    if directories is None:
        resolved_directories = sorted(entry.name for entry in source_root.iterdir() if entry.is_dir())
    else:
        if len(set(directories)) != len(directories):
            raise ValueError(f"--directories contains duplicates: {directories}")
        resolved_directories = sorted(directories)

    videos: list[VideoRef] = []
    for directory in resolved_directories:
        directory_path = source_root / directory
        if not directory_path.is_dir():
            raise FileNotFoundError(f"configured source directory is missing: {directory_path}")
        for file_path in sorted(directory_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() == ".mp4":
                videos.append(VideoRef(source_directory=directory, file_name=file_path.name, absolute_path=file_path))
    return videos


def _video_list_sha256(videos: Sequence[VideoRef]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for video in videos:
        digest.update(video.relative_path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _decode_one_video(
    video: VideoRef,
    *,
    stride: int,
    jpeg_quality: int,
    expected_shape: tuple[int, int, int],
    allow_heterogeneous_geometry: bool,
    blob_handle: Any,
) -> dict[str, Any]:
    """Decode one video sequentially, write kept frames to `blob_handle`, and
    return its full per-video record (metadata + the list of frame rows).

    Fail-closed: any geometry violation not covered by
    `allow_heterogeneous_geometry`, an unreadable video, or a video that
    decodes zero frames raises -- the caller does not catch these, so an
    unhandled exception propagates to a nonzero process exit.
    """

    capture = cv2.VideoCapture(str(video.absolute_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video for decoding: {video.absolute_path}")
    try:
        reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        established_shape: tuple[int, int, int] | None = None
        decoded_frame_count = 0
        frame_rows: list[tuple[int, int, int]] = []  # (byte_offset, byte_length, source_frame_index)
        source_frame_index = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            decoded_frame_count += 1
            if source_frame_index % stride == 0:
                if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
                    raise ValueError(
                        f"decoded frame is not an HxWx3 uint8 array: {video.absolute_path} "
                        f"source_frame_index={source_frame_index} shape={frame_bgr.shape} dtype={frame_bgr.dtype}"
                    )
                shape = (frame_bgr.shape[0], frame_bgr.shape[1], frame_bgr.shape[2])
                if established_shape is None:
                    established_shape = shape
                    if shape != expected_shape and not allow_heterogeneous_geometry:
                        raise ValueError(
                            f"video geometry {shape} differs from the corpus-wide expectation "
                            f"{expected_shape} for {video.absolute_path}; pass "
                            "--allow-heterogeneous-geometry to record and continue instead of aborting"
                        )
                elif shape != established_shape:
                    raise ValueError(
                        f"frame geometry changed mid-video for {video.absolute_path}: "
                        f"source_frame_index={source_frame_index} has shape {shape}, but this video's "
                        f"first kept frame established {established_shape}; this is stream corruption, "
                        "not corpus heterogeneity, and is never overridable"
                    )
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                payload = encode_jpeg_rgb(frame_rgb, quality=jpeg_quality)
                byte_offset = blob_handle.tell()
                blob_handle.write(payload)
                frame_rows.append((byte_offset, len(payload), source_frame_index))
            source_frame_index += 1
    finally:
        capture.release()

    if decoded_frame_count == 0:
        raise ValueError(f"video decoded zero frames (corrupt or empty): {video.absolute_path}")
    if established_shape is None:
        raise RuntimeError(f"internal inconsistency: no kept frame established a shape for {video.absolute_path}")

    expected_kept = -(-decoded_frame_count // stride)  # ceil division
    if len(frame_rows) != expected_kept:
        raise RuntimeError(
            f"internal inconsistency: kept {len(frame_rows)} frames for {video.absolute_path}, "
            f"expected ceil({decoded_frame_count}/{stride})={expected_kept}"
        )

    decode_status = "ok" if established_shape == expected_shape else "heterogeneous_geometry"
    return {
        "reported_frame_count": reported_frame_count,
        "reported_fps": reported_fps,
        "decoded_frame_count": decoded_frame_count,
        "kept_frame_count": len(frame_rows),
        "frame_height": established_shape[0],
        "frame_width": established_shape[1],
        "decode_status": decode_status,
        "frame_rows": frame_rows,
    }


def _load_or_init_state(
    state_path: Path,
    *,
    source_root: Path,
    directories: Sequence[str] | None,
    stride: int,
    jpeg_quality: int,
    expected_height: int,
    expected_width: int,
    allow_heterogeneous_geometry: bool,
    video_list_sha256: str,
    video_count: int,
) -> dict[str, Any]:
    config = {
        "source_root": str(source_root),
        "directories": sorted(directories) if directories is not None else None,
        "stride": stride,
        "jpeg_quality": jpeg_quality,
        "expected_height": expected_height,
        "expected_width": expected_width,
        "allow_heterogeneous_geometry": allow_heterogeneous_geometry,
        "video_list_sha256": video_list_sha256,
        "video_count": video_count,
    }
    if state_path.is_file():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        existing_config = {key: existing.get(key) for key in _STATE_CONFIG_FIELDS}
        if existing_config != config:
            raise ValueError(
                f"{state_path} was started with a different configuration; use a fresh --output-dir "
                f"rather than resuming with mismatched inputs (was={existing_config}, now={config})"
            )
        return existing
    state = {
        **config,
        "videos_done": 0,
        "kept_frame_total": 0,
        "heterogeneous_geometry_videos": [],
    }
    _atomic_write_json(state_path, state)
    return state


def extract_videos(
    *,
    source_root: Path,
    directories: Sequence[str] | None,
    output_dir: Path,
    stride: int = DEFAULT_STRIDE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    expected_height: int = DEFAULT_EXPECTED_HEIGHT,
    expected_width: int = DEFAULT_EXPECTED_WIDTH,
    allow_heterogeneous_geometry: bool = False,
    limit_videos: int | None = None,
    max_new_videos: int | None = None,
) -> dict[str, Any] | None:
    """Extract kept frames from every discovered video into `output_dir`.

    Returns the closure dict if this call completed the extraction and wrote
    `extract_closure.json` (either just now, or -- impossible, since an
    existing closure makes this function raise immediately -- previously);
    returns None if work remains (either because `max_new_videos` capped
    this call, or because `limit_videos` has not yet been reached).
    """

    if stride < 1:
        raise ValueError("stride must be a positive integer")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    if expected_height < 1 or expected_width < 1:
        raise ValueError("expected_height and expected_width must be positive integers")
    if limit_videos is not None and limit_videos < 1:
        raise ValueError("limit_videos must be a positive integer or None")
    if max_new_videos is not None and max_new_videos < 1:
        raise ValueError("max_new_videos must be a positive integer or None")

    source_root = source_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    blob_path = output_dir / "frames.jpegblob"
    frame_rows_path = output_dir / "frame_index_rows.tmp.csv"
    videos_csv_path = output_dir / "videos.csv"
    state_path = output_dir / "extract_state.json"
    closure_path = output_dir / "extract_closure.json"
    frame_index_path = output_dir / "frame_index.npy"

    if closure_path.is_file():
        raise FileExistsError(
            f"extraction is already closed at {closure_path}; use a fresh --output-dir for a different extraction"
        )

    all_videos = discover_videos(source_root, directories)
    if limit_videos is not None:
        all_videos = all_videos[:limit_videos]
    if not all_videos:
        raise ValueError(f"no .mp4 files discovered under {source_root} for directories={directories}")

    video_list_sha256 = _video_list_sha256(all_videos)
    expected_shape = (expected_height, expected_width, 3)
    state = _load_or_init_state(
        state_path,
        source_root=source_root,
        directories=directories,
        stride=stride,
        jpeg_quality=jpeg_quality,
        expected_height=expected_height,
        expected_width=expected_width,
        allow_heterogeneous_geometry=allow_heterogeneous_geometry,
        video_list_sha256=video_list_sha256,
        video_count=len(all_videos),
    )

    videos_done = state["videos_done"]
    total_videos = len(all_videos)
    per_directory_counts: dict[str, int] = {}
    for video in all_videos:
        per_directory_counts[video.source_directory] = per_directory_counts.get(video.source_directory, 0) + 1

    if videos_done < total_videos:
        write_header = not videos_csv_path.is_file()
        blob_mode = "r+b" if blob_path.is_file() else "wb"
        if not blob_path.is_file():
            blob_path.touch()
            blob_mode = "r+b"
        pending = min(total_videos - videos_done, max_new_videos) if max_new_videos is not None else total_videos - videos_done
        log_phase(
            f"extracting videos {videos_done}..{videos_done + pending - 1} of {total_videos} "
            f"(stride={stride}, jpeg_quality={jpeg_quality})"
        )
        progress = BatchProgress(pending, label="extract videos", every=1, min_interval_seconds=15.0)
        import csv as csv_module

        with blob_path.open(blob_mode) as blob_handle, \
                videos_csv_path.open("a", newline="", encoding="utf-8") as videos_handle, \
                frame_rows_path.open("a", encoding="utf-8") as frame_rows_handle:
            videos_writer = csv_module.DictWriter(videos_handle, fieldnames=VIDEO_CSV_FIELDS)
            if write_header:
                videos_writer.writeheader()
                videos_handle.flush()
                os.fsync(videos_handle.fileno())
            for offset in range(pending):
                video_id = videos_done + offset
                video = all_videos[video_id]
                blob_handle.seek(0, os.SEEK_END)
                file_sha256, _, file_bytes = sha256_file(video.absolute_path)
                record = _decode_one_video(
                    video,
                    stride=stride,
                    jpeg_quality=jpeg_quality,
                    expected_shape=expected_shape,
                    allow_heterogeneous_geometry=allow_heterogeneous_geometry,
                    blob_handle=blob_handle,
                )
                first_row_index = state["kept_frame_total"]
                for byte_offset, byte_length, source_frame_index in record["frame_rows"]:
                    frame_rows_handle.write(f"{byte_offset},{byte_length},{video_id},{source_frame_index}\n")
                frame_rows_handle.flush()
                os.fsync(frame_rows_handle.fileno())
                videos_writer.writerow(
                    {
                        "video_id": video_id,
                        "source_directory": video.source_directory,
                        "file_name": video.file_name,
                        "relative_path": video.relative_path,
                        "file_bytes": file_bytes,
                        "file_sha256": file_sha256,
                        "reported_frame_count": record["reported_frame_count"],
                        "reported_fps": record["reported_fps"],
                        "decoded_frame_count": record["decoded_frame_count"],
                        "kept_frame_count": record["kept_frame_count"],
                        "frame_height": record["frame_height"],
                        "frame_width": record["frame_width"],
                        "first_row_index": first_row_index,
                        "decode_status": record["decode_status"],
                    }
                )
                videos_handle.flush()
                os.fsync(videos_handle.fileno())
                blob_handle.flush()
                os.fsync(blob_handle.fileno())

                state["videos_done"] = video_id + 1
                state["kept_frame_total"] = first_row_index + record["kept_frame_count"]
                if record["decode_status"] == "heterogeneous_geometry":
                    state["heterogeneous_geometry_videos"].append(
                        {
                            "video_id": video_id,
                            "relative_path": video.relative_path,
                            "frame_height": record["frame_height"],
                            "frame_width": record["frame_width"],
                        }
                    )
                _atomic_write_json(state_path, state)
                progress.step()
        progress.done()

    if state["videos_done"] < total_videos:
        log_phase(f"stopped after {state['videos_done']}/{total_videos} videos (max_new_videos budget); resume later to finish")
        return None

    log_phase("all videos decoded; building frame_index.npy from the intermediate row log")
    rows: list[tuple[int, int, int, int]] = []
    with frame_rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            byte_offset_s, byte_length_s, video_id_s, source_frame_index_s = line.split(",")
            rows.append((int(byte_offset_s), int(byte_length_s), int(video_id_s), int(source_frame_index_s)))
    if len(rows) != state["kept_frame_total"]:
        raise RuntimeError(
            f"internal inconsistency: {frame_rows_path} has {len(rows)} rows, "
            f"expected {state['kept_frame_total']} from state"
        )
    frame_index = np.zeros(len(rows), dtype=_FRAME_INDEX_DTYPE)
    if rows:
        frame_index["byte_offset"] = [row[0] for row in rows]
        frame_index["byte_length"] = [row[1] for row in rows]
        frame_index["video_id"] = [row[2] for row in rows]
        frame_index["source_frame_index"] = [row[3] for row in rows]
    offsets = frame_index["byte_offset"]
    lengths = frame_index["byte_length"]
    if len(offsets) and not np.all(np.diff(offsets) >= 0):
        raise RuntimeError("internal inconsistency: byte offsets are not non-decreasing")
    if len(offsets) and not np.array_equal(offsets[:-1] + lengths[:-1], offsets[1:]):
        raise RuntimeError("internal inconsistency: offset index is not contiguous")
    blob_bytes = blob_path.stat().st_size
    if len(offsets) and offsets[-1] + lengths[-1] != blob_bytes:
        raise RuntimeError(
            f"internal inconsistency: final offset+length {offsets[-1] + lengths[-1]} "
            f"does not equal blob size {blob_bytes}"
        )

    descriptor, temporary_name = tempfile.mkstemp(prefix=".frame_index.npy.", suffix=".npy", dir=output_dir)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.save(temporary, frame_index, allow_pickle=False)
        os.replace(temporary, frame_index_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    log_phase("packing complete; hashing final artifacts for the closure record")
    blob_sha256, _, _ = sha256_file(blob_path)
    frame_index_sha256, _, _ = sha256_file(frame_index_path)
    videos_csv_sha256, _, _ = sha256_file(videos_csv_path)

    closure = {
        "kind": "P1C official-corpus video-derived JPEG frame blob (D-34/D-35 registered research artifact)",
        "extractor_code_version": EXTRACTOR_CODE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "directories": sorted(set(video.source_directory for video in all_videos)),
        "per_directory_video_counts": per_directory_counts,
        "video_count": total_videos,
        "kept_frame_count": int(state["kept_frame_total"]),
        "stride": stride,
        "jpeg_quality": jpeg_quality,
        "color_order": "RGB",
        "expected_image_shape": [expected_height, expected_width, 3],
        "allow_heterogeneous_geometry": allow_heterogeneous_geometry,
        "heterogeneous_geometry_videos": state["heterogeneous_geometry_videos"],
        "blob_path": "frames.jpegblob",
        "blob_sha256": blob_sha256,
        "blob_bytes": blob_bytes,
        "frame_index_path": "frame_index.npy",
        "frame_index_sha256": frame_index_sha256,
        "frame_index_dtype": [list(field) for field in _FRAME_INDEX_DTYPE.descr],
        "videos_csv_path": "videos.csv",
        "videos_csv_sha256": videos_csv_sha256,
        "cv2_version": cv2.__version__,
        "cv2_ffmpeg_build_line": cv2_ffmpeg_build_line(),
        "source_video_grouping": "one row per source video in videos.csv, in the video_id order defined by discover_videos()",
        "excludes": (
            "no SHAR label assignment of any kind; source_directory is recorded verbatim as "
            "provenance only (see module docstring; B-001/B-002)"
        ),
    }
    _atomic_write_json(closure_path, closure)
    log_phase(f"closed: {closure_path}")
    return closure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--directories",
        type=str,
        nargs="+",
        default=None,
        help="Subdirectory names under --source-root to extract (default: all immediate subdirectories found)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--expected-height", type=int, default=DEFAULT_EXPECTED_HEIGHT)
    parser.add_argument("--expected-width", type=int, default=DEFAULT_EXPECTED_WIDTH)
    parser.add_argument("--allow-heterogeneous-geometry", action="store_true")
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="Shrink the discovered corpus to its first N videos (sorted order); use for a small, closeable smoke run",
    )
    parser.add_argument(
        "--max-new-videos",
        type=int,
        default=None,
        help="Process at most N not-yet-done videos this invocation, then stop without closing (incremental/testing knob)",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    log_phase(f"extracting {args.source_root} -> {args.output_dir}")
    closure = extract_videos(
        source_root=args.source_root.resolve(),
        directories=args.directories,
        output_dir=args.output_dir.resolve(),
        stride=args.stride,
        jpeg_quality=args.jpeg_quality,
        expected_height=args.expected_height,
        expected_width=args.expected_width,
        allow_heterogeneous_geometry=args.allow_heterogeneous_geometry,
        limit_videos=args.limit_videos,
        max_new_videos=args.max_new_videos,
    )
    log_phase(f"done in {time.perf_counter() - started:.1f}s")
    if closure is not None:
        print(json.dumps(closure, indent=2, sort_keys=True))
        return 0
    print(json.dumps({"status": "incomplete", "output_dir": str(args.output_dir.resolve())}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
