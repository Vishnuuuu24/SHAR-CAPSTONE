"""Shared streamed-extraction I/O for one-off P1C feature/artifact builders.

`scripts/extract_p1c_shooting_tokens.py` and `scripts/build_p1c_roi_cache.py`
both plan a small, cross-fold set of (video, frame) pairs from dataset
metadata alone, then stream a forward pass into a pre-allocated on-disk
memmap. This module holds the two pieces that are byte-for-byte identical
between them (docs/PROJECT_RULES.md's rule of two): resolving wanted (video, frame) pairs
to dataset indices without decoding a JPEG per candidate row, and finalising
a streamed memmap (flush, sha256 the array file, record its shape) so the
caller only has to build its own manifest dict around that.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from core.p1c_readiness import sha256_file


def select_rows_for_videos(
    dataset: Any, wanted: Iterable[str], seen: set[str]
) -> list[tuple[int, str, int, int]]:
    """`(dataset index, video id, source frame, target)` for `wanted` videos not yet in `seen`.

    Reads a `P1CVideoFrameDataset`'s metadata arrays directly
    (`_videos`/`_video_ref`/`_frame_index`/`_target`) rather than calling
    `__getitem__`, which would decode a JPEG per row -- for a corpus-wide
    dataset that is hundreds of thousands of wasted decodes to select a few
    thousand rows, and the arrays already exist. Does not mutate `seen`; the
    caller decides when a video counts as covered (e.g. only after a label
    filter is applied).
    """

    wanted = set(wanted)
    videos = dataset._videos  # noqa: SLF001 -- one-off extraction, metadata only
    ref, frames, targets = dataset._video_ref, dataset._frame_index, dataset._target  # noqa: SLF001
    out: list[tuple[int, str, int, int]] = []
    for index in range(len(dataset)):
        video = videos[int(ref[index])]["source_video_id"]
        if video in wanted and video not in seen:
            out.append((index, video, int(frames[index]), int(targets[index])))
    return out


def finalize_streamed_array(out_dir: Path, memmap: Any, path_name: str = "features.float16.npy") -> tuple[list[int], str]:
    """Flush a streamed memmap and return `(shape, sha256)` of the file it wrote.

    The caller still owns writing its own `rows.csv` and `manifest.json` --
    those differ enough in shape between callers (a feature manifest vs an
    ROI manifest with detector/policy fields) that folding them in here would
    trade a real duplication for a fake, over-parameterised one.
    """

    memmap.flush()
    del memmap
    path = out_dir / path_name
    # Chunked, never `path.read_bytes()`: K-30. A 32.5 GB patch-token array read
    # whole is an instant OOM inside a 16 GiB pod, and it killed the P3A cache
    # build after the entire GPU pass had already succeeded. `sha256_file` is
    # the project's existing chunked hasher -- rule of two, not a second copy.
    digest = sha256_file(path)
    shape = list(np.load(path, mmap_mode="r").shape)
    return shape, digest


def write_manifest(out_dir: Path, manifest: dict[str, Any], *, filename: str = "manifest.json") -> None:
    """Write `manifest` (plus a UTC `created_at`) as pretty, sorted JSON."""

    payload = {**manifest, "created_at": datetime.now(timezone.utc).isoformat()}
    (out_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
