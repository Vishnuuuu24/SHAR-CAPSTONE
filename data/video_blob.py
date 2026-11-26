"""Read the D-34/D-35 video-derived JPEG frame blob.

`scripts/extract_p1c_video_frames.py` decodes the official UCF-Crime MP4 corpus
into three artifacts: one concatenated JPEG payload (`frames.jpegblob`), a
structured offset index (`frame_index.npy`), and per-video provenance
(`videos.csv`), closed by `extract_closure.json`. This module is the read side.

It is deliberately **protocol-agnostic and label-free**. The blob carries pixels
and provenance, never a SHAR class assignment: `source_directory` is recorded
verbatim by the extractor and the mapping from directory to label depends on
which official protocol a given experiment targets (see the two-protocol row in
`docs/EVIDENCE_REGISTER.md` -- Action Recognition's 14 balanced classes over 700
videos, versus Anomaly Detection's 1,900). Nothing here decides that. A dataset
class that joins these rows to a manifest belongs above this module, not in it.

Why a separate module rather than another `P1CPackedFrameDataset` subclass: the
D-33 Kaggle blob holds fixed-size raw records, so row `i` lives at `i *
item_bytes` and reading it is pure offset arithmetic. These records are
variable-length JPEG, so every read needs an explicit `(offset, length)` lookup
plus a decode. That is a different access model, and expressing it as a subclass
of a fixed-stride reader would mean inheriting arithmetic that does not apply.

Verification posture matches D-33's: the blob's SHA-256 is checked **once** at
construction (`verify_digest=True`, the default) rather than per row, because
per-row hashing was the cost the packed formats exist to remove.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core import pagecache
from PIL import Image

FRAME_INDEX_DTYPE = np.dtype(
    [
        ("byte_offset", "<i8"),
        ("byte_length", "<i8"),
        ("video_id", "<i8"),
        ("source_frame_index", "<i8"),
    ]
)

EXPECTED_COLOR_ORDER = "RGB"

# Blobs whose SHA-256 this process has already confirmed, keyed by identity
# (path, size, mtime_ns) so a file replaced under the same path is re-verified.
#
# Without this, a run that builds a train and a validation dataset hashes the
# same 84 GiB blob twice for no additional assurance -- roughly four minutes of
# pure CPU. The check itself is not weakened: the first construction still
# fails closed on a mismatch, and any change to the file changes its identity
# key and forces a fresh hash.
_VERIFIED_BLOBS: set[tuple[str, int, int]] = set()


def _blob_identity(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns)


def reset_verified_blob_cache() -> None:
    """Forget which blobs this process has verified. For tests."""
    _VERIFIED_BLOBS.clear()


def _sha256_file(path: Path, *, chunk_bytes: int = 1 << 23) -> str:
    """Hash a file without letting its pages accumulate in the page cache.

    The whole file is still read and hashed -- the integrity guarantee is
    unchanged.  What changes is only what the kernel keeps afterwards, which
    matters because this blob is 84.3 GiB and, inside a pod, page cache is
    charged to the container's memory limit.  See `core.pagecache` and K-15.
    """
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        fd = handle.fileno()
        offset = dropped_to = 0
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
            offset += len(chunk)
            if offset - dropped_to >= pagecache.DROP_INTERVAL_BYTES:
                pagecache.drop_range(fd, dropped_to, offset - dropped_to)
                dropped_to = offset
        pagecache.drop_range(fd, dropped_to, offset - dropped_to)
    return digest.hexdigest()


@dataclass(frozen=True)
class VideoBlobClosure:
    """The fields of an `extract_closure.json` that the read side needs."""

    closure_path: Path
    blob_path: Path
    blob_sha256: str
    blob_bytes: int
    frame_index_path: Path
    frame_index_sha256: str
    videos_csv_path: Path
    kept_frame_count: int
    video_count: int
    stride: int
    jpeg_quality: int
    expected_image_shape: tuple[int, int, int]
    color_order: str
    extractor_code_version: str

    @classmethod
    def load(cls, closure_path: Path | str) -> "VideoBlobClosure":
        closure_path = Path(closure_path).resolve()
        closure: dict[str, Any] = json.loads(closure_path.read_text(encoding="utf-8"))
        parent = closure_path.parent

        shape = closure.get("expected_image_shape")
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError(f"{closure_path} has an invalid expected_image_shape")

        color_order = str(closure.get("color_order", ""))
        if color_order != EXPECTED_COLOR_ORDER:
            # Not cosmetic. cv2 decodes BGR, and a blob written without the
            # conversion would look structurally perfect while every pixel had
            # red and blue transposed (K-05). Refuse rather than guess.
            raise ValueError(
                f"{closure_path} declares color_order={color_order!r}; "
                f"expected {EXPECTED_COLOR_ORDER!r}"
            )

        return cls(
            closure_path=closure_path,
            blob_path=(parent / closure["blob_path"]).resolve(),
            blob_sha256=str(closure["blob_sha256"]),
            blob_bytes=int(closure["blob_bytes"]),
            frame_index_path=(parent / closure["frame_index_path"]).resolve(),
            frame_index_sha256=str(closure["frame_index_sha256"]),
            videos_csv_path=(parent / closure["videos_csv_path"]).resolve(),
            kept_frame_count=int(closure["kept_frame_count"]),
            video_count=int(closure["video_count"]),
            stride=int(closure["stride"]),
            jpeg_quality=int(closure["jpeg_quality"]),
            expected_image_shape=(int(shape[0]), int(shape[1]), int(shape[2])),
            color_order=color_order,
            extractor_code_version=str(closure["extractor_code_version"]),
        )


class VideoBlobReader:
    """Random access to decoded RGB frames in a video-derived JPEG blob.

    Row order is the extractor's: videos in `discover_videos()` order, and
    within a video, ascending `source_frame_index`. That ordering is stable
    across runs, which is what lets a manifest reference rows by integer.
    """

    def __init__(
        self,
        closure_path: Path | str,
        *,
        verify_digest: bool = True,
    ) -> None:
        self.closure = VideoBlobClosure.load(closure_path)

        self._index: np.ndarray | None = self._load_index()

        actual_bytes = self.closure.blob_path.stat().st_size
        if actual_bytes != self.closure.blob_bytes:
            raise ValueError(
                f"{self.closure.blob_path} is {actual_bytes} bytes; the closure "
                f"declares {self.closure.blob_bytes}. Refusing a truncated blob."
            )

        if verify_digest:
            identity = _blob_identity(self.closure.blob_path)
            if identity not in _VERIFIED_BLOBS:
                actual = _sha256_file(self.closure.blob_path)
                if actual != self.closure.blob_sha256:
                    raise ValueError(
                        f"{self.closure.blob_path} sha256 {actual} does not match the "
                        f"closure's {self.closure.blob_sha256}"
                    )
                _VERIFIED_BLOBS.add(identity)

        # mmap rather than read(): the blob is far larger than RAM at full
        # corpus scale, and workers should share page cache rather than each
        # holding a private copy.  Opened lazily so that this object can cross
        # a process boundary -- see __getstate__.
        self._blob: np.memmap | None = None

    # -- pickling ----------------------------------------------------------
    #
    # The P1C loaders use the "spawn" start method, so a Dataset holding a
    # reader is *pickled* into every worker rather than inherited by fork.
    # Two attributes must never travel that way:
    #
    #   _blob   a np.memmap, which is an ndarray subclass with no pickle
    #           override -- pickling it serialises the mapped bytes *by value*,
    #           i.e. it tries to push all 84.3 GiB through a buffer.  That is
    #           what OOM-killed CALIBRATION-006: anon memory climbed ~1 GiB/s
    #           from the moment the loader was built until the cgroup limit.
    #   _index  85 MiB, and reloadable from its own file far more cheaply than
    #           it can be serialised into six workers.
    #
    # Both are reopened on first use in the child.  The digest is deliberately
    # NOT re-verified there: the parent verified this exact blob before the
    # fork-out, and re-hashing 84.3 GiB once per worker would be far worse than
    # the problem it guards against.  The cheap structural checks still run.

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_blob"] = None
        state["_index"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    @property
    def index(self) -> np.ndarray:
        if self._index is None:
            self._index = self._load_index()
        return self._index

    @property
    def blob(self) -> np.memmap:
        if self._blob is None:
            self._blob = np.memmap(
                self.closure.blob_path,
                dtype=np.uint8,
                mode="r",
                shape=(self.closure.blob_bytes,),
            )
        return self._blob

    def _load_index(self) -> np.ndarray:
        index = np.load(self.closure.frame_index_path)
        if index.dtype != FRAME_INDEX_DTYPE:
            raise ValueError(
                f"{self.closure.frame_index_path} has dtype {index.dtype}, "
                f"expected {FRAME_INDEX_DTYPE}"
            )
        if index.shape[0] != self.closure.kept_frame_count:
            raise ValueError(
                f"frame index holds {index.shape[0]} rows but the closure "
                f"declares kept_frame_count={self.closure.kept_frame_count}"
            )
        return index

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def payload(self, row: int) -> bytes:
        """Raw JPEG bytes for one row, undecoded."""
        entry = self.index[row]
        start = int(entry["byte_offset"])
        end = start + int(entry["byte_length"])
        return bytes(self.blob[start:end])

    def frame(self, row: int) -> np.ndarray:
        """Decode one row to an HxWx3 uint8 RGB array."""
        image = Image.open(io.BytesIO(self.payload(row)))
        if image.mode != EXPECTED_COLOR_ORDER:
            image = image.convert(EXPECTED_COLOR_ORDER)
        return np.asarray(image, dtype=np.uint8)

    def provenance(self, row: int) -> dict[str, int]:
        """The row's source video and its index in that video's decode order.

        `source_frame_index` is the original index in the source video, not the
        compacted position in the blob. With that video's own fps from
        `videos.csv` it converts to a timestamp exactly. Use that video's fps
        and no other: the corpus is not uniformly 30 fps (K-04).
        """
        entry = self.index[row]
        return {
            "video_id": int(entry["video_id"]),
            "source_frame_index": int(entry["source_frame_index"]),
        }

    def rows_for_video(self, video_id: int) -> np.ndarray:
        """Row numbers belonging to one source video, ascending.

        Rows for a video are contiguous by construction, but this does not rely
        on that -- a boolean select stays correct if the extractor's ordering
        ever changes.
        """
        return np.flatnonzero(self.index["video_id"] == video_id)
