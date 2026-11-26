"""Training dataset over the D-34/D-35 video blob.

Joins a manifest built by `scripts/build_p1c_video_manifests.py` to the packed
JPEG blob and yields the same sample dict shape the D-33 loaders produce, so a
runner can consume either source without special-casing.

Why not a subclass of `P1CFrameDataset`: that class is built around a manifest
whose `filepath` names a decodable image on disk, and it resolves and optionally
digest-verifies that file per row. Here `filepath` names the source MP4 -- a
container holding thousands of frames, not this row's pixels -- so inheriting
that machinery would mean inheriting assumptions that are false for this source.
The shared preprocessing contract is reused directly instead
(`FramePreprocessing`), which is the part that actually has to agree.

Row resolution: a manifest row identifies its pixels by
`(source_video_id, source_frame_index)`, never by a blob row number. The lookup
table is built once at construction from the blob's own frame index plus
`videos.csv`. That indirection is deliberate -- see `data/video_manifests.py`.

Cost note (D-37): unlike the D-33 raw path, every sample here costs a JPEG
decode of a 320x240 record plus a downscale to the training resolution. At
64x64 the accelerator is fast enough that this is the plausible bottleneck, so
the next calibration must measure it rather than assume it.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.augmentation import AugmentationPolicy, NONE_AUGMENTATION, augmentation_seed
from data.p1c_loader import FramePreprocessing, _INTERPOLATION
from data.video_blob import VideoBlobReader
from data.video_manifests import read_videos_csv


class P1CVideoFrameDataset(Dataset[dict[str, Any]]):
    """Manifest rows joined to packed JPEG frames."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        closure_path: Path | str,
        preprocessing: FramePreprocessing,
        class_names: Sequence[str],
        verify_digest: bool = True,
        split: str | None = None,
        augmentation: AugmentationPolicy | None = None,
        augmentation_run_seed: int = 0,
    ) -> None:
        if augmentation is not None:
            if not isinstance(augmentation, AugmentationPolicy):
                raise TypeError("augmentation must be an AugmentationPolicy or None")
            if split != "train":
                # Structural, not incidental (data/augmentation.py): this class has
                # no manifest-driven split guard of its own the way P1CFrameDataset
                # does, so `split` must be explicitly asserted "train" here before
                # a real augmentation policy is accepted at all -- an omitted or
                # non-train `split` refuses construction rather than assuming.
                raise ValueError("augmentation may only be registered for split=train; validation/test must never be augmented")
        if isinstance(augmentation_run_seed, bool) or not isinstance(augmentation_run_seed, int) or augmentation_run_seed < 0:
            raise ValueError("augmentation_run_seed must be a non-negative integer")
        self._manifest_path = Path(manifest_path)
        self._preprocessing = preprocessing
        self._label_to_index = {name: i for i, name in enumerate(class_names)}
        if len(self._label_to_index) != len(class_names):
            raise ValueError("class_names contains duplicates")
        self._augmentation = augmentation if augmentation is not None else NONE_AUGMENTATION
        self._augmentation_run_seed = augmentation_run_seed
        self._epoch = 0

        self._reader = VideoBlobReader(closure_path, verify_digest=verify_digest)
        videos = read_videos_csv(self._reader.closure.videos_csv_path)

        # (blob video_id, source_frame_index) -> blob row, built once.
        index = self._reader.index
        by_key: dict[tuple[int, int], int] = {}
        for row_number, (video_id, frame_index) in enumerate(
            zip(index["video_id"].tolist(), index["source_frame_index"].tolist())
        ):
            by_key[(video_id, frame_index)] = row_number

        # Keyed by (directory, stem): stems repeat across directories in this
        # corpus (see data/video_manifests.read_videos_csv). The manifest's
        # filepath carries the directory, so the pair is always recoverable.
        key_to_video_id = {key: row.video_id for key, row in videos.items()}

        # Rows are stored as parallel arrays plus a small per-video table, not
        # as a list of dicts.
        #
        # This is a memory decision, not a style one.  The P1C loaders use the
        # "spawn" start method, so the whole dataset is pickled into every
        # worker rather than shared copy-on-write.  Held as dicts, 391,014
        # manifest rows cost ~0.56 GiB *per worker*; six workers plus the
        # parent did not fit in the pod's 16 GiB (K-13, K-15).  Held this way
        # the same rows cost single-digit MiB, because every field except the
        # frame index is a property of the source *video* -- and there are 476
        # of those, not 391,014.  Nothing is lost: `__getitem__` rebuilds the
        # identical sample dict.
        videos_seen: dict[tuple[str, str], int] = {}
        video_table: list[dict[str, str]] = []
        video_refs: list[int] = []
        targets: list[int] = []
        frame_indices: list[int] = []
        blob_rows: list[int] = []

        with self._manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                label = record["label"]
                if label not in self._label_to_index:
                    raise ValueError(
                        f"{self._manifest_path}: label {label!r} is not in class_names"
                    )
                stem = record["source_video_id"]
                directory = record["filepath"].split("/", 1)[0]
                video_id = key_to_video_id.get((directory, stem))
                if video_id is None:
                    raise KeyError(
                        f"{self._manifest_path}: source video {stem!r} is not in the blob "
                        f"under {directory!r}"
                    )
                source_frame_index = int(record["source_frame_index"])
                blob_row = by_key.get((video_id, source_frame_index))
                if blob_row is None:
                    raise KeyError(
                        f"{self._manifest_path}: {stem} frame "
                        f"{record['source_frame_index']} is not in the blob"
                    )

                video_key = (directory, stem)
                reference = videos_seen.get(video_key)
                if reference is None:
                    reference = len(video_table)
                    videos_seen[video_key] = reference
                    video_table.append({
                        "source_dataset": record["source_dataset"],
                        "source_video_id": stem,
                        "filepath": record["filepath"],
                        "file_digest": record["file_digest"],
                        "split": record["split"],
                        "label": label,
                    })
                video_refs.append(reference)
                targets.append(self._label_to_index[label])
                frame_indices.append(source_frame_index)
                blob_rows.append(blob_row)

        if not blob_rows:
            raise ValueError(f"{self._manifest_path} contains no rows")

        self._videos = video_table
        self._video_ref = np.asarray(video_refs, dtype=np.int32)
        self._target = np.asarray(targets, dtype=np.int16)
        self._frame_index = np.asarray(frame_indices, dtype=np.int64)
        self._blob_rows = np.asarray(blob_rows, dtype=np.int64)

    @property
    def augmentation(self) -> AugmentationPolicy:
        """The resolved policy this dataset applies -- `NONE_AUGMENTATION` if none was registered."""

        return self._augmentation

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch augmentation seeding derives from. See sibling in `data.p1c_loader`."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self._epoch = epoch

    def __len__(self) -> int:
        return int(self._blob_rows.shape[0])

    @property
    def class_names(self) -> list[str]:
        return list(self._label_to_index)

    def source_video_groups(self) -> frozenset[tuple[str, str]]:
        """Distinct ``(source_dataset, source_video_id)`` keys in this split.

        The runners intersect this across train and validation and refuse any
        overlap. Implementing it is what lets the video source participate in
        that leakage check rather than silently skipping it -- the manifest
        builder already guarantees the grouping, but the runner verifies it
        independently at load, which is the point.

        Note the key is the pair, not the video id alone: the same stem can
        appear under two directories in this corpus (K-11).
        """
        return frozenset(
            (video["source_dataset"], video["source_video_id"]) for video in self._videos
        )

    def raw_jpeg(self, index: int) -> bytes:
        """Row `index`'s stored JPEG bytes at native resolution: no resize, normalisation or augmentation."""

        return bytes(self._reader.payload(int(self._blob_rows[index])))

    def __getitem__(self, index: int) -> dict[str, Any]:
        video = self._videos[int(self._video_ref[index])]
        source_frame_index = int(self._frame_index[index])
        payload = self._reader.payload(int(self._blob_rows[index]))
        with Image.open(io.BytesIO(payload)) as opened:
            image = opened.convert(self._preprocessing.color_mode)
            # Identity must vary per FRAME, not per video: `video["filepath"]` and
            # `video["file_digest"]` are constant across the ~800 frames sampled
            # from one source video (D-37's near-duplicate problem this whole
            # experiment exists to attack), so seeding on them alone would give
            # every frame of a video the identical crop/flip/jitter -- the frames
            # would stay near-duplicates of each other, just all offset the same
            # way. Folding in `source_frame_index` is what actually decorrelates
            # them.
            seed = augmentation_seed(
                f"{video['filepath']}#frame{source_frame_index}",
                video["file_digest"],
                self._augmentation_run_seed,
                self._epoch,
                self._augmentation,
            )
            image = self._augmentation.apply(
                image,
                resize_width=self._preprocessing.resize_width,
                resize_height=self._preprocessing.resize_height,
                resample=_INTERPOLATION[self._preprocessing.interpolation],
                seed=seed,
            )
            pixels = np.asarray(image, dtype=np.float32)

        expected = (
            self._preprocessing.resize_height,
            self._preprocessing.resize_width,
            3,
        )
        if pixels.shape != expected:
            raise ValueError(
                f"decoded RGB frame has shape {pixels.shape}, expected {expected}: "
                f"{video['source_video_id']} frame {source_frame_index}"
            )

        tensor = torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).div_(255.0)
        mean = torch.tensor(self._preprocessing.mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self._preprocessing.std, dtype=tensor.dtype).view(3, 1, 1)
        return {
            "image": (tensor - mean) / std,
            "target": int(self._target[index]),
            "source_dataset": video["source_dataset"],
            "source_video_id": video["source_video_id"],
            "source_frame_index": source_frame_index,
            "filepath": video["filepath"],
            "file_digest": video["file_digest"],
            "split": video["split"],
        }
