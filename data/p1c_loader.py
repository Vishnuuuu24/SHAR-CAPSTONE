"""Manifest-backed, deterministic frame loading for P1C controls.

This module deliberately owns no augmentation or model-specific scientific
choice.  The caller must provide the complete resize, colour, interpolation,
and normalization configuration before a dataset can be constructed.
"""

from __future__ import annotations

import json
import math
import random
import hashlib
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.augmentation import AugmentationPolicy, NONE_AUGMENTATION, augmentation_seed
from data.manifest import ManifestRow, iter_manifest, resolve_manifest_path


_PREPROCESSING_KEYS = {
    "resize_height",
    "resize_width",
    "interpolation",
    "color_mode",
    "mean",
    "std",
}
_INTERPOLATION = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}
_P1C_SPLITS = {"train", "validation", "test"}


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _channel_values(value: object, name: str, *, positive: bool) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three finite values")
    values: list[float] = []
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise ValueError(f"{name}[{index}] must be finite")
        converted = float(raw)
        if positive and converted <= 0:
            raise ValueError(f"{name}[{index}] must be greater than zero")
        values.append(converted)
    return tuple(values)  # type: ignore[return-value]


@dataclass(frozen=True)
class FramePreprocessing:
    """The complete, explicit image transform contract for a P1C run."""

    resize_height: int
    resize_width: int
    interpolation: str
    color_mode: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def __post_init__(self) -> None:
        _positive_integer(self.resize_height, "resize_height")
        _positive_integer(self.resize_width, "resize_width")
        if self.interpolation not in _INTERPOLATION:
            raise ValueError(f"interpolation must be one of {sorted(_INTERPOLATION)}")
        if self.color_mode != "RGB":
            raise ValueError("P1C frame controls require explicit color_mode=RGB")
        object.__setattr__(self, "mean", _channel_values(self.mean, "mean", positive=False))
        object.__setattr__(self, "std", _channel_values(self.std, "std", positive=True))

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "FramePreprocessing":
        """Parse an exact preprocessing block without silently adding defaults."""

        keys = set(config)
        if keys != _PREPROCESSING_KEYS:
            missing = sorted(_PREPROCESSING_KEYS - keys)
            unexpected = sorted(keys - _PREPROCESSING_KEYS)
            raise ValueError(
                "preprocessing configuration keys must be exactly "
                f"{sorted(_PREPROCESSING_KEYS)}; missing={missing}; unexpected={unexpected}"
            )
        return cls(
            resize_height=_positive_integer(config["resize_height"], "resize_height"),
            resize_width=_positive_integer(config["resize_width"], "resize_width"),
            interpolation=str(config["interpolation"]),
            color_mode=str(config["color_mode"]),
            mean=_channel_values(config["mean"], "mean", positive=False),
            std=_channel_values(config["std"], "std", positive=True),
        )


@dataclass(frozen=True)
class FrameLoaderOptions:
    """Explicit runtime options for a deterministic P1C frame DataLoader."""

    batch_size: int
    shuffle: bool
    num_workers: int
    seed: int
    epoch: int
    pin_memory: bool
    drop_last: bool
    persistent_workers: bool

    def __post_init__(self) -> None:
        _positive_integer(self.batch_size, "batch_size")
        if not isinstance(self.shuffle, bool):
            raise TypeError("shuffle must be bool")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if not isinstance(self.pin_memory, bool) or not isinstance(self.drop_last, bool):
            raise TypeError("pin_memory and drop_last must be bool")
        if not isinstance(self.persistent_workers, bool):
            raise TypeError("persistent_workers must be bool")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers greater than zero")


class P1CFrameDataset(Dataset[dict[str, Any]]):
    """Load P1B frames with an explicit opt-in for held-out Test access.

    Training callers can construct only Train or validation datasets by
    default.  Test loading is deliberately possible solely for the separate
    post-selection evaluation path and must opt in at construction time.
    """

    def __init__(
        self,
        manifest_path: Path,
        dataset_roots: Mapping[str, Path],
        *,
        split: Literal["train", "validation", "test"],
        class_names: Sequence[str],
        preprocessing: FramePreprocessing,
        verify_files: bool = True,
        verify_digests: bool = True,
        allow_held_out_test: bool = False,
        augmentation: AugmentationPolicy | None = None,
        augmentation_run_seed: int = 0,
    ) -> None:
        if split not in _P1C_SPLITS:
            raise ValueError("P1C frame loading accepts only split=train, split=validation, or split=test")
        if not isinstance(allow_held_out_test, bool):
            raise TypeError("allow_held_out_test must be bool")
        if split == "test" and not allow_held_out_test:
            raise PermissionError("held-out Test loading requires explicit allow_held_out_test=True")
        if not isinstance(preprocessing, FramePreprocessing):
            raise TypeError("preprocessing must be an explicit FramePreprocessing configuration")
        if not isinstance(verify_files, bool) or not isinstance(verify_digests, bool):
            raise TypeError("verify_files and verify_digests must be bool")
        if augmentation is not None:
            if not isinstance(augmentation, AugmentationPolicy):
                raise TypeError("augmentation must be an AugmentationPolicy or None")
            if split != "train":
                # Structural, not incidental: a validation/test dataset cannot be
                # handed a real augmentation policy even by an editing mistake at
                # the call site -- construction itself refuses. See
                # data/augmentation.py's module docstring.
                raise ValueError("augmentation may only be registered for split=train; validation/test must never be augmented")
        if isinstance(augmentation_run_seed, bool) or not isinstance(augmentation_run_seed, int) or augmentation_run_seed < 0:
            raise ValueError("augmentation_run_seed must be a non-negative integer")
        self._label_to_index = self._label_mapping(class_names)
        self._dataset_roots = dict(dataset_roots)
        self._manifest_path = manifest_path.resolve()
        self._verify_files = verify_files
        self._verify_digests = verify_digests
        self.split = split
        self._index_path = self._manifest_path.parent / ".p1c_index" / f"{self._manifest_path.name}.{split}.sqlite3"
        self._connection: sqlite3.Connection | None = None
        self._record_count = self._create_or_verify_index()
        self._preprocessing = preprocessing
        self._augmentation = augmentation if augmentation is not None else NONE_AUGMENTATION
        self._augmentation_run_seed = augmentation_run_seed
        self._epoch = 0

    @property
    def augmentation(self) -> AugmentationPolicy:
        """The resolved policy this dataset applies -- `NONE_AUGMENTATION` if none was registered."""

        return self._augmentation

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch augmentation seeding derives from.

        Called once per epoch, before the dataset is handed to a new
        DataLoader (the runner already recreates the loader every epoch to
        thread a fresh shuffle seed through `FrameLoaderOptions`, so this
        follows the same call pattern). A no-op on augmentation-free datasets.
        """

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self._epoch = epoch

    @staticmethod
    def _index_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE records (
              position INTEGER PRIMARY KEY,
              filepath TEXT NOT NULL,
              source_dataset TEXT NOT NULL,
              source_video_id TEXT NOT NULL,
              source_frame_index INTEGER,
              label TEXT NOT NULL,
              label_scope TEXT NOT NULL,
              label_source TEXT NOT NULL,
              annotation_version TEXT NOT NULL,
              inside_official_interval INTEGER,
              split TEXT NOT NULL,
              file_digest TEXT NOT NULL
            );
            CREATE INDEX records_source_video ON records(source_dataset, source_video_id);
            """
        )

    def _manifest_digest(self) -> str:
        digest = hashlib.sha256()
        with self._manifest_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _row_values(row: ManifestRow, position: int) -> tuple[object, ...]:
        interval = None if row.inside_official_interval is None else int(row.inside_official_interval)
        return (
            position, row.filepath, row.source_dataset, row.source_video_id, row.source_frame_index,
            row.label, row.label_scope, row.label_source, row.annotation_version, interval, row.split, row.file_digest,
        )

    @staticmethod
    def _row_from_values(values: tuple[object, ...]) -> ManifestRow:
        return ManifestRow(
            filepath=str(values[0]), source_dataset=str(values[1]), source_video_id=str(values[2]),
            source_frame_index=values[3] if isinstance(values[3], int) else None, label=str(values[4]),
            label_scope=str(values[5]), label_source=str(values[6]), annotation_version=str(values[7]),
            inside_official_interval=None if values[8] is None else bool(values[8]), split=str(values[9]),
            file_digest=str(values[10]),
        )

    def _verify_cached_files(self, connection: sqlite3.Connection) -> None:
        if not self._verify_files and not self._verify_digests:
            # Neither integrity gate is requested, so walking 1.1M rows to
            # resolve paths and discard them is pure cost.
            return
        for values in connection.execute(
            "SELECT filepath, source_dataset, source_video_id, source_frame_index, label, label_scope, label_source, "
            "annotation_version, inside_official_interval, split, file_digest FROM records ORDER BY position"
        ):
            row = self._row_from_values(tuple(values))
            path = resolve_manifest_path(row, self._dataset_roots)
            if self._verify_files and not path.is_file():
                raise FileNotFoundError(f"cached P1C manifest file is missing: {path}")
            if self._verify_digests:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != row.file_digest:
                    raise ValueError(f"cached P1C manifest digest mismatch for {row.filepath}")

    def _create_or_verify_index(self) -> int:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        digest = self._manifest_digest()
        if self._index_path.is_file():
            connection = sqlite3.connect(self._index_path)
            try:
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
                if metadata.get("manifest_sha256") == digest and metadata.get("split") == self.split:
                    unknown_labels = [
                        label for (label,) in connection.execute("SELECT DISTINCT label FROM records")
                        if label not in self._label_to_index
                    ]
                    if unknown_labels:
                        raise ValueError(f"manifest labels are absent from class_names: {sorted(unknown_labels)}")
                    self._verify_cached_files(connection)
                    count = int(metadata["record_count"])
                    if count < 1:
                        raise ValueError(f"manifest has no rows for P1C split={self.split}")
                    return count
            finally:
                connection.close()
            self._index_path.unlink()
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{self._index_path.name}.", suffix=".tmp", dir=self._index_path.parent)
        os.close(descriptor)
        Path(temporary_name).unlink(missing_ok=True)
        temporary = Path(temporary_name)
        connection = sqlite3.connect(temporary)
        try:
            self._index_schema(connection)
            position = 0
            unknown_labels: set[str] = set()
            for row in iter_manifest(
                self._manifest_path,
                self._dataset_roots,
                verify_files=self._verify_files,
                verify_digests=self._verify_digests,
            ):
                if row.split != self.split:
                    continue
                if row.label not in self._label_to_index:
                    unknown_labels.add(row.label)
                connection.execute(
                    "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._row_values(row, position),
                )
                position += 1
            if unknown_labels:
                raise ValueError(f"manifest labels are absent from class_names: {sorted(unknown_labels)}")
            if position < 1:
                raise ValueError(f"manifest has no rows for P1C split={self.split}")
            connection.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                (("manifest_sha256", digest), ("split", self.split), ("record_count", str(position))),
            )
            connection.commit()
        except Exception:
            connection.close()
            temporary.unlink(missing_ok=True)
            raise
        connection.close()
        temporary.replace(self._index_path)
        return position

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(self._index_path)
        return self._connection

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_connection"] = None
        return state

    @staticmethod
    def _label_mapping(class_names: Sequence[str]) -> dict[str, int]:
        if isinstance(class_names, (str, bytes)) or not isinstance(class_names, Sequence) or not class_names:
            raise ValueError("class_names must be a non-empty ordered sequence")
        labels = list(class_names)
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("class_names must contain non-empty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("class_names must not contain duplicates")
        return {label: index for index, label in enumerate(labels)}

    @property
    def rows(self) -> tuple[ManifestRow, ...]:
        """Immutable selected rows for provenance-aware callers and tests."""

        return tuple(
            self._row_from_values(tuple(values))
            for values in self._db().execute(
                "SELECT filepath, source_dataset, source_video_id, source_frame_index, label, label_scope, label_source, "
                "annotation_version, inside_official_interval, split, file_digest FROM records ORDER BY position"
            )
        )

    def source_video_groups(self) -> frozenset[tuple[str, str]]:
        """Return compact distinct source-video keys without materializing every row."""

        return frozenset((str(dataset), str(video)) for dataset, video in self._db().execute(
            "SELECT DISTINCT source_dataset, source_video_id FROM records"
        ))

    def __len__(self) -> int:
        return self._record_count

    def _row_at(self, index: int) -> ManifestRow:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self._record_count:
            raise IndexError("P1C frame index is outside the selected manifest split")
        values = self._db().execute(
            "SELECT filepath, source_dataset, source_video_id, source_frame_index, label, label_scope, label_source, "
            "annotation_version, inside_official_interval, split, file_digest FROM records WHERE position = ?",
            (index,),
        ).fetchone()
        if values is None:
            raise RuntimeError("P1C manifest index is internally inconsistent")
        return self._row_from_values(tuple(values))

    def _sample_from_image(self, image: Any, row: ManifestRow) -> dict[str, Any]:
        """Finish one already-decoded PIL image into the exact sample dict.

        Both the per-file path (``Image.open``) and the packed-blob path
        (``Image.fromarray`` over a memmap slice) call this so that resize,
        normalization, and metadata assembly are one code path -- the
        guarantee that the two loading strategies produce bit-identical
        tensors for the same manifest row rests on that sharing, not on the
        two implementations happening to agree.
        """

        image = image.convert(self._preprocessing.color_mode)
        seed = augmentation_seed(row.filepath, row.file_digest, self._augmentation_run_seed, self._epoch, self._augmentation)
        image = self._augmentation.apply(
            image,
            resize_width=self._preprocessing.resize_width,
            resize_height=self._preprocessing.resize_height,
            resample=_INTERPOLATION[self._preprocessing.interpolation],
            seed=seed,
        )
        pixels = np.asarray(image, dtype=np.float32)
        if pixels.shape != (self._preprocessing.resize_height, self._preprocessing.resize_width, 3):
            raise ValueError(f"decoded RGB image has unexpected shape {pixels.shape}: {row.filepath}")
        tensor = torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).div_(255.0)
        mean = torch.tensor(self._preprocessing.mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self._preprocessing.std, dtype=tensor.dtype).view(3, 1, 1)
        return {
            "image": (tensor - mean) / std,
            "target": self._label_to_index[row.label],
            "source_dataset": row.source_dataset,
            "source_video_id": row.source_video_id,
            "source_frame_index": row.source_frame_index,
            "filepath": row.filepath,
            "file_digest": row.file_digest,
            "split": row.split,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._row_at(index)
        path = resolve_manifest_path(row, self._dataset_roots)
        try:
            with Image.open(path) as opened:
                return self._sample_from_image(opened, row)
        except OSError as exc:
            raise OSError(f"cannot decode P1C manifest image: {path}") from exc


@dataclass(frozen=True)
class PackedBlobInfo:
    """The subset of a pack-closure artifact (D-33) needed to read one split."""

    blob_path: Path
    blob_sha256: str
    item_bytes: int
    image_shape: tuple[int, int, int]
    row_count: int
    blob_row_offset: int
    split_row_count: int
    manifest_sha256: str

    @classmethod
    def from_closure(cls, closure_path: Path, *, split: str) -> "PackedBlobInfo":
        closure_path = closure_path.resolve()
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        splits = closure.get("splits")
        if not isinstance(splits, dict) or split not in splits:
            raise ValueError(f"pack closure {closure_path} has no split={split!r}")
        info = splits[split]
        image_shape = closure.get("image_shape")
        if not isinstance(image_shape, list) or len(image_shape) != 3:
            raise ValueError(f"pack closure {closure_path} has an invalid image_shape")
        blob_path = (closure_path.parent / closure["blob_path"]).resolve()
        return cls(
            blob_path=blob_path,
            blob_sha256=str(closure["blob_sha256"]),
            item_bytes=int(closure["item_bytes"]),
            image_shape=(int(image_shape[0]), int(image_shape[1]), int(image_shape[2])),
            row_count=int(closure["row_count"]),
            blob_row_offset=int(info["blob_row_offset"]),
            split_row_count=int(info["row_count"]),
            manifest_sha256=str(info["manifest_sha256"]),
        )


class P1CPackedFrameDataset(P1CFrameDataset):
    """Packed-blob-backed P1C frame dataset.

    Reads pixels from a contiguous ``numpy.memmap`` offset instead of opening
    and decoding one PNG per row, which is the fix for the measured
    loader-bound NFS/PNG-decode cost. Row metadata (label, source_video_id,
    provenance) still comes from the same manifest-backed sqlite index the
    per-file dataset uses, so both classes agree on row ordering, labels, and
    source-video grouping by construction -- only the pixel source differs.

    Per-file existence/digest verification is skipped here (the blob's own
    SHA-256 already covers every packed byte); the blob digest itself is
    verified exactly once, at construction, never per row and never per
    forked/spawned worker process.
    """

    def __init__(
        self,
        manifest_path: Path,
        dataset_roots: Mapping[str, Path],
        *,
        split: Literal["train", "validation", "test"],
        class_names: Sequence[str],
        preprocessing: FramePreprocessing,
        pack_closure_path: Path,
        allow_held_out_test: bool = False,
        augmentation: AugmentationPolicy | None = None,
        augmentation_run_seed: int = 0,
    ) -> None:
        super().__init__(
            manifest_path,
            dataset_roots,
            split=split,
            class_names=class_names,
            preprocessing=preprocessing,
            verify_files=False,
            verify_digests=False,
            allow_held_out_test=allow_held_out_test,
            augmentation=augmentation,
            augmentation_run_seed=augmentation_run_seed,
        )
        blob_info = PackedBlobInfo.from_closure(pack_closure_path, split=split)
        if blob_info.manifest_sha256 != self._manifest_digest():
            raise ValueError(
                f"pack closure {pack_closure_path} was built from a different {split} manifest "
                f"than {self._manifest_path}"
            )
        if blob_info.split_row_count != self._record_count:
            raise ValueError(
                f"pack closure {pack_closure_path} records {blob_info.split_row_count} {split} rows, "
                f"but the manifest has {self._record_count}"
            )
        if blob_info.item_bytes != math.prod(blob_info.image_shape):
            raise ValueError(f"pack closure {pack_closure_path} item_bytes does not match image_shape")
        if not blob_info.blob_path.is_file():
            raise FileNotFoundError(f"packed blob is missing: {blob_info.blob_path}")
        actual_blob_bytes = blob_info.blob_path.stat().st_size
        expected_blob_bytes = blob_info.row_count * blob_info.item_bytes
        if actual_blob_bytes != expected_blob_bytes:
            raise ValueError(
                f"packed blob {blob_info.blob_path} is {actual_blob_bytes} bytes, "
                f"expected {expected_blob_bytes} for {blob_info.row_count} rows"
            )
        actual_blob_sha256 = hashlib.sha256(blob_info.blob_path.read_bytes()).hexdigest()
        if actual_blob_sha256 != blob_info.blob_sha256:
            raise ValueError(
                f"packed blob {blob_info.blob_path} failed SHA-256 verification: "
                f"expected {blob_info.blob_sha256}, got {actual_blob_sha256}"
            )
        self._blob_info = blob_info
        self._memmap: np.memmap | None = None

    def _blob(self) -> np.memmap:
        if self._memmap is None:
            info = self._blob_info
            self._memmap = np.memmap(
                info.blob_path, dtype=np.uint8, mode="r", shape=(info.row_count, *info.image_shape)
            )
        return self._memmap

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        # A memmap handle is process-local; a forked/spawned DataLoader worker
        # reopens it lazily via ``_blob()`` on first access instead of
        # inheriting (or re-verifying) this process's handle or digest.
        state["_memmap"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._row_at(index)
        blob_row = self._blob_info.blob_row_offset + index
        array = np.array(self._blob()[blob_row])  # copy out of the memmap page
        image = Image.fromarray(array, mode="RGB")
        return self._sample_from_image(image, row)


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python, NumPy, and torch from the DataLoader-assigned worker seed."""

    if isinstance(worker_id, bool) or not isinstance(worker_id, int) or worker_id < 0:
        raise ValueError("worker_id must be a non-negative integer")
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_p1c_frame_dataloader(
    dataset: Dataset[Any], *, options: FrameLoaderOptions
) -> DataLoader[Any]:
    """Build a deterministic P1C frame loader with explicit worker behavior."""

    if not isinstance(options, FrameLoaderOptions):
        raise TypeError("options must be FrameLoaderOptions")
    # Every epoch has a reproducible, distinct order. A resumed run recreates
    # its loader using the same (run seed, epoch) pair instead of attempting
    # to serialize private sampler/worker state.
    payload = f"p1c-frame-loader\0{options.seed}\0{options.epoch}".encode("utf-8")
    epoch_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    generator = torch.Generator()
    generator.manual_seed(epoch_seed)
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": options.batch_size,
        "shuffle": options.shuffle,
        "num_workers": options.num_workers,
        "pin_memory": options.pin_memory,
        "drop_last": options.drop_last,
        "persistent_workers": options.persistent_workers,
        "generator": generator,
        "worker_init_fn": seed_dataloader_worker,
    }
    if options.num_workers:
        # Use a platform-independent start method: worker initialization is
        # therefore proven rather than inherited from the parent process.
        arguments["multiprocessing_context"] = "spawn"
    return DataLoader(**arguments)
