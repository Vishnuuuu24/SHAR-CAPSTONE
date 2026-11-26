#!/usr/bin/env python3
"""Pack P1B Train/validation manifests into one contiguous uint8 frame blob.

Background (see docs/DECISIONS.md D-33). P1C training on the DGX is
loader-bound: the registered Train/validation manifests reference 1,136,536
and 129,809 individual 64x64x3 PNG files respectively, each an open+decode
over NFS. This script performs the expensive per-file work exactly once,
verifying every file's SHA-256 against its manifest ``file_digest`` and every
decoded image's shape/dtype, then writes image ``i`` (in manifest row order)
to byte offset ``i * 12288`` of one contiguous blob. ``data.p1c_loader``'s
``P1CPackedFrameDataset`` then reads that blob with ``numpy.memmap`` instead
of opening and decoding a PNG per row.

Train and validation only. There is deliberately no ``--test-manifest``
option: the Test split is gated (B-001) and its manifests are stale, so
packing it is out of scope for this tool, not merely unused.

Fail-closed: any SHA-256 mismatch or any decoded image that is not exactly
64x64x3 uint8 RGB aborts the whole run (an unhandled exception propagates to
a nonzero exit). Nothing about scientific scope is decided here.

Resumable: the expensive per-file verify+decode+write loop tracks completed
row counts per split in ``pack_state.json`` and skips already-completed rows
on the next invocation without re-touching their source files. The cheap
metadata/label copy (no file opens, no hashing) is rebuilt in full on every
invocation because it is fast and this keeps it trivially idempotent. The
closure artifact (``pack_closure.json``) is written once, only after every
row in both splits is verified and packed, and refuses to be silently
overwritten by a second run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.progress import BatchProgress, log_phase  # noqa: E402
from data.contracts import MANIFEST_FIELDS, validate_manifest_row  # noqa: E402
from data.inventory import sha256_file  # noqa: E402
from data.manifest import (  # noqa: E402
    ManifestRow,
    _nullable_boolean,
    _nullable_integer,
    resolve_manifest_path,
)

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 3
ITEM_BYTES = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
PACKER_CODE_VERSION = "p1c-pack-v1"
LABEL_DTYPE = "<U32"
DEFAULT_DATASET_KEY = "ucf_crime_kaggle_frames"
DEFAULT_DATASET_ROOT = REPO_ROOT / "Datasets" / "UCF Crime Dataset Kaggle"
DEFAULT_TRAIN_MANIFEST = (
    REPO_ROOT
    / "results/p1b/p1b-grouped-train-validation-57f3a29b-10pct-s0/train_manifest.csv"
)
DEFAULT_VALIDATION_MANIFEST = (
    REPO_ROOT
    / "results/p1b/p1b-grouped-train-validation-57f3a29b-10pct-s0/validation_manifest.csv"
)
_SPLIT_NAMES = ("train", "validation")
_STATE_FIELDS = {
    "train_manifest_sha256",
    "validation_manifest_sha256",
    "train_row_count",
    "validation_row_count",
    "train_rows_done",
    "validation_rows_done",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _read_header(manifest_path: Path) -> list[str]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def _manifest_row_count(manifest_path: Path) -> int:
    header = _read_header(manifest_path)
    if header != MANIFEST_FIELDS:
        raise ValueError(f"manifest header/order must be exactly {MANIFEST_FIELDS}; found {header}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _decode_verified_image(path: Path) -> np.ndarray:
    """Decode exactly as ``data.p1c_loader.P1CFrameDataset`` does before resize."""

    try:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            array = np.asarray(image, dtype=np.uint8)
    except OSError as exc:
        raise OSError(f"cannot decode manifest image: {path}") from exc
    if array.shape != IMAGE_SHAPE or array.dtype != np.uint8:
        raise ValueError(f"decoded image is not {IMAGE_SHAPE} uint8 RGB (got shape={array.shape}, dtype={array.dtype}): {path}")
    return array


def _iter_verified_rows(
    manifest_path: Path, dataset_roots: dict[str, Path], *, skip: int
) -> Iterator[tuple[int, ManifestRow, Path]]:
    """Yield ``(row_index, row, resolved_path)`` from ``skip`` onward.

    Rows before ``skip`` are not parsed at all (a bare line skip on the csv
    reader), so a resumed run pays no cost -- not even a digest -- for rows a
    previous run already verified and wrote.
    """

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise ValueError(f"manifest header/order must be exactly {MANIFEST_FIELDS}; found {reader.fieldnames}")
        for index, raw in enumerate(reader):
            if index < skip:
                continue
            line = index + 2
            typed = {
                **raw,
                "source_frame_index": _nullable_integer(raw["source_frame_index"], "source_frame_index", line),
                "inside_official_interval": _nullable_boolean(
                    raw["inside_official_interval"], "inside_official_interval", line
                ),
            }
            issues = validate_manifest_row(typed)
            if issues:
                raise ValueError(f"line {line}: {'; '.join(issues)}")
            row = ManifestRow(**typed)
            path = resolve_manifest_path(row, dataset_roots)
            if not path.is_file():
                raise FileNotFoundError(f"line {line}: manifest file is missing: {path}")
            digest, _, _ = sha256_file(path)
            if digest != row.file_digest:
                raise ValueError(
                    f"line {line}: content digest mismatch for {row.filepath}: "
                    f"manifest={row.file_digest} actual={digest}"
                )
            yield index, row, path


def _write_metadata_and_labels(
    train_manifest: Path, validation_manifest: Path, *, metadata_path: Path, labels_path: Path
) -> tuple[int, int]:
    """Rebuild the packed-order metadata/label copy in full, atomically.

    This never opens an image file -- it is a pure CSV-to-CSV copy of the two
    already-frozen manifests concatenated in packed row order (train then
    validation), so it is cheap enough to redo unconditionally on every
    invocation rather than tracking its own resume state.
    """

    labels: list[str] = []
    descriptor, temporary_metadata_name = tempfile.mkstemp(prefix=f".{metadata_path.name}.", dir=metadata_path.parent)
    temporary_metadata = Path(temporary_metadata_name)
    counts: dict[Path, int] = {}
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for manifest_path in (train_manifest, validation_manifest):
                header = _read_header(manifest_path)
                if header != MANIFEST_FIELDS:
                    raise ValueError(f"manifest header/order must be exactly {MANIFEST_FIELDS}; found {header}")
                count = 0
                with manifest_path.open("r", encoding="utf-8", newline="") as source:
                    reader = csv.DictReader(source)
                    for raw in reader:
                        writer.writerow(raw)
                        labels.append(raw["label"])
                        count += 1
                counts[manifest_path] = count
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_metadata, metadata_path)
    except Exception:
        temporary_metadata.unlink(missing_ok=True)
        raise
    label_array = np.array(labels, dtype=LABEL_DTYPE)
    if any(label_array[i] != labels[i] for i in range(0, len(labels), max(1, len(labels) // 64) or 1)):
        raise ValueError(f"LABEL_DTYPE={LABEL_DTYPE} truncated a label; widen it")
    descriptor2, temporary_labels_name = tempfile.mkstemp(prefix=f".{labels_path.name}.", suffix=".npy", dir=labels_path.parent)
    os.close(descriptor2)
    temporary_labels = Path(temporary_labels_name)
    try:
        np.save(temporary_labels, label_array, allow_pickle=False)
        os.replace(temporary_labels, labels_path)
    except Exception:
        temporary_labels.unlink(missing_ok=True)
        raise
    return counts[train_manifest], counts[validation_manifest]


def pack_frames(
    *,
    train_manifest: Path,
    validation_manifest: Path,
    dataset_root: Path,
    dataset_key: str,
    output_dir: Path,
    checkpoint_every: int = 5000,
) -> dict[str, Any]:
    """Pack Train+validation manifests into one blob; return the closure record.

    Idempotent/resumable: safe to re-invoke with the same arguments after an
    interruption. Raises (fail-closed) on any digest mismatch, malformed
    image, size mismatch against a prior partial run, or an attempt to
    re-pack into a directory that already holds a completed closure.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    blob_path = output_dir / "frames.bin"
    metadata_path = output_dir / "metadata.csv"
    labels_path = output_dir / "labels.npy"
    state_path = output_dir / "pack_state.json"
    closure_path = output_dir / "pack_closure.json"

    if closure_path.is_file():
        raise FileExistsError(
            f"pack is already closed at {closure_path}; use a fresh --output-dir for a different pack"
        )

    dataset_roots = {dataset_key: dataset_root}
    train_manifest_sha256 = _sha256(train_manifest)
    validation_manifest_sha256 = _sha256(validation_manifest)
    train_row_count = _manifest_row_count(train_manifest)
    validation_row_count = _manifest_row_count(validation_manifest)
    if train_row_count < 1 or validation_row_count < 1:
        raise ValueError("both manifests must contain at least one row")
    total_rows = train_row_count + validation_row_count
    expected_blob_bytes = total_rows * ITEM_BYTES

    log_phase("rebuilding metadata.csv and labels.npy (cheap CSV copy, no image reads)")
    meta_train_count, meta_validation_count = _write_metadata_and_labels(
        train_manifest, validation_manifest, metadata_path=metadata_path, labels_path=labels_path
    )
    if meta_train_count != train_row_count or meta_validation_count != validation_row_count:
        raise ValueError("metadata pass row counts disagree with the manifest row counts")

    existing_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
    if existing_state is not None:
        if set(existing_state) != _STATE_FIELDS:
            raise ValueError(f"pack_state.json has unexpected fields: {sorted(set(existing_state) ^ _STATE_FIELDS)}")
        if (
            existing_state["train_manifest_sha256"] != train_manifest_sha256
            or existing_state["validation_manifest_sha256"] != validation_manifest_sha256
            or existing_state["train_row_count"] != train_row_count
            or existing_state["validation_row_count"] != validation_row_count
        ):
            raise ValueError(
                f"{state_path} was started from different manifests; use a fresh --output-dir "
                "rather than resuming with mismatched inputs"
            )
        state = dict(existing_state)
    else:
        state = {
            "train_manifest_sha256": train_manifest_sha256,
            "validation_manifest_sha256": validation_manifest_sha256,
            "train_row_count": train_row_count,
            "validation_row_count": validation_row_count,
            "train_rows_done": 0,
            "validation_rows_done": 0,
        }
        _atomic_write_json(state_path, state)

    if blob_path.is_file():
        actual_size = blob_path.stat().st_size
        if actual_size != expected_blob_bytes:
            raise ValueError(
                f"existing {blob_path} is {actual_size} bytes, expected {expected_blob_bytes}; "
                "resume is unsafe here, use a fresh --output-dir"
            )
    else:
        with blob_path.open("wb") as handle:
            handle.truncate(expected_blob_bytes)

    verification_pass_count = state["train_rows_done"] + state["validation_rows_done"]
    with blob_path.open("r+b") as blob_handle:
        for split, manifest_path, row_count, blob_row_offset in (
            ("train", train_manifest, train_row_count, 0),
            ("validation", validation_manifest, validation_row_count, train_row_count),
        ):
            done_key = f"{split}_rows_done"
            done = state[done_key]
            if done >= row_count:
                log_phase(f"{split}: already fully packed ({done}/{row_count}); skipping")
                continue
            log_phase(f"{split}: packing rows {done}..{row_count - 1} (resuming)" if done else f"{split}: packing {row_count} rows")
            progress = BatchProgress(row_count, label=f"pack {split}", every=checkpoint_every, min_interval_seconds=15.0)
            for _ in range(done):
                progress.step()
            last_checkpoint = done
            for index, row, path in _iter_verified_rows(manifest_path, dataset_roots, skip=done):
                array = _decode_verified_image(path)
                blob_handle.seek((blob_row_offset + index) * ITEM_BYTES)
                blob_handle.write(array.tobytes())
                verification_pass_count += 1
                progress.step()
                if (index + 1) - last_checkpoint >= checkpoint_every:
                    state[done_key] = index + 1
                    _atomic_write_json(state_path, state)
                    last_checkpoint = index + 1
            state[done_key] = row_count
            _atomic_write_json(state_path, state)
            progress.done()
            blob_handle.flush()
            os.fsync(blob_handle.fileno())

    if verification_pass_count != total_rows:
        raise RuntimeError(
            f"internal inconsistency: verified {verification_pass_count} rows, expected {total_rows}"
        )

    log_phase("packing complete; hashing the final blob for the closure record")
    blob_sha256 = _sha256(blob_path)

    closure = {
        "kind": "P1C packed frame blob (D-33 registered research artifact); Train+validation only",
        "packer_code_version": PACKER_CODE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": total_rows,
        "per_file_verification_pass_count": verification_pass_count,
        "item_bytes": ITEM_BYTES,
        "image_shape": list(IMAGE_SHAPE),
        "image_dtype": "uint8",
        "blob_path": "frames.bin",
        "blob_sha256": blob_sha256,
        "blob_bytes": expected_blob_bytes,
        "metadata_path": "metadata.csv",
        "labels_path": "labels.npy",
        "dataset_root": str(dataset_root),
        "dataset_key": dataset_key,
        "splits": {
            "train": {
                "row_count": train_row_count,
                "blob_row_offset": 0,
                "manifest_path": str(train_manifest),
                "manifest_sha256": train_manifest_sha256,
            },
            "validation": {
                "row_count": validation_row_count,
                "blob_row_offset": train_row_count,
                "manifest_path": str(validation_manifest),
                "manifest_sha256": validation_manifest_sha256,
            },
        },
        "excludes": "test split is out of scope for this packer; no --test-manifest option exists",
        "source_video_grouping": "preserved verbatim from the source manifests via metadata.csv and labels.npy, in packed row order",
    }
    _atomic_write_json(closure_path, closure)
    log_phase(f"closed: {closure_path}")
    return closure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-key", type=str, default=DEFAULT_DATASET_KEY)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    args = parser.parse_args()

    train_manifest = args.train_manifest.resolve()
    validation_manifest = args.validation_manifest.resolve()
    dataset_root = args.dataset_root.resolve()
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be a positive integer")

    output_dir = args.output_dir
    if output_dir is None:
        train_tag = _sha256(train_manifest)[:8]
        validation_tag = _sha256(validation_manifest)[:8]
        output_dir = REPO_ROOT / "results" / "p1c" / f"p1c-packed-frames-{train_tag}-{validation_tag}"
    output_dir = output_dir.resolve()

    started = time.perf_counter()
    log_phase(f"packing {train_manifest} + {validation_manifest} -> {output_dir}")
    closure = pack_frames(
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        dataset_root=dataset_root,
        dataset_key=args.dataset_key,
        output_dir=output_dir,
        checkpoint_every=args.checkpoint_every,
    )
    log_phase(f"done in {time.perf_counter() - started:.1f}s")
    print(json.dumps(closure, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
