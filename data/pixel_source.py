"""Pixel-source resolution shared by the P1C calibration and baseline runners.

``scripts/calibrate_p1c_frame_loader.py`` and ``scripts/run_p1c_frame_baseline.py``
each resolve a registered ``fixed_protocol.data`` mapping into exactly one
training pixel source before building datasets:

- absent ``packed_frames``/``video_frames`` -> per-file PNG/JPEG loading
  (``data.p1c_loader.P1CFrameDataset``), the original behaviour;
- ``data.packed_frames`` -> the D-33 Kaggle-sourced packed blob
  (``data.p1c_loader.P1CPackedFrameDataset``);
- ``data.video_frames`` -> the D-34/D-35 official-video-sourced packed JPEG
  blob (``data.video_dataset.P1CVideoFrameDataset``).

Both scripts used to duplicate this field parsing, the both-registered
refusal, and the ``verify_digests`` enforcement. Per docs/PROJECT_RULES.md's rule of two,
the second copy is extracted here so the two runners cannot silently drift
about what a config field means -- a typo or a loosened field must fail the
same way regardless of which script reads it.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Protocol

from data.p1c_loader import PackedBlobInfo
from data.video_blob import VideoBlobClosure
from data.video_manifests import SOURCE_DATASET


def packed_frames_field(data: dict[str, Any]) -> dict[str, Any] | None:
    """Return the registered ``data.packed_frames`` block, or ``None`` if absent.

    Absence must leave every existing behavior unchanged (per-file loading);
    presence must be an exact, explicit ``pack_closure_path`` registration
    rather than a loosely-shaped mapping this code happens to tolerate.
    """

    packed = data.get("packed_frames")
    if packed is None:
        return None
    if (
        not isinstance(packed, dict)
        or set(packed) != {"pack_closure_path"}
        or not isinstance(packed.get("pack_closure_path"), str)
        or not packed["pack_closure_path"]
    ):
        raise ValueError("data.packed_frames must contain exactly a non-empty string pack_closure_path")
    return packed


def video_frames_field(data: dict[str, Any]) -> dict[str, Any] | None:
    """Return the registered ``data.video_frames`` block, or ``None`` if absent.

    D-34/D-35 source. Shaped exactly like ``data.packed_frames`` and validated
    the same way -- an exact key set, not a loosely-shaped mapping this code
    happens to tolerate -- so that a typo in a config is a failure rather than
    a silent fallback to a different pixel source.

    Registering both ``packed_frames`` and ``video_frames`` is refused by
    ``refuse_if_both_registered``: they are two different pixel sources and a
    config must say which one it means.
    """

    video = data.get("video_frames")
    if video is None:
        return None
    if (
        not isinstance(video, dict)
        or set(video) != {"extract_closure_path"}
        or not isinstance(video.get("extract_closure_path"), str)
        or not video["extract_closure_path"]
    ):
        raise ValueError(
            "data.video_frames must contain exactly a non-empty string extract_closure_path"
        )
    return video


def pack_closure_path(data: dict[str, Any]) -> Path | None:
    packed = packed_frames_field(data)
    if packed is None:
        return None
    return Path(packed["pack_closure_path"])


def extract_closure_path(data: dict[str, Any]) -> Path | None:
    video = video_frames_field(data)
    if video is None:
        return None
    return Path(video["extract_closure_path"])


def refuse_if_both_registered(data: dict[str, Any]) -> None:
    """Fail closed when a config registers two pixel sources at once."""

    if pack_closure_path(data) is not None and extract_closure_path(data) is not None:
        raise ValueError(
            "data.packed_frames (D-33, Kaggle-sourced) and data.video_frames (D-34/D-35, "
            "official-video-sourced) are different pixel sources; register exactly one"
        )


_VERIFY_DIGESTS_REASON = {
    "packed_frames": (
        "data.packed_frames requires training.loader.verify_digests=true: under the packed-blob "
        "path (D-33) this field no longer gates a per-file check -- the blob's own SHA-256 is "
        "always verified once at dataset construction instead -- but it must still be registered "
        "true so a config cannot claim a verification posture weaker than what the run performs"
    ),
    "video_frames": (
        "data.video_frames requires training.loader.verify_digests=true: the blob's own "
        "SHA-256 is verified once at dataset construction, and a config must not claim a "
        "verification posture weaker than what the run performs"
    ),
}


def require_verify_digests_true(protocol: dict[str, Any], *, field: str) -> None:
    """Fail closed unless ``training.loader.verify_digests`` is registered ``true``.

    Both blob paths verify their own SHA-256 exactly once, at dataset
    construction, in place of a per-file digest walk. ``verify_digests`` no
    longer gates that walk under either branch, but it must still read
    ``true`` -- a config that registered ``false`` would claim a weaker
    verification posture than the run actually performs, which docs/PROJECT_RULES.md's
    reproducibility rule does not allow.
    """

    if protocol["training"]["loader"]["verify_digests"] is not True:
        raise ValueError(_VERIFY_DIGESTS_REASON[field])


def packed_blob_sha256(protocol: dict[str, Any]) -> str | None:
    """The registered D-33 packed blob's own SHA-256, or ``None`` when unpacked.

    Read from the pack closure's own declared field (cheap JSON parse), the
    same way a run manifest records a manifest's registered SHA-256 rather
    than a value recomputed at this point -- the dataset itself performs the
    actual once-at-construction verification against the real blob bytes.
    """

    path = pack_closure_path(protocol["data"])
    if path is None:
        return None
    return PackedBlobInfo.from_closure(path, split="train").blob_sha256


def video_blob_sha256(protocol: dict[str, Any]) -> str | None:
    """The registered D-34/D-35 video blob's own SHA-256, or ``None`` when absent.

    Mirrors ``packed_blob_sha256``: a cheap read of the extract closure's own
    declared field, not the (expensive, 84 GiB) blob recomputed here.
    """

    path = extract_closure_path(protocol["data"])
    if path is None:
        return None
    return VideoBlobClosure.load(path).blob_sha256


class _GroupedDataset(Protocol):
    def source_video_groups(self) -> frozenset[tuple[str, str]]: ...


def assert_no_source_video_leakage(train: _GroupedDataset, validation: _GroupedDataset, *, message: str) -> None:
    """Refuse a Train/validation pair that shares a source video.

    Works identically across every pixel source -- per-file, packed-blob, and
    video-blob datasets all implement ``source_video_groups()`` -- so the
    check runs once here regardless of which dataset class built the split.
    """

    if train.source_video_groups() & validation.source_video_groups():
        raise ValueError(message)


def loader_mode(*, packed_blob_sha256: str | None, video_blob_sha256: str | None) -> str:
    """The resolved pixel source as one of the three names both scripts log.

    Both runners independently computed this three-way ``video_blob`` /
    ``packed_blob`` / ``per_file`` label from the same two SHA-256 fields
    (once each for a log line, once each for their artifact); per docs/PROJECT_RULES.md's
    rule of two, one copy lives here so the log and the artifact cannot say
    different things.
    """

    if video_blob_sha256 is not None:
        return "video_blob"
    if packed_blob_sha256 is not None:
        return "packed_blob"
    return "per_file"


def _manifest_source_dataset(manifest_path: Path, *, label: str) -> str:
    """Read the ``source_dataset`` of a manifest's first data row, and no more.

    ``data/video_manifests.py`` writes the same ``SOURCE_DATASET`` value into
    every row it emits, so one row settles the yes/no provenance question
    this exists to answer. The failure this guards against
    (``refuse_if_blob_manifest_lacks_pixel_source`` below) was itself caused
    by a loop over every manifest row -- 391,014 of them, each re-hashing a
    ~50 MB source MP4 -- so this helper must not do that either.
    """

    if not manifest_path.is_file():
        raise ValueError(f"{label}: manifest file not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{label}: manifest is empty (no header row): {manifest_path}")
        if "source_dataset" not in reader.fieldnames:
            raise ValueError(f"{label}: manifest has no source_dataset column: {manifest_path}")
        try:
            row = next(reader)
        except StopIteration:
            raise ValueError(f"{label}: manifest has a header row but no data rows: {manifest_path}") from None
    return row["source_dataset"]


def refuse_if_blob_manifest_lacks_pixel_source(manifest_path: Path, data: dict[str, Any], *, label: str) -> None:
    """Refuse a blob-derived manifest that registers no pixel source.

    K-21: ``P1C-VIDEO-FRAME-BASELINE-011-AR-FOLD1.yaml`` was drafted without a
    ``data.video_frames`` block. With neither ``video_frames`` nor
    ``packed_frames`` registered, both ``_make_datasets`` branches (baseline
    runner and calibration script) resolved to ``None`` and fell through to
    ``P1CFrameDataset``, which -- because the manifest's rows point at the
    original MP4s and ``verify_digests`` was true -- re-hashed a source video
    once per manifest row. For BASELINE-011's train manifest that was 391,014
    rows against a 97.7 GiB corpus: 39 TB of reads and 7h25m of GPU-slice time
    burned before anyone noticed, and it would have run far longer. Nothing
    refused; it just got very slow -- exactly the silent-degradation failure
    ``core/precision.py`` and ``core/throughput.py`` exist to avoid.

    A manifest is fatal-if-unregistered only when it is actually blob-derived
    (``source_dataset == SOURCE_DATASET``); a genuinely per-file manifest with
    no pixel source registered is the original, still-legal behaviour.
    """

    if pack_closure_path(data) is not None or extract_closure_path(data) is not None:
        return  # a pixel source is registered; nothing to refuse
    if _manifest_source_dataset(manifest_path, label=label) != SOURCE_DATASET:
        return  # genuinely per-file manifest; per-file loading is correct
    raise ValueError(
        f"{label} is blob-derived (source_dataset={SOURCE_DATASET!r}, {manifest_path}) but this "
        "config registers neither data.video_frames.extract_closure_path nor data.packed_frames. "
        "Falling through to per-file loading would not fail -- it would silently re-hash the "
        "source MP4 once per manifest row. Measured cost of exactly this mistake "
        "(P1C-VIDEO-FRAME-BASELINE-011-AR-FOLD1, 391,014 rows): 39 TB of reads against a 97.7 GiB "
        "corpus, 7h25m of GPU-slice time before anyone noticed. Register "
        "data.video_frames.extract_closure_path (or data.packed_frames, if the packed blob is "
        "what this manifest actually came from)."
    )
