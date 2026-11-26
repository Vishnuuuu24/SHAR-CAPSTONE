"""Build P1B-shaped manifests from the D-34/D-35 video blob.

One manifest row is one **kept frame** in the blob. The row carries the source
MP4's path and digest rather than a per-frame digest, because there is no
per-frame file to hash any more -- the pixels live inside one packed blob whose
own SHA-256 is verified at load. Hashing the source video is the honest
verifiable thing: it proves the pixels came from the video the row names.

Rows do not store a blob row number. The loader resolves
`(source_video_id, source_frame_index)` to a blob row through the blob's own
frame index, so the manifest stays a statement about *data*, not about one
artifact's internal layout -- and a re-extraction that changes row ordering
does not invalidate every manifest.

Label honesty (docs/PROJECT_RULES.md rule 1): every row here is `video_inherited`. A frame
in an anomalous video is not known to contain the anomaly; it inherits the
video's label. Nothing in this module upgrades that, and
`inside_official_interval` is left empty because the temporal intervals belong
to the anomaly-detection protocol's test split and are not frame-level truth
for the classification task.

Grouping (docs/PROJECT_RULES.md, "preserve source-video grouping end to end"): the
train/validation carve is by whole source video, never by frame. Splitting one
video's frames across train and validation would leak near-identical
neighbouring frames across the boundary and inflate validation scores.
"""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from data.official_splits import (
    ACTION_RECOGNITION,
    ANOMALY_DETECTION,
    SplitEntry,
    local_directory_for,
    read_action_recognition_split,
    read_anomaly_detection_split,
)

SOURCE_DATASET = "ucf_crime_official_video_frames"
LABEL_SCOPE = "video_inherited"

MANIFEST_FIELDS = [
    "filepath",
    "source_dataset",
    "source_video_id",
    "source_frame_index",
    "label",
    "label_scope",
    "label_source",
    "annotation_version",
    "inside_official_interval",
    "split",
    "file_digest",
]


@dataclass(frozen=True)
class VideoRow:
    """One row of the extractor's `videos.csv`, narrowed to what is needed."""

    video_id: int
    source_directory: str
    file_name: str
    relative_path: str
    file_sha256: str
    kept_frame_count: int
    first_row_index: int

    @property
    def stem(self) -> str:
        return self.file_name[: -len(".mp4")] if self.file_name.endswith(".mp4") else self.file_name


def read_videos_csv(path: Path) -> dict[tuple[str, str], VideoRow]:
    """Read `videos.csv`, keyed by ``(source_directory, stem)``.

    Keyed by the pair, not the stem alone, because **stems are not unique
    across the corpus**. Measured on 2026-08-25: all 50 videos in
    `z_Normal_Videos_event` are byte-identical copies (same SHA-256) of 50 of
    the 150 files in `Testing_Normal_Videos_Anomaly`, so the 1,950 files hold
    only 1,900 distinct videos. Keying by stem alone would silently pick one of
    the two directories and could assign an Action-Recognition Normal frame the
    provenance of an Anomaly-Detection test video.

    The official split lists always name their directory, so the pair is
    unambiguous at every call site.
    """
    rows: dict[tuple[str, str], VideoRow] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            row = VideoRow(
                video_id=int(record["video_id"]),
                source_directory=record["source_directory"],
                file_name=record["file_name"],
                relative_path=record["relative_path"],
                file_sha256=record["file_sha256"],
                kept_frame_count=int(record["kept_frame_count"]),
                first_row_index=int(record["first_row_index"]),
            )
            key = (row.source_directory, row.stem)
            if key in rows:
                raise ValueError(f"{path}: duplicate video {key!r}")
            rows[key] = row
    return rows


def load_protocol_split(
    ucf_crimes_root: Path, *, protocol: str, fold: int | None
) -> dict[str, list[SplitEntry]]:
    if protocol == ACTION_RECOGNITION:
        if fold is None:
            raise ValueError("action_recognition requires an explicit --fold")
        return read_action_recognition_split(ucf_crimes_root, fold=fold)
    if protocol == ANOMALY_DETECTION:
        if fold is not None:
            raise ValueError("anomaly_detection has no folds; omit --fold")
        return read_anomaly_detection_split(ucf_crimes_root)
    raise ValueError(f"unknown protocol {protocol!r}")


def carve_validation(
    entries: Sequence[SplitEntry],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[SplitEntry], list[SplitEntry]]:
    """Split train entries into (train, validation) by whole source video.

    Stratified per class so a small validation fraction cannot silently drop a
    class entirely, and deterministic given `seed`. Every class contributes at
    least one validation video when it has more than one video available.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")

    by_class: dict[str, list[SplitEntry]] = {}
    for entry in entries:
        by_class.setdefault(entry.class_name, []).append(entry)

    rng = random.Random(seed)
    train: list[SplitEntry] = []
    validation: list[SplitEntry] = []
    for class_name in sorted(by_class):
        videos = sorted(by_class[class_name], key=lambda e: e.relative_path)
        rng.shuffle(videos)
        take = int(round(len(videos) * fraction))
        take = max(1, take) if len(videos) > 1 else 0
        take = min(take, len(videos) - 1) if len(videos) > 1 else 0
        validation.extend(videos[:take])
        train.extend(videos[take:])

    train.sort(key=lambda e: e.relative_path)
    validation.sort(key=lambda e: e.relative_path)
    return train, validation


def manifest_rows(
    entries: Iterable[SplitEntry],
    *,
    videos: dict[str, VideoRow],
    frame_indices: dict[str, list[int]],
    split: str,
    label_source: str,
    annotation_version: str,
) -> Iterator[dict[str, object]]:
    """Yield one manifest row per kept frame of every entry, in stable order."""
    for entry in entries:
        local_directory = local_directory_for(entry)
        video = videos.get((local_directory, entry.video_id))
        if video is None:
            raise KeyError(
                f"{entry.relative_path} is in the official split but not in videos.csv "
                f"under {local_directory!r}; the blob does not cover this protocol"
            )
        local_relative = f"{local_directory}/{entry.video_id}.mp4"
        if video.relative_path != local_relative:
            raise ValueError(
                f"{entry.video_id}: videos.csv path {video.relative_path!r} does not match "
                f"the official split's {local_relative!r}"
            )
        for source_frame_index in frame_indices[(local_directory, entry.video_id)]:
            yield {
                "filepath": video.relative_path,
                "source_dataset": SOURCE_DATASET,
                "source_video_id": entry.video_id,
                "source_frame_index": source_frame_index,
                "label": entry.class_name,
                "label_scope": LABEL_SCOPE,
                "label_source": label_source,
                "annotation_version": annotation_version,
                "inside_official_interval": "",
                "split": split,
                "file_digest": video.file_sha256,
            }


def write_manifest(path: Path, rows: Iterable[dict[str, object]]) -> tuple[int, str]:
    """Write a manifest and return (row_count, sha256_of_the_file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return count, digest.hexdigest()


def assert_no_video_leakage(splits: dict[str, Sequence[SplitEntry]]) -> None:
    """Refuse any source video appearing in more than one split.

    This is the check that makes the grouping guarantee real rather than
    aspirational, and it runs before anything is written.
    """
    seen: dict[str, str] = {}
    for split_name in sorted(splits):
        for entry in splits[split_name]:
            previous = seen.get(entry.video_id)
            if previous is not None and previous != split_name:
                raise ValueError(
                    f"source video {entry.video_id} appears in both {previous!r} "
                    f"and {split_name!r}"
                )
            seen[entry.video_id] = split_name

def assert_consistent_label_vocabulary(splits: dict[str, Sequence[SplitEntry]]) -> None:
    """Refuse a protocol whose splits do not share one label vocabulary.

    The anomaly-detection protocol fails this, by design rather than by
    accident: its Normal videos live in `Training_Normal_Videos_Anomaly` (800)
    and `Testing_Normal_Videos_Anomaly` (150), which are different directory
    names and therefore different labels here. Used as-is for 14-class
    classification, a model would learn a class the test split never contains
    and be scored on one it never saw.

    Merging those two directories into a single `Normal` class is a defensible
    decision, but it is a *label mapping*, and docs/PROJECT_RULES.md forbids inventing one.
    So this fails closed and names the decision instead of quietly making it.
    """
    vocabularies = {
        name: {entry.class_name for entry in entries} for name, entries in splits.items()
    }
    reference_name, reference = next(iter(vocabularies.items()))
    for name, vocabulary in vocabularies.items():
        if vocabulary != reference:
            only_here = sorted(vocabulary - reference)
            only_there = sorted(reference - vocabulary)
            raise ValueError(
                f"label vocabularies differ between {reference_name!r} and {name!r}: "
                f"only in {name}={only_here}, only in {reference_name}={only_there}. "
                "The anomaly-detection protocol hits this because its Normal videos sit in "
                "two differently-named directories. Resolving it requires a registered "
                "label-mapping decision (e.g. merge both into 'Normal'); this tool will not "
                "invent one."
            )
