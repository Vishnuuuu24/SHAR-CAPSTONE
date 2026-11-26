"""Parse the two official UCF-Crime split protocols.

The official download ships **two** protocols under
`Datasets/UCF Crime Official/raw/UCF_Crimes/`, and conflating them is a
task-separation error (docs/PROJECT_RULES.md rule 3). See the two-protocol row in
`docs/EVIDENCE_REGISTER.md` for the verified structure.

**Action Recognition** (`Action_Regnition_splits/` -- the misspelling is
upstream): 14 classes listed in `ClassIDs.txt`, where class 8 is
`Normal_Videos_event`. Four folds, each 532 train + 168 test, perfectly
balanced at 38 train + 12 test per class, 700 videos in total, and the four
test sets are exactly disjoint. This is SHAR's RESULT 2 task verbatim -- "the
13 anomaly labels plus Normal".

**Anomaly Detection** (`Anomaly_Detection_splits/`): 1,610 train + 290 test =
1,900 videos, the temporal/binary task, with the interval annotations under
`Temporal_Anomaly_Annotation_For_Testing_Videos/` attached to its test split.
Its Normal videos live in `Training_Normal_Videos_Anomaly` (800) and
`Testing_Normal_Videos_Anomaly` (150), which are *not* the Action Recognition
Normal class.

This module only reads and validates the official files. It assigns no SHAR
labels beyond the class names the official `ClassIDs.txt` itself defines, and
it decides nothing about which protocol an experiment should use -- that is an
owner-level choice, deliberately expressed as a parameter rather than baked in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ACTION_RECOGNITION = "action_recognition"
ANOMALY_DETECTION = "anomaly_detection"
PROTOCOLS = (ACTION_RECOGNITION, ANOMALY_DETECTION)

# The Action Recognition directory name for the Normal class. The local corpus
# stores it as `z_Normal_Videos_event`; the `z_` prefix is a local sort artifact
# and is not part of the official class name.
AR_NORMAL_CLASS = "Normal_Videos_event"
AR_NORMAL_LOCAL_DIR = "z_Normal_Videos_event"

AR_FOLDS = (1, 2, 3, 4)


@dataclass(frozen=True)
class SplitEntry:
    """One video in an official split list."""

    relative_path: str  # e.g. "Abuse/Abuse001_x264.mp4", as written upstream
    directory: str  # e.g. "Abuse"
    video_id: str  # e.g. "Abuse001_x264" -- the stem, matching videos.csv
    class_name: str  # official class name for this protocol


def _read_list(path: Path) -> list[str]:
    """Read an official split list.

    Upstream lines carry a trailing space and use forward slashes. Blank lines
    are skipped; nothing else is normalised, so a genuinely malformed line
    surfaces rather than being silently repaired.
    """
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="strict").splitlines():
        entry = raw.strip()
        if entry:
            lines.append(entry)
    return lines


def read_class_ids(splits_root: Path) -> dict[str, int]:
    """Read `ClassIDs.txt` as {class_name: id}, preserving the official ids."""
    path = splits_root / "ClassIDs.txt"
    mapping: dict[str, int] = {}
    for line in _read_list(path):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}: cannot parse class line {line!r}")
        mapping[parts[0]] = int(parts[1])
    if len(mapping) != 14:
        raise ValueError(f"{path}: expected 14 classes, found {len(mapping)}")
    return mapping


def _entry_from_line(line: str, *, class_name: str | None = None) -> SplitEntry:
    # Anomaly_Detection lines are like "Abuse/Abuse001_x264.mp4"; Action
    # Recognition lines have the same shape. A line without a slash is a
    # structural surprise and must not be guessed at.
    if "/" not in line:
        raise ValueError(f"split line has no directory component: {line!r}")
    directory, filename = line.split("/", 1)
    if not filename.endswith(".mp4"):
        raise ValueError(f"split line is not an .mp4: {line!r}")
    return SplitEntry(
        relative_path=line,
        directory=directory,
        video_id=filename[: -len(".mp4")],
        class_name=class_name if class_name is not None else directory,
    )


def read_action_recognition_split(
    ucf_crimes_root: Path, *, fold: int
) -> dict[str, list[SplitEntry]]:
    """Return {"train": [...], "test": [...]} for one official AR fold."""
    if fold not in AR_FOLDS:
        raise ValueError(f"fold must be one of {AR_FOLDS}, got {fold}")
    splits_root = ucf_crimes_root / "Action_Regnition_splits"
    class_ids = read_class_ids(splits_root)

    result: dict[str, list[SplitEntry]] = {}
    for split, stem in (("train", "train"), ("test", "test")):
        path = splits_root / f"{stem}_{fold:03d}.txt"
        entries = [_entry_from_line(line) for line in _read_list(path)]
        for entry in entries:
            if entry.class_name not in class_ids:
                raise ValueError(
                    f"{path}: directory {entry.class_name!r} is not in ClassIDs.txt"
                )
        result[split] = entries

    # The official folds are balanced by construction. Checking it here turns a
    # silently truncated or edited split file into an immediate failure rather
    # than a quietly skewed experiment.
    #
    # The invariant is checked, not a hardcoded count: all 14 classes present
    # and equally represented. Dropping whole classes trips the class count;
    # dropping individual lines trips the balance. The real corpus's exact
    # 38-train/12-test figures are pinned in tests/test_official_splits.py
    # rather than baked in here, so this parser stays usable on any correctly
    # formed split file.
    for split in ("train", "test"):
        counts: dict[str, int] = {}
        for entry in result[split]:
            counts[entry.class_name] = counts.get(entry.class_name, 0) + 1
        if len(counts) != 14:
            raise ValueError(
                f"AR fold {fold} {split} covers {len(counts)} classes, expected 14"
            )
        sizes = set(counts.values())
        if len(sizes) != 1:
            expected = max(sizes, key=list(counts.values()).count)
            off = {k: v for k, v in counts.items() if v != expected}
            raise ValueError(
                f"AR fold {fold} {split} is not balanced: most classes have "
                f"{expected} videos, deviations={off}"
            )
    return result


def read_anomaly_detection_split(
    ucf_crimes_root: Path,
) -> dict[str, list[SplitEntry]]:
    """Return {"train": [...], "test": [...]} for the anomaly-detection protocol."""
    splits_root = ucf_crimes_root / "Anomaly_Detection_splits"
    result: dict[str, list[SplitEntry]] = {}
    for split, filename in (("train", "Anomaly_Train.txt"), ("test", "Anomaly_Test.txt")):
        path = splits_root / filename
        result[split] = [_entry_from_line(line) for line in _read_list(path)]

    total = len(result["train"]) + len(result["test"])
    if total != 1900:
        raise ValueError(
            f"anomaly-detection split totals {total} videos, expected 1900"
        )
    return result


def local_directory_for(entry: SplitEntry) -> str:
    """Map an official split directory to the directory name on local disk.

    Only one name differs: the AR Normal class is stored locally under
    `z_Normal_Videos_event`. Everything else matches byte for byte.
    """
    if entry.directory == AR_NORMAL_CLASS:
        return AR_NORMAL_LOCAL_DIR
    return entry.directory
