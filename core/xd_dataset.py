"""XD-Violence (official 6 violence classes + Normal): filename labels, source groups, grouped validation hold-out."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Sequence

CODES = {"A": "Normal", "B1": "Fighting", "B2": "Shooting", "B4": "Riot", "B5": "Abuse", "B6": "Car accident", "G": "Explosion"}
XD_CLASSES = ("Normal", "Fighting", "Shooting", "Riot", "Abuse", "Car accident", "Explosion")


def labels_of(stem: str) -> list[str]:
    """Labels in filename order ('..._label_G-B2-B6' -> Explosion, Shooting, Car accident)."""

    m = re.search(r"_label_([A-Z0-9-]+)$", stem)
    if not m:
        raise ValueError(f"no label in {stem!r}")
    codes = [c for c in m.group(1).split("-") if c != "0"]
    unknown = [c for c in codes if c not in CODES]
    if unknown or not codes:
        raise ValueError(f"{stem}: unknown label codes {unknown or codes}")
    return [CODES[c] for c in codes]


def group_of(stem: str) -> str:
    """Source film / YouTube id: clips of one source must not straddle a split."""

    return stem.split("__#", 1)[0]


def build_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "data/video").rglob("*.mp4")):
        stem = path.stem
        labs = labels_of(stem)
        rows.append({"source_dataset": "xd_violence", "source_video_id": stem, "unified_label": "|".join(labs),
                     "primary_label": labs[0], "n_labels": len(labs), "group_id": group_of(stem),
                     "split": "test" if path.parent.name == "test_videos" else "train", "fold": "",
                     "original_abs_path": str(path.resolve())})
    return rows


def assign_validation(rows: Sequence[dict], fraction: float, seed: int) -> set[str]:
    """Groups moved from train to validation: rarest class first, hash-ordered groups, until each class has
    ceil(fraction * its train count) validation videos. Uses label composition only, never model results."""

    train = [r for r in rows if r["split"] == "train"]
    groups: dict[str, list[dict]] = {}
    for r in train:
        groups.setdefault(r["group_id"], []).append(r)
    order = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    counts = {c: sum(c in r["unified_label"].split("|") for r in train) for c in XD_CLASSES}
    chosen: set[str] = set()
    for cls in sorted(XD_CLASSES, key=lambda c: counts[c]):
        need = math.ceil(fraction * counts[cls])
        have = sum(cls in r["unified_label"].split("|") for g in chosen for r in groups[g])
        for g in order:
            if have >= need:
                break
            if g in chosen or not any(cls in r["unified_label"].split("|") for r in groups[g]):
                continue
            chosen.add(g)
            have += sum(cls in r["unified_label"].split("|") for r in groups[g])
    return chosen
