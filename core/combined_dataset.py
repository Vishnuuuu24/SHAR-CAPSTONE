"""Assemble the SHAR-Combined working dataset: symlinks plus an audit manifest.

Originals under ``Datasets/`` are never modified. Only ``os.stat`` touches them
(no video bytes are read). Each source dataset has one adapter function that
returns ``VideoRecord`` objects; adding a dataset means adding one adapter and
registering its class mappings in ``UNIFIED_LABEL``.

Science constraints encoded here (docs/PROJECT_RULES.md rules 1 and 2):
  * ``source_label`` is never overwritten; ``unified_label`` comes only from the
    explicit ``UNIFIED_LABEL`` dict and any unmapped class is a hard error.
  * Every UCF official Test video carries ``is_ucf_test=true`` and
    ``trainable=false``; ``assert_no_ucf_test_trainable`` enforces it.
  * K-11: the 50 ``z_Normal_Videos_event`` files are byte-identical copies of
    ``Testing_Normal_Videos_Anomaly`` files (same name and size here), so they
    are Test videos, flagged, and never trainable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from data.official_splits import (
    AR_NORMAL_CLASS,
    AR_NORMAL_LOCAL_DIR,
    _read_list,
)

UCF = "ucf_crime_official"
XD = "xd_violence"
RWF = "rwf2000"
CUVA = "cuva"

CUVA_UNPARSED = "Unparsed"  # annotation present but not a '[N-..]' category code
CUVA_CATEGORY = {1: "Fighting", 2: "Animals Hurting People", 3: "Water Incidents",
                 4: "Vandalism", 5: "Traffic Accidents", 6: "Robbery", 7: "Theft",
                 8: "Pedestrian Incidents", 9: "Fire", 10: "Traffic Violations",
                 11: "Forbidden to Burn"}

# Annotator free text (Chinese), translated; every other free-text string stays Unparsed.
CUVA_FREE_TEXT = {"枪战": ("FreeText:Shooting", "gunfight"), "枪击": ("FreeText:Shooting", "shooting"),
                  "逮捕": ("FreeText:Arrest", "arrest")}

# The single auditable source_label -> unified_label registry.
UNIFIED_LABEL: dict[tuple[str, str], str] = {
    (UCF, "Shooting"): "Shooting",
    (UCF, "Assault"): "Assault",
    (UCF, "Fighting"): "Fighting",
    (UCF, "Abuse"): "Abuse",
    (UCF, AR_NORMAL_CLASS): "Normal",
    (XD, "Shooting"): "Shooting",
    (XD, "Fighting"): "Fighting",
    (XD, "Abuse"): "Abuse",
    (RWF, "Fight"): "Fighting",
    # CUVA: only four categories have a UCF counterpart. '' = unmapped: kept in the
    # manifest (nothing wasted) but never trainable for the 14-class task.
    (CUVA, "Fighting"): "Fighting",
    (CUVA, "Robbery"): "Robbery",
    (CUVA, "Vandalism"): "Vandalism",
    (CUVA, "Traffic Accidents"): "RoadAccidents",
    (CUVA, "Theft"): "",
    (CUVA, "Fire"): "",
    (CUVA, "Water Incidents"): "",
    (CUVA, "Animals Hurting People"): "",
    (CUVA, "Pedestrian Incidents"): "",
    (CUVA, "Traffic Violations"): "",
    (CUVA, "Forbidden to Burn"): "",
    (CUVA, CUVA_UNPARSED): "",
    (CUVA, "FreeText:Shooting"): "Shooting",
    (CUVA, "FreeText:Arrest"): "Arrest",
}
for _c in ("Arrest", "Arson", "Burglary", "Explosion", "RoadAccidents", "Robbery",
           "Shoplifting", "Stealing", "Vandalism"):
    UNIFIED_LABEL[(UCF, _c)] = _c

DEFAULT_CLASSES: dict[str, list[str]] = {
    UCF: ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
          "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
          AR_NORMAL_CLASS],
    XD: ["Shooting", "Fighting", "Abuse"],
    RWF: ["Fight"],
    CUVA: sorted({c for (s_, c) in UNIFIED_LABEL if s_ == CUVA}),
}

MANIFEST_NAME = "manifest.csv"
SUMMARY_NAME = "manifest_summary.json"
README_NAME = "README.md"
OWNED_FILES = {MANIFEST_NAME, SUMMARY_NAME, README_NAME, "split_summary.json",
               "previous_v1_assignment.csv", "previous_v2_assignment.csv"}
FIELDS = [
    "source_dataset", "source_label", "unified_label", "source_video_id",
    "relative_link_path", "original_abs_path", "size_bytes", "split_in_source",
    "is_ucf_test", "trainable", "notes",
]


@dataclass
class VideoRecord:
    source_dataset: str
    source_label: str
    video_id: str
    original: Path
    split: str
    is_ucf_test: bool = False
    notes: list[str] = field(default_factory=list)


class UnmappedClassError(ValueError):
    pass


def check_mapping(selection: dict[str, list[str]]) -> None:
    missing = [(s, c) for s, cs in selection.items() for c in cs if (s, c) not in UNIFIED_LABEL]
    if missing:
        raise UnmappedClassError(f"classes with no unified_label mapping: {missing}")


def _mp4s(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() == ".mp4")


def ucf_adapter(datasets_root: Path, classes: Iterable[str]) -> list[VideoRecord]:
    root = datasets_root / "UCF Crime Official" / "raw" / "UCF_Crimes"
    if not root.is_dir():
        print(f"[ucf] {root} not found; skipping")
        return []
    # Official AD lists, via the repo's reader. The parser's 1,900 total check is
    # a property of the real corpus, so tests with fake lists use the raw reader.
    lists = root / "Anomaly_Detection_splits"
    split_of = {ln: "Test" for ln in _read_list(lists / "Anomaly_Test.txt")}
    split_of.update({ln: "Train" for ln in _read_list(lists / "Anomaly_Train.txt")})
    videos = root / "Videos"
    test_normal = videos / "Testing_Normal_Videos_Anomaly"
    records: list[VideoRecord] = []
    for cls in classes:
        local = AR_NORMAL_LOCAL_DIR if cls == AR_NORMAL_CLASS else cls
        for path in _mp4s(videos / local):
            rec = VideoRecord(UCF, cls, path.stem, path, split_of.get(f"{cls}/{path.name}", ""))
            if cls == AR_NORMAL_CLASS:
                twin = test_normal / path.name
                if twin.exists() and twin.stat().st_size == path.stat().st_size:
                    rec.split, rec.is_ucf_test = "Test", True
                    rec.notes.append(
                        f"K-11: duplicate of Testing_Normal_Videos_Anomaly/{path.name} "
                        "(same name and size); treated as official Test")
                else:
                    rec.split, rec.is_ucf_test = "Test", True
                    rec.notes.append("no Testing_Normal twin found; conservatively Test")
                rec.notes.append(f"origin_dir={local}")
            elif rec.split == "":
                rec.is_ucf_test = True  # unknown split: never trainable
                rec.notes.append("not in official AD split lists; conservatively Test")
            else:
                rec.is_ucf_test = rec.split == "Test"
            records.append(rec)
        if cls == AR_NORMAL_CLASS:
            records += _ucf_extra_normals(videos, cls, {r.original.name for r in records}, split_of)
    return records


# Extra sources for the UCF Normal class (registered here, not implicit). The AR
# Normal dir is Test-only (K-11), so SHAR-Combined also draws on the AD Normal dirs.
NORMAL_TEST_DIR = "Testing_Normal_Videos_Anomaly"
NORMAL_TRAIN_DIR = "Training_Normal_Videos_Anomaly"
NORMAL_TRAIN_SAMPLE = 120
NORMAL_SAMPLE_SEED = 0
APPENDED_V2 = "appended_v2: Training_Normal video outside the v1 sample"


def _ucf_extra_normals(videos: Path, cls: str, have: set[str],
                       split_of: dict[str, str]) -> list[VideoRecord]:
    out: list[VideoRecord] = []
    test_dir, train_dir = videos / NORMAL_TEST_DIR, videos / NORMAL_TRAIN_DIR
    if test_dir.is_dir():  # official Test, minus the K-11 duplicates already added
        for p in _mp4s(test_dir):
            if p.name not in have:
                out.append(VideoRecord(UCF, cls, p.stem, p, "Test", True,
                                       [f"origin_dir={NORMAL_TEST_DIR}"]))
    if train_dir.is_dir():  # sorted-name sample, fixed seed
        names = [p for p in _mp4s(train_dir)
                 if split_of.get(f"{NORMAL_TRAIN_DIR}/{p.name}") == "Train"]
        k = min(NORMAL_TRAIN_SAMPLE, len(names))
        picked = random.Random(NORMAL_SAMPLE_SEED).sample(names, k)  # the v1 120: unchanged
        for p in picked:
            out.append(VideoRecord(UCF, cls, p.stem, p, "Train", False,
                                   [f"origin_dir={NORMAL_TRAIN_DIR}",
                                    f"sampled {k}/{len(names)} seed={NORMAL_SAMPLE_SEED}"]))
        chosen = set(picked)
        for p in names:  # the rest, appended after v1 (folded last, see combined_split)
            if p not in chosen:
                out.append(VideoRecord(UCF, cls, p.stem, p, "Train", False,
                                       [f"origin_dir={NORMAL_TRAIN_DIR}", APPENDED_V2]))
    return out


def xd_adapter(datasets_root: Path, classes: Iterable[str]) -> list[VideoRecord]:
    root = datasets_root / "XD-Violence" / "train"
    if not root.is_dir():
        print(f"[xd] {root} not found; skipping")
        return []
    return [VideoRecord(XD, c, p.stem, p, "train")
            for c in classes if (root / c).is_dir() for p in _mp4s(root / c)]


def rwf_adapter(datasets_root: Path, classes: Iterable[str]) -> list[VideoRecord]:
    root = datasets_root / "RWF-2000"
    if not root.is_dir():
        print(f"[rwf2000] {root} does not exist; adapter is a no-op")
        return []
    out = []
    for split in ("train", "val"):
        for c in classes:
            d = root / split / c
            if d.is_dir():
                out += [VideoRecord(RWF, c, p.stem, p, split) for p in _mp4s(d)]
    return out


CUVA_ANNOTATION = "cuva_gt_cleaned.json"


def parse_cuva_label(output: str) -> str:
    """'[7-1,5]' -> 'Theft'; anything not starting '[<1..11>' -> CUVA_UNPARSED."""
    m = re.match(r"\[(\d+)", output.strip())
    if m:
        return CUVA_CATEGORY.get(int(m.group(1)), CUVA_UNPARSED)
    return CUVA_FREE_TEXT.get(output.strip(), (CUVA_UNPARSED,))[0]


def cuva_adapter(datasets_root: Path, classes: Iterable[str]) -> list[VideoRecord]:
    root = datasets_root / "CUVA"
    ann = root / "annotations" / CUVA_ANNOTATION
    if not ann.is_file():
        print(f"[cuva] {ann} not found; skipping")
        return []
    label_of: dict[str, tuple[str, str]] = {}
    for r in json.loads(ann.read_text(encoding="utf-8")):
        if r.get("task") == "Classification":
            label_of[r["visual_input"]] = (parse_cuva_label(r["output"]), r["output"])
    wanted = set(classes)
    by_name: dict[str, list[Path]] = {}
    for p in sorted((root / "raw").glob("group_*/*.mp4")):
        by_name.setdefault(p.name, []).append(p)
    out: list[VideoRecord] = []
    for name, paths in sorted(by_name.items()):
        if len(paths) > 1:  # ids collide across groups: key by group dir too
            raise ValueError(f"CUVA id collision for {name}: {[str(p) for p in paths]}")
        label, raw = label_of.get(name, (CUVA_UNPARSED, "(no annotation record)"))
        if label not in wanted:
            continue
        out.append(VideoRecord(CUVA, label, paths[0].stem, paths[0], "",
                               notes=[f"cuva raw output={raw!r}", f"dir={paths[0].parent.name}",
                                      "no parent-video info: each file is its own group"]
                               + ([f"free-text annotator label, translated: {CUVA_FREE_TEXT[raw.strip()][1]}"]
                                  if raw.strip() in CUVA_FREE_TEXT else [])
                               + ([] if UNIFIED_LABEL[(CUVA, label)] else ["no UCF counterpart"])))
    return out


ADAPTERS: dict[str, Callable[[Path, Iterable[str]], list[VideoRecord]]] = {
    UCF: ucf_adapter, XD: xd_adapter, RWF: rwf_adapter, CUVA: cuva_adapter,
}


def manifest_row(rec: VideoRecord) -> dict[str, object]:
    try:
        size: object = rec.original.stat().st_size
    except OSError:
        size = ""
        rec.notes.append("BROKEN: original missing")
    unified = UNIFIED_LABEL[(rec.source_dataset, rec.source_label)]
    return {
        "source_dataset": rec.source_dataset,
        "source_label": rec.source_label,
        "unified_label": unified,
        "source_video_id": rec.video_id,
        "relative_link_path": f"{rec.source_dataset}/{rec.source_label}/{rec.original.name}",
        "original_abs_path": str(rec.original.resolve()) if size != "" else str(rec.original),
        "size_bytes": size,
        "split_in_source": rec.split,
        "is_ucf_test": str(rec.is_ucf_test).lower(),
        "trainable": str(not rec.is_ucf_test and unified != "").lower(),
        "notes": "; ".join(rec.notes),
    }


def assert_no_ucf_test_trainable(rows: Iterable[dict[str, object]]) -> None:
    for r in rows:
        is_test = r["is_ucf_test"] == "true" or (
            r["source_dataset"] == UCF and r["split_in_source"] != "Train")
        if is_test and r["trainable"] != "false":
            raise AssertionError(f"UCF Test video marked trainable: {r['relative_link_path']}")
        if r["source_dataset"] == UCF and r["split_in_source"] == "Test" and r["is_ucf_test"] != "true":
            raise AssertionError(f"UCF Test video lacks is_ucf_test: {r['relative_link_path']}")


def check_output_dir(out: Path) -> None:
    """Refuse to touch a directory holding anything we did not create."""
    if not out.exists():
        return
    for p in out.rglob("*"):
        if p.is_symlink():
            continue
        if p.is_dir():
            continue
        if p.parent == out and p.name in OWNED_FILES:
            continue
        raise RuntimeError(f"refusing to run: {p} is not a symlink or owned manifest file")


def clear_links(out: Path) -> None:
    if not out.exists():
        return
    for p in out.rglob("*"):
        if p.is_symlink():
            p.unlink()
    for p in sorted((d for d in out.rglob("*") if d.is_dir()), reverse=True):
        p.rmdir()


def _git_head(repo: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build(datasets_root: Path, out: Path, selection: dict[str, list[str]],
          repo_root: Path | None = None) -> dict[str, object]:
    check_mapping(selection)
    unknown = [s for s in selection if s not in ADAPTERS]
    if unknown:
        raise ValueError(f"unknown source datasets: {unknown}")
    check_output_dir(out)
    records: list[VideoRecord] = []
    for src, classes in selection.items():
        records += ADAPTERS[src](datasets_root, classes)
    rows = [manifest_row(r) for r in records]
    seen: set[str] = set()
    for r in rows:
        if r["relative_link_path"] in seen:
            raise ValueError(f"link path collision: {r['relative_link_path']}")
        seen.add(str(r["relative_link_path"]))
    assert_no_ucf_test_trainable(rows)

    clear_links(out)
    out.mkdir(parents=True, exist_ok=True)
    broken: list[str] = []
    for r in rows:
        if r["size_bytes"] == "":
            broken.append(str(r["relative_link_path"]))
            continue
        link = out / str(r["relative_link_path"])
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(str(r["original_abs_path"]))
    for r in rows:  # post-check every created link resolves
        link = out / str(r["relative_link_path"])
        if link.is_symlink() and not link.exists():
            broken.append(str(r["relative_link_path"]))

    manifest = out / MANIFEST_NAME
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    counts: dict[str, dict[str, int]] = {}
    for r in rows:
        d = counts.setdefault(str(r["source_dataset"]), {})
        key = str(r["source_label"])
        d[key] = d.get(key, 0) + 1
    ucf_split: dict[str, dict[str, int]] = {}
    for r in rows:
        if r["source_dataset"] == UCF:
            d = ucf_split.setdefault(str(r["source_label"]), {})
            d[str(r["split_in_source"])] = d.get(str(r["split_in_source"]), 0) + 1
    summary = {
        "counts_source_x_class": counts,
        "ucf_split_counts": ucf_split,
        "total_links": len(rows) - len(set(broken)),
        "broken_links": sorted(set(broken)),
        "class_selection": selection,
        "unified_label_map": {f"{s}/{c}": u for (s, c), u in UNIFIED_LABEL.items()},
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "git_head": _git_head(repo_root or Path.cwd()),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / README_NAME).write_text(README_TEXT, encoding="utf-8")
    return summary


README_TEXT = """# SHAR-Combined

Working dataset assembled by `scripts/build_combined_dataset.py`. Every video is a
symlink to the untouched original under `Datasets/`; nothing is copied. Layout:
`<source_dataset>/<source_label>/<filename>`.

- `manifest.csv` is the source of truth. `source_label` is the original class;
  `unified_label` comes from the explicit map in `core/combined_dataset.py`.
- **UCF official Test videos have `is_ucf_test=true` and `trainable=false`.**
  Exclude them from any training. This includes the 50 `Normal_Videos_event`
  files, which are byte-identical duplicates of `Testing_Normal_Videos_Anomaly`
  files (K-11), so the UCF Normal class here is Test-only.
- In SHAR-Combined's own split (`scripts/build_combined_split.py`) the UCF official
  Test is replaced by our sealed test: official-Test videos not sealed become
  trainable; the sealed `split=test` rows are never trainable. CUVA categories
  with no UCF counterpart have `unified_label=''` and are never trainable.
- **UCF Normal now has ~920 trainable videos (all 800 Training_Normal plus surplus)
  against 30-130 per anomaly class: training MUST use class-balanced sampling or
  class weights, and report macro metrics.**
- Labels are video-level (weak, inherited); XD-Violence and RWF-2000 are
  separate domains with their own ontology and licences. Never report a pooled
  number as a UCF headline (docs/PROJECT_RULES.md rules 1 and 3).
- `manifest_summary.json` has counts, manifest sha256, git revision and time.
- Re-running the script rebuilds links; the directory may hold only symlinks
  and these three files.
"""
