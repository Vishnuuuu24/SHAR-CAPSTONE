"""Deterministic, leakage-safe train-pool / sealed-test assignment for SHAR-Combined.

SHAR-Combined is our own benchmark: results on it are never comparable to the UCF
14-class headline, and must always be reportable per source dataset (the UCF/XD
domain gap is near-perfectly separable, so an XD-only test would be inflated).

Grouping (the leakage guard): UCF group = source video id; XD group = the parent
movie, i.e. the filename with the class prefix and everything from ``___`` on
removed (``Fighting_Bad.Boys.1995___01-07-01_..`` -> ``Bad.Boys.1995``). A movie
can appear under several XD classes, so groups are global per source, and a group
chosen for test in one class is test in every class.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from core.combined_dataset import (
    APPENDED_V2,
    CUVA, FIELDS, MANIFEST_NAME, UCF, XD, _git_head,
)

SPLIT_FIELDS = FIELDS + ["group_id", "split", "fold", "test_sealed"]
SUMMARY = "split_summary.json"
TEST_DIR = "test"
N_FOLDS = 4
UCF_TEST_TARGET = 20
UCF_SMALL_CLASS = 40
UCF_SMALL_FRACTION = 0.4
XD_TEST_FRACTION = 0.15
CUVA_TEST_FRACTION = 0.15
UNLABELED_POOL = "unlabeled_pool"


def group_id(row: dict) -> str:
    src, vid = row["source_dataset"], row["source_video_id"]
    if src == XD:
        stem = vid.split("___", 1)[0]
        prefix = row["source_label"] + "_"
        parent = stem[len(prefix):] if stem.startswith(prefix) else stem
        return f"{src}:{parent}"
    return f"{src}:{vid}"


def _rng(seed: int, *parts: str) -> random.Random:
    return random.Random(f"{seed}|" + "|".join(parts))


def _stratum(row: dict) -> tuple[str, str]:
    return row["source_dataset"], row["unified_label"]


def _ucf_target(n_videos: int) -> int:
    return UCF_TEST_TARGET if n_videos >= UCF_SMALL_CLASS else math.ceil(UCF_SMALL_FRACTION * n_videos)


def assign_splits(rows: list[dict], seed: int = 0) -> list[dict]:
    """Return copies of ``rows`` with group_id, split, fold, test_sealed set.

    Own-benchmark semantics: the sealed test replaces the UCF official test, so
    official-Test videos that are not sealed become trainable. Unmapped rows
    (unified_label '') get split 'unlabeled_pool' and are never trainable or tested.
    Each (source, class) draws from its own RNG so adding classes or sources never
    moves existing test/fold assignments.
    """
    rows = [dict(r, group_id=group_id(r)) for r in rows]
    for r in rows:  # earlier tiers keep their fold order; later tiers are folded in after
        # tier 0 = v1-trainable, 1 = former-untrainable surplus, 2 = appended in v2
        r["_core"] = (2 if APPENDED_V2 in r["notes"] else 0 if r["is_ucf_test"] != "true" else 1)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["unified_label"] != "":
            strata[_stratum(r)].append(r)
    test_groups: set[str] = set()
    # Smallest strata first so multi-class XD movies are claimed deterministically.
    for key in sorted(strata, key=lambda k: (len(strata[k]), k)):
        src, label = key
        members = sorted(strata[key], key=lambda r: r["source_video_id"])
        groups = sorted({r["group_id"] for r in members})
        rng = _rng(seed, "test", src, label)
        if src == UCF:
            target = _ucf_target(len(groups))
            official = [g for r in members if r["is_ucf_test"] == "true" for g in [r["group_id"]]]
            official = sorted(set(official))
            if len(official) >= target:
                test_groups.update(rng.sample(official, target))
            else:
                pool = [g for g in groups if g not in set(official)]
                test_groups.update(official)
                test_groups.update(rng.sample(pool, min(target - len(official), len(pool))))
        else:
            frac = CUVA_TEST_FRACTION if src == CUVA else XD_TEST_FRACTION
            target = max(1, round(frac * len(groups)))
            have = [g for g in groups if g in test_groups]
            pool = [g for g in groups if g not in test_groups]
            need = max(0, target - len(have))
            test_groups.update(rng.sample(pool, min(need, len(pool))))
    for r in rows:
        is_test = r["group_id"] in test_groups
        r["fold"] = ""
        if r["unified_label"] == "":
            r["split"], r["test_sealed"], r["trainable"] = UNLABELED_POOL, "false", "false"
            continue
        r["split"] = "test" if is_test else "train_pool"
        r["test_sealed"] = str(is_test).lower()
        r["trainable"] = "false" if is_test else "true"
    _assign_folds(rows, seed)
    assert_groups_unsplit(rows)
    assert_no_twin_straddle(rows)
    for r in rows:
        del r["_core"]
    return rows


def assert_no_twin_straddle(rows: Iterable[dict]) -> None:
    """K-11 guard: byte-identical duplicates (same filename + size) share split and fold."""
    seen: dict[tuple, tuple] = {}
    for r in rows:
        if r["size_bytes"] == "":
            continue
        key = (r["source_dataset"], Path(r["relative_link_path"]).name, r["size_bytes"])
        val = (r["split"], r["fold"])
        if seen.setdefault(key, val) != val:
            raise AssertionError(f"duplicate pair straddles splits/folds: {key}")


def _assign_folds(rows: list[dict], seed: int) -> None:
    fold_of: dict[str, int] = {}
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["split"] == "train_pool" and r["trainable"] == "true":
            strata[_stratum(r)].append(r)
    core_n = {k: sum(r["_core"] == 0 for r in v) for k, v in strata.items()}
    for key in sorted(strata, key=lambda k: (core_n[k], k)):
        load = [0] * N_FOLDS
        # pass 1 reproduces the v1 assignment (v1-trainable rows only); pass 2 adds the
        # surplus (former untrainable official-Test rows) on top, balancing the load.
        for tag, tier in (("fold", 0), ("fold_surplus", 1), ("fold_appended", 2)):
            size: dict[str, int] = defaultdict(int)
            for r in strata[key]:
                if r["_core"] == tier and (tier == 0 or r["group_id"] not in fold_of):
                    size[r["group_id"]] += 1
            order = sorted(size)
            _rng(seed, tag, *key).shuffle(order)
            order.sort(key=lambda g: -size[g])  # stable: big groups first, ties stay shuffled
            for g in order:
                if g in fold_of:
                    load[fold_of[g] - 1] += size[g]
                    continue
                f = min(range(N_FOLDS), key=lambda i: load[i])  # first minimum: deterministic
                fold_of[g] = f + 1
                load[f] += size[g]
    for r in rows:
        if r["group_id"] in fold_of and r["split"] == "train_pool" and r["trainable"] == "true":
            r["fold"] = str(fold_of[r["group_id"]])


def assert_groups_unsplit(rows: Iterable[dict]) -> None:
    splits: dict[str, set] = defaultdict(set)
    folds: dict[str, set] = defaultdict(set)
    for r in rows:
        splits[r["group_id"]].add(r["split"])
        if r["fold"] != "":
            folds[r["group_id"]].add(r["fold"])
    bad = [g for g, s in splits.items() if len(s) > 1] + [g for g, s in folds.items() if len(s) > 1]
    if bad:
        raise AssertionError(f"groups straddle splits/folds: {sorted(set(bad))[:5]}")


def assert_test_sealed(rows: Iterable[dict]) -> None:
    for r in rows:
        if r.get("split") == "test" or r.get("test_sealed") == "true":
            raise AssertionError(f"sealed test row reached a trainable path: {r['relative_link_path']}")


def load_trainable(manifest: Path, fold: int) -> tuple[list[dict], list[dict]]:
    """(train, validation) rows for cross-validation fold ``fold``; never test rows."""
    if fold not in range(1, N_FOLDS + 1):
        raise ValueError(f"fold must be 1..{N_FOLDS}, got {fold}")
    with Path(manifest).open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "train_pool"
                and r["trainable"] == "true" and r["fold"] != ""
                and r["unified_label"] != ""]
    train = [r for r in rows if int(r["fold"]) != fold]
    val = [r for r in rows if int(r["fold"]) == fold]
    assert_test_sealed(train + val)
    return train, val


def _write_test_links(out: Path, rows: list[dict]) -> int:
    root = out / TEST_DIR
    if root.exists():
        for p in root.rglob("*"):
            if p.is_symlink():
                p.unlink()
        for d in sorted((d for d in root.rglob("*") if d.is_dir()), reverse=True):
            d.rmdir()
    n = 0
    for r in rows:
        if r["split"] != "test":
            continue
        link = root / r["source_dataset"] / r["source_label"] / Path(r["relative_link_path"]).name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(r["original_abs_path"])
        n += 1
    return n


def split_manifest(out: Path, seed: int = 0, repo_root: Path | None = None) -> dict:
    manifest = out / MANIFEST_NAME
    with manifest.open(newline="", encoding="utf-8") as fh:
        base = [{k: r[k] for k in FIELDS} for r in csv.DictReader(fh)]
    rows = assign_splits(base, seed)
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SPLIT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_links = _write_test_links(out, rows)
    counts: dict = {}
    groups: dict = {}
    for r in rows:
        c = counts.setdefault(r["source_dataset"], {}).setdefault(r["unified_label"], defaultdict(int))
        c[r["split"] + ("" if r["trainable"] == "true" or r["split"] in ("test", UNLABELED_POOL) else "_untrainable")] += 1
        groups.setdefault(r["source_dataset"], {}).setdefault(r["split"], set()).add(r["group_id"])
    table: dict = {}
    for r in rows:
        t = table.setdefault(r["source_dataset"], {}).setdefault(r["unified_label"] or "(unmapped)", defaultdict(int))
        t["total"] += 1
        t[r["split"]] += 1
        t["trainable"] += r["trainable"] == "true"
    summary = {
        "seed": seed,
        "table_source_x_unified_label": {s_: {c: dict(v) for c, v in d.items()} for s_, d in table.items()},
        "counts_source_x_class_x_split": {s: {c: dict(v) for c, v in d.items()} for s, d in counts.items()},
        "group_counts": {s: {k: len(v) for k, v in d.items()} for s, d in groups.items()},
        "n_files_per_source": {s: sum(sum(v.values()) for v in d.values()) for s, d in counts.items()},
        "test_links": n_links,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "git_head": _git_head(repo_root or Path.cwd()),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
