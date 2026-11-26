#!/usr/bin/env python3
"""Write the XD-Violence manifest (official train/test; ~10% of train held out as validation, grouped by source)."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.xd_dataset import XD_CLASSES, assign_validation, build_rows  # noqa: E402

FIELDS = ["index", "source_dataset", "source_video_id", "unified_label", "primary_label", "n_labels", "group_id", "split", "fold", "original_abs_path"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("Datasets/XD-Violence-full/xd-violence"))
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=Path("results/xd/manifest.csv"))
    args = ap.parse_args(argv)
    rows = build_rows(args.root)
    val_groups = assign_validation(rows, args.fraction, args.seed)
    for r in rows:
        if r["split"] == "train" and r["group_id"] in val_groups:
            r["split"] = "validation"
    test_groups = {r["group_id"] for r in rows if r["split"] == "test"}
    trainval_groups = {r["group_id"] for r in rows if r["split"] != "test"}
    counts = collections.Counter((r["split"], c) for r in rows for c in r["unified_label"].split("|"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=FIELDS)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"index": i, **r})
    report = {"videos": collections.Counter(r["split"] for r in rows), "validation_groups": len(val_groups),
              "official_groups_shared_by_train_and_test": len(test_groups & trainval_groups),
              "per_class": {s: {c: counts[(s, c)] for c in XD_CLASSES} for s in ("train", "validation", "test")},
              "fraction": args.fraction, "seed": args.seed}
    (args.output.with_suffix(".json")).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
