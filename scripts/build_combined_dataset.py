#!/usr/bin/env python3
"""Build Datasets/SHAR-Combined (symlinks + manifest). Logic lives in core.combined_dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.combined_dataset import ADAPTERS, DEFAULT_CLASSES, build  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-root", type=Path, default=REPO / "Datasets")
    ap.add_argument("--out", type=Path, default=REPO / "Datasets" / "SHAR-Combined")
    ap.add_argument("--classes", action="append", default=[], metavar="SOURCE=A,B",
                    help=f"override classes for a source ({', '.join(ADAPTERS)}); repeatable")
    args = ap.parse_args()
    selection = {k: list(v) for k, v in DEFAULT_CLASSES.items()}
    for spec in args.classes:
        src, _, names = spec.partition("=")
        selection[src] = [n for n in names.split(",") if n]
    summary = build(args.datasets_root, args.out, selection, REPO)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # unmapped class, dirty output dir, ...
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
