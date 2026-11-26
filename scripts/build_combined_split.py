#!/usr/bin/env python3
"""Add the sealed-test / train-pool / fold columns to SHAR-Combined/manifest.csv.

Run after scripts/build_combined_dataset.py. Logic lives in core.combined_split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.combined_split import split_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "Datasets" / "SHAR-Combined")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(json.dumps(split_manifest(args.out, args.seed, REPO), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
