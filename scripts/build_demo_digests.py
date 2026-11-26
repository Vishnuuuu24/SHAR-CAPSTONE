#!/usr/bin/env python3
"""Write results/demo/protocol_digests.json: SHA-256 of each UCF protocol MP4 -> its video id, from the fold-1 manifests.

Fold 1's train, validation and test manifests together list all 700 protocol videos. `core.demo_inference` uses the map
to recognise a protocol video by its contents, so renaming a copy cannot make a training video look like unseen footage.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "results/demo/protocol_digests.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests", type=Path, default=REPO_ROOT / "results/p1b/p1b-video-ar-fold1")
    args = ap.parse_args(argv)
    digests: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        with (args.manifests / f"{split}_manifest.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if digests.setdefault(row["file_digest"], row["source_video_id"]) != row["source_video_id"]:
                    raise SystemExit(f"digest {row['file_digest']} names two videos")
    membership = json.loads((REPO_ROOT / "results/demo/membership.json").read_text())["videos"]
    if set(digests.values()) != set(membership):
        raise SystemExit("manifest videos differ from results/demo/membership.json")
    OUTPUT.write_text(json.dumps({"note": "sha256 of the source MP4 -> protocol video id", "digests": digests}, indent=0, sort_keys=True) + "\n")
    print(json.dumps({"videos": len(digests), "output": str(OUTPUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
