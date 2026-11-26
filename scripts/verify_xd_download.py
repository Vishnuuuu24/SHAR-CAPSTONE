#!/usr/bin/env python3
"""Verify a Hugging Face mirror download of XD-Violence (jherng/xd-violence) before any use.

Checks: every video's SHA-256 equals the LFS digest the HF client recorded at download time; no file is missing or
truncated; filename labels parse to the official codes; every officially annotated test video is present.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.xd_dataset import labels_of  # noqa: E402

def _sha(path: str) -> tuple[str, str, int]:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return path, h.hexdigest(), Path(path).stat().st_size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("Datasets/XD-Violence-full/xd-violence"))
    ap.add_argument("--annotations", type=Path, default=Path("Datasets/XD-Violence-full/official/annotations.txt"))
    ap.add_argument("--output", type=Path, default=Path("results/xd/download-verify.json"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    meta_root = args.root / ".cache/huggingface/download"
    expected = {}
    for m in meta_root.rglob("*.mp4.metadata"):
        rev, etag = m.read_text().splitlines()[:2]
        expected[str(m.relative_to(meta_root))[: -len(".metadata")]] = (rev, etag)
    present = {str(p.relative_to(args.root)): p for p in (args.root / "data/video").rglob("*.mp4")}
    missing = sorted(set(expected) - set(present))
    unexpected = sorted(set(present) - set(expected))
    with ProcessPoolExecutor(args.workers) as ex:
        hashed = list(ex.map(_sha, [str(present[k]) for k in sorted(set(present) & set(expected))], chunksize=8))
    mismatch = [p for p, h, _ in hashed if h != expected[str(Path(p).relative_to(args.root))][1]]
    split_counts, label_counts, multi = collections.Counter(), collections.Counter(), 0
    for rel in present:
        split = "test" if "/test_videos/" in rel else "train"
        labs = labels_of(Path(rel).stem)
        split_counts[split] += 1
        multi += len(labs) > 1
        for lab in labs:
            label_counts[(split, lab)] += 1
    test_names = {Path(r).stem for r in present if "/test_videos/" in r}
    # 14 of the official annotation lines carry a ".mp4" suffix and the rest do not (checked 2026-10-29).
    annotated = [line.split()[0].removesuffix(".mp4") for line in args.annotations.read_text().splitlines() if line.strip()]
    not_found = sorted(set(annotated) - test_names)
    report = {
        "mirror": "huggingface.co/datasets/jherng/xd-violence", "mirror_revisions": sorted({r for r, _ in expected.values()}),
        "files_expected": len(expected), "files_present": len(present), "missing": missing, "unexpected": unexpected,
        "sha256_mismatches": mismatch, "total_bytes": sum(s for _, _, s in hashed),
        "split_counts": dict(split_counts), "official_split_counts": {"train": 3954, "test": 800},
        "label_counts": {f"{s}/{l}": n for (s, l), n in sorted(label_counts.items())}, "multi_label_videos": multi,
        "official_annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "annotated_test_videos": len(annotated), "annotated_not_in_download": not_found,
        "file_sha256": {str(Path(p).relative_to(args.root)): h for p, h, _ in hashed},
        "ok": not missing and not mismatch and not not_found,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
