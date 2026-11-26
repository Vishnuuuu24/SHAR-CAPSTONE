"""Built-in example clips for the demo page: a manifest of named MP4s plus HTTP byte-range helpers.

The clips are licensed UCF-Crime videos (no redistribution), so the server only serves them to requests carrying the
demo's access token, and only names listed in the manifest (never an arbitrary path). The folder is git-ignored.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MANIFEST = "results/demo/samples/samples.json"


def load_samples(repo: Path, manifest: str = MANIFEST) -> dict[str, dict]:
    """{file name: entry} for every listed clip that exists on disk; {} when there is no manifest."""

    path = repo / manifest
    if not path.exists():
        return {}
    entries = json.loads(path.read_text())["samples"]
    out = {}
    for e in entries:
        f = path.parent / e["file"]
        if Path(e["file"]).name == e["file"] and f.is_file():
            out[e["file"]] = {**e, "path": f, "bytes": f.stat().st_size}
    return out


def public_entry(e: dict) -> dict:
    return {k: v for k, v in e.items() if k not in ("path",)}


def byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """(start, end inclusive) for a single 'bytes=a-b' / 'bytes=a-' / 'bytes=-n' Range header; None = whole file.

    Raises ValueError for a range that cannot be satisfied.
    """

    if not header:
        return None
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not m or (not m.group(1) and not m.group(2)):
        return None
    a, b = m.group(1), m.group(2)
    if not a:
        n = int(b)
        if n == 0:
            raise ValueError("empty suffix range")
        return max(0, size - n), size - 1
    start = int(a)
    end = min(int(b), size - 1) if b else size - 1
    if start >= size or start > end:
        raise ValueError("range not satisfiable")
    return start, end
