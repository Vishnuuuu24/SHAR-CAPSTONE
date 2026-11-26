#!/usr/bin/env python3
"""Drop the given paths from the page cache and report the cgroup's memory.

Staging bulk data into pod-local scratch pulls every copied byte through the
page cache twice -- once reading the source, once writing the destination --
and inside a container that cache is charged to `memory.max`. The copy itself
survives, because writeback and reclaim keep pace, but it leaves the container
sitting at its memory ceiling before the job command has run a single line.
The next substantial allocation is then OOM-killed with no process's RSS ever
looking suspicious (K-15).

These pages are worthless to us: the source is not read again, and the
destination is about to be read in a random-access pattern that will fault in
whatever it needs. So drop them, and print what that recovered.

`sync` first, from the caller: POSIX_FADV_DONTNEED only drops clean pages, so
anything still dirty from the copy would otherwise simply stay.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import pagecache  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        print("usage: drop_page_cache.py <path> [path ...]", file=sys.stderr)
        return 2
    if not pagecache.AVAILABLE:
        print("drop-cache: posix_fadvise unavailable; skipping", flush=True)
        return 0

    before = pagecache.cgroup_current_bytes()
    dropped = 0
    for path in paths:
        if not path.exists():
            print(f"drop-cache: {path} does not exist; skipping", flush=True)
            continue
        dropped += pagecache.drop_tree(path)
    after = pagecache.cgroup_current_bytes()

    gib = 2**30
    if before is None or after is None:
        print(f"drop-cache: hinted {dropped} files (cgroup usage unavailable)", flush=True)
    else:
        print(
            f"drop-cache: hinted {dropped} files; cgroup {before / gib:.2f} GiB "
            f"-> {after / gib:.2f} GiB (recovered {(before - after) / gib:.2f} GiB)",
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
