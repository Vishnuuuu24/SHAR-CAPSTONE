"""Keep bulk file reads from being charged to the pod's memory cgroup.

Inside a container, **file-backed page cache counts against
`memory.max`**. That makes any full pass over bulk data -- copying the
84.3 GiB frame blob into scratch, or hashing it for its integrity check --
quietly consume the entire memory budget, even though no process's RSS ever
grows. The pages are clean and reclaimable, so this is survivable when the
kernel can reclaim as fast as the reader dirties; on local xfs it cannot, and
the next allocation of any size is OOM-killed. Measured 2026-08-25: hashing the
blob drove `memory.current` from 0.28 GiB to 15.99 GiB of a 16 GiB limit.
See K-15.

The fix is to tell the kernel what we already know -- these pages are read once
and never revisited -- so it can drop them as they are consumed rather than
evicting something useful to keep them. `POSIX_FADV_DONTNEED` only ever drops
*clean* pages, so it cannot discard unwritten data; callers that have just
written should `sync` first, or the still-dirty pages simply stay.

This is advisory and Linux-only. Where it is unavailable every function here
becomes a no-op and the caller degrades to ordinary buffered behaviour, which
is exactly the previous state -- slower to reclaim, never wrong.
"""

from __future__ import annotations

import os
from pathlib import Path

# How much to read between drop hints during a streaming pass. Large enough
# that the syscall is negligible against the read, small enough that the
# resident window stays trivial next to any sane memory limit.
DROP_INTERVAL_BYTES = 64 << 20

_FADVISE = getattr(os, "posix_fadvise", None)
_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", None)

AVAILABLE = _FADVISE is not None and _DONTNEED is not None


def drop_range(fd: int, offset: int, length: int) -> None:
    """Ask the kernel to drop `length` bytes of `fd`'s cached pages.

    `length` must be positive. The syscall itself treats 0 as "to end of
    file", but this wrapper deliberately does not: an accidental zero is far
    more likely to be an empty range than a request to drop everything, and
    silently promoting it to "the whole file" is the more dangerous reading.
    Callers wanting a whole file use `drop_file`, which passes the real size.
    """
    if not AVAILABLE or length <= 0:
        return
    try:
        _FADVISE(fd, offset, length, _DONTNEED)
    except OSError:
        # Advisory: a filesystem that refuses the hint is not an error, it just
        # means we get the old behaviour for this file.
        pass


def drop_file(path: Path | str) -> None:
    """Drop the whole of `path` from the page cache, if it is cached at all.

    The length is the file's actual size rather than the syscall's "0 means to
    EOF" shorthand -- `drop_range` rejects non-positive lengths, and relying on
    that shorthand once already produced a silent no-op that looked exactly
    like a working call (staging reported "recovered 0.00 GiB" while the cache
    stayed pinned at the cgroup limit).
    """
    if not AVAILABLE:
        return
    try:
        fd = os.open(os.fspath(path), os.O_RDONLY)
    except OSError:
        return
    try:
        size = os.fstat(fd).st_size
        if size > 0:
            drop_range(fd, 0, size)
    finally:
        os.close(fd)


def drop_tree(root: Path | str) -> int:
    """Drop every regular file under `root`. Returns how many were visited."""
    root = Path(root)
    if root.is_file():
        drop_file(root)
        return 1
    visited = 0
    for path in root.rglob("*"):
        if path.is_file():
            drop_file(path)
            visited += 1
    return visited


def cgroup_current_bytes() -> int | None:
    """This container's charged memory, or None outside a cgroup-v2 container.

    Reported rather than enforced: it is the number that explains an OOM kill
    when no process's RSS does.
    """
    for candidate in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return int(Path(candidate).read_text().strip())
        except (OSError, ValueError):
            continue
    return None
