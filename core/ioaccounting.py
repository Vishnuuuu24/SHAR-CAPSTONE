"""Snapshot the I/O a run actually performs, so a pathological read is visible.

## Why this exists

On 2026-08-27 a registered run spent 7h25m of GPU-slice time and produced
zero epochs. Its config omitted `data.video_frames`, so it fell through to the
per-file loader and, with `verify_digests: true`, re-hashed a ~50 MB source
MP4 once per manifest row: 391,014 rows against a 97.7 GiB corpus. The bug was
diagnosed only because someone thought to read `/proc/<pid>/io` by hand and
found `rchar: 39,049,767,064,207` -- 39 TB of reads. Nothing in the run's own
artifacts recorded it. K-21.

This module puts that number in the artifact. A run that reads two orders of
magnitude more than its corpus is now self-evidently wrong from its own
manifest, without anyone having to catch it live.

## The two counter families, and why both

`/proc/self/io` (always present on Linux)
    `rchar`/`wchar` count bytes moved by read()/write() syscalls, including
    those served from page cache. `read_bytes`/`write_bytes` count what
    actually went to a block device. The gap between them IS the cache hit
    rate, which is the quantity K-21's post-mortem had to estimate rather
    than measure.

NFS per-mount counters in `/proc/self/mountstats` (present only when an NFS
mount exists)
    The `bytes:` line carries normal-read/normal-write and direct and page
    totals per mount. On a network filesystem this is the only honest measure
    of what crossed the wire -- `rchar` cannot distinguish a cached re-read
    from a fetch. Absent on a node with no NFS client mounts, which includes
    the storage server itself and every CPU-only test environment.

## Absence is not failure

Unlike `core.precision` and `core.throughput`, this module never refuses.
Those two govern what a run computes, where a silently-wrong setting
invalidates a result. This one only observes. A missing counter file, an
unreadable mount, a kernel that words the fields differently -- none of that
makes a run's science wrong, and aborting a multi-hour run over an
observability read would be a far worse failure than the gap it reports.
Every field is therefore optional and every parse is defensive; callers get
`None` and a recorded reason, never an exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROC_IO = Path("/proc/self/io")
PROC_MOUNTSTATS = Path("/proc/self/mountstats")

# The /proc/self/io fields worth carrying. rchar/wchar are syscall-level and
# include cache hits; read_bytes/write_bytes are block-device level.
_IO_FIELDS = ("rchar", "wchar", "read_bytes", "write_bytes")

# NFS `bytes:` line fields, in kernel order (fs/nfs/super.c).
_NFS_BYTES_FIELDS = (
    "normal_read", "normal_write",
    "direct_read", "direct_write",
    "server_read", "server_write",
    "readpages", "writepages",
)


def _read_proc_io() -> tuple[dict[str, int], str | None]:
    try:
        text = PROC_IO.read_text(encoding="utf-8")
    except OSError as exc:
        return {}, f"{PROC_IO} unreadable: {exc.__class__.__name__}"
    counters: dict[str, int] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() in _IO_FIELDS:
            try:
                counters[key.strip()] = int(value.strip())
            except ValueError:
                continue
    return counters, None


def _read_nfs_bytes() -> tuple[dict[str, dict[str, int]], str | None]:
    """Per-NFS-mount byte counters, keyed by mount point.

    Returns an empty mapping with no reason on a host that simply has no NFS
    client mounts -- that is the ordinary case on the storage server and in
    CPU-only tests, not a problem to report.
    """
    try:
        text = PROC_MOUNTSTATS.read_text(encoding="utf-8")
    except OSError as exc:
        return {}, f"{PROC_MOUNTSTATS} unreadable: {exc.__class__.__name__}"
    mounts: dict[str, dict[str, int]] = {}
    mount_point: str | None = None
    for line in text.splitlines():
        parts = line.split()
        if parts[:1] == ["device"]:
            # `device <src> mounted on <mnt> with fstype nfs4 ...`
            mount_point = None
            if "fstype" in parts:
                fstype = parts[parts.index("fstype") + 1] if parts.index("fstype") + 1 < len(parts) else ""
                if fstype.startswith("nfs") and fstype != "nfsd" and "on" in parts:
                    mount_point = parts[parts.index("on") + 1]
        elif mount_point and parts[:1] == ["bytes:"]:
            values = []
            for raw in parts[1:]:
                try:
                    values.append(int(raw))
                except ValueError:
                    break
            mounts[mount_point] = dict(zip(_NFS_BYTES_FIELDS, values))
            mount_point = None
    return mounts, None


@dataclass(frozen=True)
class IoSnapshot:
    """One reading of this process's I/O counters."""

    proc_io: dict[str, int] = field(default_factory=dict)
    nfs_bytes: dict[str, dict[str, int]] = field(default_factory=dict)
    unavailable: list[str] = field(default_factory=list)

    @classmethod
    def capture(cls) -> "IoSnapshot":
        proc_io, io_reason = _read_proc_io()
        nfs, nfs_reason = _read_nfs_bytes()
        return cls(
            proc_io=proc_io,
            nfs_bytes=nfs,
            unavailable=[r for r in (io_reason, nfs_reason) if r],
        )


def io_delta(start: IoSnapshot, end: IoSnapshot) -> dict[str, Any]:
    """What happened between two snapshots, in the shape a manifest records.

    `cache_hit_fraction` is the headline: bytes the syscalls asked for that
    never reached a block device. A frame loader reading a staged blob should
    sit high; a run re-hashing source media per row -- K-21's failure -- shows
    a rchar far above any plausible multiple of the corpus size, which is the
    signal this whole module exists to surface.
    """

    proc = {k: end.proc_io[k] - start.proc_io[k] for k in _IO_FIELDS
            if k in start.proc_io and k in end.proc_io}
    nfs: dict[str, dict[str, int]] = {}
    for mount, after in end.nfs_bytes.items():
        before = start.nfs_bytes.get(mount, {})
        nfs[mount] = {k: after[k] - before.get(k, 0) for k in after}

    report: dict[str, Any] = {"proc_io": proc, "nfs_bytes_by_mount": nfs}
    rchar, read_bytes = proc.get("rchar"), proc.get("read_bytes")
    if rchar is not None and read_bytes is not None and rchar > 0:
        report["cache_hit_fraction"] = round(1.0 - min(read_bytes, rchar) / rchar, 6)
    if nfs:
        report["nfs_read_bytes_total"] = sum(
            m.get("normal_read", 0) + m.get("direct_read", 0) for m in nfs.values()
        )
    else:
        # Recorded explicitly: "no NFS mount" and "counters unavailable" are
        # different states and a reader must be able to tell them apart.
        report["nfs_read_bytes_total"] = None
        report["nfs_note"] = "no NFS client mount visible to this process"
    unavailable = sorted(set(start.unavailable) | set(end.unavailable))
    if unavailable:
        report["unavailable"] = unavailable
    return report
