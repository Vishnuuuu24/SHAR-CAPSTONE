"""Tests for core.ioaccounting.

The module's contract is unusual for this repository: it must NEVER raise.
Every other resolve-or-refuse module here fails closed because a wrong value
would invalidate a result; this one only observes, so an unavailable counter
must degrade to a recorded absence rather than kill a multi-hour run.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from core.ioaccounting import IoSnapshot, io_delta

MOUNTSTATS = """device rootfs mounted on / with fstype rootfs
device master:/home mounted on /home with fstype nfs4 statvers=1.1
\tbytes: 100 200 0 0 300 400 5 6
device nfsd mounted on /proc/fs/nfsd with fstype nfsd
device /dev/sda1 mounted on /scratch with fstype xfs
"""


class IoSnapshotTests(unittest.TestCase):
    def test_capture_on_this_host_never_raises(self) -> None:
        snapshot = IoSnapshot.capture()
        self.assertIsInstance(snapshot.proc_io, dict)
        self.assertIsInstance(snapshot.nfs_bytes, dict)

    def test_parses_nfs_mounts_and_ignores_nfsd_and_non_nfs(self) -> None:
        with mock.patch.object(Path, "read_text", return_value=MOUNTSTATS):
            snapshot = IoSnapshot.capture()
        self.assertEqual(list(snapshot.nfs_bytes), ["/home"])
        self.assertEqual(snapshot.nfs_bytes["/home"]["normal_read"], 100)
        self.assertEqual(snapshot.nfs_bytes["/home"]["server_read"], 300)

    def test_unreadable_counters_are_recorded_not_raised(self) -> None:
        with mock.patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            snapshot = IoSnapshot.capture()
        self.assertEqual(snapshot.proc_io, {})
        self.assertEqual(len(snapshot.unavailable), 2)
        # And a delta over two such snapshots still produces a usable record.
        report = io_delta(snapshot, snapshot)
        self.assertIn("unavailable", report)


class IoDeltaTests(unittest.TestCase):
    def test_delta_subtracts_and_derives_cache_hit_fraction(self) -> None:
        start = IoSnapshot(proc_io={"rchar": 1_000, "wchar": 0, "read_bytes": 0, "write_bytes": 0})
        end = IoSnapshot(proc_io={"rchar": 11_000, "wchar": 0, "read_bytes": 1_000, "write_bytes": 0})
        report = io_delta(start, end)
        self.assertEqual(report["proc_io"]["rchar"], 10_000)
        # 1,000 of 10,000 syscall bytes reached the device: 90% served from cache.
        self.assertAlmostEqual(report["cache_hit_fraction"], 0.9)

    def test_k21_shape_is_visible_in_the_record(self) -> None:
        """The 2026-08-27 burn: 39 TB of rchar, nearly all cache-served."""
        start = IoSnapshot(proc_io={"rchar": 0, "wchar": 0, "read_bytes": 0, "write_bytes": 0})
        end = IoSnapshot(proc_io={"rchar": 39_049_767_064_207, "wchar": 0,
                                  "read_bytes": 104_868_019_448, "write_bytes": 0})
        report = io_delta(start, end)
        self.assertGreater(report["proc_io"]["rchar"], 39e12)
        self.assertGreater(report["cache_hit_fraction"], 0.99)

    def test_absent_nfs_is_distinguished_from_unavailable(self) -> None:
        report = io_delta(IoSnapshot(), IoSnapshot())
        self.assertIsNone(report["nfs_read_bytes_total"])
        self.assertIn("nfs_note", report)
        self.assertNotIn("unavailable", report)

    def test_nfs_read_total_sums_normal_and_direct_across_mounts(self) -> None:
        start = IoSnapshot(nfs_bytes={"/home": {"normal_read": 10, "direct_read": 1}})
        end = IoSnapshot(nfs_bytes={"/home": {"normal_read": 110, "direct_read": 21},
                                    "/data": {"normal_read": 5, "direct_read": 0}})
        report = io_delta(start, end)
        self.assertEqual(report["nfs_read_bytes_total"], 100 + 20 + 5)

    def test_zero_rchar_does_not_divide_by_zero(self) -> None:
        report = io_delta(IoSnapshot(proc_io={"rchar": 0, "read_bytes": 0}),
                          IoSnapshot(proc_io={"rchar": 0, "read_bytes": 0}))
        self.assertNotIn("cache_hit_fraction", report)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
