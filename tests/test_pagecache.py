"""Tests for `core.pagecache`.

The hints these functions issue are advisory, so a test cannot assert that the
kernel actually dropped anything -- it may decline. What is worth pinning is
that the helpers are safe: they never corrupt data, never raise on the ragged
inputs they will meet in a pod (missing paths, empty files, unreadable
directories), and degrade to a no-op where the syscall is unavailable.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import pagecache  # noqa: E402


class DropTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_dropping_does_not_change_file_contents(self) -> None:
        # The property that actually matters: this is a cache hint, not a write.
        path = self.tmp / "payload.bin"
        payload = os.urandom(200_000)
        path.write_bytes(payload)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        pagecache.drop_file(path)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        self.assertEqual(path.read_bytes(), payload)

    def test_missing_path_is_ignored_rather_than_raising(self) -> None:
        # Staging drops both source and destination; a caller should not have
        # to pre-check either.
        pagecache.drop_file(self.tmp / "does-not-exist")
        self.assertEqual(pagecache.drop_tree(self.tmp / "no-such-dir"), 0)

    def test_empty_file_and_empty_directory_are_handled(self) -> None:
        (self.tmp / "empty.bin").write_bytes(b"")
        pagecache.drop_file(self.tmp / "empty.bin")
        (self.tmp / "emptydir").mkdir()
        self.assertEqual(pagecache.drop_tree(self.tmp / "emptydir"), 0)

    def test_drop_tree_counts_regular_files_recursively(self) -> None:
        (self.tmp / "a").mkdir()
        (self.tmp / "a" / "b").mkdir()
        for name in ("a/one.bin", "a/two.bin", "a/b/three.bin"):
            (self.tmp / name).write_bytes(b"x" * 1024)
        # Directories and symlinks must not be counted as files.
        (self.tmp / "a" / "link.bin").symlink_to(self.tmp / "a" / "one.bin")
        self.assertEqual(pagecache.drop_tree(self.tmp / "a"), 4)

    def test_drop_tree_accepts_a_single_file(self) -> None:
        path = self.tmp / "solo.bin"
        path.write_bytes(b"y" * 16)
        self.assertEqual(pagecache.drop_tree(path), 1)

    def test_drop_range_tolerates_degenerate_lengths(self) -> None:
        path = self.tmp / "payload.bin"
        path.write_bytes(b"z" * 4096)
        with path.open("rb") as handle:
            pagecache.drop_range(handle.fileno(), 0, 0)
            pagecache.drop_range(handle.fileno(), 0, -1)
            pagecache.drop_range(handle.fileno(), 1 << 40, 4096)

    def test_cgroup_current_is_an_int_or_none(self) -> None:
        # None off a cgroup-v2 host; an int inside a pod. Both are valid, and
        # callers must handle either -- so pin exactly that contract.
        value = pagecache.cgroup_current_bytes()
        self.assertTrue(value is None or (isinstance(value, int) and value >= 0))


class SyscallIsActuallyIssuedTests(unittest.TestCase):
    """The hint must reach the kernel, not just fail to raise.

    Every other test here passes whether or not the syscall happens, because
    an advisory hint has no observable effect the caller can assert on. That
    blind spot shipped a real bug: `drop_file` passed length 0 (the syscall's
    "to end of file" shorthand) into a wrapper that rejects non-positive
    lengths, so it silently did nothing while looking perfectly healthy --
    staging reported "recovered 0.00 GiB" with the cache pinned at the cgroup
    limit. So record the calls and assert on them.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.calls: list[tuple[int, int]] = []
        self._real = pagecache._FADVISE
        self._real_available = pagecache.AVAILABLE
        self._real_dontneed = pagecache._DONTNEED

        def recording(fd, offset, length, advice):  # noqa: ANN001
            self.calls.append((offset, length))
            if self._real is not None:
                return self._real(fd, offset, length, advice)
            return None

        pagecache._FADVISE = recording
        pagecache._DONTNEED = 4
        pagecache.AVAILABLE = True

    def tearDown(self) -> None:
        pagecache._FADVISE = self._real
        pagecache._DONTNEED = self._real_dontneed
        pagecache.AVAILABLE = self._real_available
        self._tmp.cleanup()

    def test_drop_file_requests_the_whole_file(self) -> None:
        path = self.tmp / "payload.bin"
        path.write_bytes(b"q" * 5000)
        pagecache.drop_file(path)
        self.assertEqual(self.calls, [(0, 5000)])

    def test_drop_tree_requests_every_file(self) -> None:
        (self.tmp / "one.bin").write_bytes(b"a" * 100)
        (self.tmp / "two.bin").write_bytes(b"b" * 250)
        pagecache.drop_tree(self.tmp)
        self.assertEqual(sorted(self.calls), [(0, 100), (0, 250)])

    def test_empty_file_issues_no_call(self) -> None:
        (self.tmp / "empty.bin").write_bytes(b"")
        pagecache.drop_file(self.tmp / "empty.bin")
        self.assertEqual(self.calls, [])

    def test_non_positive_lengths_are_refused_rather_than_meaning_everything(self) -> None:
        path = self.tmp / "payload.bin"
        path.write_bytes(b"r" * 128)
        with path.open("rb") as handle:
            pagecache.drop_range(handle.fileno(), 0, 0)
            pagecache.drop_range(handle.fileno(), 0, -5)
        self.assertEqual(self.calls, [])


class UnavailableTests(unittest.TestCase):
    """Where posix_fadvise is missing, every helper degrades to a no-op."""

    def setUp(self) -> None:
        self._available = pagecache.AVAILABLE
        pagecache.AVAILABLE = False

    def tearDown(self) -> None:
        pagecache.AVAILABLE = self._available

    def test_helpers_are_silent_no_ops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            path.write_bytes(b"w" * 2048)
            pagecache.drop_file(path)
            pagecache.drop_range(0, 0, 0)
            self.assertEqual(path.read_bytes(), b"w" * 2048)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
