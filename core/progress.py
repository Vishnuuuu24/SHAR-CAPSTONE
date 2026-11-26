"""Lightweight phase-progress logging for long-running scripts.

Long P1C/P2A scripts can go silent for minutes at a time during slow,
legitimate I/O (cold NFS reads of many small files on the DGX, digest
verification over a large manifest, a multi-hour training loop) with no
signal that anything is happening. This module gives every such script a
single, consistent way to say "here is what I am doing right now" without
guessing whether it has hung.

This is progress reporting only. It writes exclusively to stderr, is never
read back by any script, and must never be treated as evidence or a
measurement -- only the JSON artifacts a script writes under ``results/`` are
that. Nothing here should sit inside a tightly-timed region in a way that
would perturb what is being measured; call sites log immediately before or
after a phase, or after a batch's timing has already been recorded.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

_START = time.perf_counter()


def log_phase(message: str) -> None:
    """Print one immediately-flushed, timestamped progress line to stderr."""

    elapsed = time.perf_counter() - _START
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} +{elapsed:8.1f}s] {message}", file=sys.stderr, flush=True)


class BatchProgress:
    """Prints a progress line every ``every`` steps or ``min_interval_seconds``,
    whichever comes first, so a fast run does not spam and a slow one is
    never silent for long. Call ``step()`` once per item/batch processed.
    """

    def __init__(
        self,
        total: int | None,
        *,
        label: str,
        every: int = 50,
        min_interval_seconds: float = 15.0,
    ) -> None:
        self._total = total
        self._label = label
        self._every = max(1, every)
        self._min_interval = min_interval_seconds
        self._last_time = time.perf_counter()
        self._count = 0

    def step(self, n: int = 1) -> None:
        self._count += n
        now = time.perf_counter()
        due_by_count = self._count % self._every == 0
        due_by_time = (now - self._last_time) >= self._min_interval
        if due_by_count or due_by_time:
            self._last_time = now
            total_part = f"/{self._total}" if self._total is not None else ""
            log_phase(f"{self._label}: {self._count}{total_part}")

    def done(self) -> None:
        total_part = f"/{self._total}" if self._total is not None else ""
        log_phase(f"{self._label}: {self._count}{total_part} (done)")
