"""Throttle — storm control for home-channel forwarding.

Pure logic with an injectable clock so it is fully testable without sleeping.

Policy:
  * dedup     — an identical message is admitted at most once per dedup_window
                (a persistent error still resurfaces each window as a heartbeat,
                rather than flooding or vanishing forever).
  * rate-cap  — at most ``rate`` admits per ``window`` seconds; excess suppressed.
  * summary   — the first admit following any suppression leads with a short
                "N suppressed" line so operators know they aren't seeing all.
"""
from __future__ import annotations

from typing import Callable, List


class Throttle:
    def __init__(
        self,
        rate: int,
        window: float,
        dedup_window: float,
        clock: Callable[[], float],
    ) -> None:
        self.rate = rate
        self.window = window
        self.dedup_window = dedup_window
        self._clock = clock

        self._window_start = clock()
        self._count = 0  # admits in the current rate window
        self._last_seen: dict[str, float] = {}  # message -> last admit time
        self._last_prune = self._window_start
        self._suppressed = 0  # suppressions since last summary

    def admit(self, message: str) -> list[str]:
        now = self._clock()

        # Roll the rate window if it has elapsed.
        if now - self._window_start >= self.window:
            self._window_start = now
            self._count = 0

        # Dedup: identical message admitted at most once per dedup_window.
        last = self._last_seen.get(message)
        if last is not None and now - last < self.dedup_window:
            self._suppressed += 1
            return []

        # Rate-cap: cap admits per window.
        if self._count >= self.rate:
            self._suppressed += 1
            return []

        # Admit.
        self._count += 1
        self._last_seen[message] = now
        self._prune(now)

        out: list[str] = []
        if self._suppressed > 0:
            out.append(self._summary(self._suppressed))
            self._suppressed = 0
        out.append(message)
        return out

    def flush_summary(self) -> list[str]:
        """Emit a pending "N suppressed" summary outside the admit path.

        A burst that ends with suppression (no following admit) would otherwise
        never report its count. The worker calls this on idle so the tail of a
        storm is still surfaced.
        """
        if self._suppressed <= 0:
            return []
        out = [self._summary(self._suppressed)]
        self._suppressed = 0
        return out

    def _summary(self, n: int) -> str:
        noun = "message" if n == 1 else "messages"
        return f"… {n} more log {noun} suppressed"

    def _prune(self, now: float) -> None:
        # Drop dedup entries past their window so the map can't grow unbounded.
        # Runs at most once per dedup_window, not on every admit: an expired
        # entry is harmless (admit re-admits it), so eager pruning only wastes
        # an O(n) scan on the hot path during high-cardinality storms.
        if now - self._last_prune < self.dedup_window:
            return
        self._last_prune = now
        expired = [m for m, ts in self._last_seen.items() if now - ts >= self.dedup_window]
        for m in expired:
            del self._last_seen[m]
