"""Event-loop freeze watchdog — last-resort detection + self-heal.

A hung *synchronous* call anywhere on the gateway's event-loop thread (a
blocking socket read, a stuck subprocess, a lock held by another process)
silently stalls every coroutine at once — SSE listeners, health monitors,
message handling, the housekeeping thread's scheduled work — with near-zero
CPU, since the process is blocked on I/O rather than spinning. Nothing else
in the gateway can detect this: the SSE-specific health monitor
(``gateway.platforms.signal._health_monitor`` and its per-platform
equivalents) only notices when *its own* task stops running, which never
happens because the freeze is upstream of it too.

This module is a coarse, platform-agnostic safety net: periodically prove
the loop is still processing a trivial coroutine, and if it stays
unresponsive past a threshold, dump every thread's current stack (this
works even while the loop itself is frozen — ``faulthandler`` reads thread
state directly via the C level, it doesn't need the loop's cooperation)
and hard-exit. The gateway runs under s6 supervision in production
(``s6-supervise gateway-default``, auto-restart on crash), so a hard exit
restores service automatically instead of the process staying dead for
hours with nothing in the logs to explain why.

This does not fix the underlying freeze — it only bounds how long the
gateway stays dead once one happens, and captures the evidence needed to
root-cause it next time.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import faulthandler
import logging
import os
import threading
import time
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Callable, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_CHECK_INTERVAL = 30.0
DEFAULT_UNRESPONSIVE_THRESHOLD = 90.0
DEFAULT_PROBE_TIMEOUT = 15.0


def probe_loop_responsive(loop: asyncio.AbstractEventLoop, timeout: float) -> bool:
    """Return True if *loop* processes a trivial coroutine within *timeout* seconds.

    Called from a different thread than the loop itself — that's the whole
    point: a frozen loop can't answer its own liveness check.
    """
    coro = asyncio.sleep(0)
    try:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:
        # Loop isn't running / already closed — nothing to watch. Close the
        # coroutine explicitly: run_coroutine_threadsafe raised before
        # scheduling it, so it would otherwise never be awaited or closed.
        coro.close()
        return False
    try:
        future.result(timeout=timeout)
        return True
    except concurrent.futures.TimeoutError:
        future.cancel()
        return False
    except Exception:
        return False


def dump_diagnostics(dump_path: Path) -> None:
    """Write an all-thread stack dump to *dump_path*.

    Uses ``faulthandler.dump_traceback``, not ``traceback``/``sys._current_frames``
    wrapped in asyncio helpers — those would themselves need the frozen loop
    to cooperate. ``faulthandler`` reads thread state directly and works
    regardless of what any individual thread is blocked on.
    """
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "w", encoding="utf-8") as fh:
        fh.write(
            f"Event loop freeze detected at "
            f"{datetime.now(UTC).isoformat()}\n\n"
        )
        faulthandler.dump_traceback(file=fh, all_threads=True)


def _default_dump_path() -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return get_hermes_home() / "logs" / f"event_loop_freeze_{ts}.log"


class EventLoopWatchdog:
    """Detects a fully-frozen event loop and self-heals by exiting the process.

    The decision logic (:meth:`check_once`) is pure/injectable so it can be
    tested without real threads, real sleeps, or an actual process exit —
    see ``tests/gateway/test_event_loop_watchdog.py``. :meth:`run` is the
    thin real-world wrapper meant to be started on its own daemon thread.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        check_interval: float = DEFAULT_CHECK_INTERVAL,
        unresponsive_threshold: float = DEFAULT_UNRESPONSIVE_THRESHOLD,
        probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
        probe_fn: Callable[[asyncio.AbstractEventLoop, float], bool] = probe_loop_responsive,
        dump_fn: Callable[[Path], None] = dump_diagnostics,
        dump_path_fn: Callable[[], Path] = _default_dump_path,
        exit_fn: Callable[[int], None] = lambda code: os._exit(code),
        clock_fn: Callable[[], float] = time.monotonic,
    ):
        self._loop = loop
        self._check_interval = check_interval
        self._unresponsive_threshold = unresponsive_threshold
        self._probe_timeout = probe_timeout
        self._probe_fn = probe_fn
        self._dump_fn = dump_fn
        self._dump_path_fn = dump_path_fn
        self._exit_fn = exit_fn
        self._clock_fn = clock_fn
        self._first_unresponsive_at: Optional[float] = None

    def check_once(self) -> bool:
        """Run one probe cycle.

        Returns True if a freeze was declared (dump + exit were invoked) —
        callers should stop looping once this returns True.
        """
        now = self._clock_fn()
        if self._probe_fn(self._loop, self._probe_timeout):
            self._first_unresponsive_at = None
            return False

        if self._first_unresponsive_at is None:
            self._first_unresponsive_at = now
            logger.warning("Event loop watchdog: probe unresponsive, starting freeze timer")
            return False

        stalled_for = now - self._first_unresponsive_at
        if stalled_for < self._unresponsive_threshold:
            return False

        dump_path = self._dump_path_fn()
        logger.critical(
            "Event loop watchdog: loop unresponsive for %.0fs (threshold %.0fs) — "
            "dumping thread stacks to %s and exiting so the supervisor restarts us",
            stalled_for, self._unresponsive_threshold, dump_path,
        )
        try:
            self._dump_fn(dump_path)
        except Exception:
            logger.exception("Event loop watchdog: diagnostic dump failed")
        self._exit_fn(1)
        return True

    def run(self, stop_event: threading.Event) -> None:
        """Blocking loop — run this on its own daemon thread."""
        logger.info(
            "Event loop watchdog started (check_interval=%.0fs, unresponsive_threshold=%.0fs)",
            self._check_interval, self._unresponsive_threshold,
        )
        while not stop_event.is_set():
            if self.check_once():
                return
            stop_event.wait(timeout=self._check_interval)
        logger.info("Event loop watchdog stopped")


def start_event_loop_watchdog(
    loop: asyncio.AbstractEventLoop,
    stop_event: threading.Event,
    **kwargs,
) -> threading.Thread:
    """Create, start, and return the watchdog's daemon thread."""
    watchdog = EventLoopWatchdog(loop, **kwargs)
    thread = threading.Thread(
        target=watchdog.run,
        args=(stop_event,),
        daemon=True,
        name="event-loop-watchdog",
    )
    thread.start()
    return thread
