"""HomeLogWorker — background daemon that turns queued records into home sends.

The handler only enqueues; this worker owns all the slow/fallible work: it drains
the queue, runs each message through the Throttle, and calls ``sender`` under the
re-entrancy guard so the send's own log output never feeds back. A send failure is
swallowed — losing a forwarded log must never crash the agent.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

from .guard import ReentrancyGuard
from .throttle import Throttle

logger = logging.getLogger(__name__)


class HomeLogWorker:
    def __init__(
        self,
        out_queue: queue.Queue[str],
        throttle: Throttle,
        sender: Callable[[str], None],
        guard: ReentrancyGuard,
        poll_timeout: float = 0.5,
    ) -> None:
        self.out_queue = out_queue
        self.throttle = throttle
        self.sender = sender
        self.guard = guard
        self.poll_timeout = poll_timeout
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def process(self, message: str) -> None:
        """Throttle one message and send what survives. Synchronous, testable."""
        self._send_all(self.throttle.admit(message))

    def idle_flush(self) -> None:
        """Deliver any pending suppression summary when the queue goes quiet."""
        self._send_all(self.throttle.flush_summary())

    def _send_all(self, messages) -> None:
        for out in messages:
            with self.guard:
                try:
                    self.sender(out)
                except Exception as exc:  # never let a bad send escape
                    logger.debug("home_log_router send failed: %s", exc)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="home-log-router", daemon=True
        )
        self._thread.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=join_timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                message = self.out_queue.get(timeout=self.poll_timeout)
            except queue.Empty:
                self.idle_flush()
                continue
            self.process(message)
