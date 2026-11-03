"""HomeLogHandler — a logging.Handler that enqueues forwardable records.

emit() is deliberately cheap and non-blocking: it gates on the policy and the
re-entrancy guard, formats the record, and does a single ``put_nowait``,
dropping silently if the queue is full. All network I/O happens later, on the
worker thread — so logging is never blocked by a slow send.
"""
from __future__ import annotations

import logging
import queue

from .guard import ReentrancyGuard
from .policy import RoutePolicy

_DEFAULT_FORMAT = "%(levelname)s %(name)s: %(message)s"


class HomeLogHandler(logging.Handler):
    def __init__(
        self,
        policy: RoutePolicy,
        out_queue: queue.Queue[str],
        guard: ReentrancyGuard,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.out_queue = out_queue
        self.guard = guard
        self.setFormatter(logging.Formatter(_DEFAULT_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        if self.guard.active:
            return
        if not self.policy.should_forward(record):
            return
        try:
            self.out_queue.put_nowait(self.format(record))
        except queue.Full:
            pass  # storm: drop rather than block the logging thread
        except Exception:  # pragma: no cover - never let logging raise
            self.handleError(record)
