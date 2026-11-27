"""Per-agent iteration budget — thread-safe consume/refund counter.

Extracted from ``run_agent.py``.  Each ``AIAgent`` instance (parent or
subagent) holds an :class:`IterationBudget`; the parent's cap comes from
``max_iterations`` (default 90), each subagent's cap comes from
``delegation.max_iterations`` (default 50).

``run_agent`` re-exports ``IterationBudget`` so existing
``from run_agent import IterationBudget`` imports keep working unchanged.
"""

from __future__ import annotations

import math
import threading

#: Default share of the budget at which the model is told to wrap up.
DEFAULT_STEP_LIMIT_WARNING_AT = 0.8


class IterationBudget:
    """Thread-safe iteration counter for an agent.

    Each agent (parent or subagent) gets its own ``IterationBudget``.
    The parent's budget is capped at ``max_iterations`` (default 90).
    Each subagent gets an independent budget capped at
    ``delegation.max_iterations`` (default 50) — this means total
    iterations across parent + subagents can exceed the parent's cap.
    Users control the per-subagent limit via ``delegation.max_iterations``
    in config.yaml.

    ``execute_code`` (programmatic tool calling) iterations are refunded via
    :meth:`refund` so they don't eat into the budget.
    """

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Try to consume one iteration.  Returns True if allowed."""
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """Give back one iteration (e.g. for execute_code turns)."""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)


def step_limit_warning_threshold(max_total: int, fraction: float) -> int | None:
    """Return the used-step count at which the wrap-up note fires.

    ``None`` means "never": the budget is too small for a warning to leave
    the model any room, or ``fraction`` is outside ``(0, 1)``.  The
    threshold is always below ``max_total`` so at least one step remains
    after the note is seen.
    """
    try:
        fraction = float(fraction)
    except (TypeError, ValueError):
        return None
    if not (0.0 < fraction < 1.0) or max_total < 2:
        return None
    return min(max_total - 1, max(1, math.ceil(max_total * fraction)))


def should_warn_step_limit(used: int, max_total: int, fraction: float) -> bool:
    """True once ``used`` has reached the wrap-up threshold."""
    threshold = step_limit_warning_threshold(max_total, fraction)
    return threshold is not None and used >= threshold


def step_limit_warning_text(used: int, max_total: int) -> str:
    """The note appended for the model when the wrap-up threshold is hit."""
    return (
        f"[System note: You have used {used} of {max_total} steps for this turn. "
        "Finish the current piece, then reply to the user with what you did "
        "and what's left. Do not start new work.]"
    )


__all__ = [
    "DEFAULT_STEP_LIMIT_WARNING_AT",
    "IterationBudget",
    "should_warn_step_limit",
    "step_limit_warning_text",
    "step_limit_warning_threshold",
]
