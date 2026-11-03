"""Inbound turn throttle + daily ceilings (R6).

Platform-agnostic rate limiting for inbound messages that would become LLM
turns. Sits at the single choke point in ``BasePlatformAdapter.handle_message``
so every platform adapter (Discord, Telegram, Signal, ...) is covered by one
gate.

Three layers, checked in order:

1. **Per-user sliding windows** (per-minute + per-hour) — one flooding user
   cannot monopolize the agent.
2. **Per-scope sliding windows** (guild / workspace / chat) — a coordinated
   raid across many users in one server still hits a ceiling.
3. **Daily ceilings** (turn count + spend USD) — a persistent ledger caps the
   worst-case daily cost of the agent regardless of who is talking. The
   ledger survives restarts.

FAIL CLOSED: if the persistent ledger cannot be read or written, the verdict
is DENY — a broken limiter must not become an open gate. Spend accounting is
also fail-closed: turns with unknown pricing consume a conservative fallback
cost instead of being free.

All limits are env-configurable and re-read at check time so container
recreates (HSM env edits) take effect without a code change. Malformed env
values fall back to the conservative defaults with one WARNING.

In-memory state (sliding windows, notify cooldowns) is process-local; the
daily ledger at ``{HERMES_HOME}/throttle/ledger.json`` is the only persistent
state (atomic 0600 writes, same pattern as ``gateway/pairing.py``).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Deque, Dict, Optional, Set

from hermes_constants import get_hermes_home
from utils import atomic_replace

logger = logging.getLogger(__name__)

# Conservative defaults. Sized so a single human operator in conversation
# with the agent never trips them (a human rarely sends >4 messages/minute
# sustained), while a flood — one hostile user or a raided guild — hits a
# wall within seconds.
_INT_DEFAULTS = {
    "GATEWAY_THROTTLE_USER_PER_MINUTE": 4,
    "GATEWAY_THROTTLE_USER_PER_HOUR": 30,
    "GATEWAY_THROTTLE_SCOPE_PER_MINUTE": 12,
    "GATEWAY_THROTTLE_SCOPE_PER_HOUR": 120,
    "GATEWAY_THROTTLE_DAILY_TURN_CEILING": 300,
}
_FLOAT_DEFAULTS = {
    "GATEWAY_THROTTLE_DAILY_SPEND_USD": 10.0,
    "GATEWAY_THROTTLE_FALLBACK_TURN_COST_USD": 0.05,
}

# Seconds between repeated "throttled" notices to the same user key. The
# denial notice must never amplify the flood it is suppressing.
NOTIFY_COOLDOWN_SECONDS = 60.0

# Seconds between repeated ledger-error log lines (the deny itself happens
# on every call — only the logging is rate limited).
_LEDGER_ERROR_LOG_INTERVAL = 60.0


@dataclass(frozen=True)
class Verdict:
    """Outcome of a throttle check."""

    allowed: bool
    reason: str = ""


_ALLOW = Verdict(True, "")


def _utc_day() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _secure_write(path: Path, data: str) -> None:
    """Atomic 0600 write — same pattern as ``gateway.pairing._secure_write``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass  # Windows doesn't support chmod the same way
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class InboundThrottle:
    """Process-wide inbound turn limiter. Use :func:`get_throttle`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._user_windows: dict[str, deque[float]] = {}
        self._scope_windows: dict[str, deque[float]] = {}
        self._notify_last: dict[str, float] = {}
        self._last_ledger_error_log = 0.0
        self._warned_env_vars: set[str] = set()
        # Last cumulative session total recorded per session key — see
        # record_session_spend. Process-local, like the sliding windows.
        self._session_spend_baseline: dict[str, float] = {}

    # ── Config (read at check time, malformed → default + one WARNING) ──

    def _env_int(self, name: str) -> int:
        default = _INT_DEFAULTS[name]
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = int(raw.strip())
            if value < 0:
                raise ValueError("negative")
            return value
        except (TypeError, ValueError):
            self._warn_env_once(name, raw, default)
            return default

    def _env_float(self, name: str) -> float:
        default = _FLOAT_DEFAULTS[name]
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = float(raw.strip())
            if value < 0:
                raise ValueError("negative")
            return value
        except (TypeError, ValueError):
            self._warn_env_once(name, raw, default)
            return default

    def _warn_env_once(self, name: str, raw: str, default) -> None:
        if name in self._warned_env_vars:
            return
        self._warned_env_vars.add(name)
        logger.warning(
            "Malformed %s=%r — using conservative default %s", name, raw, default
        )

    @staticmethod
    def _enabled() -> bool:
        return os.getenv("GATEWAY_THROTTLE_ENABLED", "true").strip().lower() not in {
            "false",
            "0",
            "no",
        }

    @staticmethod
    def _exempt_users() -> set[str]:
        raw = os.getenv("GATEWAY_THROTTLE_EXEMPT_USERS", "")
        return {part.strip() for part in raw.split(",") if part.strip()}

    # ── Persistent daily ledger (fail closed) ──

    @property
    def _ledger_path(self) -> Path:
        # Resolved per call, not cached: tests re-point HERMES_HOME per test.
        return get_hermes_home() / "throttle" / "ledger.json"

    def _read_ledger(self) -> dict:
        """Read + day-rollover the ledger. Raises on any IO/parse error."""
        path = self._ledger_path
        if not path.exists():
            # First run: no ledger yet is not an error — start a fresh day.
            return {"day": _utc_day(), "turns": 0, "spend_usd": 0.0}
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("ledger is not a JSON object")
        day = data.get("day")
        turns = data.get("turns")
        spend = data.get("spend_usd")
        if not isinstance(day, str) or not isinstance(turns, int) or not isinstance(
            spend, (int, float)
        ):
            raise ValueError("ledger schema mismatch")
        today = _utc_day()
        if day != today:
            return {"day": today, "turns": 0, "spend_usd": 0.0}
        return {"day": day, "turns": turns, "spend_usd": float(spend)}

    def _write_ledger(self, ledger: dict) -> None:
        _secure_write(self._ledger_path, json.dumps(ledger))

    def _log_ledger_error(self, err: BaseException) -> None:
        now = time.monotonic()
        if now - self._last_ledger_error_log >= _LEDGER_ERROR_LOG_INTERVAL:
            self._last_ledger_error_log = now
            logger.error(
                "Inbound throttle ledger unreadable/unwritable at %s (%s) — "
                "FAILING CLOSED: all throttled turns are denied until the "
                "ledger is repaired or removed.",
                self._ledger_path,
                err,
            )

    # ── Sliding windows ──

    @staticmethod
    def _window_counts(window: deque[float], now: float) -> tuple:
        """Prune entries older than 1h; return (count_last_minute, count_last_hour)."""
        cutoff_hour = now - 3600.0
        while window and window[0] <= cutoff_hour:
            window.popleft()
        cutoff_minute = now - 60.0
        minute_count = sum(1 for ts in window if ts > cutoff_minute)
        return minute_count, len(window)

    # ── Public API ──

    def check_and_consume(
        self, platform: str, user_id: str, scope: Optional[str]
    ) -> Verdict:
        """Gate one inbound turn. Order: enabled → user window → scope window
        → daily turn ceiling → daily spend ceiling. Fail closed on ledger
        errors. On allow, consumes window slots and one daily turn."""
        if not self._enabled():
            return _ALLOW

        user_key = f"{platform}:{user_id}"
        scope_key = f"{platform}:{scope}" if scope else None
        exempt = user_id and str(user_id) in self._exempt_users()

        with self._lock:
            now = time.monotonic()

            if not exempt:
                user_window = self._user_windows.setdefault(user_key, deque())
                u_min, u_hour = self._window_counts(user_window, now)
                if u_min >= self._env_int("GATEWAY_THROTTLE_USER_PER_MINUTE"):
                    return Verdict(False, "user_rate_per_minute")
                if u_hour >= self._env_int("GATEWAY_THROTTLE_USER_PER_HOUR"):
                    return Verdict(False, "user_rate_per_hour")

                if scope_key is not None:
                    scope_window = self._scope_windows.setdefault(scope_key, deque())
                    s_min, s_hour = self._window_counts(scope_window, now)
                    if s_min >= self._env_int("GATEWAY_THROTTLE_SCOPE_PER_MINUTE"):
                        return Verdict(False, "scope_rate_per_minute")
                    if s_hour >= self._env_int("GATEWAY_THROTTLE_SCOPE_PER_HOUR"):
                        return Verdict(False, "scope_rate_per_hour")

            # Daily ceilings bind for EVERYONE, including exempt users — they
            # protect the wallet, not the conversation.
            try:
                ledger = self._read_ledger()
            except Exception as err:
                self._log_ledger_error(err)
                return Verdict(False, "ledger_error")

            if ledger["turns"] >= self._env_int("GATEWAY_THROTTLE_DAILY_TURN_CEILING"):
                return Verdict(False, "daily_turn_ceiling")
            if ledger["spend_usd"] >= self._env_float("GATEWAY_THROTTLE_DAILY_SPEND_USD"):
                return Verdict(False, "daily_spend_ceiling")

            ledger["turns"] += 1
            try:
                self._write_ledger(ledger)
            except Exception as err:
                # If the consumed turn cannot be persisted the ceiling can be
                # bypassed by crash-looping — deny instead (fail closed).
                self._log_ledger_error(err)
                return Verdict(False, "ledger_error")

            if not exempt:
                self._user_windows[user_key].append(now)
                if scope_key is not None:
                    self._scope_windows[scope_key].append(now)

        return _ALLOW

    def record_spend(
        self, cost_usd: Optional[float], cost_status: Optional[str]
    ) -> None:
        """Add a completed turn's cost to the daily spend ledger.

        Unknown/zero cost consumes GATEWAY_THROTTLE_FALLBACK_TURN_COST_USD so
        turns with broken pricing still draw down the budget (fail-closed
        spend accounting)."""
        if not self._enabled():
            return
        if not cost_usd or cost_status in {None, "unknown"}:
            cost = self._env_float("GATEWAY_THROTTLE_FALLBACK_TURN_COST_USD")
        else:
            try:
                cost = float(cost_usd)
            except (TypeError, ValueError):
                cost = self._env_float("GATEWAY_THROTTLE_FALLBACK_TURN_COST_USD")

        with self._lock:
            try:
                ledger = self._read_ledger()
                ledger["spend_usd"] = float(ledger["spend_usd"]) + cost
                self._write_ledger(ledger)
            except Exception as err:
                # Cannot record spend — check_and_consume will fail closed on
                # the same error, so this does not open the gate.
                self._log_ledger_error(err)

    def record_session_spend(
        self, session_key: str, session_total_usd, cost_status: Optional[str]
    ) -> None:
        """Record one turn's spend given the agent's CUMULATIVE session total.

        ``agent_result["estimated_cost_usd"]`` (agent/turn_finalizer.py) is a
        running total for the whole session — agent_init zeroes it once and
        conversation_loop does ``+=`` per API call. Charging it directly every
        turn sums running totals (quadratic overcount) and trips the daily
        spend ceiling on ordinary single-operator use. This converts the
        total to a per-turn delta against the last total seen for
        ``session_key``; a rebuilt/reset agent restarts its total from zero,
        in which case the fresh total is charged whole. Non-numeric totals
        fall through to :meth:`record_spend`'s fallback pricing."""
        if isinstance(session_total_usd, bool) or not isinstance(
            session_total_usd, (int, float)
        ):
            self.record_spend(None, cost_status)
            return
        total = float(session_total_usd)
        with self._lock:
            prev = self._session_spend_baseline.get(session_key, 0.0)
            delta = total - prev if total >= prev else total
            self._session_spend_baseline[session_key] = total
        self.record_spend(delta, cost_status)

    def should_notify(self, user_key: str) -> bool:
        """True at most once per NOTIFY_COOLDOWN_SECONDS per user key —
        denial notices must not amplify the flood they suppress."""
        with self._lock:
            now = time.monotonic()
            last = self._notify_last.get(user_key)
            if last is not None and now - last < NOTIFY_COOLDOWN_SECONDS:
                return False
            self._notify_last[user_key] = now
            return True

    def reset_state_for_tests(self) -> None:
        """Clear in-memory windows/cooldowns (test helper — ledger untouched)."""
        with self._lock:
            self._user_windows.clear()
            self._scope_windows.clear()
            self._notify_last.clear()
            self._session_spend_baseline.clear()
            self._warned_env_vars.clear()
            self._last_ledger_error_log = 0.0


_singleton: Optional[InboundThrottle] = None
_singleton_lock = threading.Lock()


def get_throttle() -> InboundThrottle:
    """Process-wide throttle singleton."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = InboundThrottle()
    return _singleton
