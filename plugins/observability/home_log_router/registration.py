"""register(ctx) — wire the plugin into the running agent.

Enabling the plugin (``hermes plugins enable observability/home_log_router``) is
the opt-in; this then attaches a ``HomeLogHandler`` to the root logger and starts
a ``HomeLogWorker`` that forwards throttled records to the home channel through the
``send_message`` tool. A bare platform target (e.g. ``"signal"``) resolves to
``get_home_channel()``; if no home is configured the tool returns an error which
the worker silently ignores.

The handler and worker are *process-lifetime* resources, so teardown is bound to
``atexit`` — NOT to ``on_session_end``, which fires at the end of every
conversation turn and would tear the plugin down after the first message.
Installation is idempotent so repeated registration never stacks handlers.

``HERMES_HOME_LOG_ENABLED=0`` (or any non-truthy value) is a kill switch.
Everything else defaults sensibly; no knobs needed.
"""
from __future__ import annotations

import atexit
import logging
import os
import queue
import time
from typing import NamedTuple, Tuple

from .guard import ReentrancyGuard
from .handler import HomeLogHandler
from .policy import RoutePolicy
from .throttle import Throttle
from .worker import HomeLogWorker

logger = logging.getLogger(__name__)

# Loggers whose suppressed records are worth surfacing to home, grounded in the
# real module logger names: Signal reconnect/health, model cascade fallbacks, and
# provider errors during model calls. Prefix-matched, so submodules are covered.
#
# agent.degradation is a dedicated channel carrying ONLY main-cascade
# degradation events — right now the "Fallback activated" WARNING emitted by
# chat_completion_helpers.try_activate_fallback(). Without it, a silent
# main-model degradation to a GLM/local fallback (often = Anthropic credits
# exhausted) never reaches home. We route this dedicated logger rather than
# the whole agent.chat_completion_helpers module: that module also emits stream
# TTFB/stale timeouts, partial-stream drops, and schema-sanitize warnings, and
# prefix-matching the module would spam the home channel with that lifecycle
# noise (violates the no-lifecycle-spam rule).
DEFAULT_LOGGERS: tuple[str, ...] = (
    "plugins.platforms.signal",  # Signal adapter (moved from gateway.platforms.signal upstream)
    "agent.conversation_loop",
    "agent.degradation",
    "model_tools",
    "agent.auxiliary_client",
)


class Config(NamedTuple):
    enabled: bool
    platform: str
    level: int
    loggers: tuple[str, ...]
    rate: int
    window: int
    dedup_window: int
    queue_max: int


# Matches the project's shared truthy set (utils.TRUTHY_STRINGS). Kept local so
# this thin-patch plugin stays self-contained and doesn't couple to a core file
# that churns on upstream rebases (utils.py also uses 3.11-only syntax).
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _enabled() -> bool:
    # Active by default once the plugin is enabled; any non-truthy value is a
    # kill switch. Unset (None) stays on; empty/"0"/"false"/"off" turn it off.
    val = os.getenv("HERMES_HOME_LOG_ENABLED")
    if val is None:
        return True
    return val.strip().lower() in _TRUTHY


def _level(raw: str) -> int:
    raw = (raw or "").strip()
    if raw.isdigit():  # numeric levels like "10"/"30" must be honored
        return int(raw)
    resolved = logging.getLevelName(raw.upper())
    return resolved if isinstance(resolved, int) else logging.WARNING


def _positive(key: str, default: int) -> int:
    # A non-positive rate/window/dedup silently breaks throttling (mutes
    # everything, or disables the rate cap). Fall back to the default instead.
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _read_config() -> Config:
    raw_loggers = os.getenv("HERMES_HOME_LOG_LOGGERS", "")
    loggers = tuple(s.strip() for s in raw_loggers.split(",") if s.strip()) or DEFAULT_LOGGERS
    return Config(
        enabled=_enabled(),
        platform=os.getenv("HERMES_HOME_LOG_PLATFORM", "signal").strip() or "signal",
        level=_level(os.getenv("HERMES_HOME_LOG_LEVEL", "WARNING")),
        loggers=loggers,
        rate=_positive("HERMES_HOME_LOG_RATE", 20),
        window=_positive("HERMES_HOME_LOG_WINDOW", 60),
        dedup_window=_positive("HERMES_HOME_LOG_DEDUP_WINDOW", 300),
        queue_max=_positive("HERMES_HOME_LOG_QUEUE", 1000),
    )


# Process-lifetime singletons. register() is idempotent against these.
_handler: HomeLogHandler | None = None
_worker: HomeLogWorker | None = None
_atexit_registered = False


def register(ctx) -> None:
    global _handler, _worker, _atexit_registered
    if _handler is not None:
        return  # already installed — don't stack handlers

    cfg = _read_config()
    if not cfg.enabled:
        return  # kill switch engaged

    guard = ReentrancyGuard()
    out_queue: queue.Queue[str] = queue.Queue(maxsize=cfg.queue_max)
    handler = HomeLogHandler(RoutePolicy(cfg.loggers, cfg.level), out_queue, guard)
    throttle = Throttle(cfg.rate, cfg.window, cfg.dedup_window, time.monotonic)
    worker = HomeLogWorker(out_queue, throttle, _make_sender(cfg.platform), guard)

    logging.getLogger().addHandler(handler)
    worker.start()
    _handler, _worker = handler, worker

    if not _atexit_registered:
        atexit.register(_teardown)
        _atexit_registered = True

    logger.info(
        "home_log_router active: platform=%s level=%s loggers=%s",
        cfg.platform, logging.getLevelName(cfg.level), ",".join(cfg.loggers),
    )


def _teardown() -> None:
    global _handler, _worker
    if _handler is not None:
        logging.getLogger().removeHandler(_handler)
        _handler = None
    if _worker is not None:
        _worker.stop()
        _worker = None


def _send_to_home(platform: str, message: str) -> None:
    # Invoke the send_message tool FUNCTION directly. In the gateway plugin/worker
    # context the tool is NOT in the dispatchable registry (ctx.dispatch_tool
    # returns ``{"error": "Unknown tool: send_message"}``), so we call the function.
    # A bare platform target resolves to get_home_channel(); return value (incl.
    # "no home" errors) is intentionally ignored — forwarding is best-effort.
    from tools.send_message_tool import send_message_tool

    send_message_tool({"target": platform, "message": message})


def _make_sender(platform: str):
    def sender(message: str) -> None:
        _send_to_home(platform, message)

    return sender
