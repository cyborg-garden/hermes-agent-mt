"""disappearing_messages — delete what the agent keeps from a chat on that chat's timer.

When a Signal chat has disappearing messages on, everything this agent stores
from it (messages and replies in state.db, attachments, session snapshots,
request dumps) is deleted on the same timer. Log lines from those chats keep
their routing details but lose the message preview.

Opt-in per agent: add ``disappearing_messages`` to ``plugins.enabled``.

Hooks:
  * ``pre_gateway_dispatch`` — read the timer the Signal adapter put in
    ``event.metadata["signal"]`` and record the chat and its attachments.
    Sender timestamps are never used; everything runs on this host's clock.
  * ``pre_tool_call`` — refuse long-term ``memory`` writes in a disappearing
    chat; memory is shared across chats and cannot expire with one of them.
A background thread sweeps every ``DISAPPEARING_SWEEP_SECONDS`` (default 30).

Limits, stated plainly: this deletes our copies on this host. It cannot reach
other people's devices, Signal itself, model providers, external memory
providers, files a tool copied elsewhere, or host backups. See README.md.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import weakref
from pathlib import Path
from typing import Any, Dict, Optional, Set

from .retention import Retention

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = {"signal"}
MEMORY_TOOLS = {"memory"}
MEMORY_READ_ACTIONS = {"read", "list", "view", "search", "get"}

_retention: Optional[Retention] = None
_init_lock = threading.Lock()
_stop = threading.Event()
_filter: Optional["RedactFilter"] = None
_gateway_ref: Optional["weakref.ReferenceType"] = None
SWEEPER_THREAD_NAME = "disappearing-sweeper"


def _home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


def _get() -> Retention:
    global _retention
    with _init_lock:
        if _retention is None:
            home = _home()
            _retention = Retention(home=home, state_db=home / "state.db")
        return _retention


# --------------------------------------------------------------------- logging

_PREVIEW_RE = re.compile(r"\b(msg|reply_to_text)=('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")")
_CHAT_RE = re.compile(r"\bchat=([^\s,;)\]]+)")
_SESSION_RE = re.compile(r"\bsession=([^\s,;)\]]+)")


class RedactFilter(logging.Filter):
    """Blank message previews in log lines about a disappearing chat."""

    def __init__(self) -> None:
        super().__init__("disappearing_messages")
        self.chats: Set[str] = set()

    def _session_is_disappearing(self, sid: str) -> bool:
        # Never touch state.db from inside logging: use the map sweeps keep.
        r = _retention
        owner = r.session_owner.get(sid) if r is not None else None
        return bool(owner and owner[1] in self.chats)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.chats:
            return True
        try:
            text = record.getMessage()
        except Exception:
            return True
        if "msg=" not in text and "reply_to_text=" not in text:
            return True
        m = _CHAT_RE.search(text)
        hit = bool(m and m.group(1) in self.chats)
        if not hit:
            s = _SESSION_RE.search(text)
            hit = bool(s and self._session_is_disappearing(s.group(1)))
        if hit:
            record.msg = _PREVIEW_RE.sub(lambda mm: f"{mm.group(1)}='<disappearing>'", text)
            record.args = None
        return True


def _install_filter() -> None:
    """Attach to every handler so records from child loggers are covered."""
    global _filter
    if _filter is None:
        _filter = RedactFilter()
    seen = set()
    for lg in [logging.getLogger()] + [
        logging.getLogger(n) for n in list(logging.root.manager.loggerDict)
    ]:
        for h in getattr(lg, "handlers", []):
            if id(h) in seen:
                continue
            seen.add(id(h))
            if _filter not in h.filters:
                h.addFilter(_filter)


def _refresh_filter_chats() -> None:
    if _filter is None:
        return
    r = _get()
    _filter.chats = {cid for plat, cid in r.chats() if r.is_disappearing(plat, cid)}


# --------------------------------------------------------------------- hooks

def _platform_name(source: Any) -> str:
    plat = getattr(source, "platform", None)
    return str(getattr(plat, "value", plat) or "").lower()


def _on_pre_gateway_dispatch(event: Any = None, gateway: Any = None, session_store: Any = None, **_: Any):
    try:
        source = getattr(event, "source", None)
        platform = _platform_name(source)
        if platform not in SUPPORTED_PLATFORMS:
            return None
        chat_id = getattr(source, "chat_id", None)
        if not chat_id:
            return None
        meta = (getattr(event, "metadata", None) or {}).get(platform) or {}
        if "expires_in_seconds" not in meta:
            return None  # adapter predates the timer field
        global _gateway_ref
        if gateway is not None:
            try:
                _gateway_ref = weakref.ref(gateway)
            except TypeError:
                pass
        if session_store is not None and getattr(session_store, "sessions_dir", None):
            r = _get()
            d = Path(session_store.sessions_dir)
            if d not in r.dump_dirs:
                r.dump_dirs.append(d)
        # Sender timestamps are deliberately not used: they can be wrong or forged.
        ttl = _get().observe(
            platform,
            chat_id,
            meta.get("expires_in_seconds"),
            list(getattr(event, "media_urls", None) or []),
        )
        if _filter is not None:
            if ttl and ttl > 0:
                _filter.chats.add(chat_id)
            else:
                _filter.chats.discard(chat_id)
        _install_filter()
    except Exception:
        # Never block a message; the sweep is the safety net.
        logger.exception("disappearing: could not record a message")
    return None


def _on_pre_tool_call(tool_name: str = "", args: Optional[Dict[str, Any]] = None,
                      session_id: str = "", **_: Any):
    if tool_name not in MEMORY_TOOLS:
        return None
    action = str((args or {}).get("action") or "").lower()
    if action in MEMORY_READ_ACTIONS:
        return None
    try:
        owner = _get().session_chat(session_id)
    except Exception:
        owner = None
    if owner and owner[0] in SUPPORTED_PLATFORMS and _get().is_disappearing(*owner):
        return {
            "action": "block",
            "message": (
                "This chat has disappearing messages on, so nothing from it is saved to "
                "long-term memory. Carry on without saving it."
            ),
        }
    return None


# --------------------------------------------------------------------- sweeper

def _sweep_interval() -> float:
    try:
        return max(5.0, float(os.environ.get("DISAPPEARING_SWEEP_SECONDS", "30")))
    except ValueError:
        return 30.0


def _evict(session_keys) -> None:
    """Drop cached agents so nothing deleted lingers in memory (e.g. a summary)."""
    gw = _gateway_ref() if _gateway_ref is not None else None
    evict = getattr(gw, "_evict_cached_agent", None)
    if evict is None:
        return
    for key in session_keys:
        try:
            evict(key)
        except Exception:
            logger.debug("disappearing: could not evict a cached agent", exc_info=True)


def _sweep_once() -> None:
    result = _get().sweep()
    if result.touched_session_keys:
        _evict(result.touched_session_keys)
    _refresh_filter_chats()
    _install_filter()


def _sweep_loop() -> None:
    interval = _sweep_interval()
    while True:
        # First pass straight away: anything that fell due while we were down.
        try:
            _sweep_once()
        except Exception:
            logger.exception("disappearing: sweep failed; will retry")
        if _stop.wait(interval):
            return


def _start_sweeper() -> None:
    """One sweeper per process, even if the plugin module is loaded again."""
    for t in threading.enumerate():
        if t.name == SWEEPER_THREAD_NAME and t.is_alive():
            return
    threading.Thread(target=_sweep_loop, name=SWEEPER_THREAD_NAME, daemon=True).start()


def register(ctx) -> None:
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    _install_filter()
    _refresh_filter_chats()
    _start_sweeper()
