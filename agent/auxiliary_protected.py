"""Protected sessions in auxiliary calls (agent/protected_session.py).

A request in a protected session is pinned to the running session's configured
primary, whatever its task is configured to use, and never falls back.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Optional

logger = logging.getLogger("agent.auxiliary_client")


# Protected sessions (agent/protected_session.py). These two mark the running context
# protected without an agent (callers and tests that have none).
def set_protected_session(protected: bool) -> contextvars.Token:
    from agent import protected_session

    return protected_session._SESSION.set(
        protected_session.EgressSession(None, None, protected=bool(protected)))


def reset_protected_session(token: contextvars.Token) -> None:
    from agent import protected_session

    protected_session.reset(token)


def _protected_text_pin(messages: Any, main_runtime: dict[str, Any], task: Any,
                        explicit_runtime: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    """None outside a protected session; otherwise the runtime the call is pinned to.

    A request holding a marker marks the running session (the scan only sets the
    flag). The pin is the running session's configured primary, never the live
    "main" runtime, which a fallback can have moved; with no running session, a
    runtime the caller passed explicitly. Raises (sending nothing) when there is
    neither."""
    try:
        from agent import protected_session

        protected = protected_session.scan_running(messages, source="auxiliary")
        primary = protected_session.running_primary()
    except Exception:  # health: allow BLE001 -- fail closed: an error counts as protected (agent/protected_session.py)
        protected, primary = True, None  # fail closed
    if not protected:
        return None
    pin = primary or explicit_runtime
    if not (pin or {}).get("model") or not (pin or {}).get("provider"):
        raise protected_unavailable(task)
    logger.info("Auxiliary %s: protected session; pinned to the configured primary, no fallback",
                task or "call")
    return pin


def protected_unavailable(task: Any) -> RuntimeError:
    return RuntimeError(
        f"Auxiliary {task or 'call'}: the session holds protected text and its configured "
        "primary model is unavailable; nothing was sent to another model.")


def pinned_route(messages: Any, task: Any, explicit_runtime: Optional[dict[str, Any]], provider: Any,
                 model: Any, base_url: Any, api_key: Any, api_mode: Any) -> tuple[bool, tuple]:
    """``(protected, (provider, model, base_url, api_key, api_mode))``: the call's route, or
    the configured primary's when the session is protected (see :func:`_protected_text_pin`)."""
    pin = _protected_text_pin(messages, explicit_runtime or {}, task, explicit_runtime)
    if pin is None:
        return False, (provider, model, base_url, api_key, api_mode)
    return True, (pin["provider"], pin["model"], pin.get("base_url") or None,
                  pin.get("api_key") or None, pin.get("api_mode") or api_mode)
