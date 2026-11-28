"""Protected sessions on the gateway (agent/protected_session.py): which chats hold
protected text, and keeping their replies from making the gateway fetch a URL the model wrote."""

from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger("gateway.run")


async def session_is_protected(runner: Any, session_id: Any) -> bool:
    """Whether a session holds protected text: the in-process registry, else its record,
    read off the loop (AsyncSessionDB offloads the read; a plain stub answers directly).
    Fails closed."""
    if not isinstance(session_id, str) or not session_id:
        return False
    from agent import protected_session as _ps
    if _ps.session_id_is_protected(session_id):
        return True
    db = getattr(runner, "_session_db", None)
    if db is None or not hasattr(db, "session_is_protected"):
        return False
    try:
        answer = db.session_is_protected(session_id)
        if inspect.isawaitable(answer):
            answer = await answer
        return answer is True
    except Exception:  # health: allow BLE001 -- fail closed: an unreadable record counts as protected
        logger.debug("protected-flag read failed; treating the session as protected", exc_info=True)
        return True


async def defang_protected_reply(runner: Any, result: Any, session_id: Any) -> Any:
    """A protected session's reply never makes the gateway fetch an image URL the model
    wrote: image tags become plain alt text."""
    if not (isinstance(result, dict) and isinstance(result.get("final_response"), str)):
        return result
    from agent import protected_session as _ps
    for sid in {session_id, result.get("session_id")}:
        if await session_is_protected(runner, sid):
            result["final_response"] = _ps.defang_images(result["final_response"])
            break
    return result


async def chat_is_protected(runner: Any, source: Any) -> bool:
    """Whether the chat ``source`` belongs to has a protected session (no session yet: no)."""
    store = getattr(runner, "session_store", None)
    if store is None:
        return False
    try:
        sid = store.peek_session_id(runner._session_key_for_source(source))
    except Exception:  # health: allow BLE001 -- a chat with no session entry yet has nothing protected
        logger.debug("protected-chat lookup failed", exc_info=True)
        return False
    return await session_is_protected(runner, sid if isinstance(sid, str) else None)
