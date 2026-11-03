"""Fork: per-channel memory context ids (context_id scoping).

Kept free of tool registration so gateway code can derive a context id without
importing tools.memory_tool.
"""

from __future__ import annotations

from typing import Optional

#: Chat types that denote a thread/sub-conversation hanging off a parent
#: channel. Memory for these pools into the parent, not the thread.
THREAD_CHAT_TYPES = frozenset({"thread"})


def derive_context_id(platform: Optional[str], chat_type: Optional[str],
                      chat_id: Optional[str],
                      parent_chat_id: Optional[str] = None) -> Optional[str]:
    """Build a memory context id from a messaging source, or None if unscopable.

    Scopes EVERY chat type (including DMs) and qualifies by platform so the
    same raw chat_id on two platforms never collides:

        f"{platform}:{scope}:{chat_id}"

    Two deliberate properties:

    **Threads pool into their parent channel.** A Discord thread (and a Slack
    threaded reply) reports its own ``chat_id`` — for Discord, ``chat_id`` is
    literally the ``thread_id``. Scoping on that would give every thread its
    own island of memory, so an agent whose main surface is threads could
    never accumulate anything: each thread starts blank and its learnings die
    with it. When ``parent_chat_id`` is present for a thread we scope to the
    parent, so everything said in a channel shares one memory regardless of
    which thread it happened in. A thread with no known parent falls back to
    its own id (still scoped, just not pooled).

    **``scope`` is coarse on purpose: ``dm`` or ``chat``.** It exists only to
    stop a DM and a group that happen to share a chat_id from colliding. Using
    the raw ``chat_type`` would defeat the pooling above, because the same
    Discord channel reports ``group`` for a direct post and ``thread`` for a
    threaded one — two ids for one conversation. Collapsing every non-DM type
    (group / channel / forum / thread) to ``chat`` also makes the id stable
    across adapters that disagree on which word to use.

    Returns None when ``chat_id`` is empty/None, which routes memory to the
    unscoped global store (the interactive-CLI and no-chat case). The
    ``MemoryStore`` sanitizer maps any residual path-unsafe characters to ``_``
    so the derived id always stays under contexts/.
    """
    ctype = (chat_type or "").strip().lower()
    effective_id = chat_id
    if ctype in THREAD_CHAT_TYPES and parent_chat_id:
        effective_id = parent_chat_id
    if not effective_id:
        return None
    plat = platform if platform else "unknown"
    scope = "dm" if ctype == "dm" else "chat"
    return f"{plat}:{scope}:{effective_id}"
