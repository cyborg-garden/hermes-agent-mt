"""Protected-content gate, run first in the pre-API guard chain.

Protected text (a Papa Reo transcript arriving in a tool result, a steer, plugin
context, recalled memory or the system prompt) may only reach the configured primary.
Right before the call, after the steer is appended and the request is assembled, this
moves a cheap-routed turn or child (``agent/routing_override.py``), or an agent left on
a fallback or a model directive, to its primary and restarts the iteration so the
request is rebuilt for it, or ends the turn without any model call when the primary
cannot be reached.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any, Optional

from agent.message_metadata import append_message
from agent.routing_override import (
    PROTECTED_BLOCK_MESSAGE, PROTECTED_BLOCKED, PROTECTED_SWITCHED, agent_extra_text,
    escalate_if_protected,
    restore_primary_if_protected,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PROTECTED_BLOCKED",
    "PROTECTED_BLOCK_MESSAGE",
    "PROTECTED_SWITCHED",
    "protected_content_gate",
    "protected_outcome",
]


def _refund_iteration(agent: Any, v: Any) -> None:
    v.api_call_count -= 1
    agent._api_call_count = v.api_call_count
    with suppress(Exception):
        agent.iteration_budget.refund()


def protected_outcome(agent: Any, messages: Any, plugin_context: Any = None) -> Optional[str]:
    """``PROTECTED_SWITCHED``, ``PROTECTED_BLOCKED`` or None (nothing to do). Never raises."""
    # This turn's plugin context and memory recall travel with the messages.
    extras = (plugin_context, *tuple(getattr(agent, "_turn_protected_extras", None) or ()))
    if getattr(agent, "_routing_protected_watch", False) is not True:
        # Not a cheap route, but maybe still on a fallback (one that outlived its
        # turn, or a model directive): protected text goes back to the primary.
        return restore_primary_if_protected(agent, messages, extras)
    return escalate_if_protected(agent, messages, agent_extra_text(agent, extras))


def protected_content_gate(agent: Any, v: Any, messages: Any, plugin_context: Any) -> Optional[Any]:
    """``v`` (the preflight verdict) set to ``continue`` or ``break``, or None to go on."""
    outcome = protected_outcome(agent, messages, plugin_context)
    if outcome is None:
        return None
    _refund_iteration(agent, v)
    if outcome == PROTECTED_SWITCHED:
        from agent.conversation_loop import _sync_failover_system_message
        v.active_system_prompt = _sync_failover_system_message(agent, None, v.active_system_prompt)
        v.action = "continue"  # rebuild the request for the primary
        return v
    v.final_response, v.failed = PROTECTED_BLOCK_MESSAGE, True
    v._turn_exit_reason = "protected_content_blocked"
    append_message(messages, {"role": "assistant", "content": v.final_response})
    v.action = "break"
    return v
