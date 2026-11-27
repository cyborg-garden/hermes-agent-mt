"""Protected-content gate for cheap-routed turns, run first in the pre-API guard chain.

A cheap-routed turn or child (``agent/routing_override.py``) must never send protected
text (a Papa Reo transcript arriving in a tool result, a steer or plugin context) to the
cheap model. Right before the call, after the steer is appended and the request is
assembled, this moves the agent to its primary and restarts the iteration so the
request is rebuilt for it, or ends the turn without any model call when the primary
cannot be reached.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any, Optional

from agent.message_metadata import append_message
from agent.routing_override import (
    PROTECTED_BLOCK_MESSAGE, PROTECTED_SWITCHED, escalate_if_protected,
)

logger = logging.getLogger(__name__)


def _refund_iteration(agent: Any, v: Any) -> None:
    v.api_call_count -= 1
    agent._api_call_count = v.api_call_count
    with suppress(Exception):
        agent.iteration_budget.refund()


def protected_content_gate(agent: Any, v: Any, messages: Any, plugin_context: Any) -> Optional[Any]:
    """``v`` (the preflight verdict) set to ``continue`` or ``break``, or None to go on."""
    outcome = escalate_if_protected(agent, messages, plugin_context)
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
