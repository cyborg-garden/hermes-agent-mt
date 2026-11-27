"""Refusal escalation for cheap-routed turns (``agent/routing_override.py``).

A cheap-routed turn that answers with a short refusal is retried on the primary before
the refusal is accepted as the answer. The refusal row is never appended: the retry
re-sends the same request on the primary.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

from agent.routing_override import looks_like_refusal, routed_escalation_pending


def escalate_routed_refusal(agent: Any, final_response: Any, verdict: Callable[[str], Any],
                            api_messages: Any) -> Optional[Any]:
    """A ``continue`` verdict on the primary, or None when this is not a routed refusal."""
    if not (
        routed_escalation_pending(agent)
        and looks_like_refusal(agent._strip_think_blocks(final_response or ""))
        and agent._try_activate_fallback()
    ):
        return None
    from agent.conversation_loop import _sync_failover_system_message
    from agent.turn_context_compaction import _refund_api_call

    base = verdict("continue")
    return dataclasses.replace(
        base, final_response=None, _preflight_compression_blocked=False,
        active_system_prompt=_sync_failover_system_message(agent, api_messages, base.active_system_prompt),
        api_call_count=_refund_api_call(agent, base.api_call_count),
    )
