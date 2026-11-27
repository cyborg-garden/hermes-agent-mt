"""Refusal escalation for cheap-routed turns (``agent/routing_override.py``).

A cheap-routed turn that answers with a short refusal is retried on the primary before
the refusal is accepted as the answer. The refusal row is never appended: the retry
re-sends the same request on the primary.
"""

from __future__ import annotations

from typing import Any

from agent.routing_override import looks_like_refusal, routed_escalation_pending


def escalate_routed_refusal(agent: Any, final_response: Any) -> bool:
    """True when this was a routed refusal and the agent is now on its primary."""
    try:
        return bool(
            routed_escalation_pending(agent)
            and looks_like_refusal(agent._strip_think_blocks(final_response or ""))
            and agent._try_activate_fallback()
        )
    except Exception:  # boundary: a failed check leaves the reply as it was
        return False
