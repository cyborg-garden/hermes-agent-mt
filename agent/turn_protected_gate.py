"""Protected-content gate for cheap-routed turns, run at the top of each loop iteration.

A cheap-routed turn or child (``agent/routing_override.py``) must never send protected
text (a Papa Reo transcript arriving in a tool result, a steer, plugin context or
recalled memory) to the cheap model. The turn loop calls this after the steer drain and
before it assembles the request, runs MoA advisors, compresses or calls the model. It
moves the agent to its primary, or reports that the turn must end without any model
call when the primary cannot be reached.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any, Optional

from agent.routing_override import (
    PROTECTED_BLOCK_MESSAGE, PROTECTED_BLOCKED, PROTECTED_SWITCHED, escalate_if_protected,
)

logger = logging.getLogger(__name__)

__all__ = ["PROTECTED_BLOCK_MESSAGE", "PROTECTED_BLOCKED", "PROTECTED_SWITCHED",
           "protected_content_gate", "refund_iteration"]


def _extra_text(agent: Any, extras: tuple) -> list:
    texts = [e for e in extras if e]
    with suppress(Exception):
        if getattr(agent, "ephemeral_system_prompt", None):
            texts.append(agent.ephemeral_system_prompt)
    with suppress(Exception):
        for pfm in getattr(agent, "prefill_messages", None) or []:
            if isinstance(pfm, dict):
                texts.append(pfm.get("content"))
    return texts


def protected_content_gate(agent: Any, messages: Any, *extras: Any) -> Optional[str]:
    """``PROTECTED_SWITCHED``, ``PROTECTED_BLOCKED`` or None (nothing to do).

    ``extras`` is any other text about to be sent with ``messages`` (plugin context,
    memory recall). Inert unless the turn runs on a cheap route. Never raises.
    """
    if getattr(agent, "_routing_protected_watch", False) is not True:
        return None
    return escalate_if_protected(agent, messages, _extra_text(agent, extras))


def refund_iteration(agent: Any, api_call_count: int) -> int:
    """Give back the call count and iteration budget of a pass that never reached a model."""
    api_call_count -= 1
    agent._api_call_count = api_call_count
    with suppress(Exception):
        agent.iteration_budget.refund()
    return api_call_count
