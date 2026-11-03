"""Model self-awareness: surface the live (post-fallback) model each turn.

The runtime always *knows* which model is live — ``try_activate_fallback``
mutates ``agent.model`` / ``agent.provider`` in place when it swaps backends —
but the LLM never sees it, so "what model are you?" gets answered by
hallucination. The pre-migration ``hermes-swarm`` repo fixed this by appending
a ``[Current model: X via Y]`` line to the ephemeral per-turn context; that was
lost in the ``hermes-swarm`` → ``hermes-agent-mt`` migration and is re-ported
here.

Sourcing (the load-bearing detail): the line reads ``agent.model`` /
``agent.provider`` — the live runtime fields — at per-API-call assembly time,
NOT the ``_RUNTIME_MAIN_*`` process globals (which are published once at turn
start, before ``restore_primary_runtime``, and are *not* updated by
``try_activate_fallback``, so they go stale on a degraded turn). Reading the
agent fields means a fallback turn shows the fallback backend, not the
configured primary.

Placement: it is appended to the current turn's user message and stamped into
that message's ``api_content`` sidecar at turn start, exactly like the
memory-prefetch / ``pre_llm_call`` injections, so what turn N sends is what
turn N+1 replays and the prompt-cache prefix stays byte-stable. If a fallback
swaps the model mid-turn, the wire assembly rewrites the line (and the live
sidecar) to the new model; that turn's cache is already lost to the model
change. It never touches the cached system prompt.

Set ``HERMES_MODEL_AWARENESS=0`` to disable (the test suite does, so
upstream's byte-exact wire tests are unaffected).

Contents are two non-secret slugs (model + provider); no credentials, no PII.
"""

from __future__ import annotations

import os
from typing import Any, List

_LINE_PREFIX = "[Current model:"
_SEPARATOR = "\n\n"


def model_awareness_enabled() -> bool:
    raw = os.environ.get("HERMES_MODEL_AWARENESS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def format_current_model_line(model: str, provider: str) -> str:
    """Return ``[Current model: <model> via <provider>]``.

    Falls back to ``[Current model: <model>]`` when the provider is unknown,
    and to ``""`` when the model itself is unknown (inject nothing rather than
    a half-formed line).
    """
    model = (model or "").strip()
    provider = (provider or "").strip()
    if not model:
        return ""
    if provider:
        return f"[Current model: {model} via {provider}]"
    return f"[Current model: {model}]"


def with_current_model_line(content: Any, agent: Any) -> Any:
    """Return ``content`` ending in exactly one current-model line.

    Idempotent: a trailing line from an earlier stamp (possibly naming a
    different model, after a fallback) is replaced, never duplicated.
    Non-string content, an unknown model, or the feature being disabled
    returns ``content`` unchanged.
    """
    if not isinstance(content, str) or not model_awareness_enabled():
        return content
    line = format_current_model_line(
        getattr(agent, "model", "") or "",
        getattr(agent, "provider", "") or "",
    )
    if not line:
        return content
    idx = content.rfind(_SEPARATOR + _LINE_PREFIX)
    if idx != -1 and content.endswith("]") and "\n" not in content[idx + len(_SEPARATOR):]:
        content = content[:idx]
    return content + _SEPARATOR + line
