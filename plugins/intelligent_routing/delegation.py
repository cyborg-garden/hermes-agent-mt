"""``delegation.backend: task-routing`` — pick each subagent's model from its task.

A delegation backend (``agent/delegation_provider.py``) answers one question per
``delegate_task`` task: which route does this child take. This one runs the same
task classifier as the main turn over the child's goal and context:

* cheap-safe task (lookup, formatting, summarising, gathering, mechanical) ->
  ``routing.cheap_model`` on ``routing.cheap_provider``, labelled ``cheap · <type>``,
  with ``escalate: True`` and a fallback chain that starts with the profile's
  primary model, so an error, empty reply or refusal is retried on the primary;
* anything else -> the default route (``delegation.*``, else the parent's own
  model), exactly what the bundled ``default`` backend would do.

No LLM call is made here: the goal is written by the parent model and is explicit
enough for the heuristic, and a spawn should not wait on a router.

Enable with ``delegation.backend: task-routing``. It is independent of
``routing.intelligent``, which governs the main turn only.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from agent.delegation_provider import DelegationProvider

from .config import (
    cheap_reasoning_effort,
    cheap_tier_target,
    escalation_enabled,
    extra_protected_markers,
    primary_route,
    protected_markers,
    top_level_fallback_chain,
)
from .task_router import SCOPE_SUBAGENT, TaskSignals, classify_task

logger = logging.getLogger(__name__)

BACKEND_NAME = "task-routing"


def _default_route(config: Mapping[str, Any]) -> dict[str, Any]:
    # Same as plugins/delegation/default: the delegation section, minus a speed tier.
    return {k: v for k, v in config.items() if k != "service_tier"}


def _escalation_chain(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Primary first, then whatever chain the child would otherwise have had."""
    chain: list[dict[str, Any]] = []
    pinned = {k: config[k] for k in ("model", "provider", "base_url") if config.get(k)}
    first = pinned if pinned.get("model") else primary_route()
    if first.get("model"):
        if not first.get("provider"):
            # No provider named: let the runtime resolve it, as it does for the primary.
            first = {**first, "provider": config.get("provider") or "auto"}
        chain.append(first)
    inherited = config.get("fallback_providers")
    rest = inherited if isinstance(inherited, list) else top_level_fallback_chain()
    for entry in rest:
        if isinstance(entry, dict) and entry not in chain:
            chain.append(dict(entry))
    return chain


def _norm(text: Any) -> str:
    """Lower-case, with every whitespace run as one space (core's marker matching)."""
    return " ".join(str(text or "").lower().split())


class TaskRoutingDelegationProvider(DelegationProvider):
    @property
    def name(self) -> str:
        return BACKEND_NAME

    @property
    def display_name(self) -> str:
        return "Task routing"

    @property
    def summary(self) -> str:
        return ("Lookups, formatting, summaries and gathering run on routing.cheap_model; "
                "everything else on the normal subagent model.")

    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        goal, context = str(task.get("goal") or ""), str(task.get("context") or "")
        text = _norm(goal + "\n" + context)
        decision = classify_task(TaskSignals(
            instruction=goal, context=context, scope=SCOPE_SUBAGENT,
            is_protected=any(_norm(m) in text for m in protected_markers()),
        ))
        cheap_model, cheap_provider = cheap_tier_target()
        route: dict[str, Any]
        if decision.is_cheap and cheap_model:
            route = {
                "model": cheap_model, "provider": cheap_provider,
                "fallback_providers": _escalation_chain(config),
                "label": f"cheap · {decision.task_type}",
            }
            if escalation_enabled():
                route["escalate"] = True
            route["protected_watch"] = True
            extra = extra_protected_markers()
            if extra:
                route["protected_markers"] = extra
            effort = cheap_reasoning_effort()
            if effort:
                route["reasoning_effort"] = effort
            model, provider = cheap_model, cheap_provider
        else:
            route = _default_route(config)
            model, provider = config.get("model") or "inherit", config.get("provider") or "inherit"
        _log(decision, model, provider, task.get("index"))
        return route


def _log(decision, model, provider, index) -> None:
    from .registration import log_routing_decision

    log_routing_decision(
        tier=decision.tier, model=model, provider=provider,
        reason=f"{decision.task_type} → {decision.tier}", primary_model=None,
        task_type=decision.task_type, why=decision.reason, scope="subagent",
        confident=decision.confident, task_index=index, router="heuristic",
    )
