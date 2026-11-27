"""Pre-turn routing-override interface extension.

This is the minimal reference implementation of the plugin-interface extension
teknium1 offered on upstream PR #43534 ("Happy to support adding to the plugin
interface to make this work"). It lets a ``pre_llm_call`` plugin hook RETURN a
routing decision that the runtime actually ACTUATES for the turn — closing the
gap that the existing hook can only inject context, never change which model
runs (the model is frozen by ``restore_primary_runtime`` before the hook fires).

── Contract ──
A ``pre_llm_call`` hook result may be a dict carrying a ``route`` key::

    {"context": "...optional injected text...",
     "route": {"model": "deepseek/deepseek-v3.2", "provider": "openrouter"}}

``model`` is required; ``provider`` (and optional ``base_url`` / ``api_key`` /
``api_mode``) are passed through to ``switch_model``. An optional ``extra_body``
dict is NOT passed to ``switch_model`` (its signature has no such parameter) —
it is merged into ``agent.request_overrides["extra_body"]`` for the turn and
reverted with the rest of the override. That is what lets a routed turn carry
provider-specific BODY params for the cheap tier (e.g. ollama
``{"options": {"num_ctx": N}}``, or an aggregator's ``provider`` preferences).

⚠️ The OpenAI SDK merges ``extra_body`` into the JSON body LAST, so a key here
silently OVERRIDES a same-named TYPED parameter the transport already emitted —
measured, not assumed (see
``tests/plugins/intelligent_routing/test_cheap_extra_body_wire_semantics.py``).
Do NOT use it to disable thinking on a local tier: ``think: false`` is ignored
by ollama's OpenAI-compat ``/v1`` endpoint (ollama#14820), and the switch that
does work (``reasoning_effort``) is already resolved PER-MODEL by
``switch_model`` from ``agent.reasoning_overrides``.

An optional ``escalate: True`` arms escalation for the turn: if the routed model
errors, returns an empty reply or refuses, :func:`escalate_routed_turn` moves the
turn back to the route it started on (the configured primary) before any
fallback-chain entry is tried. See "Escalation" below.

``turn_context`` extracts the first well-formed override and applies it right
after the hook fires, before the turn's first API call assembles.

── Turn-scoping (the load-bearing detail) ──
Proactive routing is a PER-TURN decision, exactly like reactive fallback — it
must NOT pin the session to the cheap model. We reuse the agent's existing,
tested ``switch_model`` (full atomic client rebuild + rollback), which persists
the swap into ``_primary_runtime``. Crucially we then scope it with a DEDICATED
flag, NOT by overloading reactive-fallback state:

  1. Stash the pre-swap premium ``_primary_runtime`` in
     ``_routing_override_saved_primary``.
  2. Set ``_routing_override_active = True``.
  3. LEAVE ``_primary_runtime`` pointing at the CHEAP runtime for the turn, and
     do NOT touch ``_fallback_activated``.

``restore_primary_runtime`` then reverts the swap at the top of the NEXT turn via
a dedicated block that runs BEFORE the ``_fallback_activated`` and rate-limit
cooldown gates, so a routed turn always reverts regardless of cooldown state.

Why the dedicated flag (three audit-confirmed bugs the old overload caused):
  - Overloading ``_rate_limited_until``/``_fallback_activated`` for scoping let a
    cheap turn LEAK past its turn when a cooldown gate skipped restoration
    (Blocker 1); pointing ``_primary_runtime`` at premium made in-turn transient
    recovery jump the routed turn onto premium (Major 2); and pre-arming
    ``_fallback_activated`` corrupted the reactive cooldown accounting so a cheap
    429 got no cooldown (Major 3). Keeping ``_primary_runtime`` == cheap and using
    a dedicated flag fixes all three: reactive fallback runs normally UNDERNEATH a
    routed turn with the cheap runtime as its "primary".

── Escalation ──
A cheap turn that fails must not cost the user an answer. ``try_activate_fallback``
calls :func:`escalate_routed_turn` first; the empty-response ladder and the final
text check (refusals) call it directly. It fires at most once per turn, switches
back via ``switch_model`` to the pre-turn route, and logs one
``routing_escalation {json}`` line (reason only, never text). The revert at the
top of the next turn clears it with the rest of the override state.

Delegated children routed cheap by the task-routing delegation backend carry
``_routing_escalate_pending`` with no target: their fallback chain already starts
with the parent's primary, so escalation there is the chain's first hop.

── Fail-safe ──
Any error (bad override, ``switch_model`` raising on a bad key/network) leaves
the agent untouched and returns ``False`` — a routing miss must degrade to
"answer on the configured primary", never break the turn.

⚠️ AUDIT NOTE: this module + its single call site in ``turn_context.py`` are the
ONLY core-runtime changes for intelligent routing. They touch the
model-selection path (user-visible: they change which model answers a turn).
Everything else lives in the ``intelligent_routing`` plugin. Reviewers should
focus here.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Fields we forward from an override into switch_model (model/provider plus the
# optional endpoint/credential fields switch_model already accepts).
_PASSTHROUGH_FIELDS = ("base_url", "api_key", "api_mode")

# Distinguishes "there was no extra_body before" from "it was explicitly {}",
# so restore removes the key entirely rather than leaving an empty dict behind.
_NOT_SET = object()

# THIRD state, and the one a single sentinel got wrong: "this routed turn never
# wrote extra_body at all". routing.cheap_extra_body is empty by default, so
# that is the COMMON case — and conflating it with _NOT_SET made the revert pop
# an extra_body the override never set. request_overrides["extra_body"] is built
# once at agent construction for custom providers and for fast/priority mode, so
# that pop silently destroyed the primary provider's own body params for the
# rest of the session. Revert must be a no-op on this value.
_UNTOUCHED = object()


def extract_routing_override(results: Iterable[Any]) -> Optional[dict]:
    """Return the first well-formed ``route`` override from hook results, or None.

    A well-formed override is a dict with a non-empty ``model``. First match
    wins (deterministic; matches how the ``pre_llm_call`` context parts are
    consumed in registration order).
    """
    if not results:
        return None
    for r in results:
        if not isinstance(r, dict):
            continue
        route = r.get("route")
        if isinstance(route, dict) and str(route.get("model") or "").strip():
            return route
    return None


def apply_routing_override(agent: Any, override: Optional[dict]) -> bool:
    """Actuate a routing override for THIS turn. Return True iff a swap happened.

    Turn-scoped and fail-safe (see module docstring). No-op (returns False) when
    the override is empty or already matches the live model/provider.
    """
    if not override or not isinstance(override, dict):
        return False

    target_model = str(override.get("model") or "").strip()
    if not target_model:
        return False
    target_provider = str(override.get("provider") or "").strip()

    cur_model = str(getattr(agent, "model", "") or "").strip()
    cur_provider = str(getattr(agent, "provider", "") or "").strip()
    # Avoid churn: nothing to do if we're already on the target.
    if target_model == cur_model and (
        not target_provider or target_provider == cur_provider
    ):
        return False

    # Stash the configured-primary runtime so restore_primary_runtime can revert
    # the swap next turn. switch_model will overwrite _primary_runtime with the
    # cheap runtime — we deliberately KEEP that (so in-turn recovery paths that
    # read _primary_runtime stay on cheap) and use the stash + a dedicated flag
    # for the revert, NOT the reactive-fallback state.
    primary_snapshot = getattr(agent, "_primary_runtime", None)
    # The route the agent is on right now: what escalation returns to if the routed
    # turn fails. Captured before switch_model replaces it.
    escalation_target = {
        "model": cur_model, "provider": cur_provider,
        "base_url": str(getattr(agent, "base_url", "") or ""),
        "api_key": getattr(agent, "api_key", "") or "",
        "api_mode": str(getattr(agent, "api_mode", "") or ""),
    }

    kwargs = {
        k: override[k] for k in _PASSTHROUGH_FIELDS if override.get(k)
    }
    # When the route switches provider but the override carries no explicit
    # base_url, resolve the new provider's canonical endpoint up front. Without
    # this, switch_model()'s ``if base_url:`` guard keeps the CURRENT base_url,
    # so the routed model is dispatched to the previous provider's endpoint —
    # e.g. an OpenRouter model sent to api.anthropic.com/chat/completions → 404,
    # silently cascading to the bottom of the fallback chain. Mirrors the
    # reactive-fallback path in chat_completion_helpers. (cyborg routed-turn
    # 404 regression.) Fail-safe: a resolve miss must never break the turn —
    # fall through and let switch_model use provider defaults.
    if (
        "base_url" not in kwargs
        and target_provider
        and target_provider.strip().lower() != cur_provider.strip().lower()
    ):
        try:
            from agent.auxiliary_client import resolve_provider_client

            _client, _ = resolve_provider_client(target_provider, model=target_model)
            if _client is not None:
                _resolved_base = str(getattr(_client, "base_url", "") or "").strip()
                if _resolved_base:
                    kwargs["base_url"] = _resolved_base
                if "api_key" not in kwargs:
                    _resolved_key = getattr(_client, "api_key", None)
                    if _resolved_key:
                        kwargs["api_key"] = _resolved_key
        except Exception:
            logger.debug(
                "intelligent_routing: could not pre-resolve endpoint for %s; "
                "switch_model will fall back to provider defaults",
                target_provider, exc_info=True,
            )
    try:
        agent.switch_model(target_model, target_provider, **kwargs)
    except Exception as exc:
        logger.warning(
            "intelligent_routing: switch to %s/%s failed (%s); staying on %s/%s",
            target_model, target_provider or "?", exc, cur_model, cur_provider,
        )
        return False

    # Mark the override turn-scoped with a DEDICATED flag. Do NOT restore the
    # premium snapshot into _primary_runtime (keep it == cheap) and do NOT arm
    # _fallback_activated (that would corrupt reactive cooldown accounting).
    # Turn-scoped request-body params (e.g. {"options": {"num_ctx": N}} for a
    # local tier). The SDK merges these into the JSON body LAST, so they
    # override same-named typed params. Thinking control does NOT belong here —
    # use agent.reasoning_overrides (see the module docstring).
    # Merged over any existing extra_body so custom-provider params survive; the
    # prior value is stashed for restore_primary_runtime to revert next turn.
    # Best-effort: a malformed extra_body must not undo an applied swap.
    saved_extra_body = _UNTOUCHED
    overrides_ref = None
    try:
        routed_extra_body = override.get("extra_body")
        if isinstance(routed_extra_body, dict) and routed_extra_body:
            overrides = getattr(agent, "request_overrides", None)
            if isinstance(overrides, dict):
                saved_extra_body = overrides.get("extra_body", _NOT_SET)
                base = saved_extra_body if isinstance(saved_extra_body, dict) else {}
                # Shallow copy: saved and live are always DISTINCT dicts (so the
                # merge can't corrupt the snapshot at the top level), but they
                # share nested values. Never mutate a nested value in place here
                # or in flight — that would reach through into the snapshot and
                # into the loaded config that cheap_extra_body() copied from.
                merged = dict(base)
                merged.update(routed_extra_body)
                overrides["extra_body"] = merged
                # Remember WHICH dict we dirtied. The gateway assigns a freshly
                # built per-turn request_overrides onto the cached agent before
                # the next turn's revert runs, so re-reading the attribute there
                # would clean the wrong object: leaving cheap params on the dict
                # we actually mutated and stomping the new turn's own extra_body.
                overrides_ref = overrides
        elif routed_extra_body not in (None, {}):
            logger.debug(
                "intelligent_routing: ignoring non-dict extra_body (%s)",
                type(routed_extra_body).__name__,
            )
    except Exception:
        logger.debug("intelligent_routing: could not apply extra_body", exc_info=True)

    try:
        # Only the FIRST apply of a turn snapshots. A second apply without an
        # intervening restore would otherwise stash the first route's own
        # runtime and merged extra_body as the "pristine" pre-turn state, so the
        # cheap params would survive the revert. The single in-tree call site
        # applies once per turn, but the snapshot is the whole safety property.
        # `is not True` deliberately, not falsiness: the flag is set to exactly
        # True here and cleared to False by the revert, so anything else means
        # "no snapshot has been taken" rather than "a routed turn is live".
        if getattr(agent, "_routing_override_active", False) is not True:
            agent._routing_override_saved_primary = primary_snapshot
            agent._routing_override_saved_extra_body = saved_extra_body
            agent._routing_override_overrides_ref = overrides_ref
        elif overrides_ref is not None and getattr(
            agent, "_routing_override_saved_extra_body", _UNTOUCHED
        ) is _UNTOUCHED:
            # First apply wrote no extra_body, this one did: the snapshot is
            # still pristine, but the revert now needs the dirtied dict + the
            # value that was there before this apply touched it.
            agent._routing_override_saved_extra_body = saved_extra_body
            agent._routing_override_overrides_ref = overrides_ref
        agent._routing_override_active = True
        # Escalation is armed only by a route that asks for it (the plugin sets
        # it on cheap routes; a user's "use <model>" directive does not), and only
        # on the first apply of a turn, so the target is the pre-turn route.
        escalate = override.get("escalate") is True
        watch = escalate or override.get("protected_watch") is True
        if watch and getattr(agent, "_routing_escalation_target", None) is None:
            agent._routing_escalation_target = escalation_target
            agent._routing_escalate_pending = escalate
            agent._routing_protected_watch = True
            agent._routing_protected_markers = _markers(override.get("protected_markers"))
            agent._routing_cheap_model = target_model
    except Exception:
        logger.debug("intelligent_routing: could not scope override to turn",
                     exc_info=True)

    logger.info(
        "intelligent_routing: routed this turn to %s/%s (was %s/%s)",
        target_model, target_provider or "?", cur_model, cur_provider,
    )
    return True


def revert_routing_override(agent: Any) -> bool:
    """Undo a routed turn at the top of the next one. True iff a routed turn was reverted.

    Called by ``restore_primary_runtime`` before its fallback and cooldown gates, so a
    routed turn always reverts. The caller then rebuilds the runtime from the restored
    (premium) ``_primary_runtime``.

    Request-body params have three states, not two: ``_UNTOUCHED`` means the routed turn
    never wrote extra_body (the default) and must be left strictly alone, since popping
    there destroyed the primary provider's own body params for the rest of the session;
    ``_NOT_SET`` means the key did not exist before, so it is removed rather than left as
    ``{}``; anything else is the pre-turn value to reinstate. It acts on the dict
    ``apply_routing_override`` actually mutated, NOT a fresh read of
    ``agent.request_overrides``: the gateway assigns a new per-turn overrides dict onto
    the cached agent before this revert runs.
    """
    if not getattr(agent, "_routing_override_active", False):
        return False
    saved = getattr(agent, "_routing_override_saved_primary", None)
    if isinstance(saved, dict):
        agent._primary_runtime = saved
    saved_eb = getattr(agent, "_routing_override_saved_extra_body", _UNTOUCHED)
    overrides = getattr(agent, "_routing_override_overrides_ref", None)
    if saved_eb is not _UNTOUCHED and isinstance(overrides, dict):
        if saved_eb is _NOT_SET:
            overrides.pop("extra_body", None)
        else:
            overrides["extra_body"] = saved_eb
    # Clear the scoping state so this runs exactly once.
    agent._routing_override_active = False
    agent._routing_override_saved_primary = None
    agent._routing_override_saved_extra_body = None
    agent._routing_override_overrides_ref = None
    clear_escalation_state(agent)
    return True


# ── Escalation ────────────────────────────────────────────────────────────────

# Text that marks content as protected. The defaults are the footer the
# papareo_transcribe plugin puts on every Papa Reo transcript (Kaitiakitanga
# License): that text must never reach a model other than the configured primary.
# A route may add more with ``protected_markers``.
DEFAULT_PROTECTED_MARKERS: tuple = (
    "transcribed by papa reo",
    "kaitiakitanga license",
    "kaitiakitanga licence",
)

# Short refusal openers. Matched only against a SHORT reply from a cheap-routed
# model with no tool calls: a long answer that happens to say "I can't" is not one.
_REFUSAL_MAX_CHARS = 600
_REFUSAL_PREFIXES = (
    "i can't help", "i cannot help", "i can't assist", "i cannot assist", "i can't do that",
    "i cannot do that", "i'm unable to", "i am unable to", "i'm not able to", "i am not able to",
    "i won't be able to", "i will not be able to", "i can't comply", "i cannot comply",
    "sorry, i can't", "sorry, but i can't", "i'm sorry, but i can't", "i'm sorry, i can't",
    "i'm sorry, but i cannot", "i apologize, but i can't", "i apologise, but i can't",
    "as an ai", "i don't have access", "i do not have access", "i'm not allowed",
)


def looks_like_refusal(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip().lower().replace("\u2019", "'")
    if not t or len(t) > _REFUSAL_MAX_CHARS:
        return False
    return t.startswith(_REFUSAL_PREFIXES)


def routed_escalation_pending(agent: Any) -> bool:
    """True when this agent runs a cheap-routed turn that has not escalated yet."""
    return getattr(agent, "_routing_escalate_pending", False) is True


def _log_escalation(agent: Any, trigger: str, to_model: str, ok: bool) -> None:
    try:
        import json

        logger.info("routing_escalation %s", json.dumps({
            "trigger": trigger, "from_model": str(getattr(agent, "model", "") or ""),
            "to_model": to_model, "ok": ok,
            "session_id": getattr(agent, "session_id", None),
            "child": bool(getattr(agent, "_delegate_depth", 0)),
        }, default=str))
    except Exception:
        logger.debug("intelligent_routing: escalation metric failed", exc_info=True)


def _switch_to(agent: Any, target: dict) -> bool:
    """Switch the agent back to ``target`` (the pre-turn route). True on success."""
    to_model = str(target["model"])
    kwargs = {k: target[k] for k in ("api_key", "base_url", "api_mode") if target.get(k)}
    try:
        agent.switch_model(to_model, str(target.get("provider") or ""), **kwargs)
    except Exception:  # boundary: a failed switch must leave the turn on its current route
        logger.warning("intelligent_routing: escalation to %s failed", to_model, exc_info=True)
        return False
    # The cheap route's body params must not ride along to the primary: put the
    # pre-turn extra_body back now and leave nothing for next turn's revert.
    try:
        saved_eb = getattr(agent, "_routing_override_saved_extra_body", _UNTOUCHED)
        overrides = getattr(agent, "_routing_override_overrides_ref", None)
        if saved_eb is not _UNTOUCHED and isinstance(overrides, dict):
            if saved_eb is _NOT_SET:
                overrides.pop("extra_body", None)
            else:
                overrides["extra_body"] = saved_eb
            agent._routing_override_saved_extra_body = _UNTOUCHED
    except Exception:
        logger.debug("intelligent_routing: could not revert extra_body on escalation",
                     exc_info=True)
    try:
        from agent.chat_completion_helpers import rewrite_prompt_model_identity

        rewrite_prompt_model_identity(agent, agent.model, agent.provider)
    except Exception:
        logger.debug("intelligent_routing: could not rewrite model identity", exc_info=True)
    return True


def escalate_routed_turn(agent: Any, trigger: Any = None) -> bool:
    """Move a cheap-routed turn back to the route it started on. True iff it switched.

    Fires once per turn. A child (no target) returns False after disarming, so the
    caller walks its fallback chain, whose first entry is the parent's primary.
    Never raises.
    """
    try:
        if not routed_escalation_pending(agent):
            return False
        agent._routing_escalate_pending = False
        trigger_name = str(getattr(trigger, "value", trigger) or "unknown")
        target = getattr(agent, "_routing_escalation_target", None)
        if not isinstance(target, dict) or not target.get("model"):
            _log_escalation(agent, trigger_name, "fallback-chain", True)
            return False
        to_model = str(target["model"])
        if not _switch_to(agent, target):
            _log_escalation(agent, trigger_name, to_model, False)
            return False
        _log_escalation(agent, trigger_name, to_model, True)
        status = getattr(agent, "_buffer_diagnostic_status", None)
        if callable(status):
            status(f"↻ Retrying on {to_model}")
        return True
    except Exception:
        logger.debug("intelligent_routing: escalation failed", exc_info=True)
        return False


def try_activate_fallback(agent: Any, reason: Any = None, reset_at: Any = None) -> bool:
    """The agent's ``_try_activate_fallback``: a failing cheap-routed turn goes back to the
    route it started on (the configured primary) before any fallback-chain entry."""
    if escalate_routed_turn(agent, reason):
        return True
    from agent.chat_completion_helpers import try_activate_fallback as walk_fallback_chain
    return walk_fallback_chain(agent, reason, reset_at)


def clear_escalation_state(agent: Any) -> None:
    agent._routing_escalate_pending = False
    agent._routing_escalation_target = None
    agent._routing_protected_watch = False
    agent._routing_cheap_model = None


def _markers(extra: Any) -> tuple:
    if isinstance(extra, str):
        extra = [extra]
    if not isinstance(extra, (list, tuple)):
        extra = []
    return DEFAULT_PROTECTED_MARKERS + tuple(str(m).strip().lower() for m in extra if str(m).strip())


def _content_text(content: Any, _depth: int = 0) -> str:
    """All text in a message's content, including nested parts (tool_result blocks)."""
    if isinstance(content, str):
        return content
    if _depth > 6:
        return ""
    if isinstance(content, dict):
        parts = [str(content.get("text") or "")]
        for key in ("content", "output", "result"):
            if key in content:
                parts.append(_content_text(content[key], _depth + 1))
        return " ".join(p for p in parts if p)
    if isinstance(content, (list, tuple)):
        return " ".join(_content_text(p, _depth + 1) for p in content)
    return "" if content is None else str(content)


PROTECTED_SWITCHED = "switched"
PROTECTED_BLOCKED = "blocked"

PROTECTED_BLOCK_MESSAGE = (
    "I stopped here: this conversation now contains protected text that may only be "
    "handled by this agent's main model, and that model could not be reached. "
    "Nothing was sent to another model. Please try again shortly."
)


def escalate_if_protected(agent: Any, messages: Any, extra_text: Any = "") -> Optional[str]:
    """Run right before each API call of a cheap-routed turn or child.

    If anything about to be sent (``messages`` plus ``extra_text``, e.g. plugin context)
    carries a protected marker while the agent is still on the cheap model, move to the
    primary first. Returns ``"switched"``, ``"blocked"`` (the primary could not be
    reached: the caller must end the turn without calling any model), or None.
    Armed by every cheap route, independent of ``escalate``. Never raises.
    """
    try:
        if getattr(agent, "_routing_protected_watch", False) is not True:
            return None
        cheap_model = getattr(agent, "_routing_cheap_model", None)
        if cheap_model and str(getattr(agent, "model", "") or "") != str(cheap_model):
            return None  # already off the cheap model
        markers = getattr(agent, "_routing_protected_markers", None) or DEFAULT_PROTECTED_MARKERS
        texts = [_content_text(m.get("content")) for m in (messages or []) if isinstance(m, dict)]
        texts.append(_content_text(extra_text))
        if not any(any(mk in t.lower() for mk in markers) for t in texts if t):
            return None
        agent._routing_escalate_pending = False
        target = getattr(agent, "_routing_escalation_target", None)
        if isinstance(target, dict) and target.get("model"):
            ok = _switch_to(agent, target)
            _log_escalation(agent, "protected_content", str(target["model"]), ok)
        else:
            # A cheap child: its fallback chain starts with the parent's primary. Accept the
            # hop only if it really landed there.
            primary = getattr(agent, "_routing_primary_model", None)
            ok = bool(agent._try_activate_fallback()) and bool(primary) and (
                str(getattr(agent, "model", "") or "") == str(primary))
            _log_escalation(agent, "protected_content", str(primary or "?"), ok)
        if ok:
            agent._routing_protected_watch = False
            return PROTECTED_SWITCHED
        logger.warning("intelligent_routing: protected content on a cheap route and the primary "
                       "is unreachable; ending the turn without a model call")
        return PROTECTED_BLOCKED
    except Exception:
        logger.warning("intelligent_routing: protected-content check failed; ending the turn",
                       exc_info=True)
        return PROTECTED_BLOCKED
