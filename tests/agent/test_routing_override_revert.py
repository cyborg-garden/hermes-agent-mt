"""Regression tests driving the REAL restore_primary_runtime after a routing
override — the error-adjacent paths the fake-agent unit tests missed.

Covers the three CONFIRMED audit blockers:
  BLOCKER 1 — cheap model must not leak past its turn when the primary is in
              rate-limit cooldown (restore_primary_runtime's cooldown early-return
              gate must NOT skip an override revert).
  MAJOR 2  — during the routed turn, _primary_runtime must point at the CHEAP
             runtime, so in-turn transient-transport recovery rebuilds cheap, not
             premium.
  MAJOR 3  — the override must NOT pre-arm _fallback_activated, so a cheap-model
             429 arms its own cooldown correctly (fallback accounting intact).

These drive the real ``restore_primary_runtime`` (not a fake) so they prove the
revert actually happens, not merely that preconditions were set.
"""
import time

import pytest

from agent.agent_runtime_helpers import restore_primary_runtime
from agent.routing_override import apply_routing_override


class _Comp:
    def update_model(self, **kw):
        self.__dict__.update(kw)
    model = "x"; base_url = ""; api_key = ""; provider = "anthropic"
    context_length = 200000; api_mode = ""; threshold_tokens = 0


def _premium_runtime():
    return {
        "model": "claude-sonnet-5", "provider": "anthropic", "base_url": "",
        "api_mode": "anthropic_messages", "api_key": "pk", "client_kwargs": {},
        "use_prompt_caching": True, "use_native_cache_layout": True,
        "anthropic_api_key": "pk", "anthropic_base_url": "", "is_anthropic_oauth": False,
        "compressor_model": "claude-sonnet-5", "compressor_base_url": "",
        "compressor_api_key": "pk", "compressor_provider": "anthropic",
        "compressor_context_length": 200000, "compressor_api_mode": "anthropic_messages",
        "compressor_threshold_tokens": 0,
    }


def _make_agent():
    class A:
        pass
    a = A()
    a.model = "claude-sonnet-5"; a.provider = "anthropic"; a.base_url = ""
    a.api_mode = "anthropic_messages"; a.api_key = "pk"; a.client = None
    a._anthropic_client = "AC"; a._anthropic_api_key = "pk"; a._anthropic_base_url = ""
    a._is_anthropic_oauth = False; a._client_kwargs = {}; a._use_prompt_caching = True
    a._use_native_cache_layout = True; a._transport_cache = {}
    a._fallback_activated = False; a._fallback_index = 0; a._fallback_chain = []
    a._rate_limited_until = 0; a._credential_pool = None
    a.context_compressor = _Comp()
    a._primary_runtime = _premium_runtime()

    def switch(nm, np, **kw):
        a.model = nm; a.provider = np; a.api_mode = "chat_completions"
        a.base_url = "https://cheap/v1"; a.api_key = "ck"; a.client = "CHEAP"
        a._anthropic_client = None
        a._client_kwargs = {"api_key": "ck", "base_url": a.base_url}
        a._use_prompt_caching = False; a._use_native_cache_layout = False
        # switch_model persists the cheap runtime into _primary_runtime.
        a._primary_runtime = {
            "model": nm, "provider": np, "base_url": a.base_url, "api_mode": a.api_mode,
            "api_key": "ck", "client_kwargs": dict(a._client_kwargs),
            "use_prompt_caching": False, "use_native_cache_layout": False,
            "compressor_model": nm, "compressor_base_url": a.base_url,
            "compressor_api_key": "ck", "compressor_provider": np,
            "compressor_context_length": 128000, "compressor_api_mode": a.api_mode,
            "compressor_threshold_tokens": 0,
        }
        a._fallback_activated = False; a._fallback_index = 0

    a.switch_model = switch
    a._create_openai_client = lambda kw, reason="", shared=True: "REBUILT"
    return a


# ─────────────────────────── BLOCKER 1: the leak repro ───────────────────────

def test_routed_cheap_model_reverts_even_when_primary_in_cooldown():
    """The auditor's leak repro, now expected to PASS: a routed cheap turn must
    revert to premium next turn even if a rate-limit cooldown is armed."""
    a = _make_agent()
    apply_routing_override(a, {"model": "deepseek/deepseek-v3.2", "provider": "openrouter"})
    assert a.model == "deepseek/deepseek-v3.2"

    # Cheap model 429'd this turn, arming a primary cooldown:
    a._rate_limited_until = time.monotonic() + 60

    # Next turn's top-of-loop restore MUST revert despite the cooldown gate.
    restore_primary_runtime(a)
    assert a.model == "claude-sonnet-5", f"LEAK: turn2 answered by {a.model}"
    assert a.provider == "anthropic"
    # _primary_runtime is back to premium and the override flag is cleared.
    assert a._primary_runtime["model"] == "claude-sonnet-5"
    assert getattr(a, "_routing_override_active", False) is False


def test_revert_happens_with_no_cooldown_too():
    """Baseline: revert also works on the ordinary (no-cooldown) path."""
    a = _make_agent()
    apply_routing_override(a, {"model": "deepseek/deepseek-v3.2", "provider": "openrouter"})
    restore_primary_runtime(a)
    assert a.model == "claude-sonnet-5"
    assert getattr(a, "_routing_override_active", False) is False


# ─────────────────── MAJOR 2: _primary_runtime = cheap in-turn ───────────────

def test_primary_runtime_points_at_cheap_during_routed_turn():
    """During the routed turn, _primary_runtime must be the CHEAP runtime so
    in-turn transient-transport recovery (which rebuilds from _primary_runtime)
    stays on cheap and does not jump to premium."""
    a = _make_agent()
    apply_routing_override(a, {"model": "deepseek/deepseek-v3.2", "provider": "openrouter"})
    # Mid-turn, before the next-turn restore:
    assert a._primary_runtime["model"] == "deepseek/deepseek-v3.2"
    assert a._primary_runtime["provider"] == "openrouter"
    # The saved premium snapshot is stashed separately for the revert.
    assert a._routing_override_saved_primary["model"] == "claude-sonnet-5"


# ─────────────────── MAJOR 3: _fallback_activated NOT pre-armed ──────────────

def test_override_does_not_prearm_fallback_activated():
    """The override must not set _fallback_activated — that flag carries reactive
    cooldown-accounting semantics (chat_completion_helpers:1195-1198)."""
    a = _make_agent()
    apply_routing_override(a, {"model": "deepseek/deepseek-v3.2", "provider": "openrouter"})
    assert a._fallback_activated is False
    assert a._routing_override_active is True


def test_reactive_fallback_engages_under_a_routed_turn():
    """Reactive fallback must still work UNDERNEATH a routed turn: if the cheap
    model fails mid-turn, try_activate_fallback treats cheap as the current
    'primary' (since _primary_runtime == cheap and _fallback_activated is False)
    and arms the cheap-provider cooldown correctly."""
    from agent.chat_completion_helpers import try_activate_fallback
    from agent.error_classifier import FailoverReason

    a = _make_agent()
    apply_routing_override(a, {"model": "deepseek/deepseek-v3.2", "provider": "openrouter"})
    # Empty chain -> exhausted, but the rate_limit branch must still arm cooldown
    # because _fallback_activated is False and current provider == the (cheap)
    # _primary_runtime provider.
    before = getattr(a, "_rate_limited_until", 0)
    try_activate_fallback(a, reason=FailoverReason.rate_limit)
    assert a._rate_limited_until > before, "cheap 429 must arm a cooldown"
