"""End-to-end: plugin routing DECISION -> actual model SWAP, turn-scoped.

Proves the whole chain the coordinator asked for:
  plugin.on_pre_llm_call() returns a `route` override
    -> extract_routing_override() pulls it (as turn_context does)
    -> apply_routing_override() actuates the swap via the agent's switch_model
    -> the swap is turn-scoped (primary snapshot restored, fallback armed).

Uses a fake agent that mimics switch_model's field mutation, so we exercise the
real plugin + real routing_override glue without a live model/provider.
"""
from unittest.mock import MagicMock, patch

from agent.routing_override import apply_routing_override, extract_routing_override
from plugins.intelligent_routing import registration


def _fake_agent(model="claude-sonnet-5", provider="anthropic"):
    agent = MagicMock()
    agent.model = model
    agent.provider = provider
    agent._primary_runtime = {"model": model, "provider": provider}
    agent._fallback_activated = False

    def _switch(new_model, new_provider, **kw):
        agent.model = new_model
        agent.provider = new_provider
        agent._primary_runtime = {"model": new_model, "provider": new_provider}

    agent.switch_model.side_effect = _switch
    return agent


def _run_hook(user_message, **kw):
    """Invoke the plugin hook with routing enabled; return the list of results
    the way turn_context collects them from invoke_hook."""
    with patch(
        "plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
        return_value=True,
    ):
        result = registration.on_pre_llm_call(
            user_message=user_message, model="claude-sonnet-5",
            provider="anthropic", platform="cli", **kw,
        )
    return [result]


def test_mechanical_turn_swaps_to_cheap_end_to_end():
    agent = _fake_agent()
    results = _run_hook("grep for TODO in the repo")

    override = extract_routing_override(results)
    assert override == {"model": "deepseek/deepseek-v4-flash-0731", "provider": "openrouter", "escalate": True}

    swapped = apply_routing_override(agent, override)
    assert swapped is True
    # The agent is now on the cheap tier for THIS turn.
    assert agent.model == "deepseek/deepseek-v4-flash-0731"
    assert agent.provider == "openrouter"
    # ...turn-scoped via a DEDICATED flag: _primary_runtime stays cheap (so
    # in-turn recovery stays on cheap), the premium snapshot is stashed for the
    # revert, and reactive-fallback state is untouched. (The actual next-turn
    # revert against the real restore_primary_runtime is proven in
    # tests/agent/test_routing_override_revert.py.)
    assert agent._routing_override_active is True
    assert agent._routing_override_saved_primary == {"model": "claude-sonnet-5",
                                                     "provider": "anthropic"}
    assert agent._fallback_activated is False


def test_judgment_turn_does_not_swap_end_to_end():
    agent = _fake_agent()
    results = _run_hook("design a system for multi-tenant routing across the fleet")

    override = extract_routing_override(results)
    assert override is None  # premium -> no route emitted

    swapped = apply_routing_override(agent, override)
    assert swapped is False
    assert agent.model == "claude-sonnet-5"  # untouched
    agent.switch_model.assert_not_called()


def test_public_facing_never_swaps_end_to_end():
    agent = _fake_agent()
    results = _run_hook("grep the announcement", is_public_facing=True)
    override = extract_routing_override(results)
    assert override is None
    assert apply_routing_override(agent, override) is False
    assert agent.model == "claude-sonnet-5"


def test_disabled_plugin_emits_nothing_to_actuate():
    agent = _fake_agent()
    # Toggle OFF -> hook returns None -> nothing to extract/apply.
    result = registration.on_pre_llm_call(
        user_message="grep for TODO", model="claude-sonnet-5", platform="cli",
    )
    with patch(
        "plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
        return_value=False,
    ):
        result = registration.on_pre_llm_call(
            user_message="grep for TODO", model="claude-sonnet-5", platform="cli",
        )
    assert result is None
    assert extract_routing_override([result]) is None
    assert apply_routing_override(agent, None) is False
    assert agent.model == "claude-sonnet-5"
