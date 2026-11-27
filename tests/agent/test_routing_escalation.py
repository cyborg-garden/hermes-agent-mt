"""Escalation: a cheap-routed turn that errors, comes back empty or refuses is
retried on the route it started on (the configured primary).

Drives the real apply_routing_override / try_activate_fallback /
restore_primary_runtime, on the fake agent the routing-override revert suite
already uses.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.agent_runtime_helpers import restore_primary_runtime
from agent.routing_override import (
    apply_routing_override,
    try_activate_fallback,
    escalate_routed_turn,
    looks_like_refusal,
    routed_escalation_pending,
)
from tests.agent.test_routing_override_revert import _make_agent

CHEAP = {"model": "deepseek/deepseek-v4-flash-0731", "provider": "openrouter", "escalate": True}
_RW = "agent.chat_completion_helpers.rewrite_prompt_model_identity"


def _agent():
    a = _make_agent()
    a.request_overrides = {}
    a._primary_runtime["request_overrides"] = {}
    a.session_id = "s1"
    a._buffer_status = lambda *_a, **_k: None
    return a


def test_status_line_failure_after_the_switch_still_counts_as_escalated():
    """The switch to the primary already happened: a broken status line must not make the
    caller think escalation failed and skip the primary in the fallback chain."""
    a = _agent()
    apply_routing_override(a, CHEAP)

    def broken_status(*_a, **_k):
        raise RuntimeError("status sink gone")
    a._buffer_status = broken_status
    a._buffer_diagnostic_status = broken_status
    with patch(_RW):
        assert escalate_routed_turn(a, "error") is True
    assert a.model == "claude-sonnet-5"


def test_cheap_route_arms_escalation_to_the_pre_turn_route():
    a = _agent()
    apply_routing_override(a, CHEAP)
    assert routed_escalation_pending(a)
    assert a._routing_escalation_target["model"] == "claude-sonnet-5"
    assert a._routing_escalation_target["provider"] == "anthropic"


def test_route_without_escalate_does_not_arm():
    """A user's "use <model>" directive is not escalated away from."""
    a = _agent()
    apply_routing_override(a, {"model": "claude-fable-5", "provider": "anthropic"})
    assert not routed_escalation_pending(a)


def test_escalation_switches_back_once():
    a = _agent()
    apply_routing_override(a, CHEAP)
    with patch("agent.chat_completion_helpers.rewrite_prompt_model_identity"):
        assert escalate_routed_turn(a, "empty_response") is True
    assert a.model == "claude-sonnet-5" and a.provider == "anthropic"
    assert not routed_escalation_pending(a)
    assert escalate_routed_turn(a, "again") is False  # once per turn


def test_try_activate_fallback_escalates_before_the_chain():
    """An empty chain would normally mean "no fallback": a routed turn still escalates."""
    a = _agent()
    apply_routing_override(a, CHEAP)
    assert a._fallback_chain == []
    with patch("agent.chat_completion_helpers.rewrite_prompt_model_identity"):
        assert try_activate_fallback(a) is True
    assert a.model == "claude-sonnet-5"


def test_unrouted_turn_never_escalates():
    a = _agent()
    assert escalate_routed_turn(a, "x") is False
    assert a.model == "claude-sonnet-5"


def test_escalation_drops_the_cheap_routes_extra_body():
    a = _agent()
    a.request_overrides = {"extra_body": {"tags": ["primary"]}}
    a._primary_runtime["request_overrides"] = {"extra_body": {"tags": ["primary"]}}
    apply_routing_override(a, {**CHEAP, "extra_body": {"options": {"num_ctx": 8192}}})
    assert "options" in a.request_overrides["extra_body"]
    with patch("agent.chat_completion_helpers.rewrite_prompt_model_identity"):
        escalate_routed_turn(a, "error")
    assert a.request_overrides["extra_body"] == {"tags": ["primary"]}
    restore_primary_runtime(a)  # next turn: nothing left to revert, and nothing destroyed
    assert a.request_overrides["extra_body"] == {"tags": ["primary"]}


def test_next_turn_revert_clears_escalation_state():
    a = _agent()
    apply_routing_override(a, CHEAP)
    restore_primary_runtime(a)
    assert not routed_escalation_pending(a)
    assert a._routing_escalation_target is None


def test_child_without_target_disarms_and_walks_the_chain():
    a = _agent()
    a._routing_escalate_pending = True  # set by delegate_tool on a cheap child
    assert escalate_routed_turn(a, "refusal") is False
    assert not routed_escalation_pending(a)


def test_refusal_detection():
    assert looks_like_refusal("I'm sorry, but I can't help with that.")
    assert looks_like_refusal("I can’t assist with this request.")
    assert looks_like_refusal("As an AI, I don't have opinions.")
    assert not looks_like_refusal("Here are the 3 files: a.py, b.py, c.py")
    assert not looks_like_refusal("I can't find any TODOs, the repo is clean.")
    assert not looks_like_refusal("I'm unable to " + "x" * 700)  # long answers are answers
    assert not looks_like_refusal(None)


# ── protected content arriving mid-turn ──

from agent.routing_override import (
    PROTECTED_BLOCKED, PROTECTED_SWITCHED, escalate_if_protected,
)

PAPA_REO = ("kōrero ...\n\nThis text transcribed by Papa Reo API. Please be sure to comply with the terms "
            "of the Kaitiakitanga license in use of this material.")


def test_protected_tool_result_moves_a_cheap_turn_to_the_primary_before_the_next_call():
    a = _agent()
    apply_routing_override(a, CHEAP)
    msgs = [{"role": "user", "content": "summarise notes/hui.txt"}]
    assert escalate_if_protected(a, msgs) is None
    msgs.append({"role": "tool", "content": PAPA_REO})
    with patch(_RW):
        assert escalate_if_protected(a, msgs) == PROTECTED_SWITCHED
    assert a.model == "claude-sonnet-5"
    assert escalate_if_protected(a, msgs) is None  # on the primary now: nothing more to do


def test_watch_is_armed_without_escalation():
    """routing.escalate: false must not switch the protected check off."""
    a = _agent()
    apply_routing_override(a, {"model": "cheap/m", "provider": "openrouter", "protected_watch": True})
    assert not routed_escalation_pending(a)
    with patch(_RW):
        assert escalate_if_protected(a, [{"role": "tool", "content": PAPA_REO}]) == PROTECTED_SWITCHED
    assert a.model == "claude-sonnet-5"


def test_failed_switch_blocks_instead_of_calling_the_cheap_model():
    a = _agent()
    apply_routing_override(a, CHEAP)

    def boom(*_a, **_k):
        raise RuntimeError("primary down")
    a.switch_model = boom
    a._try_activate_fallback = MagicMock(return_value=True)
    assert escalate_if_protected(a, [{"role": "tool", "content": PAPA_REO}]) == PROTECTED_BLOCKED
    a._try_activate_fallback.assert_not_called()  # never hops to a non-primary chain entry


def test_check_sees_everything_after_compression_and_nested_parts():
    a = _agent()
    apply_routing_override(a, CHEAP)
    msgs = [{"role": "user", "content": "x"}] * 12
    assert escalate_if_protected(a, msgs) is None
    msgs = msgs[:8] + [{"role": "tool", "content": [
        {"type": "tool_result", "content": [{"type": "text", "text": PAPA_REO}]}]}] + msgs[:4]
    with patch(_RW):
        assert escalate_if_protected(a, msgs) == PROTECTED_SWITCHED


def test_plugin_context_is_checked_too():
    a = _agent()
    apply_routing_override(a, CHEAP)
    with patch(_RW):
        assert escalate_if_protected(a, [], extra_text=PAPA_REO) == PROTECTED_SWITCHED


def test_configured_marker_on_the_route_is_honoured():
    a = _agent()
    apply_routing_override(a, {**CHEAP, "protected_markers": ["Do-Not-Route"]})
    with patch(_RW):
        assert escalate_if_protected(a, [{"role": "tool", "content": "x DO-NOT-ROUTE y"}]) == PROTECTED_SWITCHED


def _child(primary="kimi", lands_on="kimi"):
    a = MagicMock()
    a.model = "cheap/m"
    a._routing_protected_watch = True
    a._routing_cheap_model = "cheap/m"
    a._routing_escalation_target = None
    a._routing_protected_markers = None
    a._routing_primary_model = primary

    def hop():
        a.model = lands_on
        return True
    a._try_activate_fallback.side_effect = hop
    return a


def test_protected_content_on_a_cheap_child_moves_it_to_the_primary():
    a = _child()
    assert escalate_if_protected(a, [{"role": "tool", "content": [{"type": "text", "text": PAPA_REO}]}]) \
        == PROTECTED_SWITCHED


def test_child_hop_that_misses_the_primary_is_blocked():
    a = _child(lands_on="some-other-fallback")
    assert escalate_if_protected(a, [{"role": "tool", "content": PAPA_REO}]) == PROTECTED_BLOCKED


def test_unrouted_turn_is_not_touched_by_the_protected_check():
    a = _agent()
    assert escalate_if_protected(a, [{"role": "tool", "content": PAPA_REO}]) is None
    assert a.model == "claude-sonnet-5"


def test_revert_disarms_the_watch():
    a = _agent()
    apply_routing_override(a, CHEAP)
    restore_primary_runtime(a)
    assert escalate_if_protected(a, [{"role": "tool", "content": PAPA_REO}]) is None


def test_has_pending_fallback_counts_a_routed_turn():
    from run_agent import AIAgent
    a = SimpleNamespace(_fallback_index=0, _fallback_chain=[], _routing_escalate_pending=True,
                        _routing_escalation_target={"model": "m"})
    assert AIAgent._has_pending_fallback(a) is True
    a._routing_escalate_pending = False
    assert AIAgent._has_pending_fallback(a) is False



def test_loop_runs_the_protected_gate_before_anything_that_sends_text():
    """On fleet-live the gate sits at the top of the turn loop: before request assembly,
    MoA advisors, in-loop compression and the model call."""
    import inspect
    from agent import conversation_loop
    src = inspect.getsource(conversation_loop.run_conversation)
    gate = src.index("_protected = protected_content_gate(")
    assert gate > src.index("_pre_api_steer = agent._drain_pending_steer()")  # steer already appended
    for later in ("api_messages = []", "aggregate_moa_context(", "_ollama_context_limit_error(",
                  "agent._compress_context(", "agent._interruptible_api_call("):
        assert gate < src.index(later), later


def test_refusal_helper_retries_only_routed_turns():
    from agent.turn_refusal_escalation import escalate_routed_refusal
    a = MagicMock()
    a._strip_think_blocks.side_effect = lambda t: t or ""
    a._try_activate_fallback.return_value = True
    a._routing_escalate_pending = False
    assert escalate_routed_refusal(a, "I'm sorry, but I can't help with that.") is False
    a._try_activate_fallback.assert_not_called()
    a._routing_escalate_pending = True
    assert escalate_routed_refusal(a, "Here are the files.") is False
    assert escalate_routed_refusal(a, "I'm sorry, but I can't help with that.") is True


def test_gate_is_inert_off_a_cheap_route():
    from agent.turn_protected_gate import protected_content_gate
    a = _agent()
    assert protected_content_gate(a, [{"role": "tool", "content": PAPA_REO}], PAPA_REO) is None


def test_gate_checks_memory_recall_and_ephemeral_prompt():
    from agent.turn_protected_gate import PROTECTED_SWITCHED, protected_content_gate
    a = _agent()
    apply_routing_override(a, CHEAP)
    with patch(_RW):
        assert protected_content_gate(a, [], "", PAPA_REO) == PROTECTED_SWITCHED
    b = _agent()
    b.ephemeral_system_prompt = PAPA_REO
    apply_routing_override(b, CHEAP)
    with patch(_RW):
        assert protected_content_gate(b, []) == PROTECTED_SWITCHED


# ── audit follow-ups ──

def test_user_model_directive_still_guards_protected_text():
    from plugins.intelligent_routing import registration
    with patch.object(registration, "is_intelligent_routing_enabled", return_value=True):
        out = registration.on_pre_llm_call(user_message="we use kimi for this. " + PAPA_REO,
                                           model="claude-sonnet-5", provider="anthropic", platform="cli")
    assert out["route"]["protected_watch"] is True
    a = _agent()
    apply_routing_override(a, out["route"])
    with patch(_RW):
        assert escalate_if_protected(a, [{"role": "user", "content": PAPA_REO}]) == PROTECTED_SWITCHED
    assert a.model == "claude-sonnet-5"


def test_marker_in_tool_call_arguments_is_seen():
    a = _agent()
    apply_routing_override(a, CHEAP)
    msgs = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "write_file", "arguments": '{"text": "' + PAPA_REO.replace("\n", " ") + '"}'}}]}]
    with patch(_RW):
        assert escalate_if_protected(a, msgs) == PROTECTED_SWITCHED
    from plugins.intelligent_routing.config import protected_markers
    from plugins.intelligent_routing.registration import _is_protected
    assert _is_protected("list the files in /tmp", msgs, protected_markers())


def test_whitespace_variants_of_the_marker_still_match():
    for sep in ("\n", " ", "  "):
        a = _agent()
        apply_routing_override(a, CHEAP)
        text = f"x transcribed{sep}by papa reo api"
        with patch(_RW):
            assert escalate_if_protected(a, [{"role": "tool", "content": text}]) == PROTECTED_SWITCHED, repr(sep)


def test_gate_checks_the_system_prompt():
    from agent.turn_protected_gate import PROTECTED_SWITCHED as SW, protected_content_gate
    a = _agent()
    a._cached_system_prompt = "memory: " + PAPA_REO
    apply_routing_override(a, CHEAP)
    with patch(_RW):
        assert protected_content_gate(a, []) == SW


def test_watch_stays_on_after_a_hop_to_a_non_primary_entry():
    """Off the cheap model but not on the primary (a fallback entry): still guarded."""
    a = _agent()
    apply_routing_override(a, {**CHEAP, "escalate": False, "protected_watch": True})
    a.model = "some/other-fallback"
    with patch(_RW):
        assert escalate_if_protected(a, [{"role": "tool", "content": PAPA_REO}]) == PROTECTED_SWITCHED
    assert a.model == "claude-sonnet-5"


def test_refusal_variants():
    for t in ("Unfortunately, I can't help with that.", "I'm sorry, but I'm unable to do that.",
              "Sorry — I can't do that."):
        assert looks_like_refusal(t), t


def test_blocked_agent_skips_compression():
    from run_agent import AIAgent
    a = _agent()
    apply_routing_override(a, CHEAP)
    a.switch_model = MagicMock(side_effect=RuntimeError("down"))
    a._cached_system_prompt = "sys"
    msgs = [{"role": "tool", "content": PAPA_REO}]
    with patch("agent.conversation_compression.compress_context") as cc:
        out = AIAgent._compress_context(a, msgs, "sys")
    cc.assert_not_called()
    assert out[0] is msgs


def test_blocked_agent_skips_background_review():
    from agent.routing_override import protected_on_routed_model
    a = _agent()
    apply_routing_override(a, CHEAP)
    msgs = [{"role": "tool", "content": PAPA_REO}]
    assert protected_on_routed_model(a, msgs) is True
    import inspect
    from agent import turn_finalizer
    src = inspect.getsource(turn_finalizer)
    assert "not protected_on_routed_model(agent, messages)" in src
