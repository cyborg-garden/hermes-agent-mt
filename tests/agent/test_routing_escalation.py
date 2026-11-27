"""Escalation: a cheap-routed turn that errors, comes back empty or refuses is
retried on the route it started on (the configured primary).

Drives the real apply_routing_override / try_activate_fallback /
restore_primary_runtime / empty-response ladder / final-response check, on the
fake agent the routing-override revert suite already uses.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent import turn_empty_response as ter
from agent.agent_runtime_helpers import restore_primary_runtime
from agent.chat_completion_helpers import try_activate_fallback
from agent.routing_override import (
    apply_routing_override,
    escalate_routed_turn,
    looks_like_refusal,
    routed_escalation_pending,
)
from tests.agent.test_routing_override_revert import _make_agent

CHEAP = {"model": "deepseek/deepseek-v4-flash-0731", "provider": "openrouter", "escalate": True}


def _agent():
    a = _make_agent()
    a.request_overrides = {}
    a._primary_runtime["request_overrides"] = {}
    a.session_id = "s1"
    a._buffer_diagnostic_status = lambda *_a, **_k: None
    return a


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


def test_empty_reply_on_routed_turn_escalates_without_cheap_retries():
    a = MagicMock()
    a._current_streamed_assistant_text = ""
    a._has_content_after_think_block.return_value = False
    a._strip_think_blocks.side_effect = lambda t: t or ""
    a._last_content_with_tools = None
    a._thinking_prefill_retries = 2
    a._fallback_chain = []
    a._routing_escalate_pending = True
    a._try_activate_fallback.return_value = True
    msg = SimpleNamespace(reasoning=None, reasoning_content=None, reasoning_details=None)
    with patch.object(ter, "_retry_empty") as retry, \
            patch("agent.conversation_loop._sync_failover_system_message", return_value="sys"):
        v = ter.recover_empty_response(
            a, msg, None, "stop", final_response="", messages=[{"role": "user", "content": "hi"}],
            api_messages=[], conversation_history=[], active_system_prompt="sys", api_call_count=5,
            turn_exit_reason=None, preflight_compression_blocked=False,
        )
    assert v.action == "continue"
    retry.assert_not_called()
    a._try_activate_fallback.assert_called_once()


def test_refusal_detection():
    assert looks_like_refusal("I'm sorry, but I can't help with that.")
    assert looks_like_refusal("I can’t assist with this request.")
    assert looks_like_refusal("As an AI, I don't have opinions.")
    assert not looks_like_refusal("Here are the 3 files: a.py, b.py, c.py")
    assert not looks_like_refusal("I can't find any TODOs, the repo is clean.")
    assert not looks_like_refusal("I'm unable to " + "x" * 700)  # long answers are answers
    assert not looks_like_refusal(None)


def _finish(agent, text):
    from agent import turn_final_response as tfr
    msg = SimpleNamespace(content=text, tool_calls=None, reasoning_details=None)
    with patch("agent.conversation_loop._sync_failover_system_message", return_value="sys"), \
            patch("agent.turn_context_compaction._refund_api_call", side_effect=lambda _a, n: n - 1):
        return tfr.finish_text_response(
            agent, assistant_message=msg, response=None, finish_reason="stop", messages=[],
            api_messages=[], conversation_history=[], api_call_count=3, user_message="list files",
            active_system_prompt="sys", final_response=None, _turn_exit_reason=None,
            _preflight_compression_blocked=False, codex_ack_continuations=0,
            truncated_response_parts=[], length_continue_retries=0,
            _pending_verification_response=None, _pending_verification_response_previewed=False,
            effective_task_id="t",
        )


def _text_agent(pending):
    a = MagicMock()
    a._has_content_after_think_block.return_value = True
    a._strip_think_blocks.side_effect = lambda t: t or ""
    a._routing_escalate_pending = pending
    a._try_activate_fallback.return_value = True
    return a


def test_refusal_on_routed_turn_is_retried_on_the_primary():
    a = _text_agent(pending=True)
    v = _finish(a, "I'm sorry, but I can't help with that.")
    assert v.action == "continue"
    assert v.final_response is None
    a._try_activate_fallback.assert_called_once()


def test_refusal_on_unrouted_turn_is_left_alone():
    """The check is reached and declines; later guards may trip on a bare MagicMock."""
    import agent.routing_override as ro
    a = _text_agent(pending=False)
    seen = []
    real = ro.routed_escalation_pending
    with patch.object(ro, "routed_escalation_pending", side_effect=lambda ag: seen.append(ag) or real(ag)):
        try:
            _finish(a, "I'm sorry, but I can't help with that.")
        except Exception:
            pass
    assert seen == [a]
    a._try_activate_fallback.assert_not_called()


# ── protected content arriving mid-turn ──

from agent.routing_override import (
    PROTECTED_BLOCKED, PROTECTED_SWITCHED, escalate_if_protected,
)

PAPA_REO = ("kōrero ...\n\nThis text transcribed by Papa Reo API. Please be sure to comply with the terms "
            "of the Kaitiakitanga license in use of this material.")
_RW = "agent.chat_completion_helpers.rewrite_prompt_model_identity"


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


def test_deterministic_error_on_a_cheap_turn_still_escalates():
    """should_fallback=False (the model's own malformed tool-call JSON) normally ends the
    turn; on a cheap route the primary gets a try."""
    from tests.agent.test_malformed_tool_args_no_fallback import _Agent, _Err, _settle
    agent = _Agent()
    agent._routing_escalate_pending = True
    _settle(agent, _Err("invalid tool call arguments"), "openrouter", 400)
    assert agent.activated == [True]


def test_content_filter_stall_on_a_cheap_turn_escalates_with_an_empty_chain():
    from agent import turn_truncation as tt
    a = MagicMock()
    a._fallback_index, a._fallback_chain = 0, []
    a._has_pending_fallback.return_value = True
    a._try_activate_fallback.return_value = True
    st = MagicMock()
    st.agent = a
    st.response = SimpleNamespace(_content_filter_terminated=True)
    st.truncated_response_parts = []
    tt._content_filter_fallback(st, MagicMock())
    a._try_activate_fallback.assert_called_once()


def test_turn_loop_runs_the_protected_check_before_each_call():
    import inspect
    from agent import conversation_loop
    src = inspect.getsource(conversation_loop)
    i = src.index("escalate_if_protected(agent, s.messages, s._plugin_user_context)")
    assert src.rfind("_run_phase(prepare_iteration", 0, i) != -1  # after any steer is appended
    assert i < src.index("_run_phase(assemble_api_request", i)
    assert "PROTECTED_BLOCK_MESSAGE" in src[i:src.index("_run_phase(assemble_api_request", i)]
