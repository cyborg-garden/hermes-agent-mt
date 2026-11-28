"""Cheap-routed turns on the real turn loop: protected text, refusals, empty replies,
errors and exceptions all land on (or return to) the primary.

Drives ``AIAgent.run_conversation`` with the model call patched, so every request
the loop would send is recorded with the model it was addressed to. The route is
injected the way the intelligent_routing plugin returns it from ``pre_llm_call``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.routing_override import PROTECTED_BLOCK_MESSAGE
from run_agent import AIAgent

PRIMARY, CHEAP = "primary/model", "cheap/model"
ROUTE = {"model": CHEAP, "provider": "custom", "escalate": True, "protected_watch": True}
PAPA_REO = ("kōrero ...\n\nThis text transcribed by Papa Reo API. Please be sure to comply "
            "with the terms of the Kaitiakitanga license in use of this material.")


def _agent():
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        a = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:9/v1", provider="custom",
            model=PRIMARY, quiet_mode=True, skip_context_files=True, skip_memory=True,
            enabled_toolsets=["memory"], max_iterations=6,
        )
    a.switches = []

    def switch(new_model, new_provider, **_kw):
        if getattr(a, "fail_switch_to", None) == new_model:
            raise RuntimeError("primary unreachable")
        a.switches.append(new_model)
        a.model, a.provider = new_model, new_provider

    a.switch_model = switch
    return a


def _reply(content, tool_calls=None):
    message = SimpleNamespace(
        role="assistant", content=content, tool_calls=tool_calls, reasoning=None,
        reasoning_content=None, reasoning_details=None, refusal=None, model_dump=lambda **_kw: {},
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop", index=0)],
        usage=None, model="m", id="r",
    )


def _run(agent, replies, hook_results=None, message="list the files"):
    """Run one turn; ``replies`` is a callable(model, call_index) -> response or exception."""
    calls = []

    def _call(api_kwargs, **_kw):
        calls.append({"model": api_kwargs.get("model"), "messages": api_kwargs.get("messages")})
        out = replies(api_kwargs.get("model"), len(calls) - 1)
        if isinstance(out, BaseException):
            raise out
        return out

    results = hook_results if hook_results is not None else [{"route": dict(ROUTE)}]

    def _hook(name, **_kw):
        return list(results) if name == "pre_llm_call" else []

    with (
        patch.object(agent, "_interruptible_api_call", _call),
        patch.object(agent, "_interruptible_streaming_api_call", _call),
        patch("hermes_cli.plugins.invoke_hook", _hook),
        patch("agent.chat_completion_helpers.rewrite_prompt_model_identity"),
        patch("time.sleep"),
    ):
        result = agent.run_conversation(message)
    return result, calls


def _sent_text(call):
    return " ".join(str(m.get("content")) for m in call["messages"] or [])


def test_cheap_turn_answers_on_the_cheap_model():
    a = _agent()
    result, calls = _run(a, lambda m, i: _reply("a.py b.py"))
    assert [c["model"] for c in calls] == [CHEAP]
    assert result["final_response"] == "a.py b.py"


def test_protected_plugin_context_never_reaches_the_cheap_model():
    a = _agent()
    hook = [{"route": dict(ROUTE)}, {"context": PAPA_REO}]
    result, calls = _run(a, lambda m, i: _reply("summary"), hook_results=hook)
    assert calls and all(c["model"] == PRIMARY for c in calls)
    assert not any("papa reo" in _sent_text(c).lower() and c["model"] == CHEAP for c in calls)
    assert result["final_response"] == "summary"


def test_protected_text_with_unreachable_primary_ends_the_turn_without_any_call():
    a = _agent()
    a.fail_switch_to = PRIMARY
    hook = [{"route": dict(ROUTE)}, {"context": PAPA_REO}]
    result, calls = _run(a, lambda m, i: _reply("never"), hook_results=hook)
    assert calls == []
    assert result["final_response"] == PROTECTED_BLOCK_MESSAGE


def test_protected_tool_result_moves_the_turn_before_the_next_call():
    a = _agent()
    tc = SimpleNamespace(id="call_1", type="function",
                         function=SimpleNamespace(name="memory", arguments="{}"))

    def replies(model, i):
        return _reply("", tool_calls=[tc]) if i == 0 else _reply("done")

    def fake_tools(assistant_message, messages, *_a, **_kw):
        messages.append({"role": "tool", "tool_call_id": "call_1", "content": PAPA_REO})

    a.valid_tool_names = set(a.valid_tool_names or ()) | {"memory"}
    with patch.object(a, "_execute_tool_calls", fake_tools):
        result, calls = _run(a, replies)
    assert calls[0]["model"] == CHEAP
    assert len(calls) >= 2 and all(c["model"] == PRIMARY for c in calls[1:])
    assert result["final_response"] == "done"


def test_refusal_on_a_cheap_turn_is_retried_on_the_primary():
    a = _agent()
    result, calls = _run(a, lambda m, i: _reply(
        "I'm sorry, but I can't help with that." if m == CHEAP else "here you go"))
    assert [c["model"] for c in calls] == [CHEAP, PRIMARY]
    assert result["final_response"] == "here you go"


def test_empty_reply_on_a_cheap_turn_escalates_without_cheap_retries():
    a = _agent()
    result, calls = _run(a, lambda m, i: _reply("" if m == CHEAP else "answer"))
    assert [c["model"] for c in calls] == [CHEAP, PRIMARY]
    assert result["final_response"] == "answer"


class _Http400(Exception):
    status_code = 400

    def __init__(self):
        super().__init__("Error code: 400 - invalid request")
        self.response = SimpleNamespace(status_code=400, headers={})
        self.body = {"error": {"message": "invalid request"}}


class _Http500(Exception):
    status_code = 500

    def __init__(self):
        super().__init__("Error code: 500 - upstream exploded")
        self.response = SimpleNamespace(status_code=500, headers={})
        self.body = {"error": {"message": "upstream exploded"}}


@pytest.mark.parametrize("err", [_Http400, _Http500])
def test_api_error_on_a_cheap_turn_escalates_with_no_fallback_chain(err):
    a = _agent()
    assert not a._fallback_chain
    result, calls = _run(a, lambda m, i: err() if m == CHEAP else _reply("recovered"))
    assert calls[-1]["model"] == PRIMARY
    assert result["final_response"] == "recovered"


def test_exception_mid_turn_still_reverts_next_turn():
    a = _agent()
    with pytest.raises(KeyboardInterrupt):
        _run(a, lambda m, i: KeyboardInterrupt())
    assert a.model == CHEAP  # the turn died on the cheap route
    # Next turn, no route: the loop starts on the primary.
    _, calls = _run(a, lambda m, i: _reply("ok"), hook_results=[])
    assert [c["model"] for c in calls] == [PRIMARY]


def test_unrouted_turn_is_untouched():
    a = _agent()
    result, calls = _run(a, lambda m, i: _reply("I'm sorry, but I can't help with that."),
                         hook_results=[{"context": PAPA_REO}])
    assert [c["model"] for c in calls] == [PRIMARY]
    assert a.switches == []
    assert result["final_response"].startswith("I'm sorry")


def test_post_tool_compression_never_runs_on_the_cheap_model_with_protected_text():
    """Compression right after a tool batch would hand the new transcript to the
    compressor's model, which on a cheap turn is the cheap model."""
    a = _agent()
    tc = SimpleNamespace(id="call_1", type="function",
                         function=SimpleNamespace(name="memory", arguments="{}"))
    seen = []

    def fake_tools(assistant_message, messages, *_a, **_kw):
        messages.append({"role": "tool", "tool_call_id": "call_1", "content": PAPA_REO})

    def fake_compress(_agent, messages, system_message, **_kw):
        seen.append(a.model)
        return messages, a._cached_system_prompt

    a.valid_tool_names = set(a.valid_tool_names or ()) | {"memory"}
    a.compression_enabled = True
    with patch.object(a, "_execute_tool_calls", fake_tools), \
            patch("agent.conversation_compression.compress_context", fake_compress), \
            patch.object(a.context_compressor, "should_compress", side_effect=lambda *_a, **_k: len(seen) == 0):
        _run(a, lambda m, i: _reply("", tool_calls=[tc]) if i == 0 else _reply("done"))
    assert seen and seen[0] == PRIMARY


# ── Fallback hops after protected text: primary only ─────────────────────────
# The retry loop walks the fallback chain on its own; the protected check at the
# top of each pass never sees that hop. Protected text must stay on the primary.

FALLBACK = "fallback/other"


def _chain_walk(agent, reason=None, reset_at=None):
    """Stand-in for the real chain walk: one hop to FALLBACK, then exhausted."""
    if agent.model == FALLBACK:
        return False
    agent.model = FALLBACK
    return True


def _leaks(calls):
    return [c for c in calls if c["model"] != PRIMARY and "papa reo" in _sent_text(c).lower()]


def test_primary_error_after_protected_escalation_never_falls_back():
    a = _agent()
    hook = [{"route": dict(ROUTE)}, {"context": PAPA_REO}]
    with patch("agent.chat_completion_helpers.try_activate_fallback", _chain_walk):
        _result, calls = _run(a, lambda m, i: _Http500() if m == PRIMARY else _reply("ok"),
                             hook_results=hook)
    assert calls and not _leaks(calls)
    assert all(c["model"] == PRIMARY for c in calls)


def test_unrouted_protected_session_never_falls_back():
    a = _agent()
    with patch("agent.chat_completion_helpers.try_activate_fallback", _chain_walk):
        _result, calls = _run(a, lambda m, i: _Http500() if m == PRIMARY else _reply("ok"),
                             hook_results=[{"context": PAPA_REO}])
    assert calls and not _leaks(calls)
    assert all(c["model"] == PRIMARY for c in calls)


def test_unrouted_session_without_protected_text_still_falls_back():
    a = _agent()
    with patch("agent.chat_completion_helpers.try_activate_fallback", _chain_walk):
        result, calls = _run(a, lambda m, i: _Http500() if m == PRIMARY else _reply("ok"),
                             hook_results=[])
    assert calls[-1]["model"] == FALLBACK
    assert result["final_response"] == "ok"
