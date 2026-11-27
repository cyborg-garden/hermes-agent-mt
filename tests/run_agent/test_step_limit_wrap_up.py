"""Conversation-loop tests for the step-limit wrap-up note and final summary.

A fake model that never stops calling tools drives a real ``run_conversation``.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.run_agent.test_tool_call_guardrail_runtime import (
    _make_agent,
    _mock_response,
    _mock_tool_call,
)

NOTE_MARK = "[System note: You have used"


class _Driver:
    """Fake model: always asks for a tool, except when ``script`` says otherwise."""

    def __init__(self, agent, script=None):
        self.agent = agent
        self.script = dict(script or {})
        self.loop_requests = []

    def __call__(self, api_kwargs):
        self.loop_requests.append(api_kwargs)
        n = len(self.loop_requests)
        if n in self.script:
            return self.script[n]
        return _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps({"q": n}))],
        )


def _install(agent, summary_text):
    driver = _Driver(agent)
    agent._interruptible_api_call = driver
    agent._interruptible_streaming_api_call = lambda kw, **_k: driver(kw)
    summary_client = MagicMock()
    summary_client.chat.completions.create.return_value = _mock_response(content=summary_text)
    agent._ensure_primary_openai_client = MagicMock(return_value=summary_client)
    return driver, summary_client


def _run(agent, tool_log):
    def fake_tool(name, args, *a, **k):
        tool_log.append(name)
        return json.dumps({"ok": True})

    with (
        patch("run_agent.handle_function_call", side_effect=fake_tool),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        return agent.run_conversation("build the thing")


def _sent_messages(api_kwargs):
    return api_kwargs.get("messages") or []


def _notes_in(api_kwargs):
    count = 0
    for msg in _sent_messages(api_kwargs):
        content = msg.get("content")
        if isinstance(content, str):
            count += content.count(NOTE_MARK)
    return count


def test_warning_appears_once_at_eighty_percent():
    agent = _make_agent("web_search", max_iterations=10)
    driver, _ = _install(agent, "Done: searched. Not done: build. Next: build.")
    tools = []
    _run(agent, tools)

    assert len(driver.loop_requests) == 10
    # Calls 1-8 never carry the note; call 9 is the first to see it (after
    # 8 of 10 steps were used) and every later request carries exactly one.
    for req in driver.loop_requests[:8]:
        assert _notes_in(req) == 0
    for req in driver.loop_requests[8:]:
        assert _notes_in(req) == 1
    note_req = driver.loop_requests[8]
    joined = json.dumps(_sent_messages(note_req))
    assert "8 of 10 steps" in joined


def test_exhaustion_produces_final_text_reply_without_running_tools():
    agent = _make_agent("web_search", max_iterations=10)
    driver, summary_client = _install(agent, "Done: searched. Work remaining: build it.")
    tools = []
    result = _run(agent, tools)

    # (b) a real text reply comes back
    assert "Work remaining: build it." in result["final_response"]
    assert result["turn_exit_reason"] == "max_iterations_reached(10/10)"
    # (c) exactly one tool batch per loop call: the grace call ran none
    assert len(tools) == 10
    summary_client.chat.completions.create.assert_called_once()
    summary_kwargs = summary_client.chat.completions.create.call_args.kwargs
    assert "tools" not in summary_kwargs
    assert "without calling any more tools" in json.dumps(summary_kwargs["messages"])


def test_grace_reply_tool_calls_are_ignored():
    agent = _make_agent("web_search", max_iterations=5)
    driver, summary_client = _install(agent, "")
    summary_client.chat.completions.create.return_value = _mock_response(
        content="Stopping. Work remaining: none",
        finish_reason="tool_calls",
        tool_calls=[_mock_tool_call("web_search", "{}")],
    )
    tools = []
    result = _run(agent, tools)

    assert len(tools) == 5
    assert "Stopping." in result["final_response"]


def test_empty_grace_reply_falls_back_to_last_tool():
    agent = _make_agent("web_search", max_iterations=5)
    _install(agent, "")
    tools = []
    result = _run(agent, tools)

    # (d) never an empty reply: a plain fallback built from the turn
    assert "I ran out of steps before finishing" in result["final_response"]
    assert "Last thing I was doing: web_search" in result["final_response"]


def test_blank_response_from_empty_recovery_does_not_skip_summary():
    """An empty reply mid-turn (nudged) followed by tools to exhaustion."""
    agent = _make_agent("web_search", max_iterations=6)
    driver, summary_client = _install(agent, "Summary. Work remaining: none")
    driver.script[2] = _mock_response(content="", finish_reason="stop", tool_calls=None)
    tools = []
    result = _run(agent, tools)

    assert result["turn_exit_reason"] == "max_iterations_reached(6/6)"
    summary_client.chat.completions.create.assert_called_once()
    assert "Summary." in result["final_response"]


def test_summary_disabled_skips_extra_call_but_still_replies():
    agent = _make_agent("web_search", max_iterations=5)
    _, summary_client = _install(agent, "should not be used")
    agent._step_limit_summary = False
    tools = []
    result = _run(agent, tools)

    summary_client.chat.completions.create.assert_not_called()
    assert "Last thing I was doing: web_search" in result["final_response"]


def test_warning_disabled_by_config():
    agent = _make_agent(
        "web_search",
        max_iterations=10,
        config={"agent": {"step_limit_warning": False}},
    )
    assert agent._step_limit_warning is False
    driver, _ = _install(agent, "ok. Work remaining: none")
    _run(agent, [])
    assert all(_notes_in(req) == 0 for req in driver.loop_requests)


def test_config_sets_warning_fraction():
    agent = _make_agent(
        "web_search",
        max_iterations=10,
        config={"agent": {"step_limit_warning_at": 0.5}},
    )
    driver, _ = _install(agent, "ok. Work remaining: none")
    _run(agent, [])
    assert _notes_in(driver.loop_requests[4]) == 0
    assert _notes_in(driver.loop_requests[5]) == 1


def test_subagent_budget_sizes_its_own_warning():
    """A delegated child runs the same loop with its own (smaller) budget."""
    agent = _make_agent("web_search", max_iterations=5)
    agent.platform = "subagent"
    driver, _ = _install(agent, "child summary. Work remaining: none")
    result = _run(agent, [])
    # 0.8 * 5 = 4: the 5th request is the first to carry the note.
    assert _notes_in(driver.loop_requests[3]) == 0
    assert _notes_in(driver.loop_requests[4]) == 1
    assert "4 of 5 steps" in json.dumps(_sent_messages(driver.loop_requests[4]))
    assert "child summary." in result["final_response"]
