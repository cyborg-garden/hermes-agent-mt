"""Regression tests for salvaging a partial summary from a timed-out subagent.

When a child exceeds ``delegation.child_timeout_seconds`` the parent abandons
its future, so ``run_conversation`` never returns and its ``final_response``
is lost. Previously the parent received ``summary: None`` — an entire run's
worth of completed tool work was discarded, with no artifact written anywhere.

``_salvage_partial_summary`` reconstructs the child's last assistant prose and
a tool trace from the live transcript that ``run_conversation`` publishes as
``agent._session_messages``.

These tests pin:
- the salvage reads the live transcript and keeps the trailing turns
- the tool trace pairs calls to results by tool_call_id, flagging errors
- the result is clearly marked PARTIAL and provisional
- salvage never raises, whatever the child looks like
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tools.delegate_tool import _PARTIAL_SUMMARY_TURNS, _salvage_partial_summary


def _child(messages):
    child = MagicMock()
    child._session_messages = messages
    return child


class TestSalvagePartialSummary:
    def test_returns_empty_when_no_transcript(self):
        child = MagicMock()
        child._session_messages = None
        assert _salvage_partial_summary(child, 0) == ("", [])

    def test_returns_empty_for_empty_transcript(self):
        assert _salvage_partial_summary(_child([]), 0) == ("", [])

    def test_keeps_trailing_assistant_turns(self):
        msgs = [{"role": "user", "content": "go"}]
        for i in range(6):
            msgs.append({"role": "assistant", "content": f"turn {i}"})

        summary, _ = _salvage_partial_summary(_child(msgs), 0)

        # Only the last _PARTIAL_SUMMARY_TURNS survive, in order.
        kept = [f"turn {i}" for i in range(6 - _PARTIAL_SUMMARY_TURNS, 6)]
        for text in kept:
            assert text in summary
        assert "turn 0" not in summary
        assert summary.index(kept[0]) < summary.index(kept[-1])

    def test_summary_is_marked_partial_and_provisional(self):
        msgs = [{"role": "assistant", "content": "found three companies"}]
        summary, _ = _salvage_partial_summary(_child(msgs), 0)

        assert summary.startswith("[PARTIAL")
        assert "timed out" in summary
        assert "provisional" in summary
        assert "found three companies" in summary

    def test_tool_trace_pairs_calls_to_results_by_id(self):
        msgs = [
            {
                "role": "assistant",
                "content": "searching",
                "tool_calls": [
                    {"id": "a", "function": {"name": "web_search", "arguments": '{"q":"x"}'}},
                    {"id": "b", "function": {"name": "web_extract", "arguments": '{"u":"y"}'}},
                ],
            },
            # Out of order, to prove pairing is by id and not by position.
            {"role": "tool", "tool_call_id": "b", "content": '{"error": "Internal Server Error"}'},
            {"role": "tool", "tool_call_id": "a", "content": "three results"},
        ]

        summary, trace = _salvage_partial_summary(_child(msgs), 0)

        by_tool = {t["tool"]: t for t in trace}
        assert by_tool["web_search"]["status"] == "ok"
        assert by_tool["web_extract"]["status"] == "error"
        assert "2 tool call(s): 1 ok, 1 errored" in summary

    def test_trace_returned_even_when_no_assistant_prose(self):
        """A child that only made tool calls still yields a usable trace."""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "a", "function": {"name": "web_search", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "a", "content": "ok"},
        ]
        summary, trace = _salvage_partial_summary(_child(msgs), 0)
        assert summary == ""
        assert len(trace) == 1
        assert trace[0]["tool"] == "web_search"

    def test_blank_assistant_turns_are_skipped(self):
        msgs = [
            {"role": "assistant", "content": "   "},
            {"role": "assistant", "content": "real content"},
        ]
        summary, _ = _salvage_partial_summary(_child(msgs), 0)
        assert "real content" in summary
        assert summary.count("real content") == 1

    @pytest.mark.parametrize(
        "transcript",
        [
            "not a list",
            [None, 42, "junk"],
            [{"role": "assistant"}],
            [{"role": "assistant", "content": None}],
            [{"role": "assistant", "tool_calls": [{"function": {}}]}],
            [{"role": "tool", "content": None}],
        ],
        ids=["str", "junk-items", "no-content", "null-content", "malformed-call", "null-tool"],
    )
    def test_never_raises_on_malformed_transcripts(self, transcript):
        """Salvage is best-effort — a failure must not mask the timeout."""
        summary, trace = _salvage_partial_summary(_child(transcript), 0)
        assert isinstance(summary, str)
        assert isinstance(trace, list)

    def test_never_raises_when_attribute_access_explodes(self):
        class Hostile:
            @property
            def _session_messages(self):
                raise RuntimeError("boom")

        assert _salvage_partial_summary(Hostile(), 0) == ("", [])


class TestLiveTranscriptIsPublished:
    """The salvage is only possible because run_conversation publishes the
    live message list; pin that contract so it isn't silently removed."""

    def test_conversation_loop_publishes_session_messages_at_turn_start(self):
        import inspect

        from agent import conversation_loop

        # The turn body moved into ``_run_conversation_turn`` (upstream loop
        # decomposition); the publish sits right after the per-turn context is
        # settled, before the loop starts.
        src = inspect.getsource(conversation_loop._run_conversation_turn)
        head = src.split("begin_voice_turn_route(agent, _ctx.messages", 1)[1][:1200]
        assert "agent._session_messages = _ctx.messages" in head, (
            "run_conversation must publish its live transcript on the agent "
            "immediately after binding it, or a timed-out subagent's work "
            "cannot be salvaged."
        )
