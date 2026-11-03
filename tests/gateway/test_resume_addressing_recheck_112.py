"""Regression tests for hermes-agent-mt#112 — gateway resume/interrupt layers.

Incident: gateway auto-resume re-fired sessions whose only recent activity was
OBSERVED third-party group traffic (another agent's first-person progress
narration), synthesizing a full agent turn that executed — and claimed — work
addressed to a different agent. A subsequent human "stop" was treated as
advisory and the resumed task ran to completion anyway.

Fixes under test:

1. ``_resume_trigger_was_addressed`` / ``_run_startup_resume_event``: a
   session whose transcript tail is an ``observed`` user row is NOT
   auto-resumed; its resume marker is cleared and the pre-claimed runner slot
   released.

2. Terminal interrupts: a human TEXT message arriving while a synthetic
   startup-resume run is in flight forces a hard interrupt (bypassing the
   subagent/compression queue demotions), clears ``resume_pending``, and
   queues a one-shot do-not-continue system note for the next turn.

3. ``build_resume_recovery_note`` warns that '[name]: …' history lines are
   observed third-party messages, never the agent's own work.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Minimal telegram stubs so gateway imports cleanly (mirrors sibling tests).
_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.config import Platform
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SessionSource,
    build_session_key,
)
from gateway.run import GatewayRunner, build_resume_recovery_note
from gateway.session import SessionEntry
from tests.gateway.restart_test_helpers import (
    make_restart_runner,
    make_restart_source,
)

OBSERVED_ROW = {
    "role": "user",
    "content": "[agent-a]: I'm cloning the repo and writing the report now",
    "observed": True,
}
ADDRESSED_ROW = {"role": "user", "content": "please fix the deploy"}


def _pending_entry(source, session_key="agent:main:telegram:group:resume-chat"):
    return SessionEntry(
        session_key=session_key,
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="group",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )


async def _drain(runner) -> None:
    for _ in range(200):
        tasks = list(runner._background_tasks)
        if not tasks:
            break
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# 1. Addressing re-check before auto-resume
# ---------------------------------------------------------------------------

class TestResumeAddressingRecheck:

    @pytest.mark.asyncio
    async def test_observed_tail_skips_auto_resume(self):
        """A transcript ending in observed third-party traffic is not an
        interrupted addressed turn — never synthesize a resume for it."""
        runner, adapter = make_restart_runner()
        source = make_restart_source(chat_id="resume-chat", chat_type="group")
        entry = _pending_entry(source)
        runner.session_store._entries = {entry.session_key: entry}
        runner.session_store.load_transcript = MagicMock(
            return_value=[ADDRESSED_ROW, {"role": "assistant", "content": "done"},
                          OBSERVED_ROW]
        )
        runner.session_store.clear_resume_pending = MagicMock(return_value=True)
        adapter.handle_message = AsyncMock()

        scheduled = runner._schedule_resume_pending_sessions()
        await _drain(runner)

        assert scheduled == 1  # scheduling happens before the async re-check
        adapter.handle_message.assert_not_awaited()
        runner.session_store.clear_resume_pending.assert_called_once_with(
            entry.session_key
        )
        # Pre-claimed runner slot must be released so real messages dispatch.
        assert entry.session_key not in runner._running_agents
        # And the startup-resume tracking set must not leak the key.
        assert entry.session_key not in getattr(
            runner, "_startup_resume_sessions", set()
        )

    @pytest.mark.asyncio
    async def test_addressed_tail_still_auto_resumes(self):
        """A genuinely interrupted addressed turn keeps the resume behavior."""
        runner, adapter = make_restart_runner()
        source = make_restart_source(chat_id="resume-chat", chat_type="group")
        entry = _pending_entry(source)
        runner.session_store._entries = {entry.session_key: entry}
        runner.session_store.load_transcript = MagicMock(
            return_value=[OBSERVED_ROW, ADDRESSED_ROW]
        )
        adapter.handle_message = AsyncMock()

        runner._schedule_resume_pending_sessions()
        await _drain(runner)

        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_interrupted_turn_buried_under_observed_chatter_still_resumes(self):
        """Observed rows on TOP of genuinely interrupted work (an assistant
        row with pending tool_calls) must not swallow the recovery."""
        runner, adapter = make_restart_runner()
        source = make_restart_source(chat_id="resume-chat", chat_type="group")
        entry = _pending_entry(source)
        runner.session_store._entries = {entry.session_key: entry}
        runner.session_store.load_transcript = MagicMock(
            return_value=[
                ADDRESSED_ROW,
                {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
                {"role": "tool", "content": "partial", "tool_call_id": "t1"},
                OBSERVED_ROW,
                OBSERVED_ROW,
            ]
        )
        adapter.handle_message = AsyncMock()

        runner._schedule_resume_pending_sessions()
        await _drain(runner)

        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_all_observed_session_never_resumes(self):
        """The incident shape: a per-user session holding NOTHING but another
        agent's observed narration."""
        runner, adapter = make_restart_runner()
        source = make_restart_source(chat_id="resume-chat", chat_type="group")
        entry = _pending_entry(source)
        runner.session_store._entries = {entry.session_key: entry}
        runner.session_store.load_transcript = MagicMock(
            return_value=[OBSERVED_ROW, OBSERVED_ROW, OBSERVED_ROW]
        )
        runner.session_store.clear_resume_pending = MagicMock(return_value=True)
        adapter.handle_message = AsyncMock()

        runner._schedule_resume_pending_sessions()
        await _drain(runner)

        adapter.handle_message.assert_not_awaited()
        runner.session_store.clear_resume_pending.assert_called_once_with(
            entry.session_key
        )

    @pytest.mark.asyncio
    async def test_group_resume_event_carries_observed_context_prompt(self):
        """The synthetic resume event must carry the adapter's observed-group
        channel prompt so observed rows separate on replay (#112)."""
        runner, adapter = make_restart_runner()
        source = make_restart_source(chat_id="resume-chat", chat_type="group")
        entry = _pending_entry(source)
        runner.session_store._entries = {entry.session_key: entry}
        runner.session_store.load_transcript = MagicMock(
            return_value=[ADDRESSED_ROW]
        )
        adapter._group_observe_channel_prompt = (
            lambda: "identity block… observed group context rules…"
        )
        adapter.handle_message = AsyncMock()

        runner._schedule_resume_pending_sessions()
        await _drain(runner)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.channel_prompt is not None
        assert "observed group context" in event.channel_prompt

    @pytest.mark.asyncio
    async def test_transcript_read_error_fails_toward_resume(self):
        """The re-check must not break recovery for sessions it can't read."""
        runner, adapter = make_restart_runner()
        source = make_restart_source(chat_id="resume-chat", chat_type="group")
        entry = _pending_entry(source)
        runner.session_store._entries = {entry.session_key: entry}
        runner.session_store.load_transcript = MagicMock(
            side_effect=RuntimeError("db locked")
        )
        adapter.handle_message = AsyncMock()

        runner._schedule_resume_pending_sessions()
        await _drain(runner)

        adapter.handle_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# 2. Terminal interrupts for startup-resume runs
# ---------------------------------------------------------------------------

def _make_busy_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.adapters = {}
    runner.config = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._busy_input_mode = "interrupt"
    runner.session_store = MagicMock()
    runner.session_store.clear_resume_pending = MagicMock(return_value=True)
    return runner


def _make_busy_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value="telegram")
    return adapter


def _make_user_event(text: str = "stop this isnt about you") -> MessageEvent:
    source = SessionSource(
        platform=MagicMock(value="telegram"),
        chat_id="123",
        chat_type="group",
        user_id="user1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
    )


def _make_running_parent(active_children=None) -> MagicMock:
    parent = MagicMock()
    parent._active_children = list(active_children or [])
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 4,
        "max_iterations": 60,
        "current_tool": "terminal",
    }
    return parent


class TestTerminalResumeInterrupt:

    @pytest.mark.asyncio
    async def test_human_message_terminates_startup_resume_run(self):
        runner = _make_busy_runner()
        adapter = _make_busy_adapter()
        event = _make_user_event()
        sk = build_session_key(event.source)
        parent = _make_running_parent()
        runner._running_agents[sk] = parent
        runner.adapters[event.source.platform] = adapter
        runner._startup_resume_sessions = {sk}

        with patch("gateway.platforms.base.merge_pending_message_event"):
            handled = await runner._handle_active_session_busy_message(event, sk)

        assert handled is True
        parent.interrupt.assert_called_once_with(event.text)
        runner.session_store.clear_resume_pending.assert_called_once_with(sk)
        assert sk in runner._resume_cancelled_notes

    @pytest.mark.asyncio
    async def test_terminal_interrupt_overrides_subagent_demotion(self):
        """Active subagents normally demote interrupt→queue (#30170); a
        human interrupting a run nobody asked for must still terminate it."""
        runner = _make_busy_runner()
        adapter = _make_busy_adapter()
        event = _make_user_event()
        sk = build_session_key(event.source)
        parent = _make_running_parent(active_children=[MagicMock()])
        runner._running_agents[sk] = parent
        runner.adapters[event.source.platform] = adapter
        runner._startup_resume_sessions = {sk}

        with patch("gateway.platforms.base.merge_pending_message_event"):
            handled = await runner._handle_active_session_busy_message(event, sk)

        assert handled is True
        parent.interrupt.assert_called_once_with(event.text)
        assert sk in runner._resume_cancelled_notes

    @pytest.mark.asyncio
    async def test_terminal_interrupt_overrides_queue_text_mode(self):
        """busy_text_mode='queue' must not let a startup-resume run absorb a
        human message as a queued follow-up."""
        runner = _make_busy_runner()
        runner._busy_text_mode = "queue"
        adapter = _make_busy_adapter()
        event = _make_user_event()
        sk = build_session_key(event.source)
        parent = _make_running_parent()
        runner._running_agents[sk] = parent
        runner.adapters[event.source.platform] = adapter
        runner._startup_resume_sessions = {sk}

        with patch("gateway.platforms.base.merge_pending_message_event"):
            handled = await runner._handle_active_session_busy_message(event, sk)

        assert handled is True
        parent.interrupt.assert_called_once_with(event.text)

    @pytest.mark.asyncio
    async def test_internal_event_still_never_interrupts(self):
        """Internal synthetic events keep their no-interrupt invariant even
        on a startup-resume session."""
        runner = _make_busy_runner()
        adapter = _make_busy_adapter()
        event = _make_user_event("[async delegation completed]")
        object.__setattr__(event, "internal", True)
        sk = build_session_key(event.source)
        parent = _make_running_parent()
        runner._running_agents[sk] = parent
        runner.adapters[event.source.platform] = adapter
        runner._startup_resume_sessions = {sk}

        handled = await runner._handle_active_session_busy_message(event, sk)

        assert handled is False
        parent.interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_session_interrupt_unchanged(self):
        """Sessions NOT in the startup-resume set keep existing semantics —
        no resume clearing, no cancellation note."""
        runner = _make_busy_runner()
        adapter = _make_busy_adapter()
        event = _make_user_event()
        sk = build_session_key(event.source)
        parent = _make_running_parent()
        runner._running_agents[sk] = parent
        runner.adapters[event.source.platform] = adapter

        with patch("gateway.platforms.base.merge_pending_message_event"):
            handled = await runner._handle_active_session_busy_message(event, sk)

        assert handled is True
        parent.interrupt.assert_called_once_with(event.text)
        runner.session_store.clear_resume_pending.assert_not_called()
        assert sk not in getattr(runner, "_resume_cancelled_notes", set())


# ---------------------------------------------------------------------------
# 3. Recovery-note guidance for observed history lines
# ---------------------------------------------------------------------------

class TestRecoveryNoteObservedGuidance:

    def test_note_warns_about_observed_lines(self):
        note = build_resume_recovery_note("restart_interrupted")
        assert "OBSERVED" in note
        assert "never claim" in note

    def test_note_warns_on_message_resume_too(self):
        note = build_resume_recovery_note("restart_timeout", "what happened?")
        assert "OBSERVED" in note
