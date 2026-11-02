"""Gateway helper agents must not overwrite the session's stored system prompt.

Hygiene and /compress build a throwaway AIAgent with skip_memory=True and only
the memory toolset. The next real turn restores whatever prompt the session row
holds, so writing the helper's prompt there strips the memory, user profile and
skills blocks from every later turn of that session. They mark the stored prompt
instead, and the next real turn rebuilds it with current memory.
"""

import importlib
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import REBUILD_PROMPT_MARK
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource
from hermes_state import SessionDB

FULL_PROMPT = (
    "You are a test agent.\n\n## Skills (mandatory)\n<available_skills>\ngame-art\n"
    "</available_skills>\n\nMEMORY (your personal notes) [1%]\nno music in dreams\n\n"
    "USER PROFILE (who the user is) [1%]\nSam\n\nModel: test/model\nProvider: openrouter"
)
SESSION_ID = "sess-prompt"
SESSION_KEY = "agent:main:telegram:private:12345"


def _seed_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(SESSION_ID, "telegram", model="test/model")
    db.update_system_prompt(SESSION_ID, FULL_PROMPT)
    history = []
    for i in range(12):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": f"turn {i}"})
        db.append_message(session_id=SESSION_ID, role=role, content=f"turn {i}")
    return db, history


def _fake_compress(self, messages, current_tokens=None, focus_topic=None, force=False):
    self.compression_count += 1
    return [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "assistant", "content": "recent"},
    ]


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id="12345", chat_type="private", user_id="12345"
    )


def _runner(gateway_run, db, history):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="1"))
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=SESSION_KEY,
        session_id=SESSION_ID,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="private",
    )
    runner.session_store.load_transcript.return_value = history
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = SimpleNamespace(_db=db)
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._evict_cached_agent = lambda _key: None
    runner._resolve_session_agent_runtime = lambda **_kw: (
        "test/model",
        {
            "api_key": "test-key",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
        },
    )
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    return runner


@pytest.mark.asyncio
async def test_hygiene_leaves_memory_and_skills_in_stored_prompt(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(
        "compression:\n  enabled: true\n  threshold: 0.85\n  hygiene_hard_message_limit: 10\n"
    )
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(ContextCompressor, "compress", _fake_compress)
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length", lambda *a, **k: 1_000_000
    )
    db, history = _seed_db(tmp_path)
    runner = _runner(gateway_run, db, history)

    result = await runner._handle_message(
        MessageEvent(text="hello", source=_source(), message_id="1")
    )

    assert result == "ok"
    # Hygiene really ran: the old turns are soft-archived.
    archived = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? AND compacted = 1",
        (SESSION_ID,),
    ).fetchone()[0]
    assert archived == 12
    stored = db.get_session(SESSION_ID)["system_prompt"]
    assert "<available_skills>" in stored
    assert "MEMORY (your personal notes)" in stored
    assert "USER PROFILE (who the user is)" in stored


@pytest.mark.asyncio
async def test_next_real_turn_after_hygiene_gets_notes_saved_earlier(monkeypatch, tmp_path):
    """Compaction removes the saved note from context, so the prompt must hold it."""
    from agent.conversation_loop import _restore_or_build_system_prompt

    (tmp_path / "config.yaml").write_text(
        "compression:\n  enabled: true\n  threshold: 0.85\n  hygiene_hard_message_limit: 10\n"
    )
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(ContextCompressor, "compress", _fake_compress)
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length", lambda *a, **k: 1_000_000
    )
    db, history = _seed_db(tmp_path)
    runner = _runner(gateway_run, db, history)

    await runner._handle_message(MessageEvent(text="hello", source=_source(), message_id="1"))

    rebuilt = FULL_PROMPT.replace("no music in dreams", "the note saved this session")
    next_turn = SimpleNamespace(
        session_id=SESSION_ID,
        _session_db=db,
        _cached_system_prompt=None,
        model="test/model",
        provider="openrouter",
        platform="telegram",
        _build_system_prompt=lambda _message: rebuilt,
    )
    conversation = [{"role": "user", "content": "go on"}]
    _restore_or_build_system_prompt(next_turn, None, conversation)
    assert "the note saved this session" in next_turn._cached_system_prompt
    assert db.get_session(SESSION_ID)["system_prompt"] == rebuilt

    # Once only: the turn after that reuses the same bytes.
    next_turn._build_system_prompt = lambda _message: "must not be rebuilt"
    next_turn._cached_system_prompt = None
    _restore_or_build_system_prompt(next_turn, None, conversation)
    assert next_turn._cached_system_prompt == rebuilt


@pytest.mark.asyncio
async def test_compress_command_leaves_memory_and_skills_in_stored_prompt(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text("compression:\n  in_place: true\n")
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(ContextCompressor, "compress", _fake_compress)
    monkeypatch.setattr(
        ContextCompressor, "has_content_to_compress", lambda self, msgs: True
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length", lambda *a, **k: 1_000_000
    )
    db, history = _seed_db(tmp_path)
    runner = _runner(gateway_run, db, history)

    await runner._handle_compress_command(
        MessageEvent(text="/compress", source=_source(), message_id="1")
    )

    stored = db.get_session(SESSION_ID)["system_prompt"]
    # A stale helper prompt must never land on the row. Rotation leaves the new
    # row empty, in which case the next real agent builds a full prompt.
    session_id = runner.session_store.get_or_create_session.return_value.session_id
    child = db.get_session(session_id)["system_prompt"]
    assert stored.replace(REBUILD_PROMPT_MARK, "") == FULL_PROMPT
    assert child is None or child.replace(REBUILD_PROMPT_MARK, "") == FULL_PROMPT
