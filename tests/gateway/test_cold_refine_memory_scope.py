"""Fork: every agent the gateway builds for a chat carries that chat's memory scope.

Memory is scoped per chat by default. An agent built without the chat's ``context_id`` gets the
unscoped store: it reads and writes the global MEMORY.md, so a chat's ``/refine`` after a restart
(no cached agent) reviewed with the wrong memory and saved what it learned where every chat reads it.
"""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from tests.gateway.test_messaging_refine_without_cached_agent import _gateway, _is_review, _Model

_SOURCE = SessionSource(platform=Platform.TELEGRAM, chat_id="-100group", chat_type="group", user_id="u1")
_SCOPE = "telegram:chat:-100group"


def test_per_chat_memory_is_on_by_default(tmp_path, monkeypatch):
    """Flipping the default would silently pool every chat's memory into one shared store."""
    from hermes_cli.config import load_config
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["memory"]["context_scoping_enabled"] is True
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # a profile with no config.yaml of its own
    assert load_config()["memory"]["context_scoping_enabled"] is True


def test_an_agent_for_a_chat_gets_the_chats_store_under_the_default_config(tmp_path, monkeypatch):
    """The default reaches the store: with no memory config at all, a chat's agent is scoped."""
    import run_agent
    from tools import memory_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(memory_tool, "get_memory_dir", lambda: tmp_path / "memories")
    agent = run_agent.AIAgent(model="test/model", api_key="k", base_url="http://127.0.0.1:9/v1", quiet_mode=True,
                              enabled_toolsets=["memory"], skip_memory=True, context_id=_SCOPE)
    assert agent._memory_store is not None and agent._memory_store.context_id == _SCOPE


class _Captured(Exception):
    pass


def _capture_agent_kwargs(monkeypatch):
    import run_agent

    seen = {}

    def build(**kwargs):
        seen.update(kwargs)
        raise _Captured

    monkeypatch.setattr(run_agent, "AIAgent", build)
    return seen


def test_cold_review_parent_is_built_with_the_chats_memory_scope(monkeypatch):
    from gateway import session_cold_review, session_policy

    seen = _capture_agent_kwargs(monkeypatch)
    monkeypatch.setattr(session_policy, "policy_for_source", lambda runner, source: None)
    runner = MagicMock()
    runner._resolve_turn_toolsets.return_value = (["memory"], None)
    runner._resolve_session_agent_runtime.return_value = ("m", {})
    runner._resolve_turn_agent_config.return_value = {"model": "m", "runtime": {}}
    runner._provider_routing = {}
    with pytest.raises(_Captured), session_cold_review.cold_review_parent(runner, _SOURCE, "key", "sid"):
        pass
    assert seen["context_id"] == _SCOPE


@pytest.mark.asyncio
async def test_hygiene_compression_agent_is_built_with_the_chats_memory_scope(monkeypatch):
    from gateway.run import GatewayRunner

    seen = _capture_agent_kwargs(monkeypatch)
    runner = object.__new__(GatewayRunner)
    runner._session_db = SimpleNamespace(get_session=AsyncMock(return_value=None))
    with pytest.raises(_Captured):
        await runner._hmwa_hygiene_build_agent("m", {}, SimpleNamespace(session_id="sid"), _SOURCE)
    assert seen["context_id"] == _SCOPE


@pytest.mark.asyncio
async def test_manual_compression_agent_is_built_with_the_chats_memory_scope(monkeypatch):
    from gateway.run import GatewayRunner

    seen = _capture_agent_kwargs(monkeypatch)
    runner = object.__new__(GatewayRunner)
    runner._session_db = SimpleNamespace(get_session=AsyncMock(return_value=None))
    with pytest.raises(_Captured):
        await runner._build_manual_compression_agent("sid", "m", {}, _SOURCE)
    assert seen["context_id"] == _SCOPE


class _NotingModel(_Model):
    """A reviewer that saves a note and tries to delete a global entry, then stops."""

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        raw = self.rfile.read(length)
        body = json.loads(raw)
        messages = body.get('messages', [])
        if not (_is_review(body) and messages[-1].get('role') != 'tool'):
            self.rfile = _Replay(raw)
            return super().do_POST()
        self.server.requests.append(body)
        calls = [{'id': f'call_{len(self.server.requests)}_{i}', 'type': 'function', 'index': i, 'function': {
            'name': 'memory', 'arguments': json.dumps(args)}} for i, args in enumerate((
                {'action': 'add', 'target': 'memory', 'content': 'REVIEW_NOTE'},
                {'action': 'remove', 'target': 'memory', 'old_text': 'GLOBAL_FACT'}))]
        message = {'role': 'assistant', 'content': None, 'tool_calls': calls}
        if body.get('stream'):
            chunk = {'id': 'l', 'object': 'chat.completion.chunk', 'model': body.get('model'),
                     'choices': [{'index': 0, 'delta': message, 'finish_reason': 'tool_calls'}]}
            payload, kind = f'data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n'.encode(), 'text/event-stream'
        else:
            payload, kind = json.dumps({'id': 'l', 'object': 'chat.completion', 'model': body.get('model'),
                                        'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls'}],
                                        'usage': {'prompt_tokens': 1, 'completion_tokens': 1,
                                                  'total_tokens': 2}}).encode(), 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _Replay:
    def __init__(self, raw):
        self._raw = raw

    def read(self, _n=-1):
        return self._raw


@pytest.mark.platforms("linux")  # the chaos harness reaps by POSIX signals
def test_cold_refine_reviews_and_writes_only_the_chats_own_memory(tmp_path):
    """After a restart ``/refine`` reads this chat's entries (plus the read-only global layer), never
    another chat's, and what it saves lands in this chat's file: the global file and the other
    chat's file are byte-for-byte what they were."""
    peer = ThreadingHTTPServer(('127.0.0.1', 0), _NotingModel)
    peer.requests = []
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    root = Path(tmp_path) / 'gw'
    root.mkdir()

    def gateway():
        gw = _gateway(root, url)
        memories = gw.hermes_home / 'memories'
        files = {'global': memories / 'MEMORY.md',
                 'chat': memories / 'contexts' / 'chaosfake:dm:chaos-chat' / 'MEMORY.md',
                 'other': memories / 'contexts' / 'chaosfake:dm:other-chat' / 'MEMORY.md'}
        return gw, files

    try:
        warm, files = gateway()
        for name, text in (('global', 'GLOBAL_FACT'), ('chat', 'CHAT_FACT'), ('other', 'OTHER_FACT')):
            files[name].parent.mkdir(parents=True, exist_ok=True)
            files[name].write_text(text)
        try:
            warm.start()
            mark = warm.mark()
            warm.send_user('OPENED_PR_42')
            assert warm.wait_for(lambda e: e.ev == 'send' and 'ACK_OPENED_PR_42' in e.data.get('text', ''),
                                 60, 'turn reply', since=mark), warm.log_tail()
        finally:
            warm.stop()
        cold, files = gateway()  # same home and state.db; the restarted gateway holds no agent
        try:
            cold.start()
            mark, before = cold.mark(), len(peer.requests)
            cold.send_user('/refine cold pass')
            summary = cold.wait_for(lambda e: e.ev == 'send' and '💾' in e.data.get('text', ''), 60,
                                    'cold /refine summary', since=mark)
            review = next((r for r in peer.requests[before:] if _is_review(r)), None)
            assert summary is not None and review is not None, (cold.sends_since(mark), cold.log_tail())
        finally:
            cold.stop()
    finally:
        peer.shutdown()
        peer.server_close()

    seen = json.dumps(review['messages'])
    assert 'CHAT_FACT' in seen, 'the cold review did not read this chat\'s memory'
    assert 'OTHER_FACT' not in seen, 'the cold review read another chat\'s memory'
    assert 'REVIEW_NOTE' in files['chat'].read_text() and 'CHAT_FACT' in files['chat'].read_text()
    assert files['global'].read_text() == 'GLOBAL_FACT', 'the cold review wrote to the global store'
    assert files['other'].read_text() == 'OTHER_FACT', 'the cold review wrote to another chat\'s store'
