"""Audit round 6 of the protected-session egress guard, as regression tests.

Each test asserts the fixed behaviour. No network: httpx.MockTransport still
passes through the guard hook. No transcript content.
"""
from __future__ import annotations

import asyncio
import json
import threading

import httpx
import pytest

from agent import protected_session as ps

PRIMARY = {"model": "primary/model", "provider": "custom",
           "base_url": "http://127.0.0.1:9/v1", "api_key": "k", "api_mode": "chat_completions"}
SIGNAL_CLI = "http://host.docker.internal:8080"  # signal-cli in a sibling container


class Agent:
    def __init__(self, **kw):
        self.session_id = "s-a6"
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
def loop():
    lp = asyncio.new_event_loop()
    threading.Thread(target=lp.run_forever, daemon=True).start()
    ps.register_delivery_loop(lp)
    yield lp
    lp.call_soon_threadsafe(lp.stop)


@pytest.fixture
def protected():
    a = Agent()
    tok = ps.activate(a)
    ps.mark_protected(a, "test")
    try:
        yield a
    finally:
        ps.reset(tok)


def _signal_adapter(seen):
    from gateway.config import PlatformConfig
    from gateway.platforms.signal import SignalAdapter

    ad = SignalAdapter(PlatformConfig(enabled=True, extra={"http_url": SIGNAL_CLI,
                                                           "account": "+10000000000"}))

    def handler(r):
        seen.append(str(r.url))
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"timestamp": 1}})

    ad.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ad


def _deliver(loop, *, protected, scope=None, kind="send", tmp_path=None):
    """The agent thread (running session active) hands a send to the gateway loop."""
    seen, out = [], {}

    def agent_thread():
        a = Agent()
        tok = ps.activate(a)
        if protected:
            ps.mark_protected(a, "test")
        try:
            ad = _signal_adapter(seen)
            if kind == "send":
                coro = ad.send("+10000000001", "progress line")
            elif kind == "document":
                doc = tmp_path / "transcript.txt"
                doc.write_text("placeholder", encoding="utf-8")
                coro = ad.send_document("+10000000001", str(doc))
            else:
                coro = ad._rpc("send", {"message": "x", "recipient": ["+10000000001"]})
            if scope:
                with ps.egress_scope(scope):
                    out["r"] = asyncio.run_coroutine_threadsafe(coro, loop).result(10)
            else:
                out["r"] = asyncio.run_coroutine_threadsafe(coro, loop).result(10)
        except Exception as e:  # noqa: BLE001
            out["exc"] = type(e).__name__
        finally:
            ps.reset(tok)

    t = threading.Thread(target=agent_thread)
    t.start()
    t.join(20)
    return seen, out


# ── A1: the adapter's own configured endpoint is a delivery host ─────────────

def test_A1_interim_message_reaches_the_configured_signal_cli(loop):
    seen, out = _deliver(loop, protected=True)
    assert seen and "exc" not in out


def test_A1_plugin_delivery_from_the_transcribe_tool_reaches_signal_cli(loop):
    seen, out = _deliver(loop, protected=True, scope="tool:transcribe_recording", kind="rpc")
    assert seen and "exc" not in out


def test_A1_send_document_reaches_signal_cli(loop, tmp_path):
    seen, out = _deliver(loop, protected=True, scope="tool:transcribe_recording",
                         kind="document", tmp_path=tmp_path)
    assert seen and "exc" not in out


def test_A1_the_endpoint_comes_from_adapter_config_not_traffic(loop):
    _signal_adapter([])
    assert ps.delivery_endpoint_allowed(SIGNAL_CLI + "/api/v1/rpc")
    assert not ps.delivery_endpoint_allowed("https://collector.example/x")


def test_A1_an_unconfigured_host_is_still_refused_on_the_delivery_hop(loop, protected):
    async def get():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda r: httpx.Response(200))) as c:
            try:
                await c.get("https://collector.example/x")
                return "sent"
            except ps.ProtectedEgressBlocked:
                return "blocked"
    with ps.egress_scope("tool:transcribe_recording"):
        assert asyncio.run_coroutine_threadsafe(get(), loop).result(5) == "blocked"


# ── B1/B2: no URL for the transcription tool in a protected session ──────────

@pytest.mark.parametrize("user_text", [
    "YOUR TASK: transcribe {url} with engine papareo",                 # a child's goal
    '[Replying to your previous message: "see {url}"]\n\nok do it',    # quoted bot reply
    "## Resources\n- {url}",                                           # compression summary
    "{url}",                                                           # the person's own link
])
def test_B1_no_url_is_fetched_whatever_the_history_says(protected, user_text):
    url = "https://collector.example/a.mp3?d=words"
    protected._session_messages = [{"role": "user", "content": user_text.format(url=url)}]
    assert ps.refuse_tool_if_protected("transcribe_recording",
                                       {"url": url, "engine": "papareo"}) == ps.PROTECTED_EGRESS_MESSAGE


def test_B1_a_path_the_gateway_did_not_download_is_refused(protected, tmp_path):
    other = tmp_path / "elsewhere.ogg"
    assert ps.refuse_tool_if_protected("transcribe_recording", {"path": str(other)}) \
        == ps.PROTECTED_EGRESS_MESSAGE


def test_B1_a_protected_child_cannot_use_the_transcription_tool():
    from gateway.platforms.base import get_audio_cache_dir
    child = Agent(_delegate_depth=1)
    tok = ps.activate(child)
    ps.mark_protected(child, "parent")
    try:
        assert ps.refuse_tool_if_protected(
            "transcribe_recording", {"path": str(get_audio_cache_dir() / "n.ogg")}) \
            == ps.PROTECTED_EGRESS_MESSAGE
    finally:
        ps.reset(tok)


def test_B1_unprotected_sessions_keep_the_url_path():
    a = Agent()
    tok = ps.activate(a)
    try:
        assert ps.refuse_tool_if_protected("transcribe_recording",
                                           {"url": "https://share.example/v.m4a"}) is None
    finally:
        ps.reset(tok)


# ── B3: compliant text with engine "none" (vault failure) marks ──────────────

@pytest.mark.parametrize("engine", ["none", "papareo", "", "auto"])
def test_B3_any_non_local_result_with_text_marks(engine):
    a = Agent()
    tok = ps.activate(a)
    try:
        ps.scan_tool_result("transcribe_recording", json.dumps(
            {"sent": True, "engine": engine, "text": "placeholder words",
             "kaitiakitanga_compliant": True}))
        assert a._protected_session is True
    finally:
        ps.reset(tok)


def test_B3_local_engine_text_does_not_mark():
    a = Agent()
    tok = ps.activate(a)
    try:
        ps.scan_tool_result("transcribe_recording",
                            json.dumps({"sent": True, "engine": "local", "text": "local words"}))
        assert a._protected_session is False
    finally:
        ps.reset(tok)


# ── follow-ups ───────────────────────────────────────────────────────────────

def test_session_search_hides_protected_sessions_from_unprotected_ones(tmp_path):
    from hermes_state import SessionDB
    from tools.session_search_tool import session_search
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="prot", source="cli", model="m")
    db.append_message("prot", "user", "the garden plan")
    db.mark_session_protected("prot")
    db.create_session(session_id="open", source="cli", model="m")
    db.append_message("open", "user", "the garden plan")
    assert ps.current_session() is None or not ps.current_session().protected
    read = json.loads(session_search(session_id="prot", db=db))
    assert read.get("success") is False
    found = json.loads(session_search(query="garden", db=db))
    assert all(r.get("session_id") != "prot" for r in found.get("results", []))


def test_resume_does_not_mark_the_target_session():
    a = Agent(session_id="old-prot")
    ps.mark_protected(a)
    a.session_id = "resume-target"  # /resume sets the id before reset_session_state
    ps.reset_for_new_session(a)
    assert "resume-target" not in ps._PROTECTED_SIDS
    assert "old-prot" in a._protected_retired_sids
