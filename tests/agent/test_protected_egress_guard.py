"""The protected-session egress guard (agent/protected_session.py).

A protected session (one that has held Papa Reo transcript text) may only send text to
its configured primary. These tests drive the real hooks: httpx under the OpenAI and
Anthropic SDKs, requests, urllib, aiohttp, botocore, thread propagation, tool and
plugin scopes, and the sticky flag in the session record.

All transcript text here is placeholder text with the marker footer.
"""
from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from agent import protected_session as ps

FOOTER = "This text transcribed by Papa Reo API. Kaitiakitanga license applies."
TRANSCRIPT = "placeholder transcript words. " * 20 + FOOTER
PRIMARY = {"model": "primary/model", "provider": "custom",
           "base_url": "http://127.0.0.1:9/v1", "api_key": "k", "api_mode": "chat_completions"}


def _agent(**kw):
    a = SimpleNamespace(session_id="s1", _session_db=None, _session_init_model_config={},
                        _protected_session=False, _protected_primary=dict(PRIMARY),
                        _cached_system_prompt="", ephemeral_system_prompt=None,
                        prefill_messages=[])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.fixture
def protected():
    a = _agent()
    ps.mark_protected(a, "test")
    token = ps.activate(a)
    try:
        yield a
    finally:
        ps.reset(token)


@pytest.fixture
def unprotected():
    a = _agent()
    token = ps.activate(a)
    try:
        yield a
    finally:
        ps.reset(token)


def _mock_http(sent):
    def handler(request):
        sent.append(request)
        body = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}]}
        return httpx.Response(200, json=body)
    return httpx.Client(transport=httpx.MockTransport(handler))


def _openai(base_url, sent):
    from openai import OpenAI
    return OpenAI(api_key="k", base_url=base_url, max_retries=0, http_client=_mock_http(sent))


# ── model clients (httpx) ────────────────────────────────────────────────────

def test_openai_request_to_the_primary_goes_through(protected):
    sent = []
    _openai(PRIMARY["base_url"], sent).chat.completions.create(
        model=PRIMARY["model"], messages=[{"role": "user", "content": TRANSCRIPT}])
    assert len(sent) == 1


@pytest.mark.parametrize("base_url,model", [
    ("http://127.0.0.1:9/v1", "cheap/model"),          # primary endpoint, other model
    ("http://127.0.0.1:8/v1", "primary/model"),        # other endpoint, same model name
    ("https://openrouter.example/api/v1", "fallback/x"),
])
def test_openai_request_off_the_primary_is_refused_before_sending(protected, base_url, model):
    from openai import APIConnectionError
    sent = []
    with pytest.raises(APIConnectionError) as exc:
        _openai(base_url, sent).chat.completions.create(
            model=model, messages=[{"role": "user", "content": TRANSCRIPT}])
    assert sent == []
    assert isinstance(exc.value.__cause__, ps.ProtectedEgressBlocked)


def test_anthropic_request_off_the_primary_is_refused_before_sending(protected):
    anthropic = pytest.importorskip("anthropic")
    Anthropic, APIConnectionError = anthropic.Anthropic, anthropic.APIConnectionError
    sent = []
    client = Anthropic(api_key="k", base_url="https://api.anthropic.example", max_retries=0,
                       http_client=_mock_http(sent))
    with pytest.raises(APIConnectionError):
        client.messages.create(model="claude-side", max_tokens=5,
                               messages=[{"role": "user", "content": "hi"}])
    assert sent == []


def test_async_httpx_model_request_off_the_primary_is_refused(protected):
    sent = []

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda r: sent.append(r) or httpx.Response(200))) as c:
            await c.post("http://127.0.0.1:8/v1/chat/completions",
                         json={"model": "cheap/model", "messages": []})
    with pytest.raises(ps.ProtectedEgressBlocked):
        asyncio.run(go())
    assert sent == []


def test_unprotected_session_is_unchanged(unprotected):
    sent = []
    _openai("http://127.0.0.1:8/v1", sent).chat.completions.create(
        model="cheap/model", messages=[{"role": "user", "content": "hello"}])
    assert len(sent) == 1


def test_no_running_session_is_unchanged():
    assert ps.current_session() is None or not ps.current_session().protected
    sent = []
    _openai("http://127.0.0.1:8/v1", sent).chat.completions.create(
        model="cheap/model", messages=[{"role": "user", "content": "hello"}])
    assert len(sent) == 1


def test_platform_delivery_outside_any_scope_is_not_a_model_request(protected):
    """Chat platforms post to paths like /channels/<id>/messages: not a model call."""
    sent = []
    with httpx.Client(transport=httpx.MockTransport(
            lambda r: sent.append(r) or httpx.Response(200))) as c:
        c.post("https://chat.example/api/channels/1/messages", json={"content": "reply"})
    assert len(sent) == 1


def test_refusal_carries_no_content(protected):
    with pytest.raises(ps.ProtectedEgressBlocked) as exc:
        ps.check_request("https://side.example/v1/chat/completions?q=secret-query",
                         json.dumps({"model": "x", "messages": [{"content": TRANSCRIPT}]}).encode())
    text = str(exc.value) + repr(exc.value.args)
    assert "secret-query" not in text and "placeholder" not in text and "side.example" not in text


# ── tools and plugins (requests, urllib, aiohttp) ────────────────────────────

class _Adapter:
    def __init__(self, sent):
        self.sent = sent

    def send(self, request, **_kw):
        import requests
        self.sent.append(request)
        r = requests.Response()
        r.status_code, r._content, r.url = 200, b"{}", request.url
        return r

    def close(self):
        pass


def _requests_post(url, sent):
    import requests
    s = requests.Session()
    s.mount("https://", _Adapter(sent))
    s.mount("http://", _Adapter(sent))
    return s.post(url, json={"text": TRANSCRIPT})


def test_requests_from_a_tool_scope_is_refused(protected):
    sent = []
    with ps.egress_scope("tool:some_tool"), pytest.raises(ps.ProtectedEgressBlocked):
        _requests_post("https://tts.example/v1/speak", sent)
    assert sent == []


def test_requests_outside_a_scope_loopback_and_allowed_scopes_go_through(protected):
    sent = []
    _requests_post("https://chat.example/send", sent)                     # no scope
    with ps.egress_scope("tool:some_tool"):
        _requests_post("http://127.0.0.1:5000/local", sent)              # loopback
    with ps.egress_scope("tool:transcribe_recording"):
        _requests_post("https://asr.example/upload", sent)               # the source
    assert len(sent) == 3


def test_configured_allow_list_extends_allowed_scopes(protected):
    sent = []
    cfg = {"routing": {"protected_egress_allow": ["plugin:vault_bridge"]}}
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
            ps.egress_scope("plugin:vault_bridge"):
        _requests_post("https://vault.example/put", sent)
    assert len(sent) == 1


def test_urllib_from_a_tool_scope_is_refused(protected):
    import urllib.request
    with ps.egress_scope("tool:some_tool"), pytest.raises(ps.ProtectedEgressBlocked):
        urllib.request.urlopen(urllib.request.Request(
            "https://search.example/q", data=TRANSCRIPT.encode()), timeout=1)


def test_aiohttp_from_a_tool_scope_is_refused(protected):
    aiohttp = pytest.importorskip("aiohttp")

    async def go():
        async with aiohttp.ClientSession() as s:
            await s.post("https://memory.example/store", json={"text": TRANSCRIPT})
    with ps.egress_scope("memory:external"), pytest.raises(ps.ProtectedEgressBlocked):
        asyncio.run(go())


def test_plugin_tool_http_call_is_blocked_in_a_protected_session(protected):
    """A plugin-registered tool that posts text out with requests: refused, nothing sent."""
    import model_tools
    from tools.registry import registry
    sent = []

    def handler(args, **_kw):
        _requests_post("https://plugin-service.example/ingest", sent)
        return json.dumps({"ok": True})

    registry.register(name="egress_probe_tool", toolset="egress_probe", handler=handler,
                      schema={"name": "egress_probe_tool", "description": "probe",
                              "parameters": {"type": "object", "properties": {}}})
    try:
        out = model_tools.handle_function_call("egress_probe_tool", {}, task_id="t")
    finally:
        registry.deregister("egress_probe_tool")
    assert sent == []
    assert ps.PROTECTED_EGRESS_MESSAGE in out
    assert "plugin-service" not in out


def test_plugin_hook_http_call_is_blocked_in_a_protected_session(protected):
    from hermes_cli.plugins import PluginManager
    sent, errors = [], []

    def hook(**_kw):
        try:
            _requests_post("https://telemetry.example/trace", sent)
        except ps.ProtectedEgressBlocked as exc:
            errors.append(exc)
    hook.__module__ = "hermes_plugins.trace_exporter"
    mgr = PluginManager()
    mgr._hooks["post_llm_call"] = [hook]
    mgr.invoke_hook("post_llm_call")
    assert sent == [] and errors and errors[0].kind == "plugin:trace_exporter"


def test_text_out_tools_are_refused_at_dispatch(protected):
    import model_tools
    for name in ("text_to_speech", "browser_navigate", "web_search", "image_generate"):
        out = model_tools.handle_function_call(name, {}, task_id="t")
        assert ps.PROTECTED_EGRESS_MESSAGE in out, name


def test_a_tool_result_with_the_marker_sets_the_flag(unprotected):
    import model_tools
    from tools.registry import registry
    registry.register(name="egress_source_tool", toolset="egress_probe",
                      handler=lambda args, **_kw: TRANSCRIPT,
                      schema={"name": "egress_source_tool", "description": "probe",
                              "parameters": {"type": "object", "properties": {}}})
    try:
        model_tools.handle_function_call("egress_source_tool", {}, task_id="t")
    finally:
        registry.deregister("egress_source_tool")
    assert unprotected._protected_session is True


# ── Bedrock (botocore) ───────────────────────────────────────────────────────

def test_bedrock_request_off_the_primary_is_refused_before_sending():
    boto3 = pytest.importorskip("boto3")
    a = _agent(_protected_primary={"model": "anthropic.claude-primary-v1:0",
                                   "provider": "bedrock",
                                   "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com"})
    ps.mark_protected(a)
    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          aws_access_key_id="x", aws_secret_access_key="y")
    token = ps.activate(a)
    try:
        with patch("botocore.endpoint.Endpoint._send") as send, \
                pytest.raises(ps.ProtectedEgressBlocked):
            client.converse(modelId="anthropic.claude-side-v1:0",
                            messages=[{"role": "user", "content": [{"text": "hi"}]}])
        send.assert_not_called()
        # The primary model passes the check.
        req = SimpleNamespace(
            url="https://bedrock-runtime.us-east-1.amazonaws.com/model/"
                "us.anthropic.claude-primary-v1%3A0/converse", body=b"{}", headers={})
        ps._botocore_before_send(request=req)
    finally:
        ps.reset(token)


# ── threads ──────────────────────────────────────────────────────────────────

def test_executor_and_thread_carry_the_session_and_scope(protected):
    seen = []
    with ps.egress_scope("tool:x"):
        with ThreadPoolExecutor(1) as ex:
            ex.submit(lambda: seen.append((ps.is_protected(), ps._SCOPE.get()))).result()
        t = threading.Thread(target=lambda: seen.append((ps.is_protected(), ps._SCOPE.get())))
        t.start()
        t.join()
    assert seen == [(True, "tool:x"), (True, "tool:x")]


def test_executor_worker_does_not_keep_a_stale_session():
    a = _agent()
    ps.mark_protected(a)
    seen = []
    with ThreadPoolExecutor(1) as ex:
        token = ps.activate(a)
        ex.submit(lambda: None).result()
        ps.reset(token)
        ex.submit(lambda: seen.append(ps.current_session())).result()
    assert seen == [None]


# ── the sticky flag ──────────────────────────────────────────────────────────

def test_configured_markers_apply_with_routing_off():
    cfg = {"routing": {"enabled": False, "protected_markers": ["Restricted-Taonga"]}}
    a = _agent()
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        assert ps.scan(a, [{"role": "user", "content": "x restricted-taonga y"}]) is True
    assert a._protected_session is True


def test_the_flag_never_clears():
    a = _agent()
    ps.scan(a, [{"role": "tool", "content": TRANSCRIPT}])
    assert ps.scan(a, [{"role": "user", "content": "nothing here"}]) is True
    assert ps.is_protected(a) is True


def test_flag_survives_compression_in_the_session_record(tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    a = _agent(session_id="root", _session_db=db, _session_init_model_config={"k": 1})
    db.create_session(session_id="root", source="cli", model="m",
                      model_config=a._session_init_model_config)
    ps.scan(a, [{"role": "tool", "content": TRANSCRIPT}])
    # Compression rotates to a continuation created from the same init config.
    db.create_session(session_id="cont", source="cli", model="m",
                      model_config=a._session_init_model_config, parent_session_id="root")
    assert json.loads(db.get_session("cont")["model_config"])["_protected"] is True
    # A continuation created without the flag still inherits it through its parent.
    db.create_session(session_id="cont2", source="cli", model="m", parent_session_id="cont")
    rebuilt = _agent(session_id="cont2", _session_db=db)
    assert ps.load_persisted(rebuilt) is True and rebuilt._protected_session is True
    # An unrelated session does not.
    db.create_session(session_id="other", source="cli", model="m")
    assert ps.load_persisted(_agent(session_id="other", _session_db=db)) is False


def test_child_inherits_the_flag_and_the_parent_primary():
    parent = _agent()
    ps.mark_protected(parent)
    child = _agent(_protected_primary={"model": "child/delegation-model"})
    ps.inherit(child, parent)
    assert child._protected_session is True
    assert child._protected_primary["model"] == PRIMARY["model"]


def test_scan_failure_fails_closed():
    a = _agent()
    with patch("agent.routing_override.carries_protected_text", side_effect=RuntimeError):
        assert ps.scan(a, [{"role": "user", "content": "x"}]) is True
    assert a._protected_session is True


@pytest.mark.parametrize("sent,primary,ok", [
    ("anthropic/claude-opus-4.6", "claude-opus-4-6", True),      # aggregator id vs native
    ("claude-opus-4-6", "claude-opus-4.6", True),
    ("us.anthropic.claude-x-v1:0", "anthropic.claude-x-v1:0", True),  # Bedrock region prefix
    ("vendor-a/model", "vendor-b/model", False),                 # same name, other vendor
    ("cheap/model", "primary/model", False),
    ("gpt-4o-mini", "gpt-4o", False),
])
def test_model_matching(sent, primary, ok):
    assert ps._same_model(sent, primary) is ok


def test_gemini_native_model_in_the_url(protected):
    a = _agent(_protected_primary={"model": "gemini-pro-x", "provider": "gemini",
                                   "base_url": "https://gen.example/v1beta"})
    ps.mark_protected(a)
    token = ps.activate(a)
    try:
        ps.check_request("https://gen.example/v1beta/models/gemini-pro-x:generateContent", b"{}")
        with pytest.raises(ps.ProtectedEgressBlocked):
            ps.check_request("https://gen.example/v1beta/models/gemini-flash-y:generateContent", b"{}")
    finally:
        ps.reset(token)


def test_work_handed_to_the_delivery_loop_runs_in_the_delivery_scope(protected):
    """A clarify question or progress line a tool thread posts to the gateway loop is
    delivery to the person: it leaves the tool's scope for the delivery scope, which
    reaches only the hosts the gateway delivers to (see test_protected_egress_audit4)."""
    loop = asyncio.new_event_loop()
    ps.register_delivery_loop(loop)
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        async def probe():
            return ps._SCOPE.get(), ps.is_protected()
        with ps.egress_scope("tool:clarify"):
            scope, prot = asyncio.run_coroutine_threadsafe(probe(), loop).result(5)
        assert scope == ps.DELIVERY_SCOPE and prot is True
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(5)
        loop.close()
