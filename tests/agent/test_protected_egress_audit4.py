"""Audit round 4 of the protected-session egress guard, as regression tests.

Each test asserts the fixed behaviour: before the fix every one of them failed.
The papareo_transcribe plugin strings in fixtures/papareo_plugin/strings.json are
copied verbatim from that plugin (its "next" instructions and SKILL.md). They name
the Kaitiakitanga License but carry no transcript.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import threading
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from agent import protected_session as ps

PLUGIN = json.loads((pathlib.Path(__file__).parent / "fixtures" / "papareo_plugin"
                     / "strings.json").read_text(encoding="utf-8"))
FOOTER = ("This text transcribed by Papa Reo API. Please be sure to comply with the terms "
          "of the licence in use of this material.")
PRIMARY = {"model": "primary/model", "provider": "custom",
           "base_url": "http://127.0.0.1:9/v1", "api_key": "k", "api_mode": "chat_completions"}


def _agent(**kw):
    a = SimpleNamespace(session_id="s1", _session_db=None, _session_init_model_config={},
                        _protected_session=False, _protected_primary=dict(PRIMARY),
                        _cached_system_prompt="", ephemeral_system_prompt=None,
                        prefill_messages=[])
    a.__dict__.update(kw)
    return a


@pytest.fixture
def running():
    a = _agent()
    token = ps.activate(a)
    try:
        yield a
    finally:
        ps.reset(token)


@pytest.fixture
def protected(running):
    ps.mark_protected(running, "test")
    return running


# ── B1/B2: the plugin's own words never mark a session ───────────────────────

@pytest.mark.parametrize("const", ["NEXT_PAPAREO", "NEXT_PAPAREO_DIRECT"])
def test_B1_a_papareo_handle_result_leaves_the_session_unprotected(running, const):
    result = json.dumps({"sent": True, "engine": "papareo", "transcribed_by": "Papa Reo",
                         "handle": "om_handle_x", "filename": "transcript.txt",
                         "next": PLUGIN[const]}, ensure_ascii=False)
    ps.scan_running(result, source="tool_result")
    assert running._protected_session is False
    assert ps.refuse_tool_if_protected("web_search") is None


def test_B2_the_papareo_skill_leaves_the_session_unprotected(running):
    ps.scan_running(json.dumps({"success": True, "content": PLUGIN["SKILL_MD"]}),
                    source="tool_result")
    assert running._protected_session is False


def test_B2_skill_soul_memory_and_context_files_in_the_system_prompt_do_not_mark():
    prompt = "\n\n".join([PLUGIN["SKILL_MD"], PLUGIN["NEXT_PAPAREO"],
                          "SOUL: we follow the Kaitiakitanga License for Papa Reo work.",
                          "MEMORY: transcripts come under the Kaitiakitanga licence."])
    a = _agent(_cached_system_prompt=prompt, ephemeral_system_prompt=PLUGIN["SKILL_MD"])
    assert ps.scan(a, [{"role": "user", "content": "send me the recording notes"}]) is False
    assert a._protected_session is False


def test_the_transcript_footer_still_marks(running):
    ps.scan_running("placeholder words.\n\n" + FOOTER, source="tool_result")
    assert running._protected_session is True


def test_core_and_routing_plugin_markers_match_and_exclude_the_licence_name():
    from agent.routing_override import DEFAULT_PROTECTED_MARKERS as core
    from plugins.intelligent_routing.config import DEFAULT_PROTECTED_MARKERS as plugin
    assert core == plugin
    assert not any("kaitiakitanga" in m for m in core)


# ── A1/A3: subprocess and MCP tools are refused by name ──────────────────────

@pytest.mark.parametrize("tool", ["terminal", "execute_code", "mcp_github_search"])
def test_A1_A3_subprocess_and_mcp_tools_are_refused(protected, tool):
    import model_tools
    out = model_tools.handle_function_call(tool, {"command": "true"}, task_id="t")
    assert ps.PROTECTED_EGRESS_MESSAGE in out


def test_A3_any_tool_in_an_mcp_toolset_is_refused(protected):
    from tools.registry import registry
    registry.register(name="remote_search", toolset="mcp-remote",
                      handler=lambda args, **_kw: "{}",
                      schema={"name": "remote_search", "description": "x",
                              "parameters": {"type": "object", "properties": {}}})
    try:
        assert ps.refuse_tool_if_protected("remote_search") == ps.PROTECTED_EGRESS_MESSAGE
    finally:
        registry.deregister("remote_search")


def test_subprocess_tools_still_run_outside_a_protected_session(running):
    assert ps.refuse_tool_if_protected("terminal") is None


# ── A2: exporter plugins get no payload in a protected session ───────────────

def _manager_with(module):
    from hermes_cli.plugins import PluginManager
    seen = []

    def hook(**kw):
        seen.append(kw)
    hook.__module__ = module
    mgr = PluginManager()
    mgr._hooks["post_llm_call"] = [hook]
    return mgr, seen


@pytest.mark.parametrize("module", ["hermes_plugins.observability__langfuse",
                                    "hermes_plugins.observability__nemo_relay"])
def test_A2_exporter_hooks_are_not_called_in_a_protected_session(protected, module):
    mgr, seen = _manager_with(module)
    mgr.invoke_hook("post_llm_call", assistant_response="placeholder")
    assert seen == []


def test_A2_exporter_hooks_still_run_otherwise(running):
    mgr, seen = _manager_with("hermes_plugins.observability__langfuse")
    mgr.invoke_hook("post_llm_call", assistant_response="x")
    assert len(seen) == 1


# ── A4: the delivery hop only reaches hosts the gateway itself delivers to ───

def test_A4_delivery_hop_reaches_known_delivery_hosts_only(protected):
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    ps.register_delivery_loop(loop)
    sent = []

    async def post(url):
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda r: sent.append(r.url.host) or httpx.Response(200))) as c:
            try:
                await c.post(url, content=b"reply")
                return "sent"
            except ps.ProtectedEgressBlocked:
                return "blocked"

    try:
        with ps.egress_scope("tool:clarify"):
            ok = asyncio.run_coroutine_threadsafe(
                post("https://discord.com/api/v10/channels/1/messages"), loop).result(5)
            bad = asyncio.run_coroutine_threadsafe(
                post("https://collector.example/"), loop).result(5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
    assert ok == "sent" and bad == "blocked"
    assert "collector.example" not in sent


# ── A5: a gateway hook outside a turn for a protected chat ───────────────────

def test_A5_hook_outside_a_turn_for_a_protected_chat_is_guarded(tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="discord", model="m")
    db.mark_session_protected("chat")
    assert ps.current_session() is None
    with ps.protected_chat(db, "chat"), ps.egress_scope("plugin:some_plugin"):
        with pytest.raises(ps.ProtectedEgressBlocked):
            ps.check_request("https://collector.example/", b"next message", {})
    db.create_session(session_id="plain", source="discord", model="m")
    with ps.protected_chat(db, "plain"), ps.egress_scope("plugin:some_plugin"):
        ps.check_request("https://collector.example/", b"next message", {})


# ── follow-ups ───────────────────────────────────────────────────────────────

def test_a_refused_request_is_not_retried_or_failed_over():
    from openai import APIConnectionError
    from agent.error_classifier import classify_api_error
    try:
        try:
            raise ps.ProtectedEgressBlocked("model")
        except ps.ProtectedEgressBlocked as inner:
            raise APIConnectionError(request=httpx.Request("POST", "http://x/")) from inner
    except APIConnectionError as exc:
        c = classify_api_error(exc)
    assert c.retryable is False and c.should_fallback is False
    assert c.message == ps.PROTECTED_EGRESS_MESSAGE


def test_cli_new_session_clears_the_flag():
    a = _agent(_session_init_model_config={"k": 1})
    ps.mark_protected(a)
    ps.reset_for_new_session(a)
    assert a._protected_session is False and "_protected" not in a._session_init_model_config


def test_reset_session_state_clears_the_flag():
    from run_agent import AIAgent
    a = _agent(_session_init_model_config={})
    ps.mark_protected(a)
    with patch.object(AIAgent, "_transition_context_engine_session"):
        try:
            AIAgent.reset_session_state(a)
        except AttributeError:
            pass  # later lines touch attributes the stub lacks; the reset ran first
    assert a._protected_session is False


def test_primary_without_endpoint_matches_its_provider_hosts_only():
    a = _agent(_protected_primary={"model": "anthropic.claude-p-v1:0", "provider": "bedrock",
                                   "base_url": ""})
    ps.mark_protected(a)
    token = ps.activate(a)
    try:
        ps.check_request("https://bedrock-runtime.us-east-1.amazonaws.com/model/"
                         "anthropic.claude-p-v1:0/converse", b"{}")
        with pytest.raises(ps.ProtectedEgressBlocked):
            ps.check_request("https://other-host.example/model/anthropic.claude-p-v1:0/converse",
                             b"{}")
    finally:
        ps.reset(token)
