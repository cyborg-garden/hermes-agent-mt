"""Protected-session guard follow-ups (after #237).

* F1: ``read_file`` / ``search_files`` run on the terminal backend. On a backend
  that is not this machine (ssh, modal, daytona, singularity, a remote Docker
  daemon) they would read protected text off-box, so they are refused.
* F2: the ``codex_app_server`` runtime hands the whole turn to a codex
  subprocess, outside every hook here, so a protected session's turn is refused
  before the subprocess is spawned or reused.
* F3: ``PluginContext.dispatch_tool`` goes through the same allowlist. The one
  exception is the transcription tool handing its result to the vault
  (``ovenmitt_vault_send`` dispatched from inside ``transcribe_recording``).
* F4: a model-called ``ovenmitt_vault_send`` stays refused.
* F5: ``tool_search`` / ``tool_describe`` / ``tool_call`` stay refused.

No real transcript text appears here: the session is marked protected directly.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import protected_session as ps


class Agent:
    def __init__(self, **kw):
        self.session_id = "s-followups"
        self._session_db = None
        self._session_init_model_config = {}
        self._protected_session = False
        self._protected_primary = {"model": "p/m", "provider": "custom",
                                   "base_url": "http://127.0.0.1:9/v1"}
        self._cached_system_prompt = ""
        self.ephemeral_system_prompt = None
        self.prefill_messages = []
        self._session_messages = []
        self.__dict__.update(kw)


@pytest.fixture
def protected():
    a = Agent()
    tok = ps.activate(a)
    ps.mark_protected(a, "test")
    try:
        yield a
    finally:
        ps.reset(tok)


@pytest.fixture
def unprotected():
    a = Agent()
    tok = ps.activate(a)
    try:
        yield a
    finally:
        ps.reset(tok)


@pytest.fixture
def backend(monkeypatch):
    """Select a terminal backend and keep the active-environment table clean."""
    from tools import terminal_tool

    monkeypatch.delenv("DOCKER_HOST", raising=False)
    saved = dict(terminal_tool._active_environments)
    terminal_tool._active_environments.clear()

    def select(env_type: str):
        monkeypatch.setenv("TERMINAL_ENV", env_type)

    try:
        yield select
    finally:
        terminal_tool._active_environments.clear()
        terminal_tool._active_environments.update(saved)


# ── F1: file tools on an off-box terminal backend ──────────────────────────

@pytest.mark.parametrize("env_type", ["ssh", "modal", "daytona", "singularity", "vercel"])
@pytest.mark.parametrize("tool", ["read_file", "search_files"])
def test_file_tools_refused_on_an_off_box_backend(protected, backend, env_type, tool):
    backend(env_type)
    assert ps.refuse_tool_if_protected(tool, {"path": "notes.txt"}, task_id="t") \
        == ps.PROTECTED_EGRESS_MESSAGE


@pytest.mark.parametrize("env_type", ["local", "docker"])
@pytest.mark.parametrize("tool", ["read_file", "search_files"])
def test_file_tools_allowed_on_this_machine(protected, backend, env_type, tool):
    backend(env_type)
    assert ps.refuse_tool_if_protected(tool, {"path": "notes.txt"}, task_id="t") is None


@pytest.mark.parametrize("docker_host", ["tcp://10.0.0.5:2376", "ssh://builder@remote"])
def test_file_tools_refused_on_a_remote_docker_daemon(protected, backend, monkeypatch, docker_host):
    backend("docker")
    monkeypatch.setenv("DOCKER_HOST", docker_host)
    assert ps.refuse_tool_if_protected("read_file", {"path": "x"}, task_id="t") \
        == ps.PROTECTED_EGRESS_MESSAGE


def test_file_tools_allowed_on_a_local_docker_socket(protected, backend, monkeypatch):
    backend("docker")
    monkeypatch.setenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    assert ps.refuse_tool_if_protected("read_file", {"path": "x"}, task_id="t") is None


def test_an_active_remote_environment_wins_over_a_local_config(protected, backend):
    from tools import terminal_tool

    backend("local")

    class SSHEnvironment:  # what a live ssh sandbox looks like by class
        pass

    key = terminal_tool._resolve_container_task_id("t")
    terminal_tool._active_environments[key] = SSHEnvironment()
    assert ps.refuse_tool_if_protected("search_files", {"pattern": "x"}, task_id="t") \
        == ps.PROTECTED_EGRESS_MESSAGE


def test_an_unknown_backend_config_fails_closed(protected, backend, monkeypatch):
    from tools import terminal_tool

    backend("local")

    def boom():
        raise ValueError("bad terminal config")

    monkeypatch.setattr(terminal_tool, "_get_env_config", boom)
    assert ps.refuse_tool_if_protected("read_file", {"path": "x"}, task_id="t") \
        == ps.PROTECTED_EGRESS_MESSAGE


def test_off_box_backend_is_irrelevant_outside_a_protected_session(unprotected, backend):
    backend("ssh")
    assert ps.refuse_tool_if_protected("read_file", {"path": "x"}, task_id="t") is None


def test_other_allowed_tools_are_unaffected_by_the_backend(protected, backend):
    backend("ssh")
    for tool in ("clarify", "todo", "skill_view", "skills_list", "session_search",
                 "read_document"):
        assert ps.refuse_tool_if_protected(tool, {}, task_id="t") is None, tool


def test_handle_function_call_refuses_read_file_over_ssh(protected, backend, monkeypatch):
    import model_tools
    from tools.registry import registry

    backend("ssh")
    entry = registry.get_entry("read_file")
    assert entry is not None
    ran = []
    monkeypatch.setattr(entry, "handler", lambda args, **_kw: ran.append(1) or "{}")
    out = model_tools.handle_function_call("read_file", {"path": "x"}, task_id="t")
    assert ran == [] and ps.PROTECTED_EGRESS_MESSAGE in out


def test_pre_tool_block_refuses_search_files_over_modal(protected, backend):
    from hermes_cli.plugins import resolve_pre_tool_block

    backend("modal")
    assert resolve_pre_tool_block("search_files", {"pattern": "x"}, task_id="t") \
        == ps.PROTECTED_EGRESS_MESSAGE


# ── F2: codex app-server runtime ───────────────────────────────────────────

def _codex_agent(protected_flag: bool):
    agent = MagicMock()
    agent._protected_session = protected_flag
    agent._turn_protected_extras = ()
    agent._cached_system_prompt = ""
    agent.ephemeral_system_prompt = None
    agent.prefill_messages = []
    agent._routing_protected_markers = ()
    agent.api_mode = "codex_app_server"
    agent._codex_session = MagicMock()
    agent._codex_session.run_turn.side_effect = AssertionError("codex must not run")
    return agent


def test_codex_app_server_turn_refused_in_a_protected_session(monkeypatch):
    from agent import codex_runtime
    from agent.transports import codex_app_server_session as cas

    spawned = []
    monkeypatch.setattr(cas, "CodexAppServerSession",
                        lambda *a, **k: spawned.append(1) or MagicMock())
    agent = _codex_agent(True)
    agent._codex_session = None
    messages = [{"role": "user", "content": "hello"}]
    out = codex_runtime.run_codex_app_server_turn(
        agent, user_message="hello", original_user_message="hello",
        messages=messages, effective_task_id="t")
    assert spawned == []
    assert out["completed"] is False and out["api_calls"] == 0
    assert out["final_response"] == ps.PROTECTED_CODEX_RUNTIME_MESSAGE
    assert "hello" not in json.dumps(out.get("error"))


def test_codex_app_server_turn_refused_with_an_existing_session():
    from agent import codex_runtime

    agent = _codex_agent(True)
    out = codex_runtime.run_codex_app_server_turn(
        agent, user_message="x", original_user_message="x",
        messages=[{"role": "user", "content": "x"}], effective_task_id="t")
    agent._codex_session.run_turn.assert_not_called()
    assert out["final_response"] == ps.PROTECTED_CODEX_RUNTIME_MESSAGE


def test_codex_app_server_turn_refused_when_this_turns_plugin_context_is_protected():
    from agent import codex_runtime
    from agent.routing_override import DEFAULT_PROTECTED_MARKERS

    agent = _codex_agent(False)
    agent._session_db = None
    agent._session_init_model_config = {}
    # Plugin context for this turn carries the marker (synthetic, no real text).
    agent._turn_protected_extras = (f"synthetic placeholder {DEFAULT_PROTECTED_MARKERS[0]}", None)
    out = codex_runtime.run_codex_app_server_turn(
        agent, user_message="x", original_user_message="x",
        messages=[{"role": "user", "content": "x"}], effective_task_id="t")
    agent._codex_session.run_turn.assert_not_called()
    assert out["final_response"] == ps.PROTECTED_CODEX_RUNTIME_MESSAGE
    assert agent._protected_session is True


def test_codex_app_server_turn_runs_normally_when_unprotected():
    from agent import codex_runtime

    agent = _codex_agent(False)
    agent._session_db = None
    agent._session_init_model_config = {}
    session = agent._codex_session
    session.run_turn.side_effect = RuntimeError("reached codex")
    out = codex_runtime.run_codex_app_server_turn(
        agent, user_message="x", original_user_message="x",
        messages=[{"role": "user", "content": "x"}], effective_task_id="t")
    assert out["final_response"] != ps.PROTECTED_CODEX_RUNTIME_MESSAGE
    session.run_turn.assert_called_once()


# ── F3 / F4: plugin dispatch_tool and the vault hand-off ───────────────────

def _ctx(name: str = "some_plugin"):
    from hermes_cli.plugins import PluginContext, PluginManifest

    manager = SimpleNamespace(_cli_ref=None)
    return PluginContext(PluginManifest(name=name), manager)


@pytest.fixture
def fake_tools():
    import model_tools  # noqa: F401  (registers the built-in tools)
    from tools.registry import registry

    ran: list = []
    names = ("ovenmitt_vault_send", "followup_plugin_tool")
    added = []
    for name in names:
        if registry.get_entry(name) is None:
            registry.register(
                name=name, toolset="followups",
                handler=(lambda n: (lambda args, **_kw: ran.append(n) or "{}"))(name),
                schema={"name": name, "description": "x",
                        "parameters": {"type": "object", "properties": {}}})
            added.append(name)
    try:
        yield ran
    finally:
        for name in added:
            registry.deregister(name)


def test_dispatch_tool_refuses_a_tool_off_the_allowlist(protected, fake_tools):
    out = _ctx().dispatch_tool("followup_plugin_tool", {})
    assert fake_tools == [] and ps.PROTECTED_EGRESS_MESSAGE in out


def test_dispatch_tool_refuses_terminal(protected):
    from tools.registry import registry

    entry = registry.get_entry("terminal")
    if entry is None:
        pytest.skip("terminal not registered")
    out = _ctx().dispatch_tool("terminal", {"command": "true"})
    assert ps.PROTECTED_EGRESS_MESSAGE in out


def test_dispatch_tool_runs_an_allowed_tool(protected, monkeypatch):
    from tools.registry import registry

    seen = []
    monkeypatch.setattr(registry, "dispatch",
                        lambda name, args, **kw: seen.append(name) or "{}")
    assert _ctx().dispatch_tool("todo", {}) == "{}"
    assert seen == ["todo"]


def test_dispatch_tool_applies_the_backend_rule(protected, backend, monkeypatch):
    from tools.registry import registry

    backend("ssh")
    monkeypatch.setattr(registry, "dispatch", lambda *a, **k: pytest.fail("ran"))
    assert ps.PROTECTED_EGRESS_MESSAGE in _ctx().dispatch_tool("read_file", {"path": "x"})


def test_dispatch_tool_refuses_for_a_protected_parent_agent_outside_a_turn(fake_tools):
    parent = Agent()
    ps.mark_protected(parent, "test")
    assert ps.current_session() is None or not ps.is_protected()
    out = _ctx().dispatch_tool("followup_plugin_tool", {}, parent_agent=parent)
    assert fake_tools == [] and ps.PROTECTED_EGRESS_MESSAGE in out


def test_dispatch_tool_unaffected_outside_a_protected_session(unprotected, fake_tools):
    _ctx().dispatch_tool("followup_plugin_tool", {})
    assert fake_tools == ["followup_plugin_tool"]


def test_vault_send_from_inside_the_transcription_tool_still_works(protected, fake_tools):
    with ps.egress_scope("tool:transcribe_recording"), ps.tool_handler("transcribe_recording"):
        _ctx("papareo_transcribe").dispatch_tool("ovenmitt_vault_send", {"label": "x"})
    assert fake_tools == ["ovenmitt_vault_send"]


def test_vault_send_dispatched_outside_the_transcription_tool_is_refused(protected, fake_tools):
    out = _ctx("papareo_transcribe").dispatch_tool("ovenmitt_vault_send", {"label": "x"})
    assert fake_tools == [] and ps.PROTECTED_EGRESS_MESSAGE in out


def test_vault_send_from_a_plugin_hook_scope_is_refused(protected, fake_tools):
    with ps.egress_scope("tool:transcribe_recording"), ps.tool_handler("transcribe_recording"):
        with ps.egress_scope("plugin:some_plugin"):
            out = _ctx().dispatch_tool("ovenmitt_vault_send", {"label": "x"})
    assert fake_tools == [] and ps.PROTECTED_EGRESS_MESSAGE in out


def test_the_transcription_scope_does_not_open_other_tools(protected, fake_tools):
    with ps.egress_scope("tool:transcribe_recording"), ps.tool_handler("transcribe_recording"):
        out = _ctx("papareo_transcribe").dispatch_tool("followup_plugin_tool", {})
    assert fake_tools == [] and ps.PROTECTED_EGRESS_MESSAGE in out


def test_model_called_vault_send_stays_refused(protected, fake_tools):
    import model_tools
    from hermes_cli.plugins import resolve_pre_tool_block

    out = model_tools.handle_function_call("ovenmitt_vault_send", {"label": "x"}, task_id="t")
    assert fake_tools == [] and ps.PROTECTED_EGRESS_MESSAGE in out
    assert resolve_pre_tool_block("ovenmitt_vault_send", {"label": "x"}) \
        == ps.PROTECTED_EGRESS_MESSAGE


def test_model_called_vault_send_refused_even_inside_the_transcription_scope(protected, fake_tools):
    import model_tools

    with ps.egress_scope("tool:transcribe_recording"):
        out = model_tools.handle_function_call("ovenmitt_vault_send", {"label": "x"}, task_id="t")
    assert fake_tools == [] and ps.PROTECTED_EGRESS_MESSAGE in out


# ── F5: tool search bridge ─────────────────────────────────────────────────

@pytest.mark.parametrize("tool", ["tool_search", "tool_describe", "tool_call"])
def test_tool_search_bridge_is_refused(protected, tool):
    assert ps.refuse_tool_if_protected(tool, {"query": "x"}) == ps.PROTECTED_EGRESS_MESSAGE


@pytest.mark.parametrize("tool", ["tool_search", "tool_describe"])
def test_tool_search_bridge_refused_before_the_inline_catalog_read(protected, monkeypatch, tool):
    import model_tools
    from tools import tool_search as ts

    monkeypatch.setattr(ts, "dispatch_tool_search", lambda *a, **k: pytest.fail("ran"))
    monkeypatch.setattr(ts, "dispatch_tool_describe", lambda *a, **k: pytest.fail("ran"))
    out = model_tools.handle_function_call(tool, {"query": "x", "name": "x"}, task_id="t")
    assert ps.PROTECTED_EGRESS_MESSAGE in out


# ── audit follow-ups ───────────────────────────────────────────────────────

def _docker_context(tmp_path, name: str, host: str) -> None:
    import hashlib

    meta = tmp_path / "contexts" / "meta" / hashlib.sha256(name.encode()).hexdigest()
    meta.mkdir(parents=True)
    (meta / "meta.json").write_text(json.dumps(
        {"Name": name, "Endpoints": {"docker": {"Host": host}}}))


@pytest.mark.parametrize("via", ["env", "config"])
def test_file_tools_refused_on_a_remote_docker_context(protected, backend, monkeypatch,
                                                       tmp_path, via):
    backend("docker")
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    _docker_context(tmp_path, "remote-builder", "ssh://builder@remote")
    if via == "env":
        monkeypatch.setenv("DOCKER_CONTEXT", "remote-builder")
    else:
        monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
        (tmp_path / "config.json").write_text(json.dumps({"currentContext": "remote-builder"}))
    assert ps.refuse_tool_if_protected("read_file", {"path": "x"}, task_id="t") \
        == ps.PROTECTED_EGRESS_MESSAGE


def test_file_tools_refused_on_an_unknown_docker_context(protected, backend, monkeypatch,
                                                         tmp_path):
    backend("docker")
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    monkeypatch.setenv("DOCKER_CONTEXT", "nowhere")
    assert ps.refuse_tool_if_protected("read_file", {"path": "x"}, task_id="t") \
        == ps.PROTECTED_EGRESS_MESSAGE


def test_file_tools_allowed_on_a_local_docker_context(protected, backend, monkeypatch, tmp_path):
    backend("docker")
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    _docker_context(tmp_path, "desktop-linux", "unix:///tmp/docker.sock")
    (tmp_path / "config.json").write_text(json.dumps({"currentContext": "desktop-linux"}))
    assert ps.refuse_tool_if_protected("read_file", {"path": "x"}, task_id="t") is None


@pytest.fixture
def transcriber(monkeypatch, fake_tools):
    """A stand-in transcribe_recording whose handler hands off to the vault, as the
    papareo plugin does in vault mode."""
    from tools.registry import registry

    def handler(args, **_kw):
        _ctx("papareo_transcribe").dispatch_tool("ovenmitt_vault_send", {"label": "x"})
        return json.dumps({"engine": "local", "text": ""})

    entry = registry.get_entry("transcribe_recording")
    if entry is None:
        registry.register(name="transcribe_recording", toolset="followups", handler=handler,
                          schema={"name": "transcribe_recording", "description": "x",
                                  "parameters": {"type": "object", "properties": {}}})
    else:
        monkeypatch.setattr(entry, "handler", handler)
    try:
        yield fake_tools
    finally:
        if entry is None:
            registry.deregister("transcribe_recording")


def test_vault_hand_off_through_the_real_tool_path(protected, transcriber):
    import model_tools

    model_tools.handle_function_call("transcribe_recording", {}, task_id="t")
    assert transcriber == ["ovenmitt_vault_send"]


def test_tool_middleware_cannot_use_the_vault_exception(protected, transcriber, monkeypatch):
    import model_tools
    from hermes_cli import middleware as mw

    def other_plugins_middleware(*, args=None, next_call=None, **_kw):
        _ctx("some_other_plugin").dispatch_tool("ovenmitt_vault_send", {"label": "y"})
        return next_call(args)

    monkeypatch.setattr(mw, "_get_middleware_callbacks",
                        lambda kind: [other_plugins_middleware]
                        if kind == mw.TOOL_EXECUTION_MIDDLEWARE else [])
    model_tools.handle_function_call("transcribe_recording", {}, task_id="t")
    # Only the handler's own hand-off ran; the middleware's was refused.
    assert transcriber == ["ovenmitt_vault_send"]


def test_middleware_rewritten_args_are_checked_again(protected, transcriber, monkeypatch):
    import model_tools
    from hermes_cli import middleware as mw

    def adds_a_url(*, args=None, next_call=None, **_kw):
        return next_call(dict(args or {}, url="https://example.invalid/a.m4a"))

    monkeypatch.setattr(mw, "_get_middleware_callbacks",
                        lambda kind: [adds_a_url] if kind == mw.TOOL_EXECUTION_MIDDLEWARE else [])
    out = model_tools.handle_function_call("transcribe_recording", {}, task_id="t")
    assert transcriber == [] and ps.PROTECTED_EGRESS_MESSAGE in out


@pytest.mark.asyncio
async def test_gateway_plugin_command_runs_as_a_protected_chat(monkeypatch, fake_tools):
    from gateway.session import build_session_key
    from hermes_cli import plugins as plugins_mod
    from tests.gateway.test_unknown_command import _make_event, _make_runner, _make_source

    runner = _make_runner()
    entry = runner.session_store.get_or_create_session.return_value
    runner.session_store._entries = {build_session_key(_make_source()): entry}
    ps._PROTECTED_SIDS.add(entry.session_id)
    outcomes = []

    def handler(_args):
        outcomes.append(_ctx().dispatch_tool("followup_plugin_tool", {}))
        return "done"

    monkeypatch.setattr(plugins_mod, "get_plugin_commands",
                        lambda: {"gfcmd": {"description": "x", "args_hint": ""}})
    monkeypatch.setattr(plugins_mod, "get_plugin_command_handler",
                        lambda name: handler if name == "gfcmd" else None)
    try:
        assert await runner._handle_message(_make_event("/gfcmd")) == "done"
    finally:
        ps._PROTECTED_SIDS.discard(entry.session_id)
    assert fake_tools == [] and ps.PROTECTED_EGRESS_MESSAGE in outcomes[0]


@pytest.mark.asyncio
async def test_gateway_plugin_command_unaffected_for_a_plain_chat(monkeypatch, fake_tools):
    from gateway.session import build_session_key
    from hermes_cli import plugins as plugins_mod
    from tests.gateway.test_unknown_command import _make_event, _make_runner, _make_source

    runner = _make_runner()
    entry = runner.session_store.get_or_create_session.return_value
    runner.session_store._entries = {build_session_key(_make_source()): entry}

    monkeypatch.setattr(plugins_mod, "get_plugin_commands",
                        lambda: {"gfcmd": {"description": "x", "args_hint": ""}})
    monkeypatch.setattr(plugins_mod, "get_plugin_command_handler",
                        lambda name: (lambda _a: _ctx().dispatch_tool("followup_plugin_tool", {})
                                      and "done") if name == "gfcmd" else None)
    assert await runner._handle_message(_make_event("/gfcmd")) == "done"
    assert fake_tools == ["followup_plugin_tool"]
