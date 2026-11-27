"""Guard: protected sessions use a default-deny tool allowlist.

Every registered tool is classified: it runs in a protected session only if it is
in ``PROTECTED_SESSION_TOOLS`` (agent/protected_egress_allowlist.py); every other
tool, including any registered later, is refused.
"""
from __future__ import annotations

import json

import pytest

from agent import protected_session as ps
from agent.protected_egress_allowlist import PROTECTED_SESSION_TOOLS


class Agent:
    def __init__(self, **kw):
        self.session_id = "s-allow"
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


def _registered():
    import model_tools  # noqa: F401  (registers the built-in tools)
    from tools.registry import registry
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered
        _ensure_plugins_discovered()
    except Exception:
        pass
    return sorted(registry.get_all_tool_names())


def test_every_registered_tool_is_classified(protected):
    names = _registered()
    assert names
    for name in names:
        refused = ps.refuse_tool_if_protected(name, {}) is not None
        assert refused is (name not in PROTECTED_SESSION_TOOLS), name


def test_the_allowlist_names_only_real_tools():
    from model_tools import _AGENT_LOOP_TOOLS
    names = set(_registered()) | set(_AGENT_LOOP_TOOLS) | {"transcribe_recording"}
    assert PROTECTED_SESSION_TOOLS <= names, PROTECTED_SESSION_TOOLS - names


def test_a_new_tool_is_refused_by_default(protected):
    import model_tools
    from tools.registry import registry
    ran = []
    registry.register(name="brand_new_tool", toolset="brand_new",
                      handler=lambda args, **_kw: ran.append(1) or "{}",
                      schema={"name": "brand_new_tool", "description": "x",
                              "parameters": {"type": "object", "properties": {}}})
    try:
        out = model_tools.handle_function_call("brand_new_tool", {}, task_id="t")
    finally:
        registry.deregister("brand_new_tool")
    assert ran == [] and ps.PROTECTED_EGRESS_MESSAGE in out


@pytest.mark.parametrize("tool", ["memory", "delegate_task", "cronjob", "skill_manage",
                                  "write_file", "patch", "kanban_create", "person_memory"])
def test_persisting_and_agent_loop_tools_are_refused(protected, tool):
    assert ps.refuse_tool_if_protected(tool, {}) == ps.PROTECTED_EGRESS_MESSAGE


def test_the_shared_pre_tool_gate_refuses_agent_loop_tools(protected):
    """Every dispatch site (registry, the agent-loop direct calls for memory and
    delegate_task, concurrent and sequential) calls resolve_pre_tool_block."""
    from hermes_cli.plugins import resolve_pre_tool_block
    assert resolve_pre_tool_block("memory", {"action": "add", "content": "x"}) \
        == ps.PROTECTED_EGRESS_MESSAGE
    assert resolve_pre_tool_block("read_file", {"path": "/tmp/x"}) is None


def test_allowed_tools_still_run_and_nothing_is_refused_unprotected():
    a = Agent()
    tok = ps.activate(a)
    try:
        assert ps.refuse_tool_if_protected("memory", {}) is None
        assert ps.refuse_tool_if_protected("cronjob", {}) is None
    finally:
        ps.reset(tok)


def test_memory_tool_itself_refuses_in_a_protected_session(protected, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.memory_tool import MemoryStore, memory_tool
    store = MemoryStore()
    store.load_from_disk()
    res = json.loads(memory_tool(action="add", target="memory", content="placeholder", store=store))
    assert res.get("success") is False
    fresh = MemoryStore()
    fresh.load_from_disk()
    assert "placeholder" not in (fresh.format_for_system_prompt("memory") or "")
