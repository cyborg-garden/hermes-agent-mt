"""delegation.backend: task-routing — each subagent's model from its task.

Unit checks on the provider, then the real path: the bundled plugin discovered
from a temp HERMES_HOME, real delegate_task, real route cleaning and credential
resolution; only child construction and run are stubbed so the test can read
what each child was handed.
"""
import json
import logging
import types
from pathlib import Path
from unittest.mock import patch

import hermes_yaml as yaml
import pytest

from plugins.intelligent_routing import delegation as deleg

REPO_PLUGINS = Path(__file__).resolve().parents[3] / "plugins"
PRIMARY = {"model": "moonshotai/kimi-k3.1-pro", "provider": "openrouter"}


@pytest.fixture
def cfg():
    with patch.object(deleg, "cheap_tier_target", return_value=("cheap/model", "openrouter")), \
            patch.object(deleg, "primary_route", return_value=dict(PRIMARY)), \
            patch.object(deleg, "top_level_fallback_chain",
                         return_value=[{"provider": "openrouter", "model": "fb/one"}]), \
            patch.object(deleg, "cheap_reasoning_effort", return_value=None):
        yield


def _route(goal, context="", config=None):
    return deleg.TaskRoutingDelegationProvider().route_task(
        {"goal": goal, "context": context, "index": 0}, config or {})


def test_cheap_task_gets_cheap_model_escalation_and_primary_first(cfg):
    route = _route("List every file under tools/ whose name starts with delegate_.")
    assert (route["model"], route["provider"]) == ("cheap/model", "openrouter")
    assert route["escalate"] is True
    assert route["fallback_providers"][0] == PRIMARY
    assert route["fallback_providers"][1] == {"provider": "openrouter", "model": "fb/one"}
    assert route["label"] == "cheap · lookup"
    assert "reasoning_effort" not in route  # glm-5.3-flash 400s with reasoning off: never forced


def test_premium_task_takes_the_default_route(cfg):
    config = {"model": "", "provider": "", "max_iterations": 50, "service_tier": "fast"}
    route = _route("Implement the escalation helper with tests.", config=config)
    assert route == {"model": "", "provider": "", "max_iterations": 50}


def test_context_can_make_a_cheap_goal_premium(cfg):
    assert "escalate" not in _route("Summarise the context.", context="We will publish this on the blog.")


def test_pinned_delegation_model_is_the_escalation_target(cfg):
    route = _route("Count the test files.", config={"model": "pinned/m", "provider": "openrouter"})
    assert route["fallback_providers"][0] == {"model": "pinned/m", "provider": "openrouter"}


def test_decision_is_logged_without_task_text(cfg, caplog):
    with caplog.at_level(logging.INFO):
        _route("Count the lines in GOALMARK.txt", context="CONTEXTMARK")
    assert '"scope": "subagent"' in caplog.text and '"task_type": "lookup"' in caplog.text
    assert "GOALMARK" not in caplog.text and "CONTEXTMARK" not in caplog.text


def test_protected_marker_in_context_keeps_the_child_premium(cfg):
    route = _route("Summarise the transcript in the context.",
                   context="...\nThis text transcribed by Papa Reo API.")
    assert "escalate" not in route and route.get("model") != "cheap/model"
    with patch.object(deleg, "protected_markers", return_value=("do-not-route",)):
        assert _route("List the files.", context="DO-NOT-ROUTE").get("model") != "cheap/model"


def test_escalate_off_is_honoured_for_children(cfg):
    with patch.object(deleg, "escalation_enabled", return_value=False):
        route = _route("Count the test files.")
    assert route["model"] == "cheap/model" and "escalate" not in route


def test_primary_without_provider_resolves_auto(cfg):
    with patch.object(deleg, "primary_route", return_value={"model": "m"}):
        assert _route("Count the test files.")["fallback_providers"][0] == {"model": "m", "provider": "auto"}


# ── the real spawn path ──

def _home(tmp_path, monkeypatch, delegation=None):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginManager

    home = tmp_path / ".hermes"
    (home / "plugins").mkdir(parents=True)
    (home / "plugins" / "intelligent_routing").symlink_to(REPO_PLUGINS / "intelligent_routing",
                                                         target_is_directory=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["intelligent_routing"]},
        "model": dict(PRIMARY, default=PRIMARY["model"]),
        "fallback_providers": [{"provider": "openrouter", "model": "fb/one"}],
        "routing": {"cheap_model": "cheap/model", "cheap_provider": "openrouter"},
        "delegation": {"backend": "task-routing"} if delegation is None else delegation,
    }), encoding="utf-8")
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "delegation").symlink_to(REPO_PLUGINS / "delegation", target_is_directory=True)
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    manager = PluginManager()
    manager.discover_and_load()
    return manager


def test_delegate_task_routes_each_child_by_task(tmp_path, monkeypatch):
    from tools import delegate_tool as dt

    built, children = [], []

    def build(**kwargs):
        built.append(kwargs)
        child = types.SimpleNamespace(tool_progress_callback=None, model=kwargs.get("model"))
        children.append(child)
        return child

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", build)
    monkeypatch.setattr(dt, "_run_single_child", lambda task_index, goal, child=None, parent_agent=None, **kw: {
        "task_index": task_index, "status": "completed", "summary": "ok", "api_calls": 1, "duration_seconds": 0})
    parent = types.SimpleNamespace(session_id="root", model=PRIMARY["model"], provider="openrouter",
                                   tool_progress_callback=None, _delegate_spinner=None, request_overrides={},
                                   _safe_print=lambda line: None)
    manager = _home(tmp_path, monkeypatch)
    try:
        out = json.loads(dt.delegate_task(tasks=[
            {"goal": "Grep the repo for 'routing_override' and return each match with file and line."},
            {"goal": "Refactor delegate_tool.py so routing is a separate module."},
        ], parent_agent=parent))
        assert [r["status"] for r in out["results"]] == ["completed", "completed"]
        cheap, premium = built
        assert cheap["model"] == "cheap/model"
        assert cheap["routing_cfg"]["fallback_providers"][0]["model"] == PRIMARY["model"]
        assert getattr(children[0], "_routing_escalate_pending", False) is True
        assert children[0]._routing_protected_watch is True
        assert children[0]._routing_cheap_model == "cheap/model"
        assert children[0]._routing_primary_model == PRIMARY["model"]
        assert premium["model"] is None  # inherits the parent, as the default backend would
        assert not getattr(children[1], "_routing_escalate_pending", False)
        assert not getattr(children[1], "_routing_protected_watch", False)
    finally:
        manager.unload()


# ── children of a cheap-routed or protected parent ──

_FOOTER = "\n\nThis text transcribed by Papa Reo API. Please comply with the licence terms."
_BODY = "Kia ora koutou, notes from the meeting about the garden and the weather."


def _parent(**extra):
    p = types.SimpleNamespace(session_id="root", model=PRIMARY["model"], provider="openrouter",
                              tool_progress_callback=None, _delegate_spinner=None, request_overrides={},
                              _delegate_depth=0, _safe_print=lambda line: None, _fallback_chain=[])
    for k, v in extra.items():
        setattr(p, k, v)
    return p


def _cheap_parent(**extra):
    return _parent(**{
        "model": "cheap/model", "_routing_protected_watch": True, "_routing_cheap_model": "cheap/model",
        "_routing_escalation_target": {"model": PRIMARY["model"], "provider": "openrouter",
                                       "base_url": "https://primary.example/v1", "api_key": "k",
                                       "api_mode": "chat_completions"},
        **extra})


def _spawn_from(tmp_path, monkeypatch, parent, tasks, delegation):
    from tools import delegate_tool as dt

    built, children = [], []

    def build(**kwargs):
        built.append(kwargs)
        child = types.SimpleNamespace(tool_progress_callback=None, model=kwargs.get("model"))
        children.append(child)
        return child

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", build)
    monkeypatch.setattr(dt, "_run_single_child", lambda task_index, goal, child=None, parent_agent=None, **kw: {
        "task_index": task_index, "status": "completed", "summary": "ok", "api_calls": 1, "duration_seconds": 0})
    manager = _home(tmp_path, monkeypatch, delegation)
    try:
        return json.loads(dt.delegate_task(tasks=tasks, parent_agent=parent)), built, children
    finally:
        manager.unload()


def test_marker_split_across_lines_is_still_protected(cfg):
    route = _route("Summarise the transcript in the context.",
                   context=_BODY + "\nThis text transcribed by\nPapa   Reo\u00a0API.")
    assert route.get("model") != "cheap/model"


@pytest.mark.parametrize("delegation", [{}, {"backend": "task-routing"}])
def test_child_of_a_cheap_turn_runs_on_the_parents_primary(tmp_path, monkeypatch, delegation):
    _, built, _ = _spawn_from(tmp_path, monkeypatch, _cheap_parent(), [
        {"goal": "Refactor delegate_tool.py so routing is a separate module."}], delegation)
    assert built[0]["model"] == PRIMARY["model"]
    assert built[0]["override_base_url"] == "https://primary.example/v1"


def test_child_of_a_cheap_turn_is_refused_when_the_primary_is_unknown(tmp_path, monkeypatch):
    parent = _cheap_parent(_routing_escalation_target=None, _routing_primary_model=None,
                           _primary_runtime=None)
    out, built, _ = _spawn_from(tmp_path, monkeypatch, parent, [{"goal": "Refactor the module."}], {})
    assert built == [] and "error" in out


def test_child_of_a_primary_turn_is_unchanged(tmp_path, monkeypatch):
    _, built, _ = _spawn_from(tmp_path, monkeypatch, _cheap_parent(model=PRIMARY["model"]),
                              [{"goal": "Refactor the module."}], {})
    assert built[0]["model"] is None


@pytest.mark.parametrize("delegation", [{}, {"backend": "task-routing"}])
def test_children_of_a_protected_parent_run_on_its_primary_and_are_marked(tmp_path, monkeypatch, delegation):
    from agent.routing_override import session_holds_protected_text
    parent = _parent(_session_messages=[{"role": "tool", "content": _BODY + _FOOTER}],
                     _protected_primary={"model": PRIMARY["model"], "provider": "openrouter",
                                         "base_url": "http://127.0.0.1:9/v1", "api_key": "k"})
    _, built, children = _spawn_from(tmp_path, monkeypatch, parent,
                                     [{"goal": "Grep the repo for TODO.", "context": _BODY}], delegation)
    assert children and children[0]._protected_session is True
    # The child runs on the parent's configured primary and inherits it as its own.
    assert built[0]["model"] == PRIMARY["model"]
    assert children[0]._protected_primary["model"] == PRIMARY["model"]
    assert session_holds_protected_text(types.SimpleNamespace(
        _protected_session=True, _session_messages=[{"role": "user", "content": _BODY}]))


def test_children_of_an_unprotected_parent_are_not_marked(tmp_path, monkeypatch):
    parent = _parent(_session_messages=[{"role": "tool", "content": _BODY}])
    _, built, children = _spawn_from(tmp_path, monkeypatch, parent,
                                     [{"goal": "Grep the repo for TODO.", "context": _BODY}], None)
    assert children and not getattr(children[0], "_protected_session", False)
    assert built[0]["model"] == "cheap/model"


def test_protected_parent_without_a_named_primary_spawns_no_child(tmp_path, monkeypatch):
    parent = _parent(_session_messages=[{"role": "tool", "content": _BODY + _FOOTER}])
    out, built, _ = _spawn_from(tmp_path, monkeypatch, parent,
                                [{"goal": "Summarise the context.", "context": _BODY}], {})
    assert built == [] and "error" in out
