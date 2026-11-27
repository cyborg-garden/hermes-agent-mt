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

import yaml
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
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins_mod, "_ensure_plugins_discovered", lambda force=False: manager)
    return manager


def _spawn(tmp_path, monkeypatch, tasks, delegation=None):
    from tools import delegate_tool as dt

    built, children = [], []

    def build(**kwargs):
        built.append(kwargs)
        child = types.SimpleNamespace(tool_progress_callback=None, model=kwargs.get("model"))
        children.append(child)
        return child

    monkeypatch.setattr(dt, "_build_child_agent", build)
    monkeypatch.setattr(dt, "_run_single_child", lambda task_index, goal, child=None, parent_agent=None, **kw: {
        "task_index": task_index, "status": "completed", "summary": "ok", "api_calls": 1, "duration_seconds": 0})
    parent = types.SimpleNamespace(session_id="root", model=PRIMARY["model"], provider="openrouter",
                                   tool_progress_callback=None, _delegate_spinner=None, request_overrides={},
                                   _delegate_depth=0, _safe_print=lambda line: None)
    _home(tmp_path, monkeypatch, delegation)
    out = json.loads(dt.delegate_task(tasks=tasks, parent_agent=parent))
    return out, built, children


def test_delegate_task_routes_each_child_by_task(tmp_path, monkeypatch):
    out, built, children = _spawn(tmp_path, monkeypatch, [
        {"goal": "Grep the repo for 'routing_override' and return each match with file and line."},
        {"goal": "Refactor delegate_tool.py so routing is a separate module."},
    ])
    assert [r["status"] for r in out["results"]] == ["completed", "completed"]
    cheap, premium = built
    assert cheap["model"] == "cheap/model" and cheap["override_provider"] == "openrouter"
    assert cheap["override_fallback_chain"][0]["model"] == PRIMARY["model"]
    assert getattr(children[0], "_routing_escalate_pending", False) is True
    assert children[0]._routing_protected_watch is True
    assert children[0]._routing_cheap_model == "cheap/model"
    assert children[0]._routing_primary_model == PRIMARY["model"]
    assert premium["model"] is None  # inherits the parent, as without a backend
    assert "override_fallback_chain" not in premium
    assert not getattr(children[1], "_routing_escalate_pending", False)
    assert not getattr(children[1], "_routing_protected_watch", False)


def test_protected_context_keeps_the_child_on_the_default_route(tmp_path, monkeypatch):
    _, built, children = _spawn(tmp_path, monkeypatch, [
        {"goal": "Summarise the transcript in the context.",
         "context": "...\nThis text transcribed by Papa Reo API."},
    ])
    assert built[0]["model"] is None and "override_fallback_chain" not in built[0]
    assert not getattr(children[0], "_routing_protected_watch", False)


def test_no_backend_is_todays_behaviour(tmp_path, monkeypatch):
    _, built, children = _spawn(tmp_path, monkeypatch, [
        {"goal": "Grep the repo for 'routing_override'."},
    ], delegation={})
    assert built[0]["model"] is None and "override_fallback_chain" not in built[0]
    assert not hasattr(children[0], "_routing_escalate_pending")


def test_unresolvable_cheap_route_falls_back_to_the_default_route(tmp_path, monkeypatch):
    from tools import delegate_tool as dt
    real = dt._resolve_delegation_credentials

    def resolve(cfg, parent):
        if cfg.get("model") == "cheap/model":
            raise ValueError("no key")
        return real(cfg, parent)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", resolve)
    _, built, children = _spawn(tmp_path, monkeypatch, [{"goal": "Count the test files."}])
    assert built[0]["model"] is None and "override_fallback_chain" not in built[0]
    assert not hasattr(children[0], "_routing_escalate_pending")


def test_backend_error_falls_back_to_the_default_route(tmp_path, monkeypatch):
    import agent.delegation_provider as dp

    class Boom:
        def route_task(self, task, config):
            raise RuntimeError("boom")
    monkeypatch.setattr(dp, "get_provider", lambda name: Boom())
    _, built, _ = _spawn(tmp_path, monkeypatch, [{"goal": "Count the test files."}])
    assert built[0]["model"] is None


# ── children of a cheap-routed parent ──

def _cheap_parent(**extra):
    p = types.SimpleNamespace(session_id="root", model="cheap/model", provider="openrouter",
                              tool_progress_callback=None, _delegate_spinner=None, request_overrides={},
                              _delegate_depth=0, _safe_print=lambda line: None,
                              _routing_protected_watch=True, _routing_cheap_model="cheap/model",
                              _routing_escalation_target={"model": PRIMARY["model"], "provider": "openrouter",
                                                          "base_url": "https://primary.example/v1",
                                                          "api_key": "k", "api_mode": "chat_completions"},
                              _fallback_chain=[])
    for k, v in extra.items():
        setattr(p, k, v)
    return p


def _spawn_from(tmp_path, monkeypatch, parent, tasks, delegation):
    from tools import delegate_tool as dt
    built = []
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: built.append(kw) or types.SimpleNamespace(
        tool_progress_callback=None, model=kw.get("model")))
    monkeypatch.setattr(dt, "_run_single_child", lambda task_index, goal, child=None, parent_agent=None, **kw: {
        "task_index": task_index, "status": "completed", "summary": "ok", "api_calls": 1, "duration_seconds": 0})
    _home(tmp_path, monkeypatch, delegation)
    return json.loads(dt.delegate_task(tasks=tasks, parent_agent=parent)), built


@pytest.mark.parametrize("delegation", [{}, {"backend": "task-routing"}])
def test_child_of_a_cheap_turn_runs_on_the_parents_primary(tmp_path, monkeypatch, delegation):
    out, built = _spawn_from(tmp_path, monkeypatch, _cheap_parent(), [
        {"goal": "Refactor delegate_tool.py so routing is a separate module."}], delegation)
    assert built[0]["model"] == PRIMARY["model"]
    assert built[0]["override_base_url"] == "https://primary.example/v1"


def test_child_of_a_cheap_turn_is_refused_when_the_primary_is_unknown(tmp_path, monkeypatch):
    parent = _cheap_parent(_routing_escalation_target=None, _routing_primary_model=None)
    out, built = _spawn_from(tmp_path, monkeypatch, parent, [{"goal": "Refactor the module."}], {})
    assert built == [] and "error" in out


def test_child_of_a_primary_turn_is_unchanged(tmp_path, monkeypatch):
    parent = _cheap_parent(model=PRIMARY["model"])
    _, built = _spawn_from(tmp_path, monkeypatch, parent, [{"goal": "Refactor the module."}], {})
    assert built[0]["model"] is None


# ── A parent holding protected text keeps every child on the primary ─────────

_FOOTER = "\n\nTranscribed by Papa Reo. Kaitiakitanga License."
_BODY = "Kia ora koutou, notes from the meeting about the garden and the weather."


def test_marker_split_across_lines_is_still_protected(cfg):
    route = _route("Summarise the transcript in the context.",
                   context=_BODY + "\nTranscribed by\nPapa   Reo")
    assert route.get("model") != "cheap/model"


def test_parent_holding_protected_text_keeps_children_off_the_cheap_model(cfg):
    from agent import delegation_provider as dp
    from tools import delegate_tool as dt

    dp.register_provider(deleg.TaskRoutingDelegationProvider())
    parent = types.SimpleNamespace(
        model="prem/m", provider="openrouter", _fallback_chain=[],
        _session_messages=[{"role": "tool", "content": _BODY + _FOOTER}],
    )
    creds = {"model": None, "provider": None, "base_url": None, "api_key": None, "api_mode": None}
    with patch.object(dt, "_resolve_delegation_credentials", side_effect=lambda c, _p: {**creds, **c}):
        routes = dt._task_routes([{"goal": "Summarise the context.", "context": _BODY}],
                                 {"backend": deleg.BACKEND_NAME}, creds, parent)
    child_creds, route = routes[0]
    assert route is None and child_creds.get("model") != "cheap/model"


def test_parent_without_protected_text_still_routes_children(cfg):
    from agent import delegation_provider as dp
    from tools import delegate_tool as dt

    dp.register_provider(deleg.TaskRoutingDelegationProvider())
    parent = types.SimpleNamespace(
        model="prem/m", provider="openrouter", _fallback_chain=[],
        _session_messages=[{"role": "tool", "content": _BODY}],
    )
    creds = {"model": None, "provider": None, "base_url": None, "api_key": None, "api_mode": None}
    with patch.object(dt, "_resolve_delegation_credentials", side_effect=lambda c, _p: {**creds, **c}):
        routes = dt._task_routes([{"goal": "Summarise the context.", "context": _BODY}],
                                 {"backend": deleg.BACKEND_NAME}, creds, parent)
    assert routes[0][0].get("model") == "cheap/model"


def _plain_parent(messages):
    return types.SimpleNamespace(session_id="root", model=PRIMARY["model"], provider="openrouter",
                                 tool_progress_callback=None, _delegate_spinner=None,
                                 request_overrides={}, _delegate_depth=0,
                                 _safe_print=lambda line: None, _fallback_chain=[],
                                 _session_messages=messages)


def _spawn_children(tmp_path, monkeypatch, parent, tasks, delegation):
    from tools import delegate_tool as dt
    children = []
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: children.append(
        types.SimpleNamespace(tool_progress_callback=None, model=kw.get("model"))) or children[-1])
    monkeypatch.setattr(dt, "_run_single_child", lambda task_index, goal, child=None, parent_agent=None, **kw: {
        "task_index": task_index, "status": "completed", "summary": "ok", "api_calls": 1, "duration_seconds": 0})
    _home(tmp_path, monkeypatch, delegation)
    dt.delegate_task(tasks=tasks, parent_agent=parent)
    return children


@pytest.mark.parametrize("delegation", [{}, {"backend": "task-routing"}])
def test_children_of_a_protected_parent_are_marked_protected(tmp_path, monkeypatch, delegation):
    from agent.routing_override import session_holds_protected_text
    parent = _plain_parent([{"role": "tool", "content": _BODY + _FOOTER}])
    children = _spawn_children(tmp_path, monkeypatch, parent,
                               [{"goal": "Summarise the context.", "context": _BODY}], delegation)
    assert children and children[0]._protected_session is True
    assert session_holds_protected_text(types.SimpleNamespace(
        _protected_session=True, _session_messages=[{"role": "user", "content": _BODY}]))


def test_children_of_an_unprotected_parent_are_not_marked(tmp_path, monkeypatch):
    parent = _plain_parent([{"role": "tool", "content": _BODY}])
    children = _spawn_children(tmp_path, monkeypatch, parent,
                               [{"goal": "Summarise the context.", "context": _BODY}], {})
    assert children and not getattr(children[0], "_protected_session", False)
