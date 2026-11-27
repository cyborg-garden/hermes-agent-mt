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

def _home(tmp_path, monkeypatch):
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
        "delegation": {"backend": "task-routing"},
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
        assert premium["model"] is None  # inherits the parent, as the default backend would
        assert not getattr(children[1], "_routing_escalate_pending", False)
    finally:
        manager.unload()
