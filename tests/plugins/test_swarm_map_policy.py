"""Tests for swarm-map-policy plugin — HSM integration for group access control."""
import pytest
from unittest.mock import patch, MagicMock
from enum import Enum


def _make_event(platform="signal", chat_id="group-123", user_id="user-456"):
    """Create a mock MessageEvent with source attributes."""
    class MockPlatform(Enum):
        SIGNAL = "signal"
        TELEGRAM = "telegram"
        DISCORD = "discord"

    event = MagicMock()
    event.source.chat_id = chat_id
    event.source.user_id = user_id
    # Set platform as an enum with .value
    platform_enum = MockPlatform(platform) if platform in ("signal", "telegram", "discord") else MagicMock(value=platform)
    event.source.platform = platform_enum
    return event


class TestSwarmMapPolicy:

    def test_plugin_registers_hooks(self):
        """Plugin registers on_session_start, pre_tool_call, and pre_gateway_dispatch hooks."""
        from plugins.swarm_map_policy import register
        ctx = MagicMock()
        register(ctx)
        hook_names = [call.args[0] for call in ctx.register_hook.call_args_list]
        assert "on_session_start" in hook_names
        assert "pre_tool_call" in hook_names
        assert "pre_gateway_dispatch" in hook_names

    def test_hsm_url_from_env(self):
        from plugins.swarm_map_policy import _hsm_url
        with patch.dict("os.environ", {"HSM_URL": "http://localhost:3002"}):
            assert _hsm_url() == "http://localhost:3002"

    def test_hsm_url_missing_returns_none(self):
        from plugins.swarm_map_policy import _hsm_url
        with patch.dict("os.environ", {}, clear=True):
            assert _hsm_url() is None

    def test_group_check_fail_closed_on_error(self):
        from plugins.swarm_map_policy import is_group_allowed
        with patch("plugins.swarm_map_policy._hsm_url", return_value="http://dead:9999"):
            with patch("plugins.swarm_map_policy.requests") as mock_req:
                mock_req.get.side_effect = Exception("Connection refused")
                assert is_group_allowed("group-123", "signal") is False

    def test_group_check_fail_closed_no_config(self):
        from plugins.swarm_map_policy import is_group_allowed
        with patch("plugins.swarm_map_policy._hsm_url", return_value=None):
            assert is_group_allowed("group-123", "signal") is False

    def test_tool_check_fail_open(self):
        from plugins.swarm_map_policy import is_tool_allowed
        with patch("plugins.swarm_map_policy._hsm_url", return_value=None):
            assert is_tool_allowed("dangerous_tool", "group-123") is True

    def test_admin_check_fail_closed(self):
        from plugins.swarm_map_policy import is_platform_admin
        with patch("plugins.swarm_map_policy._hsm_url", return_value=None):
            assert is_platform_admin("user-123", "signal") is False


class TestApproveGroupAdd:
    """Tests for approve_group_add — HSM group auto-approval on bot add."""

    def _call(self, mock_resp=None, side_effect=None, group_id="-100123", adder="777"):
        from plugins.swarm_map_policy import approve_group_add
        with patch("plugins.swarm_map_policy._hsm_url", return_value="http://hsm:3002"), \
             patch("plugins.swarm_map_policy._harness_id", return_value="hermes-test"), \
             patch("plugins.swarm_map_policy.requests") as mock_req:
            if side_effect is not None:
                mock_req.post.side_effect = side_effect
            else:
                mock_req.post.return_value = mock_resp
            result = approve_group_add(group_id, adder)
            return result, mock_req

    @staticmethod
    def _resp(status_code=200, body=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = body if body is not None else {}
        return resp

    def test_approved(self):
        """200 + approved:true returns True."""
        result, _ = self._call(self._resp(200, {"approved": True, "restarted": True}))
        assert result is True

    def test_approved_already_allowed(self):
        """200 + approved:true + already_allowed (wildcard/listed) returns True."""
        result, _ = self._call(self._resp(200, {"approved": True, "already_allowed": True}))
        assert result is True

    def test_approved_restart_failed_still_true(self):
        """approved:true with restarted:false (env written, recreate failed) is still approved."""
        result, _ = self._call(self._resp(200, {"approved": True, "restarted": False}))
        assert result is True

    def test_not_approved(self):
        """200 + approved:false returns False."""
        result, _ = self._call(
            self._resp(200, {"approved": False, "reason": "adder is not an admin"})
        )
        assert result is False

    def test_bad_request_denied(self):
        """400 with error body is treated as not approved."""
        result, _ = self._call(self._resp(400, {"error": "missing addedByUserId"}))
        assert result is False

    def test_network_error_fail_closed(self):
        """Network failure denies (fail-closed)."""
        result, _ = self._call(side_effect=Exception("Connection refused"))
        assert result is False

    def test_missing_approved_field_denied(self):
        """200 with no approved field denies (fail-closed)."""
        result, _ = self._call(self._resp(200, {}))
        assert result is False

    def test_non_bool_approved_denied(self):
        """approved must be strictly true — truthy strings deny."""
        result, _ = self._call(self._resp(200, {"approved": "yes"}))
        assert result is False

    def test_no_config_fail_closed(self):
        """Missing HSM_URL denies without any request."""
        from plugins.swarm_map_policy import approve_group_add
        with patch("plugins.swarm_map_policy._hsm_url", return_value=None):
            assert approve_group_add("-100123", "777") is False

    def test_posts_correct_url_and_body(self):
        """Request hits the HSM group endpoint with addedByUserId body."""
        _, mock_req = self._call(self._resp(200, {"approved": True}))
        mock_req.post.assert_called_once_with(
            "http://hsm:3002/api/harnesses/hermes-test/surfaces/telegram/groups/-100123",
            json={"addedByUserId": "777"},
            timeout=5,
        )


class TestSessionContextCaching:
    """Tests for session context caching via pre_gateway_dispatch."""

    def setup_method(self):
        """Clear session context before each test."""
        from plugins.swarm_map_policy import clear_session_context
        clear_session_context()

    def test_pre_gateway_dispatch_caches_context(self):
        """pre_gateway_dispatch extracts and caches platform/chat_id/user_id."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch, get_session_context
        event = _make_event(platform="signal", chat_id="group-123", user_id="user-456")
        with patch("plugins.swarm_map_policy.is_platform_admin", return_value=False):
            result = _pre_gateway_dispatch(event=event)
        assert result is None  # Should allow normal dispatch
        ctx = get_session_context()
        assert ctx is not None
        assert ctx["platform"] == "signal"
        assert ctx["chat_id"] == "group-123"
        assert ctx["user_id"] == "user-456"

    def test_pre_gateway_dispatch_returns_none_on_no_event(self):
        """pre_gateway_dispatch returns None when no event provided."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch, get_session_context
        result = _pre_gateway_dispatch(event=None)
        assert result is None
        assert get_session_context() is None

    def test_get_session_context_none_before_dispatch(self):
        """get_session_context returns None before any dispatch."""
        from plugins.swarm_map_policy import get_session_context
        assert get_session_context() is None

    def test_clear_session_context_resets(self):
        """clear_session_context removes cached data."""
        from plugins.swarm_map_policy import (
            _pre_gateway_dispatch, get_session_context, clear_session_context
        )
        event = _make_event()
        with patch("plugins.swarm_map_policy.is_platform_admin", return_value=False):
            _pre_gateway_dispatch(event=event)
        assert get_session_context() is not None
        clear_session_context()
        assert get_session_context() is None

    def test_pre_gateway_dispatch_handles_none_source_fields(self):
        """Gracefully handles None platform/chat_id/user_id."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch, get_session_context
        event = MagicMock()
        event.source.platform = None
        event.source.chat_id = None
        event.source.user_id = None
        with patch("plugins.swarm_map_policy.is_platform_admin", return_value=False):
            result = _pre_gateway_dispatch(event=event)
        assert result is None
        ctx = get_session_context()
        assert ctx["platform"] == ""
        assert ctx["chat_id"] == ""
        assert ctx["user_id"] == ""


class TestAdminResolution:
    """Tests for admin identity resolution during pre_gateway_dispatch."""

    def setup_method(self):
        from plugins.swarm_map_policy import clear_session_context
        clear_session_context()

    def test_admin_resolved_on_dispatch(self):
        """Admin status is resolved and cached during pre_gateway_dispatch."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch, get_session_context
        event = _make_event(platform="signal", user_id="admin-user")
        with patch("plugins.swarm_map_policy.is_platform_admin", return_value=True):
            _pre_gateway_dispatch(event=event)
        ctx = get_session_context()
        assert ctx["is_admin"] is True

    def test_non_admin_resolved_on_dispatch(self):
        """Non-admin status is correctly cached."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch, get_session_context
        event = _make_event(platform="signal", user_id="regular-user")
        with patch("plugins.swarm_map_policy.is_platform_admin", return_value=False):
            _pre_gateway_dispatch(event=event)
        ctx = get_session_context()
        assert ctx["is_admin"] is False

    def test_admin_resolution_fail_closed(self):
        """Admin resolution defaults to False on HSM failure."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch, get_session_context
        event = _make_event(platform="signal", user_id="user-123")
        with patch("plugins.swarm_map_policy.is_platform_admin", side_effect=Exception("HSM down")):
            _pre_gateway_dispatch(event=event)
        ctx = get_session_context()
        assert ctx["is_admin"] is False

    def test_admin_resolution_called_with_correct_args(self):
        """is_platform_admin called with user_id and platform from event."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch
        event = _make_event(platform="telegram", user_id="tg-user-789")
        with patch("plugins.swarm_map_policy.is_platform_admin", return_value=False) as mock_admin:
            _pre_gateway_dispatch(event=event)
        mock_admin.assert_called_once_with("tg-user-789", "telegram")


class TestAdminGatedTools:
    """pre_tool_call admin gating.

    Regression for the 2026-10-06 audit: the gate listed ``approval`` and
    ``pr_approval``, which are not registered tools, so it never fired, while
    its tests (which called the hook with those made-up names) stayed green.
    Dangerous-command approval is not a tool call at all: it is the gateway
    approval queue (/approve, Discord buttons), gated in gateway/run.py and
    the Discord adapter. This hook can only gate real tool names.
    """

    def setup_method(self):
        from plugins.swarm_map_policy import clear_session_context
        clear_session_context()

    def _set_admin_context(self, is_admin=True):
        from plugins.swarm_map_policy import _pre_gateway_dispatch
        event = _make_event(platform="signal", user_id="user-1")
        with patch("plugins.swarm_map_policy.is_platform_admin", return_value=is_admin):
            _pre_gateway_dispatch(event=event)

    def test_every_default_gated_name_is_a_registered_tool(self):
        """Non-vacuity floor: a gated name that matches no tool is a dead gate."""
        from tools.registry import discover_builtin_tools, registry
        from plugins.swarm_map_policy import ADMIN_GATED_TOOLS
        discover_builtin_tools()
        known = set(registry.get_all_tool_names())
        assert known, "registry discovery returned nothing; test would be vacuous"
        assert set(ADMIN_GATED_TOOLS) <= known, (
            f"gated names that match no registered tool: "
            f"{sorted(set(ADMIN_GATED_TOOLS) - known)}"
        )

    def test_unknown_names_reported(self):
        from plugins.swarm_map_policy import unknown_gated_tools
        assert unknown_gated_tools({"terminal", "approval"}, ["terminal", "web_search"]) == ["approval"]
        assert unknown_gated_tools({"terminal"}, ["terminal"]) == []

    def test_env_gates_a_real_tool_for_non_admin(self, monkeypatch):
        from plugins.swarm_map_policy import _pre_tool_call
        monkeypatch.setenv("SWARM_MAP_ADMIN_GATED_TOOLS", "terminal, execute_code")
        self._set_admin_context(is_admin=False)
        for tool in ("terminal", "execute_code"):
            result = _pre_tool_call(tool_name=tool)
            assert result is not None and result["action"] == "block", tool

    def test_env_gated_tool_allowed_for_admin(self, monkeypatch):
        from plugins.swarm_map_policy import _pre_tool_call
        monkeypatch.setenv("SWARM_MAP_ADMIN_GATED_TOOLS", "terminal")
        self._set_admin_context(is_admin=True)
        assert _pre_tool_call(tool_name="terminal") is None

    def test_env_gated_tool_blocked_without_context(self, monkeypatch):
        """Fail-closed: no dispatch context means not known to be an admin."""
        from plugins.swarm_map_policy import _pre_tool_call
        monkeypatch.setenv("SWARM_MAP_ADMIN_GATED_TOOLS", "terminal")
        result = _pre_tool_call(tool_name="terminal")
        assert result is not None and result["action"] == "block"

    def test_non_gated_tool_allowed_without_admin(self, monkeypatch):
        from plugins.swarm_map_policy import _pre_tool_call
        monkeypatch.setenv("SWARM_MAP_ADMIN_GATED_TOOLS", "terminal")
        self._set_admin_context(is_admin=False)
        assert _pre_tool_call(tool_name="web_search") is None

    def test_non_gated_tool_allowed_without_context(self, monkeypatch):
        from plugins.swarm_map_policy import _pre_tool_call
        monkeypatch.delenv("SWARM_MAP_ADMIN_GATED_TOOLS", raising=False)
        assert _pre_tool_call(tool_name="web_search") is None

    def test_block_message_names_the_tool(self, monkeypatch):
        from plugins.swarm_map_policy import _pre_tool_call
        monkeypatch.setenv("SWARM_MAP_ADMIN_GATED_TOOLS", "terminal")
        self._set_admin_context(is_admin=False)
        assert "terminal" in _pre_tool_call(tool_name="terminal")["message"]


_STARTUP_PROBE = r"""
import json, logging, sys
records = []
class _H(logging.Handler):
    def emit(self, r):
        if "swarm" in r.name and r.levelno >= logging.WARNING:
            records.append(r.getMessage())
logging.getLogger().addHandler(_H())
logging.getLogger().setLevel(logging.INFO)
# The gateway startup path: plugins are discovered before model_tools has
# imported the built-in tools (gateway/run.py, "Discover Python plugins").
from hermes_cli.plugins import discover_plugins, get_plugin_manager
discover_plugins()
loaded = get_plugin_manager()._plugins.get("swarm-map-policy")
print(json.dumps({"loaded": bool(loaded and loaded.enabled), "warnings": records}))
"""


class TestGatedToolWarningAtStartup:
    """The unknown-name warning must not fire for real tools at gateway startup.

    At startup the registry holds only plugin tools discovered so far; the
    built-ins arrive later. A check against that partial registry calls real
    tools "not gated" while the gate in fact works, sending operators after a
    bug that is not there. Runs in a fresh interpreter so the registry starts
    as empty as it does in the gateway.
    """

    def _run(self, tmp_path, gated):
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        home = tmp_path / "hermes_home"
        home.mkdir()
        (home / "config.yaml").write_text("plugins:\n  enabled: [swarm-map-policy]\n")
        env = dict(os.environ)
        env.update(
            HERMES_HOME=str(home),
            HSM_URL="http://127.0.0.1:9",
            SWARM_MAP_ADMIN_GATED_TOOLS=gated,
            PYTHONPATH=str(repo) + os.pathsep + env.get("PYTHONPATH", ""),
        )
        out = subprocess.run(
            [sys.executable, "-c", _STARTUP_PROBE],
            cwd=str(repo), env=env, capture_output=True, text=True, timeout=180,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        result = json.loads(out.stdout.strip().splitlines()[-1])
        assert result["loaded"], "plugin did not load; probe would be vacuous"
        return result["warnings"]

    def test_real_tools_do_not_warn(self, tmp_path):
        warnings = self._run(tmp_path, "terminal,execute_code,web_search,delegate_task")
        assert not [w for w in warnings if "not gated" in w], warnings

    def test_unknown_name_still_warns(self, tmp_path):
        """Non-vacuity: a typo must still produce the warning."""
        warnings = self._run(tmp_path, "terminal,approvall")
        hits = [w for w in warnings if "not gated" in w]
        assert hits and "approvall" in hits[0] and "terminal" not in hits[0], warnings
