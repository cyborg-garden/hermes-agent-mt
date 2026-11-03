"""Tests for /approve and /deny gateway commands.

Verifies that dangerous command approvals use the blocking gateway approval
mechanism — the agent thread blocks until the user responds with /approve
or /deny, mirroring the CLI's synchronous input() flow.

Supports multiple concurrent approvals (parallel subagents, execute_code)
via a per-session queue.
"""

import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(),
        message_id="m1",
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    # Default: the approver is an individual-allowlisted admin. Admin-gating
    # behavior is covered explicitly in TestApprovalUsesDmAllowlist (handler
    # wiring) and TestIndividualAllowlist (the real predicate).
    runner._is_individual_allowlisted = lambda _source: True
    runner._set_session_env = lambda _context: None
    return runner


def _clear_approval_state():
    """Reset all module-level approval state between tests."""
    from tools import approval as mod
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()


def _wait_until(predicate, timeout=30.0, interval=0.02):
    """Wait until *predicate()* is truthy or *timeout* elapses; return its value.

    Replaces the fixed-iteration ``for _ in range(N): sleep(0.05)`` polls these
    E2E tests used to wait for a background agent thread to reach the gateway
    approval notify. Those budgets (2.5-5s) race the scheduler: under CI worker
    contention the thread can be starved past the deadline, so the notify fires
    after the poll gives up and the assertion sees an empty list. The generous
    ceiling here is only reached on genuine failure; the common path returns the
    instant the condition holds, adding no latency to green runs.
    (Ported from PR #63522 by @jethac.)
    """
    deadline = time.monotonic() + timeout
    result = predicate()
    while not result and time.monotonic() < deadline:
        time.sleep(interval)
        result = predicate()
    return result


# ------------------------------------------------------------------
# Blocking gateway approval infrastructure (tools/approval.py)
# ------------------------------------------------------------------


class TestBlockingGatewayApproval:
    """Tests for the blocking approval mechanism in tools/approval.py."""

    def setup_method(self):
        _clear_approval_state()

    def test_register_and_resolve_unblocks_entry(self):
        """resolve_gateway_approval signals the entry's event."""
        from tools.approval import register_gateway_notify, unregister_gateway_notify, resolve_gateway_approval, has_blocking_approval, _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry
        session_key = "test-session"
        register_gateway_notify(session_key, lambda d: None)

        # Simulate what check_all_command_guards does
        entry = _ApprovalEntry({"command": "rm -rf /"})
        _gateway_queues.setdefault(session_key, []).append(entry)

        assert has_blocking_approval(session_key) is True

        # Resolve from another thread
        def resolve():
            time.sleep(0.1)
            resolve_gateway_approval(session_key, "once")

        t = threading.Thread(target=resolve)
        t.start()
        resolved = entry.event.wait(timeout=5)
        t.join()

        assert resolved is True
        assert entry.result == "once"
        unregister_gateway_notify(session_key)


    def test_resolve_single_pops_oldest_fifo(self):
        """resolve_gateway_approval without resolve_all resolves oldest first."""
        from tools.approval import resolve_gateway_approval, _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry
        session_key = "test-fifo"
        e1 = _ApprovalEntry({"command": "first"})
        e2 = _ApprovalEntry({"command": "second"})
        _gateway_queues[session_key] = [e1, e2]

        count = resolve_gateway_approval(session_key, "once")
        assert count == 1
        assert e1.event.is_set()
        assert e1.result == "once"
        assert not e2.event.is_set()
        assert len(_gateway_queues[session_key]) == 1


# ------------------------------------------------------------------
# /approve command
# ------------------------------------------------------------------


class TestApproveCommand:

    def setup_method(self):
        _clear_approval_state()


    @pytest.mark.asyncio
    async def test_approve_all_resolves_multiple(self):
        """/approve all resolves all pending approvals."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        _gateway_queues[session_key] = [e1, e2]

        result = await runner._handle_approve_command(_make_event("/approve all"))
        assert "2 commands" in result
        assert e1.event.is_set()
        assert e2.event.is_set()

    @pytest.mark.asyncio
    async def test_approve_all_session(self):
        """/approve all session resolves all with session scope."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        _gateway_queues[session_key] = [e1, e2]

        result = await runner._handle_approve_command(_make_event("/approve all session"))
        assert "session" in result.lower()
        assert e1.result == "session"
        assert e2.result == "session"


# ------------------------------------------------------------------
# /deny command
# ------------------------------------------------------------------


class TestDenyCommand:

    def setup_method(self):
        _clear_approval_state()


    @pytest.mark.asyncio
    async def test_deny_with_reason_attaches_reason(self):
        """/deny <reason> attaches the reason to the resolved entry."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        result = await runner._handle_deny_command(
            _make_event("/deny that path is still in use")
        )
        assert entry.result == "deny"
        assert entry.reason == "that path is still in use"
        assert "that path is still in use" in result

    @pytest.mark.asyncio
    async def test_deny_all_with_reason(self):
        """/deny all <reason> denies everything and relays one reason."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        _gateway_queues[session_key] = [e1, e2]

        result = await runner._handle_deny_command(
            _make_event("/deny all wrong directory")
        )
        assert "2 commands" in result
        assert all(e.result == "deny" for e in [e1, e2])
        assert all(e.reason == "wrong directory" for e in [e1, e2])


# ------------------------------------------------------------------
# Bare "yes" must NOT trigger approval
# ------------------------------------------------------------------


class TestBareTextNoLongerApproves:

    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_yes_does_not_execute_pending_command(self):
        """Saying 'yes' must not trigger approval. Only /approve works."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        # "yes" is not /approve — entry should still be pending
        assert not entry.event.is_set()


# ------------------------------------------------------------------
# End-to-end blocking flow
# ------------------------------------------------------------------


class TestBlockingApprovalE2E:
    """Test the full blocking flow: agent thread blocks → user approves → agent resumes."""

    @pytest.fixture(autouse=True)
    def _manual_approval_mode(self, monkeypatch):
        monkeypatch.setattr("tools.approval_context._get_approval_mode", lambda: "manual")

    def setup_method(self):
        _clear_approval_state()
        os.environ.pop("HERMES_YOLO_MODE", None)
        os.environ.pop("HERMES_INTERACTIVE", None)
        os.environ.pop("HERMES_GATEWAY_SESSION", None)
        os.environ.pop("HERMES_EXEC_ASK", None)
        os.environ.pop("HERMES_SESSION_KEY", None)
        # These E2E tests exercise manual gateway blocking; default config is
        # approvals.mode=smart which may auto-approve/deny via aux LLM before
        # notify_cb runs (flaky on CI when the LLM is slow or unavailable).
        self._approval_mode_patch = patch(
            "tools.approval_context._get_approval_mode", return_value="manual"
        )
        self._approval_mode_patch.start()

    def teardown_method(self):
        self._approval_mode_patch.stop()


    @pytest.mark.parametrize(
        "approval_config",
        [
            {"mode": "manual", "timeout": 0},
            {"mode": "manual", "timeout": 0, "gateway_timeout": 300},
        ],
        ids=["shared-timeout-only", "shared-timeout-is-canonical"],
    )
    def test_blocking_approval_uses_canonical_timeout(self, approval_config, monkeypatch):
        """Gateway waits use approvals.timeout, without a second timeout knob."""
        from tools import approval as approval_module
        from tools.approval import check_all_command_guards, register_gateway_notify, unregister_gateway_notify
        from tools.approval_context import reset_current_session_key, set_current_session_key

        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
        session_key = "e2e-timeout"
        notified = []
        register_gateway_notify(session_key, notified.append)
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.setenv("HERMES_SESSION_KEY", session_key)
        token = set_current_session_key(session_key)
        try:
            # Zero expires in the real poll loop. A timed join followed by /deny
            # races pre-approval hooks and changes the outcome being tested.
            with patch(
                "tools.approval_context._get_approval_config",
                return_value=approval_config,
            ):
                result = check_all_command_guards("rm -rf /important", "local")

            assert len(notified) == 1
            assert result["approved"] is False
            assert result["user_consent"] is False
            assert result["outcome"] == "timeout"
            assert "timed out" in result["message"]
            assert approval_module.list_gateway_approvals(session_key) == []
        finally:
            reset_current_session_key(token)
            unregister_gateway_notify(session_key)

    def test_parallel_subagent_approvals(self):
        """Multiple threads can block concurrently and be resolved independently."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            resolve_gateway_approval, check_all_command_guards,
            _gateway_queues,
        )

        session_key = "e2e-parallel"
        notified = []
        register_gateway_notify(session_key, lambda d: notified.append(d))

        results = [None, None, None]

        def make_agent(idx, cmd):
            def run():
                from tools.approval_context import reset_current_session_key, set_current_session_key

                token = set_current_session_key(session_key)
                os.environ["HERMES_GATEWAY_SESSION"] = "1"
                os.environ["HERMES_EXEC_ASK"] = "1"
                os.environ["HERMES_SESSION_KEY"] = session_key
                try:
                    results[idx] = check_all_command_guards(cmd, "local")
                finally:
                    os.environ.pop("HERMES_GATEWAY_SESSION", None)
                    os.environ.pop("HERMES_EXEC_ASK", None)
                    os.environ.pop("HERMES_SESSION_KEY", None)
                    reset_current_session_key(token)
            return run

        threads = [
            threading.Thread(target=make_agent(0, "rm -rf /a")),
            threading.Thread(target=make_agent(1, "rm -rf /b")),
            threading.Thread(target=make_agent(2, "rm -rf /c")),
        ]
        for t in threads:
            t.start()

        # Wait for all 3 to block
        assert _wait_until(lambda: len(notified) >= 3), \
            "not all 3 agents reached the gateway approval notify"

        assert len(notified) == 3
        assert len(_gateway_queues.get(session_key, [])) == 3

        # Approve all at once
        count = resolve_gateway_approval(session_key, "session", resolve_all=True)
        assert count == 3

        for t in threads:
            t.join(timeout=5)

        assert all(r is not None for r in results)
        assert all(r["approved"] is True for r in results)
        unregister_gateway_notify(session_key)


# ------------------------------------------------------------------
# Fallback: no gateway callback (cron/batch mode)
# ------------------------------------------------------------------


class TestFallbackNoCallback:

    def setup_method(self):
        _clear_approval_state()

    def test_no_callback_returns_approval_required(self):
        """Without a registered callback, the fallback returns pending_approval.

        PR #6d495d9e7 renamed the LLM-visible status from ``approval_required``
        to ``pending_approval`` to make the state distinguishable from a
        failed tool call.
        """
        from tools.approval import check_all_command_guards

        os.environ["HERMES_EXEC_ASK"] = "1"
        os.environ["HERMES_SESSION_KEY"] = "no-callback-test"
        try:
            result = check_all_command_guards("rm -rf /important", "local")
        finally:
            os.environ.pop("HERMES_EXEC_ASK", None)
            os.environ.pop("HERMES_SESSION_KEY", None)

        assert result["approved"] is False
        assert result.get("status") == "pending_approval"
        assert result.get("approval_pending") is True


# ------------------------------------------------------------------
# Cross-session admin approval (group_sessions_per_user)
# ------------------------------------------------------------------


class TestCrossSessionAdminApproval:
    """In groups with per-user sessions, a dangerous command is keyed to the
    *triggering* participant's session — not the approver's. An admin's
    /approve must still clear that sibling session's pending approval, or a
    non-admin can trigger a command nobody can ever approve."""

    def setup_method(self):
        _clear_approval_state()

    @staticmethod
    def _group_source(user_id: str) -> SessionSource:
        return SessionSource(
            platform=Platform.TELEGRAM,
            user_id=user_id,
            chat_id="g1",
            user_name=user_id,
            chat_type="group",
        )

    @staticmethod
    def _runner_with_admin(admin_id: str):
        runner = _make_runner()
        runner.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="***",
                    extra={"group_allow_admin_from": [admin_id]},
                )
            }
        )
        return runner

    @pytest.mark.asyncio
    async def test_admin_approve_clears_other_participants_pending(self):
        """Admin /approve resolves a pending approval bound to another user's
        per-user session in the same group."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = self._runner_with_admin("u_admin")

        triggerer = self._group_source("u_cam")  # non-admin who hit the command
        trig_key = runner._session_key_for_source(triggerer)
        entry = _ApprovalEntry({"command": "ffprobe x | python3", "description": "scan"})
        _gateway_queues[trig_key] = [entry]

        admin_event = MessageEvent(
            text="/approve session",
            source=self._group_source("u_admin"),
            message_id="m1",
        )
        result = await runner._handle_approve_command(admin_event)

        assert entry.event.is_set()
        assert entry.result == "session"
        assert trig_key not in _gateway_queues
        assert "no pending" not in result.lower()
        assert "not authorized" not in result.lower()

    @pytest.mark.asyncio
    async def test_nonadmin_cannot_cross_approve(self):
        """A non-admin must NOT be able to clear another participant's pending
        approval — admin-only gating still applies to cross-session resolves."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = self._runner_with_admin("u_admin")

        triggerer = self._group_source("u_other")
        trig_key = runner._session_key_for_source(triggerer)
        entry = _ApprovalEntry({"command": "rm -rf /", "description": "scan"})
        _gateway_queues[trig_key] = [entry]

        nonadmin_event = MessageEvent(
            text="/approve",
            source=self._group_source("u_cam"),  # not in group_allow_admin_from
            message_id="m2",
        )
        result = await runner._handle_approve_command(nonadmin_event)

        assert not entry.event.is_set()
        assert trig_key in _gateway_queues
        assert "not authorized" in result.lower()

    @pytest.mark.asyncio
    async def test_admin_deny_clears_other_participants_pending(self):
        """Admin /deny must also resolve another participant's stuck pending
        approval — otherwise a non-admin-triggered command can only time out."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = self._runner_with_admin("u_admin")

        triggerer = self._group_source("u_cam")
        trig_key = runner._session_key_for_source(triggerer)
        entry = _ApprovalEntry({"command": "rm -rf /", "description": "scan"})
        _gateway_queues[trig_key] = [entry]

        admin_event = MessageEvent(
            text="/deny",
            source=self._group_source("u_admin"),
            message_id="m3",
        )
        result = await runner._handle_deny_command(admin_event)

        assert entry.event.is_set()
        assert entry.result == "deny"
        assert trig_key not in _gateway_queues
        assert "no pending" not in result.lower()


# ------------------------------------------------------------------
# Approval gate: "approved = admin" — the individual (DM) allowlist decides
# who may /approve·/deny, NOT group membership.
# ------------------------------------------------------------------


class TestApprovalUsesDmAllowlist:
    """When no explicit approval-admin list is configured (the HSM/Mare
    default), the /approve|/deny admin gate falls back to "approved = admin":
    a user counts as an admin iff they are in the individual (DM) allowlist —
    the same identity DM-auth and group-invite approval already use. A user
    who only reaches the bot through an approved *group* is a normal
    participant and cannot approve dangerous commands. Otherwise an empty
    config admin list would treat every group member as an admin (the gap this
    fixes)."""

    def setup_method(self):
        _clear_approval_state()

    @staticmethod
    def _group_source(user_id: str) -> SessionSource:
        return SessionSource(
            platform=Platform.TELEGRAM,
            user_id=user_id,
            chat_id="grp",
            user_name=user_id,
            chat_type="group",
        )

    @staticmethod
    def _runner():
        runner = _make_runner()
        # No config.yaml admin list — "approved = admin" governs.
        runner.config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
        )
        # Only u_admin is in the individual (DM) allowlist; that predicate is
        # what the gate consults. (The real predicate is exercised directly in
        # TestIndividualAllowlist.)
        runner._is_individual_allowlisted = lambda source: source.user_id == "u_admin"
        return runner

    @pytest.mark.asyncio
    async def test_group_only_user_rejected_even_for_own_session(self):
        """A group-only user (not in the DM allowlist) cannot /approve — even a
        command pending in their OWN session. Without the allowlist fallback the
        empty config list would treat everyone as admin and let this through."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = self._runner()
        cam = self._group_source("u_cam")
        key = runner._session_key_for_source(cam)
        entry = _ApprovalEntry({"command": "rm -rf /", "description": "scan"})
        _gateway_queues[key] = [entry]

        result = await runner._handle_approve_command(
            MessageEvent(text="/approve", source=cam, message_id="m1")
        )

        assert not entry.event.is_set()
        assert "not authorized" in result.lower()

    @pytest.mark.asyncio
    async def test_dm_allowlisted_user_can_approve(self):
        """A user in the individual (DM) allowlist can /approve (and clears a
        sibling per-user session via the cross-session resolver), even when the
        /approve arrives from a group."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = self._runner()
        cam = self._group_source("u_cam")
        key = runner._session_key_for_source(cam)
        entry = _ApprovalEntry({"command": "ffprobe x", "description": "scan"})
        _gateway_queues[key] = [entry]

        result = await runner._handle_approve_command(
            MessageEvent(text="/approve", source=self._group_source("u_admin"), message_id="m2")
        )

        assert entry.event.is_set()
        assert "not authorized" not in result.lower()

    @pytest.mark.asyncio
    async def test_group_only_user_deny_rejected(self):
        """Same gate applies to /deny."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = self._runner()
        cam = self._group_source("u_cam")
        key = runner._session_key_for_source(cam)
        entry = _ApprovalEntry({"command": "rm -rf /", "description": "scan"})
        _gateway_queues[key] = [entry]

        result = await runner._handle_deny_command(
            MessageEvent(text="/deny", source=cam, message_id="m3")
        )

        assert not entry.event.is_set()
        assert "not authorized" in result.lower()

    @pytest.mark.asyncio
    async def test_toggle_off_lets_anyone_approve(self):
        """When the operator turns the "Admins only" toggle off
        (approvals.admin_only = False), the gate is skipped entirely and any
        participant may /approve — the HSM setting still governs on/off,
        independent of the allowlist."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = self._runner()  # u_cam is NOT in the DM allowlist
        cam = self._group_source("u_cam")
        key = runner._session_key_for_source(cam)
        entry = _ApprovalEntry({"command": "ffprobe x", "description": "scan"})
        _gateway_queues[key] = [entry]

        with patch("tools.approval_context._get_approval_config", return_value={"admin_only": False}):
            result = await runner._handle_approve_command(
                MessageEvent(text="/approve", source=cam, message_id="m4")
            )

        assert entry.event.is_set()
        assert "not authorized" not in result.lower()


# ------------------------------------------------------------------
# The individual-allowlist predicate the approval gate falls back to.
# Exercises the REAL _is_individual_allowlisted (not a stub) so the
# fail-closed + UUID-alt guarantees are actually tested.
# ------------------------------------------------------------------

# Env keys neutralized in every case so the host environment can't leak in.
_ISO_ENV_OFF = {
    "GATEWAY_ALLOWED_USERS": "",
    "GATEWAY_ALLOW_ALL_USERS": "",
    "TELEGRAM_ALLOWED_USERS": "",
    "TELEGRAM_ALLOW_ALL_USERS": "",
    "SIGNAL_ALLOWED_USERS": "",
    "SIGNAL_ALLOW_ALL_USERS": "",
    "WHATSAPP_ALLOWED_USERS": "",
    "WHATSAPP_ALLOW_ALL_USERS": "",
}


class TestIndividualAllowlist:
    """Direct tests of _is_individual_allowlisted — the "approved = admin"
    predicate. Must be fail-closed: only the individual (DM) allowlist, the
    pairing store, an allow-all flag, or a wildcard grants approval rights;
    group membership and adapter own-policy trust must NOT."""

    @staticmethod
    def _runner():
        # Bare instance so the REAL _is_individual_allowlisted runs (not the
        # _make_runner stub). The method only needs os.environ and the source.
        from gateway.run import GatewayRunner
        return object.__new__(GatewayRunner)

    @staticmethod
    def _src(platform, user_id, *, user_id_alt=None, chat_type="group"):
        return SessionSource(
            platform=platform,
            user_id=user_id,
            chat_id="g1",
            user_name=user_id,
            chat_type=chat_type,
            user_id_alt=user_id_alt,
        )

    def test_dm_allowlisted_user_matches(self):
        runner = self._runner()
        env = {**_ISO_ENV_OFF, "TELEGRAM_ALLOWED_USERS": "u_admin"}
        with patch.dict(os.environ, env, clear=False):
            assert runner._is_individual_allowlisted(self._src(Platform.TELEGRAM, "u_admin"))
            assert not runner._is_individual_allowlisted(self._src(Platform.TELEGRAM, "u_cam"))

    def test_group_only_user_fails_closed_with_no_allowlist(self):
        """MUST-FIX: on an own-policy adapter (WhatsApp) with no individual
        allowlist, a group-only user is NOT an approval admin. The previous
        _is_user_authorized reuse fell open here via the adapter-trust shortcut."""
        runner = self._runner()
        with patch.dict(os.environ, _ISO_ENV_OFF, clear=False):
            assert not runner._is_individual_allowlisted(self._src(Platform.WHATSAPP, "u_group"))

    def test_signal_uuid_alt_matches_when_listed_by_uuid(self):
        """SHOULD-FIX: a user listed by one id form is recognized via user_id_alt
        — e.g. allowlist holds the UUID, the message carries the phone as user_id
        and the UUID as user_id_alt."""
        runner = self._runner()
        env = {**_ISO_ENV_OFF, "SIGNAL_ALLOWED_USERS": "uuid-abc-123"}
        with patch.dict(os.environ, env, clear=False):
            src = self._src(Platform.SIGNAL, "+15550001111", user_id_alt="uuid-abc-123")
            assert runner._is_individual_allowlisted(src)
            # A different user (neither id listed) is rejected.
            other = self._src(Platform.SIGNAL, "+15559998888", user_id_alt="uuid-zzz-999")
            assert not runner._is_individual_allowlisted(other)

    def test_wildcard_allows_anyone(self):
        runner = self._runner()
        env = {**_ISO_ENV_OFF, "TELEGRAM_ALLOWED_USERS": "*"}
        with patch.dict(os.environ, env, clear=False):
            assert runner._is_individual_allowlisted(self._src(Platform.TELEGRAM, "anyone"))

    def test_allow_all_flag_allows_anyone(self):
        runner = self._runner()
        env = {**_ISO_ENV_OFF, "SIGNAL_ALLOW_ALL_USERS": "true"}
        with patch.dict(os.environ, env, clear=False):
            assert runner._is_individual_allowlisted(self._src(Platform.SIGNAL, "anyone"))

# Regression: cross-session approval routing isolation (#24100)
# ------------------------------------------------------------------


class TestCrossSessionApprovalIsolation:
    """Regression for #24100.

    The gateway used to write the per-turn session key to the
    process-global ``os.environ["HERMES_SESSION_KEY"]`` inside
    ``GatewayRunner._run_agent``. Because ``os.environ`` is process-global,
    a concurrent gateway session (e.g. a second Discord thread) clobbered
    the value, and a tool worker thread whose approval contextvar was unset
    fell back to ``os.environ`` and read the *wrong* session key — routing
    the approval prompt to the wrong thread.

    The fix removes that ``os.environ`` write; routing is driven solely by
    the ``_approval_session_key`` contextvar. These tests assert that a
    worker thread carrying session A's contextvar resolves to session A
    even when ``os.environ`` has been clobbered to session B.
    """

    @pytest.fixture(autouse=True)
    def _manual_approval_mode(self, monkeypatch):
        monkeypatch.setattr("tools.approval_context._get_approval_mode", lambda: "manual")

    def setup_method(self):
        _clear_approval_state()
        os.environ.pop("HERMES_SESSION_KEY", None)

    def teardown_method(self):
        os.environ.pop("HERMES_SESSION_KEY", None)

    def test_contextvar_wins_over_clobbered_environ(self):
        """get_current_session_key honors the contextvar, not stale env."""
        from tools.approval import get_current_session_key
        from tools.approval_context import reset_current_session_key, set_current_session_key

        # Simulate a concurrent session B having written process-global env
        # last (the "last writer wins" clobber that caused #24100).
        os.environ["HERMES_SESSION_KEY"] = "session-B"

        token = set_current_session_key("session-A")
        try:
            # The worker running under session A's contextvar must resolve
            # to session A, NOT the env-pinned session B.
            assert get_current_session_key() == "session-A"
        finally:
            reset_current_session_key(token)

    def test_unset_contextvar_does_not_fall_back_to_clobbered_environ(self):
        """The resolver must not leak a concurrent session's clobbered
        ``os.environ`` value once the session-context vars are cleared (#24100).

        This exercises the resolver contract directly (not a separate worker
        thread): while the session-context var holds a key, resolution returns
        it; after ``clear_session_vars`` marks the vars explicitly cleared, the
        ``os.environ`` fallback is suppressed and resolution must NOT return the
        stale, process-global value a concurrent session wrote. Under the buggy
        gateway that value would be another live session's key; here we assert
        the resolver never surfaces it.
        """
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools.approval import get_current_session_key

        # Simulate: concurrent session B was the last to clobber os.environ
        # under the OLD buggy gateway. With the fix this write never happens,
        # but we set it here to prove the resolver no longer trusts it once
        # the session-context contextvars are explicitly cleared (as the
        # gateway does in its finally block via clear_session_vars()).
        os.environ["HERMES_SESSION_KEY"] = "session-B-stale"

        # The gateway explicitly clears its session contextvars at turn end;
        # clear_session_vars sets them to "" to *suppress* the os.environ
        # fallback. A bare worker thread therefore must NOT see session-B.
        tokens = set_session_vars(session_key="session-A")
        try:
            assert get_current_session_key() == "session-A"
        finally:
            clear_session_vars(tokens)

        # After clearing, resolution returns the explicit empty/default —
        # never the clobbered process-global value from session B.
        assert get_current_session_key() != "session-B-stale", (
            "resolver leaked a concurrent session's clobbered os.environ "
            "value — #24100 regression"
        )

    def test_approval_prompt_routes_to_originating_session(self):
        """A dangerous command in session A's worker thread notifies
        session A's callback, even though os.environ points at session B."""
        from tools.approval import check_all_command_guards, register_gateway_notify, resolve_gateway_approval, unregister_gateway_notify
        from tools.approval_context import reset_current_session_key, set_current_session_key
        notified_a = []
        notified_b = []
        register_gateway_notify("session-A", lambda d: notified_a.append(d))
        register_gateway_notify("session-B", lambda d: notified_b.append(d))

        # Concurrent session B clobbered the process-global env var last.
        os.environ["HERMES_SESSION_KEY"] = "session-B"
        os.environ["HERMES_GATEWAY_SESSION"] = "1"
        os.environ["HERMES_EXEC_ASK"] = "1"

        result_holder = [None]

        def worker_a():
            # This worker belongs to session A — only its contextvar is set;
            # it deliberately does NOT touch os.environ (mirroring the fixed
            # gateway, which no longer writes HERMES_SESSION_KEY).
            token = set_current_session_key("session-A")
            try:
                result_holder[0] = check_all_command_guards(
                    "rm -rf /important", "local"
                )
            finally:
                reset_current_session_key(token)

        t = threading.Thread(target=worker_a)
        t.start()
        try:
            _wait_until(lambda: notified_a or notified_b)

            # The prompt must land in session A (the originator), never B.
            assert len(notified_a) == 1, "approval prompt did not route to session A"
            assert len(notified_b) == 0, "approval prompt leaked to session B (#24100)"
            assert "rm -rf /important" in notified_a[0]["command"]

            resolve_gateway_approval("session-A", "once")
            t.join(timeout=5)
            assert result_holder[0] is not None
            assert result_holder[0]["approved"] is True
        finally:
            os.environ.pop("HERMES_GATEWAY_SESSION", None)
            os.environ.pop("HERMES_EXEC_ASK", None)
            unregister_gateway_notify("session-A")
            unregister_gateway_notify("session-B")

    def test_two_concurrent_sessions_route_to_own_queue_contextvar_only(self):
        """Cross-session isolation driven by contextvars ALONE (#24100).

        Two concurrent worker threads with DISTINCT session keys each set
        only ``set_current_session_key()`` — they deliberately never write
        ``os.environ["HERMES_SESSION_KEY"]``. This proves the contextvar is
        sufficient post-fix, and would FAIL if contextvar routing regressed
        (the prior 'parallel' tests share one key and dual-set env+contextvar,
        so they cannot guard this invariant). Each session's dangerous command
        must land in its OWN gateway queue, and resolving one must not resolve
        the other.
        """
        from tools.approval import _gateway_queues, check_all_command_guards, register_gateway_notify, resolve_gateway_approval, unregister_gateway_notify
        from tools.approval_context import reset_current_session_key, set_current_session_key

        # No HERMES_SESSION_KEY in os.environ at all — pure contextvar routing.
        os.environ.pop("HERMES_SESSION_KEY", None)
        os.environ["HERMES_GATEWAY_SESSION"] = "1"
        os.environ["HERMES_EXEC_ASK"] = "1"

        register_gateway_notify("sess-A", lambda d: None)
        register_gateway_notify("sess-B", lambda d: None)

        results = {"sess-A": None, "sess-B": None}

        def worker(key, cmd):
            token = set_current_session_key(key)
            try:
                results[key] = check_all_command_guards(cmd, "local")
            finally:
                reset_current_session_key(token)

        ta = threading.Thread(target=worker, args=("sess-A", "rm -rf /a-data"))
        tb = threading.Thread(target=worker, args=("sess-B", "rm -rf /b-data"))
        ta.start()
        tb.start()
        try:
            # Wait until both sessions have a pending approval in their queue.
            assert _wait_until(
                lambda: len(_gateway_queues.get("sess-A", [])) >= 1
                and len(_gateway_queues.get("sess-B", [])) >= 1
            ), "both sessions never reached a pending approval"

            # Each command must be parked in its OWN session queue.
            qa = _gateway_queues.get("sess-A", [])
            qb = _gateway_queues.get("sess-B", [])
            assert len(qa) == 1, f"sess-A queue should hold 1, got {len(qa)}"
            assert len(qb) == 1, f"sess-B queue should hold 1, got {len(qb)}"

            # Resolve ONLY sess-A; sess-B must stay blocked (no cross-leak).
            resolve_gateway_approval("sess-A", "once")
            ta.join(timeout=5)
            assert results["sess-A"] is not None
            assert results["sess-A"]["approved"] is True
            assert results["sess-B"] is None, "sess-B resolved by sess-A's approval (#24100)"
            assert len(_gateway_queues.get("sess-B", [])) == 1

            # Now resolve sess-B independently.
            resolve_gateway_approval("sess-B", "once")
            tb.join(timeout=5)
            assert results["sess-B"] is not None
            assert results["sess-B"]["approved"] is True
        finally:
            resolve_gateway_approval("sess-A", "deny")
            resolve_gateway_approval("sess-B", "deny")
            ta.join(timeout=2)
            tb.join(timeout=2)
            os.environ.pop("HERMES_GATEWAY_SESSION", None)
            os.environ.pop("HERMES_EXEC_ASK", None)
            unregister_gateway_notify("sess-A")
            unregister_gateway_notify("sess-B")


class TestApprovalHandlersStayGated:
    """Guard against the upstream-merge landmine: GatewaySlashCommandsMixin
    ships UNGATED _handle_approve_command/_handle_deny_command copies. The
    fork's admin-gated handlers on GatewayRunner shadow them only because the
    class defines its own — if a cleanup ever deletes the run.py copies, the
    mixin's ungated versions take over silently. Pin both facts."""

    def test_live_approve_and_deny_are_admin_gated(self):
        """Post-v0.22 merge: the gated handlers live once, in the mixin (no
        shadow copy on GatewayRunner). Pin that the handler GatewayRunner
        actually dispatches to carries the admin gate."""
        import inspect

        from gateway.run import GatewayRunner
        from gateway.slash_commands import GatewaySlashCommandsMixin

        for name in ("_handle_approve_command", "_handle_deny_command"):
            live = getattr(GatewayRunner, name)
            assert live is GatewaySlashCommandsMixin.__dict__.get(name)
            assert "_approval_gate" in inspect.getsource(live), (
                f"GatewayRunner.{name} lost its admin gating (_approval_gate missing)."
            )
        gate_src = inspect.getsource(GatewaySlashCommandsMixin._approval_gate)
        assert "_is_approval_admin" in gate_src
        assert "_is_discord_bot_sender" in gate_src
