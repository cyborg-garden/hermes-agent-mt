"""Tests for admin-only approval gating.

Verifies that /approve and /deny commands — and Telegram inline button
callbacks — are blocked for non-admin users when ``approvals.admin_only``
is True (the default).
"""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, build_session_key
from gateway.slash_access import SlashAccessPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ADMIN_POLICY = SlashAccessPolicy(
    enabled=True,
    admin_user_ids=frozenset({"admin1"}),
    user_allowed_commands=frozenset(),
)

_DISABLED_POLICY = SlashAccessPolicy(
    enabled=False,
    admin_user_ids=frozenset(),
    user_allowed_commands=frozenset(),
)


def _make_source(user_id: str = "u1") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id=user_id,
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str, user_id: str = "u1") -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(user_id=user_id),
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
    runner._set_session_env = lambda _context: None
    return runner


def _clear_approval_state():
    from tools import approval as mod
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()


# ===========================================================================
# /approve admin gating
# ===========================================================================


class TestApproveAdminGating:
    """Non-admin users should be blocked from /approve when admin_only=True."""

    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_non_admin_blocked_when_admin_only_true(self):
        """A non-admin user gets the denial message."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = _make_runner()
        source = _make_source(user_id="nonadmin")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "rm -rf /"})
        _gateway_queues[session_key] = [entry]

        with patch("tools.approval_context._get_approval_config", return_value={"admin_only": True}):
            with patch("gateway.slash_access.policy_for_source", return_value=_ADMIN_POLICY):
                result = await runner._handle_approve_command(
                    _make_event("/approve", user_id="nonadmin")
                )

        assert "Not authorized" in result
        assert not entry.event.is_set()

    @pytest.mark.asyncio
    async def test_admin_allowed_when_admin_only_true(self):
        """An admin user can still approve."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = _make_runner()
        source = _make_source(user_id="admin1")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        with patch("tools.approval_context._get_approval_config", return_value={"admin_only": True}):
            with patch("gateway.slash_access.policy_for_source", return_value=_ADMIN_POLICY):
                result = await runner._handle_approve_command(
                    _make_event("/approve", user_id="admin1")
                )

        assert "approved" in result.lower() or "resuming" in result.lower()
        assert entry.event.is_set()

    @pytest.mark.asyncio
    async def test_anyone_allowed_when_admin_only_false(self):
        """When admin_only is False, any authorized user can approve."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = _make_runner()
        source = _make_source(user_id="nonadmin")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        with patch("tools.approval_context._get_approval_config", return_value={"admin_only": False}):
            result = await runner._handle_approve_command(
                _make_event("/approve", user_id="nonadmin")
            )

        assert "approved" in result.lower() or "resuming" in result.lower()
        assert entry.event.is_set()

    @pytest.mark.asyncio
    async def test_policy_disabled_falls_back_to_individual_allowlist(self):
        """When no explicit slash_access admin list is configured, the gate
        falls back to "approved = admin": the individual (DM) allowlist decides.
        A non-allowlisted user is denied — previously this path treated everyone
        as admin (the bug). An allowlisted user is admitted."""
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = _make_runner()
        runner._is_individual_allowlisted = lambda source: source.user_id == "approved_user"

        # Non-allowlisted user is rejected even though slash policy is disabled.
        blocked_src = _make_source(user_id="anyone")
        blocked_key = runner._session_key_for_source(blocked_src)
        blocked_entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[blocked_key] = [blocked_entry]
        with patch("tools.approval_context._get_approval_config", return_value={"admin_only": True}):
            with patch("gateway.slash_access.policy_for_source", return_value=_DISABLED_POLICY):
                blocked = await runner._handle_approve_command(
                    _make_event("/approve", user_id="anyone")
                )
        assert "Not authorized" in blocked
        assert not blocked_entry.event.is_set()

        # Allowlisted user is admitted.
        ok_src = _make_source(user_id="approved_user")
        ok_key = runner._session_key_for_source(ok_src)
        ok_entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[ok_key] = [ok_entry]
        with patch("tools.approval_context._get_approval_config", return_value={"admin_only": True}):
            with patch("gateway.slash_access.policy_for_source", return_value=_DISABLED_POLICY):
                ok = await runner._handle_approve_command(
                    _make_event("/approve", user_id="approved_user")
                )
        assert "approved" in ok.lower() or "resuming" in ok.lower()
        assert ok_entry.event.is_set()


# ===========================================================================
# /deny admin gating
# ===========================================================================


class TestDenyAdminGating:
    """Non-admin users should be blocked from /deny when admin_only=True."""

    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_non_admin_blocked_when_admin_only_true(self):
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = _make_runner()
        source = _make_source(user_id="nonadmin")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "rm -rf /"})
        _gateway_queues[session_key] = [entry]

        with patch("tools.approval_context._get_approval_config", return_value={"admin_only": True}):
            with patch("gateway.slash_access.policy_for_source", return_value=_ADMIN_POLICY):
                result = await runner._handle_deny_command(
                    _make_event("/deny", user_id="nonadmin")
                )

        assert "Not authorized" in result
        assert not entry.event.is_set()

    @pytest.mark.asyncio
    async def test_admin_can_deny(self):
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = _make_runner()
        source = _make_source(user_id="admin1")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        with patch("tools.approval_context._get_approval_config", return_value={"admin_only": True}):
            with patch("gateway.slash_access.policy_for_source", return_value=_ADMIN_POLICY):
                result = await runner._handle_deny_command(
                    _make_event("/deny", user_id="admin1")
                )

        assert "denied" in result.lower()
        assert entry.event.is_set()
        assert entry.result == "deny"

    @pytest.mark.asyncio
    async def test_anyone_can_deny_when_admin_only_false(self):
        from tools.approval import _gateway_queues
        from tools.approval_gateway_wait import _ApprovalEntry

        runner = _make_runner()
        source = _make_source(user_id="nonadmin")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        with patch("tools.approval_context._get_approval_config", return_value={"admin_only": False}):
            result = await runner._handle_deny_command(
                _make_event("/deny", user_id="nonadmin")
            )

        assert "denied" in result.lower()
        assert entry.event.is_set()


# ===========================================================================
# Telegram button callback admin gating
# ===========================================================================


def _ensure_telegram_mock():
    """Wire up minimal mocks to import TelegramAdapter."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class TestTelegramCallbackAdminGating:
    """Telegram inline button clicks should be blocked for non-admin users."""

    @pytest.mark.asyncio
    async def test_non_admin_blocked_on_button_click(self):
        adapter = _make_adapter()
        adapter._approval_state[10] = "agent:main:telegram:group:12345:99"

        query = AsyncMock()
        query.data = "ea:once:10"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.chat = MagicMock()
        query.message.chat.type = "private"
        query.message.message_thread_id = None
        query.from_user = MagicMock()
        query.from_user.id = "777"
        query.from_user.first_name = "NonAdmin"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval_context._get_approval_config", return_value={"admin_only": True}):
                with patch.object(adapter, "_is_user_admin", return_value=False):
                    await adapter._handle_callback_query(update, context)

        query.answer.assert_called_once()
        assert "admin" in query.answer.call_args[1]["text"].lower()
        query.edit_message_text.assert_not_called()
        # State should NOT be consumed
        assert 10 in adapter._approval_state

    @pytest.mark.asyncio
    async def test_admin_allowed_on_button_click(self):
        adapter = _make_adapter()
        adapter._approval_state[11] = "agent:main:telegram:group:12345:99"

        query = AsyncMock()
        query.data = "ea:once:11"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.chat = MagicMock()
        query.message.chat.type = "private"
        query.message.message_thread_id = None
        query.from_user = MagicMock()
        query.from_user.id = "111"
        query.from_user.first_name = "Admin"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval_context._get_approval_config", return_value={"admin_only": True}):
                with patch.object(adapter, "_is_user_admin", return_value=True):
                    with patch("tools.approval.resolve_gateway_approval", return_value=1):
                        await adapter._handle_callback_query(update, context)

        # Should have been resolved — state consumed
        assert 11 not in adapter._approval_state

    @pytest.mark.asyncio
    async def test_anyone_allowed_when_admin_only_false(self):
        adapter = _make_adapter()
        adapter._approval_state[12] = "agent:main:telegram:group:12345:99"

        query = AsyncMock()
        query.data = "ea:once:12"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.chat = MagicMock()
        query.message.chat.type = "private"
        query.message.message_thread_id = None
        query.from_user = MagicMock()
        query.from_user.id = "999"
        query.from_user.first_name = "Anyone"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval_context._get_approval_config", return_value={"admin_only": False}):
                with patch("tools.approval.resolve_gateway_approval", return_value=1):
                    await adapter._handle_callback_query(update, context)

        # Should have been resolved — no admin block
        assert 12 not in adapter._approval_state

    @pytest.mark.asyncio
    async def test_admin_check_error_denies(self):
        """Fail closed: if the admin check raises, the dangerous-command
        approval must be denied, not allowed through."""
        adapter = _make_adapter()
        adapter._approval_state[13] = "agent:main:telegram:group:12345:99"

        query = AsyncMock()
        query.data = "ea:once:13"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.chat = MagicMock()
        query.message.chat.type = "private"
        query.message.message_thread_id = None
        query.from_user = MagicMock()
        query.from_user.id = "111"
        query.from_user.first_name = "Admin"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval_context._get_approval_config", return_value={"admin_only": True}):
                with patch.object(adapter, "_is_user_admin", side_effect=RuntimeError("boom")):
                    with patch("tools.approval.resolve_gateway_approval", return_value=1):
                        await adapter._handle_callback_query(update, context)

        # Denied — state NOT consumed, denial surfaced.
        assert 13 in adapter._approval_state
        assert query.answer.called
        assert "denied" in query.answer.call_args[1]["text"].lower()


# ===========================================================================
# _is_user_admin method
# ===========================================================================


class _FakeAdminRunner:
    """Stands in for GatewayRunner: exposes _is_approval_admin and a bound
    method so the adapter can reach it via _message_handler.__self__."""

    def __init__(self, verdict):
        self._verdict = verdict
        self.seen = None

    def _is_approval_admin(self, source):
        self.seen = source
        return self._verdict

    def _handler(self, *a, **k):  # used purely as a bound method
        pass


# Env baseline so the host environment can't leak into fallback assertions.
_TG_ENV_OFF = {"TELEGRAM_ALLOWED_USERS": "", "GATEWAY_ALLOW_ALL_USERS": ""}


class TestIsUserAdmin:
    """_is_user_admin gates Telegram inline-button approvals. It must mirror the
    slash /approve gate ("approved = admin"): a user's approval authority can't
    depend on whether they clicked a button or typed a command."""

    def test_delegates_to_runner_approval_admin(self):
        """When the gateway runner is wired, _is_user_admin returns the runner's
        _is_approval_admin verdict — the SAME check the slash gate uses — and
        builds a Telegram source from the button context."""
        adapter = _make_adapter()
        runner = _FakeAdminRunner(True)
        adapter._message_handler = runner._handler  # __self__ == runner

        assert adapter._is_user_admin(
            "u1", chat_id="g1", chat_type="supergroup", thread_id="7", user_name="x"
        ) is True
        assert runner.seen is not None
        assert runner.seen.user_id == "u1"
        assert runner.seen.platform == Platform.TELEGRAM
        # supergroup + thread → forum scope
        assert runner.seen.chat_type == "forum"

        # A False verdict from the runner denies.
        adapter._message_handler = _FakeAdminRunner(False)._handler
        assert adapter._is_user_admin("u1", chat_id="g1", chat_type="group") is False

    def test_fallback_any_allowlisted_user_is_admin(self):
        """No runner wired → env-only fallback: ANY user in TELEGRAM_ALLOWED_USERS
        is admin (not just the first), consistent with _is_individual_allowlisted."""
        adapter = _make_adapter()
        env = {**_TG_ENV_OFF, "TELEGRAM_ALLOWED_USERS": "111,222,333"}
        with patch.dict(os.environ, env, clear=False):
            assert adapter._is_user_admin("111") is True
            assert adapter._is_user_admin("222") is True
            assert adapter._is_user_admin("444") is False

    def test_fallback_wildcard_makes_anyone_admin(self):
        """"*" in the allowlist → open mode → anyone is admin (matches the slash
        gate / _is_individual_allowlisted)."""
        adapter = _make_adapter()
        env = {**_TG_ENV_OFF, "TELEGRAM_ALLOWED_USERS": "*"}
        with patch.dict(os.environ, env, clear=False):
            assert adapter._is_user_admin("anyone") is True

    def test_fallback_gateway_allow_all_makes_admin(self):
        adapter = _make_adapter()
        env = {**_TG_ENV_OFF, "GATEWAY_ALLOW_ALL_USERS": "true"}
        with patch.dict(os.environ, env, clear=False):
            assert adapter._is_user_admin("anyone") is True

    def test_fallback_empty_allowed_users_returns_false(self):
        adapter = _make_adapter()
        with patch.dict(os.environ, _TG_ENV_OFF, clear=False):
            assert adapter._is_user_admin("111") is False

    def test_empty_user_id_returns_false(self):
        adapter = _make_adapter()
        assert adapter._is_user_admin("") is False
        assert adapter._is_user_admin(None) is False


# ===========================================================================
# LOCAL sessions (CLI / TUI / desktop): the operator approves their own prompts
# ===========================================================================


class TestLocalSourceApprovalAdmin:
    def _local_source(self):
        return SessionSource(platform=Platform.LOCAL, user_id="local-user", chat_id="local", chat_type="dm")

    @pytest.mark.parametrize("verdict,expected", [(True, True), (False, False), (None, False)])
    def test_local_source_follows_the_local_authority(self, verdict, expected):
        runner = _make_runner()
        with patch("gateway.session_local.authorize_local_source", return_value=verdict) as auth, \
                patch("gateway.slash_access.policy_for_runner_source", return_value=_ADMIN_POLICY):
            assert runner._is_approval_admin(self._local_source()) is expected
        auth.assert_called_once()

    def test_messaging_source_never_consults_the_local_authority(self):
        runner = _make_runner()
        with patch("gateway.session_local.authorize_local_source", return_value=True) as auth, \
                patch("gateway.slash_access.policy_for_runner_source", return_value=_ADMIN_POLICY):
            assert runner._is_approval_admin(_make_source("not-admin")) is False
        auth.assert_not_called()

    def _wired_runner(self):
        """A runner whose LOCAL adapter and authority are real, so the forged-source refusal is
        proven by ``authorize_local_source`` itself rather than by a patched verdict."""
        from gateway.session_authority import LiveSession
        from gateway.session_local import LocalSessionAdapter
        runner = _make_runner()
        authority = SimpleNamespace(sessions={})
        adapter = LocalSessionAdapter(authority)
        runner.session_authority = authority
        runner._adapters_for_profile = lambda _profile: {Platform.LOCAL: adapter}
        live = self._local_source()
        adapter.register_source(live)
        authority.sessions["sid"] = LiveSession(live, "route")
        return runner, live

    def test_the_live_local_source_is_an_approval_admin(self):
        runner, live = self._wired_runner()
        with patch("gateway.slash_access.policy_for_runner_source") as policy:
            assert runner._is_approval_admin(live) is True
        policy.assert_not_called()

    def test_a_forged_copy_of_the_local_source_is_refused(self):
        # Same field values (e.g. rebuilt from relay wire JSON or persisted session data), but not
        # the object the local authority registered: never an approval admin, even with an
        # allow-all messaging policy.
        runner, live = self._wired_runner()
        forged = SessionSource.from_dict(live.to_dict())
        assert forged is not live and forged.to_dict() == live.to_dict()
        with patch("gateway.slash_access.policy_for_runner_source", return_value=_ADMIN_POLICY):
            assert runner._is_approval_admin(forged) is False

    def test_a_local_source_on_a_runtime_without_a_local_adapter_is_refused(self):
        runner, _live = self._wired_runner()
        runner._adapters_for_profile = lambda _profile: {}
        with patch("gateway.slash_access.policy_for_runner_source", return_value=_ADMIN_POLICY):
            assert runner._is_approval_admin(self._local_source()) is False
