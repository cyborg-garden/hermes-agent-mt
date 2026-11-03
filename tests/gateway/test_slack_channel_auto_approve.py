"""
Tests for Slack channel auto-approval (issue #68 — parity with Signal group
reconcile).

When an admin adds the bot to a channel, that channel should be auto-approved
into the runtime allowlist — the same way Signal auto-approves a group the bot
is added to — so it works without a manual ``SLACK_ALLOWED_CHANNELS`` edit.

Two entry points, mirroring Signal:
* ``member_joined_channel`` — the bot is added live (analogous to the Signal
  invite handler, which trusts the *action* of being added).
* ``_reconcile_channels`` — on connect, list the bot's member channels via
  ``users.conversations`` and approve them (analogous to Signal's
  ``_reconcile_groups`` add-at-creation recovery).

Approvals only matter when a static whitelist is active — with no whitelist the
bot already responds everywhere, so approving is pointless churn (this mirrors
Signal reconcile's early return when ``*`` is in the group allowlist).
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Mock slack-bolt if not installed (same pattern as test_slack_mention.py)
# ---------------------------------------------------------------------------

def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod
_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter, _APPROVED_CHANNELS_FILE


BOT_USER_ID = "U_BOT_123"
OTHER_USER_ID = "U_HUMAN_456"
CHANNEL_ID = "C0AQWDLHY9M"
OTHER_CHANNEL_ID = "C9999999999"
TEAM_ID = "T_TEAM_1"


def _make_adapter(allowed_channels=None, channel_join_policy=None):
    """Build a Slack adapter with just the state the auto-approve code needs.

    Uses ``object.__new__`` to avoid the full connect machinery, then wires the
    attributes the methods under test read (mirrors ``_make_adapter`` in
    test_slack_mention.py, plus the auto-approve state ``__init__`` sets up).
    """
    extra = {}
    if allowed_channels is not None:
        extra["allowed_channels"] = allowed_channels
    if channel_join_policy is not None:
        extra["channel_join_policy"] = channel_join_policy

    adapter = object.__new__(SlackAdapter)
    adapter.platform = Platform.SLACK
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._bot_user_id = BOT_USER_ID
    adapter._team_bot_user_ids = {TEAM_ID: BOT_USER_ID}
    adapter._channel_team = {}
    adapter._team_clients = {}
    adapter._runtime_approved_channels = set()
    return adapter


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    """Point HERMES_HOME at a tmp dir so persistence never touches real state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    monkeypatch.delenv("SLACK_CHANNEL_JOIN_POLICY", raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# _channel_join_policy
# ---------------------------------------------------------------------------

def test_join_policy_defaults_to_auto():
    assert _make_adapter()._channel_join_policy() == "auto"


def test_join_policy_disabled_via_config():
    adapter = _make_adapter(channel_join_policy="disabled")
    assert adapter._channel_join_policy() == "disabled"


def test_join_policy_env_fallback(monkeypatch):
    monkeypatch.setenv("SLACK_CHANNEL_JOIN_POLICY", "disabled")
    assert _make_adapter()._channel_join_policy() == "disabled"


def test_join_policy_unknown_value_defaults_to_auto():
    assert _make_adapter(channel_join_policy="banana")._channel_join_policy() == "auto"


# ---------------------------------------------------------------------------
# _channel_approval_active
# ---------------------------------------------------------------------------

def test_approval_inactive_without_whitelist():
    # No whitelist → bot responds everywhere → approval is a no-op.
    assert _make_adapter()._channel_approval_active() is False


def test_approval_inactive_with_wildcard():
    assert _make_adapter(allowed_channels="*")._channel_approval_active() is False


def test_approval_active_with_whitelist():
    assert _make_adapter(allowed_channels=[OTHER_CHANNEL_ID])._channel_approval_active() is True


def test_approval_inactive_when_policy_disabled():
    adapter = _make_adapter(allowed_channels=[OTHER_CHANNEL_ID], channel_join_policy="disabled")
    assert adapter._channel_approval_active() is False


# ---------------------------------------------------------------------------
# _slack_allowed_channels union behavior
# ---------------------------------------------------------------------------

def test_allowed_channels_unions_runtime_approvals():
    adapter = _make_adapter(allowed_channels=[OTHER_CHANNEL_ID])
    adapter._runtime_approved_channels = {CHANNEL_ID}
    result = adapter._slack_allowed_channels()
    assert result == {OTHER_CHANNEL_ID, CHANNEL_ID}


def test_empty_whitelist_ignores_runtime_approvals():
    # Critical: an unrestricted install must STAY unrestricted even if the
    # persisted-approvals set somehow has entries.
    adapter = _make_adapter()
    adapter._runtime_approved_channels = {CHANNEL_ID}
    assert adapter._slack_allowed_channels() == set()


def test_wildcard_whitelist_ignores_runtime_approvals():
    adapter = _make_adapter(allowed_channels="*")
    adapter._runtime_approved_channels = {CHANNEL_ID}
    assert adapter._slack_allowed_channels() == {"*"}


# ---------------------------------------------------------------------------
# _approve_channel
# ---------------------------------------------------------------------------

def test_approve_channel_adds_and_persists(tmp_path):
    adapter = _make_adapter(allowed_channels=[OTHER_CHANNEL_ID])
    assert adapter._approve_channel(CHANNEL_ID, team_id=TEAM_ID, added_by=OTHER_USER_ID) is True
    assert CHANNEL_ID in adapter._runtime_approved_channels
    # Persisted to disk.
    data = json.loads((tmp_path / _APPROVED_CHANNELS_FILE).read_text())
    assert CHANNEL_ID in data
    assert data[CHANNEL_ID]["team_id"] == TEAM_ID
    assert data[CHANNEL_ID]["added_by"] == OTHER_USER_ID


def test_approve_channel_idempotent():
    adapter = _make_adapter(allowed_channels=[OTHER_CHANNEL_ID])
    assert adapter._approve_channel(CHANNEL_ID) is True
    assert adapter._approve_channel(CHANNEL_ID) is False  # already approved


def test_approve_channel_noop_when_inactive(tmp_path):
    # No whitelist → not active → do not approve or persist.
    adapter = _make_adapter()
    assert adapter._approve_channel(CHANNEL_ID) is False
    assert CHANNEL_ID not in adapter._runtime_approved_channels
    assert not (tmp_path / _APPROVED_CHANNELS_FILE).exists()


def test_approve_channel_skips_already_configured():
    adapter = _make_adapter(allowed_channels=[CHANNEL_ID])
    # Already statically configured → nothing to add.
    assert adapter._approve_channel(CHANNEL_ID) is False


# ---------------------------------------------------------------------------
# Persistence round-trip / load
# ---------------------------------------------------------------------------

def test_load_approved_channels_round_trip(tmp_path):
    seed = _make_adapter(allowed_channels=[OTHER_CHANNEL_ID])
    seed._approve_channel(CHANNEL_ID, team_id=TEAM_ID)

    # New adapter that actually runs the loader.
    fresh = object.__new__(SlackAdapter)
    fresh._runtime_approved_channels = set()
    fresh._load_approved_channels()
    assert CHANNEL_ID in fresh._runtime_approved_channels


def test_load_corrupt_file_is_non_fatal(tmp_path):
    (tmp_path / _APPROVED_CHANNELS_FILE).write_text("{ not json")
    fresh = object.__new__(SlackAdapter)
    fresh._runtime_approved_channels = set()
    fresh._load_approved_channels()  # must not raise
    assert fresh._runtime_approved_channels == set()


# ---------------------------------------------------------------------------
# member_joined_channel handler
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_member_joined_approves_when_bot_added():
    adapter = _make_adapter(allowed_channels=[OTHER_CHANNEL_ID])
    await adapter._handle_member_joined_channel(
        {"channel": CHANNEL_ID, "user": BOT_USER_ID, "team": TEAM_ID, "inviter": OTHER_USER_ID}
    )
    assert CHANNEL_ID in adapter._runtime_approved_channels


@pytest.mark.asyncio
async def test_member_joined_ignores_other_users():
    adapter = _make_adapter(allowed_channels=[OTHER_CHANNEL_ID])
    await adapter._handle_member_joined_channel(
        {"channel": CHANNEL_ID, "user": OTHER_USER_ID, "team": TEAM_ID}
    )
    assert CHANNEL_ID not in adapter._runtime_approved_channels


@pytest.mark.asyncio
async def test_member_joined_noop_without_whitelist():
    adapter = _make_adapter()  # no whitelist → nothing to approve
    await adapter._handle_member_joined_channel(
        {"channel": CHANNEL_ID, "user": BOT_USER_ID, "team": TEAM_ID}
    )
    assert CHANNEL_ID not in adapter._runtime_approved_channels


@pytest.mark.asyncio
async def test_member_joined_malformed_event_non_fatal():
    adapter = _make_adapter(allowed_channels=[OTHER_CHANNEL_ID])
    await adapter._handle_member_joined_channel("not-a-dict")  # must not raise


# ---------------------------------------------------------------------------
# _reconcile_channels
# ---------------------------------------------------------------------------

def _client_returning(channels, next_cursor=""):
    client = MagicMock()
    client.users_conversations = AsyncMock(
        return_value={
            "channels": channels,
            "response_metadata": {"next_cursor": next_cursor},
        }
    )
    return client


@pytest.mark.asyncio
async def test_reconcile_approves_member_channels():
    adapter = _make_adapter(allowed_channels=[OTHER_CHANNEL_ID])
    adapter._team_clients = {
        TEAM_ID: _client_returning([{"id": CHANNEL_ID}, {"id": "C_SECOND"}])
    }
    await adapter._reconcile_channels()
    assert CHANNEL_ID in adapter._runtime_approved_channels
    assert "C_SECOND" in adapter._runtime_approved_channels


@pytest.mark.asyncio
async def test_reconcile_noop_without_whitelist():
    adapter = _make_adapter()  # unrestricted → skip entirely
    client = _client_returning([{"id": CHANNEL_ID}])
    adapter._team_clients = {TEAM_ID: client}
    await adapter._reconcile_channels()
    assert adapter._runtime_approved_channels == set()
    client.users_conversations.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_noop_when_disabled():
    adapter = _make_adapter(allowed_channels=[OTHER_CHANNEL_ID], channel_join_policy="disabled")
    client = _client_returning([{"id": CHANNEL_ID}])
    adapter._team_clients = {TEAM_ID: client}
    await adapter._reconcile_channels()
    assert adapter._runtime_approved_channels == set()
    client.users_conversations.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_api_failure_non_fatal():
    adapter = _make_adapter(allowed_channels=[OTHER_CHANNEL_ID])
    client = MagicMock()
    client.users_conversations = AsyncMock(side_effect=RuntimeError("slack down"))
    adapter._team_clients = {TEAM_ID: client}
    await adapter._reconcile_channels()  # must not raise
    assert adapter._runtime_approved_channels == set()


@pytest.mark.asyncio
async def test_reconcile_paginates():
    adapter = _make_adapter(allowed_channels=[OTHER_CHANNEL_ID])
    client = MagicMock()
    client.users_conversations = AsyncMock(
        side_effect=[
            {"channels": [{"id": CHANNEL_ID}], "response_metadata": {"next_cursor": "next"}},
            {"channels": [{"id": "C_PAGE2"}], "response_metadata": {"next_cursor": ""}},
        ]
    )
    adapter._team_clients = {TEAM_ID: client}
    await adapter._reconcile_channels()
    assert CHANNEL_ID in adapter._runtime_approved_channels
    assert "C_PAGE2" in adapter._runtime_approved_channels
    assert client.users_conversations.await_count == 2
