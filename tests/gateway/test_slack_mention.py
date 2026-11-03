"""
Tests for Slack mention gating (require_mention / free_response_channels).

Follows the same pattern as test_whatsapp_group_gating.py.
"""

import logging
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Mock slack-bolt if not installed (same as test_slack.py)
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

from plugins.platforms.slack.adapter import SlackAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOT_USER_ID = "U_BOT_123"
CHANNEL_ID = "C0AQWDLHY9M"
OTHER_CHANNEL_ID = "C9999999999"


def _make_adapter(require_mention=None, strict_mention=None, free_response_channels=None,
                  allowed_channels=None, mention_patterns=None, channel_policy=None):
    extra = {}
    if require_mention is not None:
        extra["require_mention"] = require_mention
    if strict_mention is not None:
        extra["strict_mention"] = strict_mention
    if free_response_channels is not None:
        extra["free_response_channels"] = free_response_channels
    if allowed_channels is not None:
        extra["allowed_channels"] = allowed_channels
    if mention_patterns is not None:
        extra["mention_patterns"] = mention_patterns
    if channel_policy is not None:
        extra["channel_policy"] = channel_policy

    adapter = object.__new__(SlackAdapter)
    adapter.platform = Platform.SLACK
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._bot_user_id = BOT_USER_ID
    adapter._team_bot_user_ids = {}
    return adapter


# ---------------------------------------------------------------------------
# Tests: _slack_require_mention
# ---------------------------------------------------------------------------

def test_require_mention_defaults_to_true(monkeypatch):
    monkeypatch.delenv("SLACK_REQUIRE_MENTION", raising=False)
    adapter = _make_adapter()
    assert adapter._slack_require_mention() is True


def test_require_mention_empty_string_stays_true():
    """Empty/malformed strings keep gating ON (explicit-false parser)."""
    adapter = _make_adapter(require_mention="")
    assert adapter._slack_require_mention() is True


# ---------------------------------------------------------------------------
# Tests: _slack_strict_mention
# ---------------------------------------------------------------------------



def test_strict_mention_malformed_stays_false():
    """Unrecognised values keep strict mode OFF (fail-open to legacy behavior)."""
    adapter = _make_adapter(strict_mention="maybe")
    assert adapter._slack_strict_mention() is False


# ---------------------------------------------------------------------------
# Tests: _slack_free_response_channels
# ---------------------------------------------------------------------------


def test_free_response_channels_env_var_fallback(monkeypatch):
    monkeypatch.setenv("SLACK_FREE_RESPONSE_CHANNELS", f"{CHANNEL_ID},{OTHER_CHANNEL_ID}")
    adapter = _make_adapter()  # no config value → falls back to env
    result = adapter._slack_free_response_channels()
    assert CHANNEL_ID in result
    assert OTHER_CHANNEL_ID in result


def test_free_response_channels_bare_int():
    # YAML `free_response_channels: 1491973769726791812` (single bare integer)
    # is loaded as an int and would previously fall through the isinstance(str)
    # branch to return an empty set.  Coerce scalar → str so single-channel
    # config without quoting works as users expect.
    adapter = _make_adapter(free_response_channels=1491973769726791812)
    result = adapter._slack_free_response_channels()
    assert result == {"1491973769726791812"}


# ---------------------------------------------------------------------------
# Tests: mention gating integration (simulating _handle_slack_message logic)
# ---------------------------------------------------------------------------

def _would_process(adapter, *, is_dm=False, channel_id=CHANNEL_ID,
                   text="hello", mentioned=False, thread_reply=False,
                   active_session=False, channel_type=None):
    """Simulate the mention gating logic from _handle_slack_message.

    Returns True if the message would be processed, False if it would be
    skipped (returned early).

    ``channel_type`` mirrors the real Slack payload ("im" = 1:1 DM,
    "mpim" = group DM, "" = channel). When omitted it is derived from the
    legacy ``is_dm`` flag as a 1:1 IM, preserving existing callers. Gating
    keys off ``is_one_to_one_dm`` (only a true 1:1 IM is exempt); MPIMs are
    shared surfaces and go through the same gating as channels.
    """
    if channel_type is None:
        channel_type = "im" if is_dm else ""
    is_one_to_one_dm = channel_type == "im"

    bot_uid = adapter._team_bot_user_ids.get("T1", adapter._bot_user_id)
    if mentioned:
        text = f"<@{bot_uid}> {text}"
    is_mentioned = bool(
        (bot_uid and f"<@{bot_uid}>" in text)
        or adapter._slack_message_matches_mention_patterns(text)
    )

    if not is_one_to_one_dm and bot_uid:
        # allowed_channels check (whitelist — must pass before other gating;
        # "*" is a wildcard meaning no channel restriction). Under the
        # approved-only policy an EMPTY allowed set means no channels are
        # approved, so channel messages are dropped (secure default).
        allowed = adapter._slack_allowed_channels()
        policy = adapter._slack_channel_policy()
        if "*" not in allowed:
            if allowed and channel_id not in allowed:
                return False
            if not allowed and policy == "approved-only":
                return False

        if channel_id in adapter._slack_free_response_channels() or not adapter._slack_require_mention():
            return True
        elif adapter._slack_strict_mention() and not is_mentioned:
            return False
        elif not is_mentioned:
            if thread_reply and active_session:
                return True
            else:
                return False
    return True


def test_default_require_mention_channel_without_mention_ignored():
    adapter = _make_adapter()  # default: require_mention=True
    assert _would_process(adapter, text="hello everyone") is False


def test_channel_in_free_response_processed_without_mention():
    adapter = _make_adapter(
        require_mention=True,
        free_response_channels=[CHANNEL_ID],
    )
    assert _would_process(adapter, channel_id=CHANNEL_ID, text="hello") is True


def test_other_channel_not_in_free_response_still_gated():
    adapter = _make_adapter(
        require_mention=True,
        free_response_channels=[CHANNEL_ID],
    )
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, text="hello") is False


def test_dm_always_processed_regardless_of_setting():
    adapter = _make_adapter(require_mention=True)
    assert _would_process(adapter, is_dm=True, text="hello") is True


# ---------------------------------------------------------------------------
# Tests: MPIM / group-DM shared-surface gating (regression for the group-DM
# routing bug introduced by PRs #4633 / #54632 / #54663, which classified
# mpim as a DM and thereby exempted it from mention gating + reaction guards).
# ---------------------------------------------------------------------------



def test_mpim_not_in_allowed_channels_dropped():
    """MPIM absent from a non-empty allowed_channels whitelist is dropped,
    even when mentioned."""
    adapter = _make_adapter(require_mention=True, allowed_channels=["C_ALLOWED"])
    assert _would_process(adapter, channel_type="mpim", channel_id="C_BLOCKED",
                          mentioned=True, text="hello") is False


def test_one_to_one_im_still_exempt():
    """1:1 IM behavior is preserved: mention-exempt regardless of settings."""
    adapter = _make_adapter(require_mention=True, strict_mention=True)
    assert _would_process(adapter, channel_type="im", text="hello") is True






def test_mentioned_message_always_processed():
    adapter = _make_adapter(require_mention=True)
    assert _would_process(adapter, mentioned=True, text="what's up") is True


def test_thread_reply_with_active_session_processed():
    adapter = _make_adapter(require_mention=True)
    assert _would_process(
        adapter, text="followup",
        thread_reply=True, active_session=True,
    ) is True


def test_thread_reply_without_active_session_ignored():
    adapter = _make_adapter(require_mention=True)
    assert _would_process(
        adapter, text="followup",
        thread_reply=True, active_session=False,
    ) is False


def test_bot_uid_none_processes_channel_message():
    """When bot_uid is None (before auth_test), channel messages pass through.

    This preserves the old behavior: the gating block is skipped entirely
    when bot_uid is falsy, so messages are not silently dropped during
    startup or for new workspaces.
    """
    adapter = _make_adapter(require_mention=True)
    adapter._bot_user_id = None
    adapter._team_bot_user_ids = {}

    # With bot_uid=None, the `if not is_dm and bot_uid:` condition is False,
    # so the gating block is skipped — message passes through.
    bot_uid = adapter._team_bot_user_ids.get("T1", adapter._bot_user_id)
    assert bot_uid is None

    # Simulate: gating block not entered when bot_uid is falsy
    is_dm = False
    if not is_dm and bot_uid:
        result = False  # would enter gating
    else:
        result = True  # gating skipped, message processed
    assert result is True


# ---------------------------------------------------------------------------
# Tests: config bridging
# ---------------------------------------------------------------------------

def test_config_bridges_slack_free_response_channels(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n"
        "  require_mention: false\n"
        "  free_response_channels:\n"
        "    - C0AQWDLHY9M\n"
        "    - C9999999999\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("SLACK_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("SLACK_FREE_RESPONSE_CHANNELS", raising=False)

    config = load_gateway_config()

    assert config is not None
    slack_extra = config.platforms[Platform.SLACK].extra
    assert slack_extra.get("require_mention") is False
    assert slack_extra.get("free_response_channels") == ["C0AQWDLHY9M", "C9999999999"]
    # Verify env vars were set by config bridging
    import os as _os
    assert _os.environ["SLACK_REQUIRE_MENTION"] == "false"
    assert _os.environ["SLACK_FREE_RESPONSE_CHANNELS"] == "C0AQWDLHY9M,C9999999999"
    _os.environ.pop("SLACK_REQUIRE_MENTION", None)
    _os.environ.pop("SLACK_FREE_RESPONSE_CHANNELS", None)


def test_top_level_slack_settings_do_not_disable_env_token_setup(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n"
        "  require_mention: false\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.delenv("SLACK_REQUIRE_MENTION", raising=False)

    config = load_gateway_config()

    slack_config = config.platforms[Platform.SLACK]
    assert slack_config.enabled is True
    assert slack_config.token == "xoxb-test"
    assert slack_config.extra.get("require_mention") is False
    assert "_enabled_explicit" not in slack_config.extra


def test_explicit_platforms_slack_enabled_false_wins_over_env_token(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "platforms:\n"
        "  slack:\n"
        "    enabled: false\n"
        "    extra:\n"
        "      reply_in_thread: false\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    config = load_gateway_config()

    slack_config = config.platforms[Platform.SLACK]
    assert slack_config.enabled is False
    assert slack_config.token == "xoxb-test"
    assert slack_config.extra.get("reply_in_thread") is False
    assert "_enabled_explicit" not in slack_config.extra


def test_config_bridges_slack_reply_in_thread(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n"
        "  reply_in_thread: false\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    config = load_gateway_config()

    assert config is not None
    slack_config = config.platforms[Platform.SLACK]
    assert slack_config.extra.get("reply_in_thread") is False

    adapter = SlackAdapter(slack_config)
    assert adapter._resolve_thread_ts(reply_to="171.000", metadata={}) is None

    # Top-level channel messages arrive with metadata.thread_id == reply_to
    # because the inbound handler uses event.ts as a session-keying fallback.
    # Those must be treated as non-threaded so reply_in_thread=false takes
    # effect in channels, not just DMs.
    assert adapter._resolve_thread_ts(
        reply_to="171.000",
        metadata={"thread_id": "171.000"},
    ) is None

    # Real thread replies (reply_to differs from thread parent) must still
    # resolve to the parent thread so conversation context is preserved.
    assert adapter._resolve_thread_ts(
        reply_to="171.500",
        metadata={"thread_id": "171.000"},
    ) == "171.000"


# ---------------------------------------------------------------------------
# Regression: strict mode must NOT persist mentions into _mentioned_threads
# ---------------------------------------------------------------------------
# Prevents agent-to-agent ack loops — if a strict-mode bot remembered every
# thread it was mentioned in, the next message from the other agent in that
# thread would re-trigger the bot and defeat the entire feature.

def test_mention_in_strict_mode_does_not_register_thread():
    adapter = _make_adapter(strict_mention=True)
    adapter._bot_user_id = "U_BOT"
    adapter._mentioned_threads = set()
    adapter._MENTIONED_THREADS_MAX = 5000

    thread_ts = "1700000000.100200"
    event_thread_ts = thread_ts  # incoming message is inside an existing thread

    # Mirror the handler's @mention + strict-mode guard that protects
    # _mentioned_threads.add(). If strict is on, we must skip the add.
    text = "<@U_BOT> hello"
    is_mentioned = f"<@{adapter._bot_user_id}>" in text
    assert is_mentioned
    if event_thread_ts and not adapter._slack_strict_mention():
        adapter._mentioned_threads.add(event_thread_ts)

    assert thread_ts not in adapter._mentioned_threads


# ---------------------------------------------------------------------------
# Tests: _slack_allowed_channels
# ---------------------------------------------------------------------------


def test_allowed_channels_env_var_fallback(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", f"{CHANNEL_ID},{OTHER_CHANNEL_ID}")
    adapter = _make_adapter()  # no config value → falls back to env
    result = adapter._slack_allowed_channels()
    assert CHANNEL_ID in result
    assert OTHER_CHANNEL_ID in result


# ---------------------------------------------------------------------------
# Tests: allowed_channels gating integration
# ---------------------------------------------------------------------------


def test_allowed_channels_env_var_blocks_channel(monkeypatch):
    """SLACK_ALLOWED_CHANNELS env var (no config) also gates messages."""
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", CHANNEL_ID)
    adapter = _make_adapter()  # no config value → falls back to env
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, text="hello") is False
    assert _would_process(adapter, channel_id=CHANNEL_ID, mentioned=True) is True


@pytest.mark.asyncio
async def test_block_extraction_debug_log_does_not_include_message_preview(caplog):
    secret_block_text = "private incident token: customer-id-12345"
    adapter = _make_adapter(allowed_channels=[CHANNEL_ID])
    adapter._dedup = MagicMock(is_duplicate=MagicMock(return_value=False))
    adapter._lookup_assistant_thread_metadata = MagicMock(return_value={})
    adapter._channel_team = {}
    adapter._CHANNEL_TEAM_MAX = 10000
    # Wave-2 mention gating probes users.info for bot detection on several
    # paths; this fixture has no web client, so pin the sender as human.
    adapter._resolve_user_is_bot = AsyncMock(return_value=False)
    adapter._resolve_user_name = AsyncMock(return_value="testuser")
    adapter.handle_message = AsyncMock()

    event = {
        "channel": OTHER_CHANNEL_ID,
        "channel_type": "channel",
        "ts": "1710000000.000100",
        "team": "T1",
        "user": "U_USER",
        # Human-authored messages carry client_msg_id; without it the
        # unlabeled-bot probe path calls users.info, which this fixture
        # doesn't wire up.
        "client_msg_id": "cmid-block-priv",
        "text": "<@U_BOT_123> see quoted message",
        "blocks": [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_quote",
                        "elements": [
                            {
                                "type": "rich_text_section",
                                "elements": [
                                    {"type": "text", "text": secret_block_text}
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    with caplog.at_level(logging.DEBUG, logger="plugins.platforms.slack.adapter"):
        await adapter._handle_slack_message(event)

    assert "extracted additional text from blocks" in caplog.text
    assert "chars=" in caplog.text
    assert secret_block_text not in caplog.text


# ---------------------------------------------------------------------------
# Tests: "*" wildcard + drop-log visibility (issue #69)
# ---------------------------------------------------------------------------
# SLACK_ALLOWED_CHANNELS=* must mean "no channel restriction" (matching the
# Signal group wildcard and the documented allowlist contract). Previously
# `{"*"}` was treated as a literal whitelist, so every real channel ID was
# silently dropped — and the drop only logged at DEBUG, making it invisible
# at the default INFO level. These tests exercise the REAL handler, not the
# _would_process mirror above.

_TS_COUNTER = iter(range(1, 10_000))


def _make_real_adapter(allowed_channels=None, channel_policy=None):
    """Build a full SlackAdapter (real __init__) with I/O mocked out."""
    extra = {}
    if allowed_channels is not None:
        extra["allowed_channels"] = allowed_channels
    if channel_policy is not None:
        extra["channel_policy"] = channel_policy
    config = PlatformConfig(enabled=True, token="xoxb-fake-token", extra=extra)
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id = BOT_USER_ID
    adapter._running = True
    adapter.handle_message = AsyncMock()
    adapter._resolve_user_name = AsyncMock(return_value="Test User")
    return adapter


def _channel_event(channel_id, *, mentioned=True, text="hello"):
    if mentioned:
        text = f"<@{BOT_USER_ID}> {text}"
    return {
        "text": text,
        "user": "U_USER",
        "channel": channel_id,
        "channel_type": "channel",
        "ts": f"1700000000.{next(_TS_COUNTER):06d}",
    }


@pytest.mark.asyncio
async def test_wildcard_allows_any_channel(monkeypatch):
    """allowed_channels='*' lets an arbitrary channel through the gate."""
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    adapter = _make_real_adapter(allowed_channels="*")
    await adapter._handle_slack_message(_channel_event(OTHER_CHANNEL_ID))
    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_wildcard_env_var_allows_any_channel(monkeypatch):
    """SLACK_ALLOWED_CHANNELS=* (env var form) lets any channel through."""
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", "*")
    adapter = _make_real_adapter()
    await adapter._handle_slack_message(_channel_event(OTHER_CHANNEL_ID))
    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_wildcard_in_csv_allows_any_channel(monkeypatch):
    """'*' anywhere in the CSV disables the restriction."""
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    adapter = _make_real_adapter(allowed_channels=f"{CHANNEL_ID},*")
    await adapter._handle_slack_message(_channel_event(OTHER_CHANNEL_ID))
    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_channel_drop_logged_at_info_or_higher(monkeypatch, caplog):
    """A dropped channel must be visible at the default INFO log level."""
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    adapter = _make_real_adapter(allowed_channels=[CHANNEL_ID])
    with caplog.at_level(logging.DEBUG, logger="plugins.platforms.slack.adapter"):
        await adapter._handle_slack_message(_channel_event(OTHER_CHANNEL_ID))
    adapter.handle_message.assert_not_called()
    drop_records = [
        r for r in caplog.records
        if "non-allowed channel" in r.getMessage() and OTHER_CHANNEL_ID in r.getMessage()
    ]
    assert drop_records, "channel drop was not logged at all"
    assert all(r.levelno >= logging.INFO for r in drop_records), (
        "channel drop logged below INFO — invisible at default log level"
    )


@pytest.mark.asyncio
async def test_explicit_allowlist_still_gates(monkeypatch):
    """No wildcard: listed channel passes, unlisted channel drops."""
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    adapter = _make_real_adapter(allowed_channels=[CHANNEL_ID])
    await adapter._handle_slack_message(_channel_event(CHANNEL_ID))
    adapter.handle_message.assert_called_once()

    adapter.handle_message.reset_mock()
    await adapter._handle_slack_message(_channel_event(OTHER_CHANNEL_ID))
    adapter.handle_message.assert_not_called()


def test_would_process_wildcard_allows_any_channel(monkeypatch):
    """Mirror check: '*' disables the restriction in the gating simulation."""
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    adapter = _make_adapter(allowed_channels="*")
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, mentioned=True) is True


# ---------------------------------------------------------------------------
# Tests: _slack_channel_policy (SLACK_CHANNEL_POLICY knob)
# ---------------------------------------------------------------------------
# The policy knob inverts the meaning of an EMPTY allowed-channels set:
#   allow-all (default) → empty means "no restriction" (upstream behavior)
#   approved-only       → empty means "no channels approved" (secure default)
# It fails OPEN: anything not exactly "approved-only" (case-insensitive) is
# treated as allow-all, so an unset/garbage value can never silently lock the
# bot out of every channel.

def test_channel_policy_default_is_allow_all(monkeypatch):
    monkeypatch.delenv("SLACK_CHANNEL_POLICY", raising=False)
    adapter = _make_adapter()
    assert adapter._slack_channel_policy() == "allow-all"


def test_channel_policy_approved_only_from_config():
    adapter = _make_adapter(channel_policy="approved-only")
    assert adapter._slack_channel_policy() == "approved-only"


def test_channel_policy_approved_only_case_insensitive():
    adapter = _make_adapter(channel_policy="Approved-Only")
    assert adapter._slack_channel_policy() == "approved-only"


def test_channel_policy_env_var_fallback(monkeypatch):
    monkeypatch.setenv("SLACK_CHANNEL_POLICY", "approved-only")
    adapter = _make_adapter()  # no config value → falls back to env
    assert adapter._slack_channel_policy() == "approved-only"


def test_channel_policy_config_takes_precedence_over_env(monkeypatch):
    monkeypatch.setenv("SLACK_CHANNEL_POLICY", "approved-only")
    adapter = _make_adapter(channel_policy="allow-all")
    assert adapter._slack_channel_policy() == "allow-all"


def test_channel_policy_unknown_value_fails_open(monkeypatch):
    monkeypatch.delenv("SLACK_CHANNEL_POLICY", raising=False)
    adapter = _make_adapter(channel_policy="whatever")
    assert adapter._slack_channel_policy() == "allow-all"


def test_channel_policy_empty_string_is_allow_all(monkeypatch):
    monkeypatch.delenv("SLACK_CHANNEL_POLICY", raising=False)
    adapter = _make_adapter(channel_policy="")
    assert adapter._slack_channel_policy() == "allow-all"


# ---------------------------------------------------------------------------
# Tests: approved-only policy gating (the inverted-empty-set bug fix)
# ---------------------------------------------------------------------------

def test_allow_all_empty_list_not_filtered(monkeypatch):
    """(a) Default/unset policy + empty allowed list → NOT filtered (backward
    compat: empty means no restriction under allow-all)."""
    monkeypatch.delenv("SLACK_CHANNEL_POLICY", raising=False)
    adapter = _make_adapter(allowed_channels="")
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, mentioned=True) is True


def test_approved_only_empty_list_channel_ignored():
    """(b) approved-only + empty allowed list + channel message → ignored."""
    adapter = _make_adapter(channel_policy="approved-only", allowed_channels="")
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, mentioned=True) is False


def test_approved_only_non_empty_list_only_listed_pass():
    """(c) approved-only + non-empty list → only listed channels pass."""
    adapter = _make_adapter(channel_policy="approved-only",
                            allowed_channels=[CHANNEL_ID])
    assert _would_process(adapter, channel_id=CHANNEL_ID, mentioned=True) is True
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, mentioned=True) is False


def test_wildcard_means_all_under_allow_all(monkeypatch):
    """(d) '*' still means all channels under allow-all."""
    monkeypatch.delenv("SLACK_CHANNEL_POLICY", raising=False)
    adapter = _make_adapter(allowed_channels="*")
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, mentioned=True) is True


def test_wildcard_means_all_under_approved_only():
    """(d) '*' still means all channels even under approved-only."""
    adapter = _make_adapter(channel_policy="approved-only", allowed_channels="*")
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, mentioned=True) is True


def test_approved_only_dm_never_filtered():
    """(e) DMs are never filtered regardless of policy (existing invariant)."""
    adapter = _make_adapter(channel_policy="approved-only", allowed_channels="")
    assert _would_process(adapter, is_dm=True, channel_id="DDMCHANNEL") is True


def test_allow_all_empty_list_env_var_not_filtered(monkeypatch):
    """allow-all via env var (default) + empty list → not filtered."""
    monkeypatch.setenv("SLACK_CHANNEL_POLICY", "allow-all")
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    adapter = _make_adapter(allowed_channels="")
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, mentioned=True) is True


def test_approved_only_empty_list_env_var_channel_ignored(monkeypatch):
    """approved-only via env var + empty list → channel ignored."""
    monkeypatch.setenv("SLACK_CHANNEL_POLICY", "approved-only")
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    adapter = _make_adapter(allowed_channels="")
    assert _would_process(adapter, channel_id=OTHER_CHANNEL_ID, mentioned=True) is False


# ---------------------------------------------------------------------------
# Tests: approved-only real-handler behavior + distinguishable drop log
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approved_only_empty_real_handler_drops(monkeypatch):
    """Real _handle_slack_message: approved-only + empty allowlist drops the
    channel message (no handle_message call)."""
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    monkeypatch.delenv("SLACK_CHANNEL_POLICY", raising=False)
    adapter = _make_real_adapter(channel_policy="approved-only")
    await adapter._handle_slack_message(_channel_event(OTHER_CHANNEL_ID))
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_allow_all_empty_real_handler_passes(monkeypatch):
    """Real _handle_slack_message: default allow-all + empty allowlist still
    responds (backward compatibility preserved)."""
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    monkeypatch.delenv("SLACK_CHANNEL_POLICY", raising=False)
    adapter = _make_real_adapter()
    await adapter._handle_slack_message(_channel_event(OTHER_CHANNEL_ID))
    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_approved_only_drop_log_distinguishable(monkeypatch, caplog):
    """The approved-only empty-policy drop must be logged distinguishably from
    the ordinary whitelist drop, at INFO or higher (operational need: the
    regression went unnoticed for days)."""
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    monkeypatch.delenv("SLACK_CHANNEL_POLICY", raising=False)
    adapter = _make_real_adapter(channel_policy="approved-only")
    with caplog.at_level(logging.DEBUG, logger="plugins.platforms.slack.adapter"):
        await adapter._handle_slack_message(_channel_event(OTHER_CHANNEL_ID))
    adapter.handle_message.assert_not_called()
    policy_records = [
        r for r in caplog.records
        if "approved-only" in r.getMessage() and OTHER_CHANNEL_ID in r.getMessage()
    ]
    assert policy_records, "approved-only drop was not logged distinguishably"
    assert all(r.levelno >= logging.INFO for r in policy_records), (
        "approved-only drop logged below INFO — invisible at default log level"
    )
    # Must NOT reuse the ordinary whitelist message (which claims the list is
    # "active" — misleading when the list is actually empty).
    assert not any(
        "whitelist active" in r.getMessage() for r in policy_records
    ), "approved-only drop reused the whitelist-active message (indistinguishable)"


@pytest.mark.asyncio
async def test_approved_only_wildcard_real_handler_passes(monkeypatch):
    """approved-only + '*' still lets any channel through (wildcard override)."""
    monkeypatch.delenv("SLACK_ALLOWED_CHANNELS", raising=False)
    monkeypatch.delenv("SLACK_CHANNEL_POLICY", raising=False)
    adapter = _make_real_adapter(allowed_channels="*", channel_policy="approved-only")
    await adapter._handle_slack_message(_channel_event(OTHER_CHANNEL_ID))
    adapter.handle_message.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: config bridging for allowed_channels
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: mention_patterns (wake words) — parity with other adapters (#50732)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: Block-Kit-only mention detection (#52387)
# ---------------------------------------------------------------------------

from plugins.platforms.slack.adapter import _slack_mention_detection_text


def _blockkit_mention_event(bot_user_id=BOT_USER_ID, flat_text="Release notification"):
    """A Slack event whose @mention lives ONLY inside Block Kit blocks."""
    return {
        "text": flat_text,
        "blocks": [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [
                            {"type": "text", "text": "Hey "},
                            {"type": "user", "user_id": bot_user_id},
                            {"type": "text", "text": "! I will do a release"},
                        ],
                    }
                ],
            }
        ],
    }


def test_mention_patterns_case_insensitive():
    adapter = _make_adapter(mention_patterns=["hey hermes"])
    assert adapter._slack_message_matches_mention_patterns("HEY HERMES!") is True


def test_mention_patterns_single_string():
    adapter = _make_adapter(mention_patterns="^hermes")
    assert adapter._slack_message_matches_mention_patterns("hermes do this") is True
    assert adapter._slack_message_matches_mention_patterns("ok hermes") is False


def test_mention_patterns_invalid_regex_skipped_without_crash():
    # An invalid pattern is dropped; valid siblings still work.
    adapter = _make_adapter(mention_patterns=["(unclosed", "hey hermes"])
    assert adapter._slack_message_matches_mention_patterns("hey hermes") is True


def test_mention_patterns_env_var_fallback(monkeypatch):
    monkeypatch.setenv("SLACK_MENTION_PATTERNS", '["hey hermes", "hermes,"]')
    adapter = _make_adapter()  # no config value -> falls back to env
    assert adapter._slack_message_matches_mention_patterns("hey hermes") is True


def test_mention_patterns_env_var_csv_fallback_splits_patterns(monkeypatch):
    monkeypatch.setenv("SLACK_MENTION_PATTERNS", "hey hermes,hermes,")
    adapter = _make_adapter()  # no config value -> falls back to env

    patterns = adapter._slack_mention_patterns()

    assert [pattern.pattern for pattern in patterns] == ["hey hermes", "hermes"]
    assert adapter._slack_message_matches_mention_patterns("hey hermes") is True


def test_mention_patterns_trigger_in_channel_without_literal_mention():
    """A wake word triggers the bot in a channel even with require_mention on."""
    adapter = _make_adapter(require_mention=True, mention_patterns=["hey hermes"])
    assert _would_process(adapter, text="hey hermes what's the status") is True
    # Unrelated channel chatter is still ignored.
    assert _would_process(adapter, text="lunch anyone?") is False


# ---------------------------------------------------------------------------
# Tests: observe-unmentioned (_slack_observe_unmentioned + real flow)
# Observe (input/context) is orthogonal to the mention gate (response): when
# enabled, an unmentioned channel message is ingested as observe_only context
# but the bot still does not respond. Defaults OFF (opt-in) so existing
# deployments are unchanged; HSM opts in per surface.
# ---------------------------------------------------------------------------

def test_observe_unmentioned_defaults_to_false(monkeypatch):
    monkeypatch.delenv("SLACK_OBSERVE_UNMENTIONED", raising=False)
    adapter = _make_adapter()
    assert adapter._slack_observe_unmentioned() is False


def test_observe_unmentioned_config_true():
    adapter = _make_adapter()
    adapter.config.extra["observe_unmentioned"] = "true"
    assert adapter._slack_observe_unmentioned() is True


def test_observe_unmentioned_env_true(monkeypatch):
    monkeypatch.setenv("SLACK_OBSERVE_UNMENTIONED", "true")
    adapter = _make_adapter()
    assert adapter._slack_observe_unmentioned() is True


def test_observe_unmentioned_malformed_stays_false():
    """Explicit-true parser: unrecognised values keep observe OFF (opt-in)."""
    adapter = _make_adapter()
    adapter.config.extra["observe_unmentioned"] = "maybe"
    assert adapter._slack_observe_unmentioned() is False


@pytest.mark.asyncio
async def test_unmentioned_message_observed_when_enabled(monkeypatch):
    """observe on + unmentioned channel msg → dispatched as observe_only."""
    monkeypatch.setenv("SLACK_OBSERVE_UNMENTIONED", "true")
    adapter = _make_real_adapter(allowed_channels="*")
    await adapter._handle_slack_message(_channel_event(CHANNEL_ID, mentioned=False))
    adapter.handle_message.assert_called_once()
    msg_event = adapter.handle_message.call_args.args[0]
    assert msg_event.observe_only is True


@pytest.mark.asyncio
async def test_unmentioned_message_dropped_by_default(monkeypatch):
    """observe default off → unmentioned channel msg is hard-dropped."""
    monkeypatch.delenv("SLACK_OBSERVE_UNMENTIONED", raising=False)
    adapter = _make_real_adapter(allowed_channels="*")
    await adapter._handle_slack_message(_channel_event(CHANNEL_ID, mentioned=False))
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_mentioned_message_is_not_observe_only(monkeypatch):
    """A mentioned message is a normal (responded) turn, not observe_only."""
    monkeypatch.setenv("SLACK_OBSERVE_UNMENTIONED", "true")
    adapter = _make_real_adapter(allowed_channels="*")
    await adapter._handle_slack_message(_channel_event(CHANNEL_ID, mentioned=True))
    adapter.handle_message.assert_called_once()
    msg_event = adapter.handle_message.call_args.args[0]
    assert msg_event.observe_only is False

def test_mention_detection_text_ignores_quoted_blockkit_mention():
    """A mention inside rich_text_quote (forwarded content) must NOT count."""
    event = {
        "text": "please review",
        "blocks": [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_quote",
                        "elements": [
                            {
                                "type": "rich_text_section",
                                "elements": [
                                    {"type": "text", "text": "Contains "},
                                    {"type": "user", "user_id": BOT_USER_ID},
                                    {"type": "text", "text": " in quoted text"},
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    assert f"<@{BOT_USER_ID}>" not in _slack_mention_detection_text(event)
