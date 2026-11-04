"""DISCORD_DM_ALLOWED_USERS: DMs are closed unless the user is listed.

Being on DISCORD_ALLOWED_USERS grants channel access. It must not also open a
private DM with the bot, and the gate must fail closed when the variable is
unset so a newly created agent starts with DMs off.
"""

from types import SimpleNamespace
from unittest.mock import patch

import discord
import pytest

from plugins.platforms.discord.adapter import (
    DiscordAdapter,
    _component_check_auth,
    _discord_message_is_dm,
)

OWNER = "100"
MEMBER = "200"


def _adapter(allowed_users=(OWNER, MEMBER)):
    adapter = object.__new__(DiscordAdapter)
    adapter._allowed_user_ids = set(allowed_users)
    adapter._allowed_role_ids = set()
    adapter._client = None
    return adapter


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "DISCORD_DM_ALLOWED_USERS",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "DISCORD_CHANNEL_SCOPED_ACCESS",
        "DISCORD_ALLOWED_CHANNELS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_unset_closes_dms_for_allowlisted_users():
    assert _adapter()._is_allowed_user(OWNER, is_dm=True) is False


def test_empty_closes_dms(monkeypatch):
    monkeypatch.setenv("DISCORD_DM_ALLOWED_USERS", "  ")
    assert _adapter()._is_allowed_user(OWNER, is_dm=True) is False


def test_listed_user_can_dm_and_others_cannot(monkeypatch):
    monkeypatch.setenv("DISCORD_DM_ALLOWED_USERS", f" {OWNER} , 999")
    adapter = _adapter()
    assert adapter._is_allowed_user(OWNER, is_dm=True) is True
    assert adapter._is_allowed_user(MEMBER, is_dm=True) is False


def test_listing_does_not_grant_access_the_bot_otherwise_refuses(monkeypatch):
    monkeypatch.setenv("DISCORD_DM_ALLOWED_USERS", OWNER)
    assert _adapter(allowed_users=(MEMBER,))._is_allowed_user(OWNER, is_dm=True) is False


def test_wildcard_keeps_the_ordinary_allowlist_behaviour(monkeypatch):
    monkeypatch.setenv("DISCORD_DM_ALLOWED_USERS", "*")
    adapter = _adapter(allowed_users=(MEMBER,))
    assert adapter._is_allowed_user(MEMBER, is_dm=True) is True
    assert adapter._is_allowed_user(OWNER, is_dm=True) is False


def test_guild_messages_are_untouched():
    assert _adapter()._is_allowed_user(MEMBER, is_dm=False) is True


@pytest.mark.parametrize(
    "bypass_env",
    ["DISCORD_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS"],
)
def test_allow_all_flags_do_not_open_dms(monkeypatch, bypass_env):
    monkeypatch.setenv(bypass_env, "true")
    assert _adapter(allowed_users=())._is_allowed_user(MEMBER, is_dm=True) is False


def test_pairing_grant_does_not_open_dms():
    adapter = _adapter(allowed_users=())
    with patch.object(DiscordAdapter, "_is_pairing_approved_user", return_value=True):
        assert adapter._is_allowed_user(MEMBER, is_dm=True) is False
        assert adapter._is_allowed_user(MEMBER, is_dm=False) is True


def test_mention_style_entries_are_cleaned(monkeypatch):
    monkeypatch.setenv("DISCORD_DM_ALLOWED_USERS", f"<@{OWNER}>, user:999")
    adapter = _adapter(allowed_users=(OWNER, "999"))
    assert adapter._is_allowed_user(OWNER, is_dm=True) is True
    assert adapter._is_allowed_user("999", is_dm=True) is True


class TestMessageClassification:
    def test_dm_channel(self):
        message = SimpleNamespace(channel=discord.DMChannel(), guild=None)
        assert _discord_message_is_dm(message) is True

    def test_group_dm_has_no_guild(self):
        message = SimpleNamespace(channel=SimpleNamespace(id=1), guild=None)
        assert _discord_message_is_dm(message) is True

    def test_guild_message(self):
        message = SimpleNamespace(channel=SimpleNamespace(id=1), guild=object())
        assert _discord_message_is_dm(message) is False

    def test_uncached_guild_is_not_a_dm(self):
        channel = SimpleNamespace(id=1, guild_id=42)
        assert _discord_message_is_dm(SimpleNamespace(channel=channel, guild=None)) is False


def _interaction(user_id, *, channel, guild_id):
    return SimpleNamespace(
        user=SimpleNamespace(id=int(user_id), name="someone"),
        guild=None,
        guild_id=guild_id,
        channel=channel,
        channel_id=getattr(channel, "id", None),
    )


class TestSlashCommands:
    @pytest.mark.parametrize(
        "channel",
        [discord.DMChannel(), SimpleNamespace(id=7), None],
        ids=["dm", "group_dm", "no_channel"],
    )
    def test_guildless_interaction_is_refused(self, channel):
        allowed, _ = _adapter()._evaluate_slash_authorization(
            _interaction(MEMBER, channel=channel, guild_id=None)
        )
        assert allowed is False

    def test_group_dm_with_open_channels_is_refused(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "*")
        allowed, _ = _adapter(allowed_users=())._evaluate_slash_authorization(
            _interaction(MEMBER, channel=SimpleNamespace(id=7), guild_id=None)
        )
        assert allowed is False

    def test_listed_user_passes_in_dm(self, monkeypatch):
        monkeypatch.setenv("DISCORD_DM_ALLOWED_USERS", OWNER)
        allowed, _ = _adapter()._evaluate_slash_authorization(
            _interaction(OWNER, channel=discord.DMChannel(), guild_id=None)
        )
        assert allowed is True

    def test_guild_interaction_is_untouched(self):
        allowed, _ = _adapter()._evaluate_slash_authorization(
            _interaction(MEMBER, channel=SimpleNamespace(id=7), guild_id=42)
        )
        assert allowed is True


class TestButtons:
    def _click(self, user_id, guild_id):
        return SimpleNamespace(
            user=SimpleNamespace(id=int(user_id)), guild_id=guild_id
        )

    def test_unlisted_click_in_dm_is_refused(self):
        assert _component_check_auth(self._click(MEMBER, None), {MEMBER}, set()) is False

    def test_allow_all_does_not_open_dm_buttons(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
        assert _component_check_auth(self._click(MEMBER, None), set(), set()) is False

    def test_listed_click_in_dm_still_works(self, monkeypatch):
        monkeypatch.setenv("DISCORD_DM_ALLOWED_USERS", OWNER)
        assert _component_check_auth(self._click(OWNER, None), {OWNER}, set()) is True

    def test_guild_click_is_untouched(self):
        assert _component_check_auth(self._click(MEMBER, 42), {MEMBER}, set()) is True
