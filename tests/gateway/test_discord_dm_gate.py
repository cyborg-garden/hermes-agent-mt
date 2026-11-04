"""DISCORD_DM_ALLOWED_USERS: DMs are closed unless the user is listed.

Being on DISCORD_ALLOWED_USERS grants channel access. It must not also open a
private DM with the bot, and the gate must fail closed when the variable is
unset so a newly created agent starts with DMs off.
"""

from unittest.mock import patch

import pytest

from plugins.platforms.discord.adapter import DiscordAdapter

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
