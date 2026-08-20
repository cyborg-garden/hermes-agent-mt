"""Category snowflakes in Discord channel config lists.

``DISCORD_ALLOWED_CHANNELS`` / ``DISCORD_IGNORED_CHANNELS`` /
``DISCORD_FREE_RESPONSE_CHANNELS`` accept a **category** snowflake and match
every channel inside that category — evaluated per message via
``_discord_channel_keys_from_channel``, so channels created inside the
category later are covered without re-rendering config.

Motivating incident (cyborg.garden, 2026-08-20): every bot carried a
hand-maintained list of Greenhouse channel ids; a newly created channel in
the ring (#venture-labs) was silently outside every bot's allowlist. One
category id in the list replaces the per-channel bookkeeping.

Category matching is deliberately ID-only — category names routinely collide
with channel names (a "studio" category beside a #studio channel), so
name-form matching is not offered for categories.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Discord module mock — borrowed from test_discord_slash_commands.py so this
# file runs on machines without discord.py installed.
# ---------------------------------------------------------------------------


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return  # real discord installed

    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        discord_mod.ForumChannel = type("ForumChannel", (), {})
        discord_mod.Interaction = object

        class _FakePermissions:
            def __init__(self, value=0, **_):
                self.value = value

        discord_mod.Permissions = _FakePermissions

        ext_mod = MagicMock()
        commands_mod = MagicMock()
        commands_mod.Bot = MagicMock
        ext_mod.commands = commands_mod

        sys.modules["discord"] = discord_mod
        sys.modules.setdefault("discord.ext", ext_mod)
        sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402

CATEGORY_ID = "7770001"
CHANNEL_IN_CATEGORY = "5551111"
OTHER_CATEGORY_ID = "7770002"


@pytest.fixture(autouse=True)
def _isolate_discord_env(monkeypatch):
    for var in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOWED_ROLES",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
        "DISCORD_ALLOW_BOTS",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    a = DiscordAdapter(config)
    a._client = SimpleNamespace(user=SimpleNamespace(id=99999, name="HermesBot"), guilds=[])
    return a


def _guild_channel(channel_id=CHANNEL_IN_CATEGORY, name="studio", category_id=CATEGORY_ID):
    """A TextChannel-shaped stub: discord.py exposes the category as
    ``category_id`` (the raw API's parent_id)."""
    return SimpleNamespace(id=channel_id, name=name, category_id=category_id, parent=None)


def _thread_in_category(thread_id="6662222", parent_id=CHANNEL_IN_CATEGORY,
                        category_id=CATEGORY_ID):
    """A Thread-shaped stub: no category_id of its own; the category is
    reached through the parent channel."""
    parent = SimpleNamespace(id=parent_id, name="studio", category_id=category_id)
    return SimpleNamespace(id=thread_id, name="a thread", parent=parent,
                           parent_id=parent_id)


# ---------------------------------------------------------------------------
# Key derivation (_discord_channel_keys_from_channel)
# ---------------------------------------------------------------------------


def test_channel_keys_include_category_id(adapter):
    keys = adapter._discord_channel_keys_from_channel(_guild_channel())
    assert CATEGORY_ID in keys
    assert CHANNEL_IN_CATEGORY in keys


def test_thread_keys_include_parent_category_id(adapter):
    thread = _thread_in_category()
    keys = adapter._discord_channel_keys_from_channel(thread, str(thread.parent_id))
    assert CATEGORY_ID in keys          # ring key via parent channel
    assert CHANNEL_IN_CATEGORY in keys  # parent channel key
    assert "6662222" in keys            # the thread itself


def test_uncategorized_channel_adds_no_category_key(adapter):
    keys = adapter._discord_channel_keys_from_channel(
        SimpleNamespace(id="123", name="lobby", category_id=None, parent=None)
    )
    assert keys == {"123", "lobby", "#lobby"}


def test_category_matching_is_id_only(adapter):
    """A category NAME must not become a key — category and channel names
    collide too easily for name-form ring matching to be safe."""
    channel = _guild_channel()
    channel.category = SimpleNamespace(id=CATEGORY_ID, name="GREENHOUSE")
    keys = adapter._discord_channel_keys_from_channel(channel)
    assert "GREENHOUSE" not in keys
    assert "#GREENHOUSE" not in keys


# ---------------------------------------------------------------------------
# End-to-end through a real gate (_evaluate_slash_authorization) — proves the
# keys are consumed by an actual allow/ignore check, not just derived.
# ---------------------------------------------------------------------------


def _interaction_in(channel):
    response = SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock())
    return SimpleNamespace(
        user=SimpleNamespace(id=100200300, roles=[]),
        channel=channel,
        channel_id=getattr(channel, "id", None),
        guild_id=42,
        response=response,
    )


def test_allowlist_category_id_admits_member_channel(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", CATEGORY_ID)
    allowed, reason = adapter._evaluate_slash_authorization(
        _interaction_in(_guild_channel())
    )
    assert allowed is True, reason


def test_allowlist_foreign_category_id_rejects(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", OTHER_CATEGORY_ID)
    allowed, reason = adapter._evaluate_slash_authorization(
        _interaction_in(_guild_channel())
    )
    assert allowed is False
    assert "DISCORD_ALLOWED_CHANNELS" in (reason or "")


def test_ignored_category_beats_allowed_channel(adapter, monkeypatch):
    """Deny beats allow at ring scope: an explicitly allowed channel inside
    an ignored category stays silenced."""
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", CHANNEL_IN_CATEGORY)
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", CATEGORY_ID)
    allowed, reason = adapter._evaluate_slash_authorization(
        _interaction_in(_guild_channel())
    )
    assert allowed is False
    assert "DISCORD_IGNORED_CHANNELS" in (reason or "")
