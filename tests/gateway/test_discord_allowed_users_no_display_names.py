"""DISCORD_ALLOWED_USERS name entries resolve only by the unique username.

``_resolve_allowed_usernames`` turns non-numeric entries into user IDs at
on_ready. It used to match ``display_name`` (server nickname) and
``global_name`` too, which anyone can set to anything — so a stranger in any
guild the bot shares could set their nickname to a listed name and be
admitted. Only ``member.name`` (the unique handle) may resolve an entry.
"""

import asyncio
import logging
import os
from types import SimpleNamespace as NS

from gateway.session import Platform

# Trigger the shared discord mock (tests/gateway/conftest.py) before import.
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


def _member(uid, name, display_name=None, global_name=None, discriminator="0", bot=False):
    return NS(
        id=uid, name=name, display_name=display_name or global_name or name,
        global_name=global_name, discriminator=discriminator, bot=bot,
    )


def _adapter(entries, members):
    adapter = object.__new__(DiscordAdapter)
    adapter._platform = Platform.DISCORD
    adapter.platform = Platform.DISCORD
    adapter._allowed_role_ids = set()
    adapter._pairing_store = None
    adapter._allowed_user_ids = set(entries)
    guild = NS(name="g", members=members, member_count=len(members))
    adapter._client = NS(guilds=[guild])
    return adapter


def _resolve(monkeypatch, entries, members):
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", ",".join(entries))
    adapter = _adapter(entries, members)
    asyncio.run(adapter._resolve_allowed_usernames())
    return adapter


def test_unique_username_resolves(monkeypatch):
    adapter = _resolve(monkeypatch, ["juniperbevensee"], [_member(1001, "juniperbevensee")])
    assert adapter._allowed_user_ids == {"1001"}
    assert os.environ["DISCORD_ALLOWED_USERS"] == "1001"


def test_at_prefix_and_case_are_ignored(monkeypatch):
    adapter = _resolve(monkeypatch, ["@JuniperBevensee"], [_member(1001, "juniperbevensee")])
    assert adapter._allowed_user_ids == {"1001"}


def test_display_name_does_not_resolve(monkeypatch):
    impostor = _member(666, "stranger", display_name="juniperbevensee")
    adapter = _resolve(monkeypatch, ["juniperbevensee"], [impostor])
    assert "666" not in adapter._allowed_user_ids
    assert adapter._is_allowed_user("666") is False


def test_global_name_does_not_resolve(monkeypatch):
    impostor = _member(666, "stranger", global_name="juniperbevensee")
    adapter = _resolve(monkeypatch, ["juniperbevensee"], [impostor])
    assert "666" not in adapter._allowed_user_ids


def test_impostor_listed_first_does_not_steal_the_entry(monkeypatch):
    """The impostor must not be admitted and the real user still resolves."""
    members = [
        _member(666, "stranger", display_name="juniperbevensee"),
        _member(1001, "juniperbevensee"),
    ]
    adapter = _resolve(monkeypatch, ["juniperbevensee"], members)
    assert adapter._allowed_user_ids == {"1001"}


def test_legacy_discriminator_needs_full_tag(monkeypatch):
    """A bare name only matches a unique (discriminator 0) account; a legacy
    account such as a bot named like an admin needs ``name#1234``."""
    bot = _member(777, "juniperbevensee", discriminator="4242", bot=True)
    adapter = _resolve(monkeypatch, ["juniperbevensee"], [bot])
    assert "777" not in adapter._allowed_user_ids
    adapter = _resolve(monkeypatch, ["juniperbevensee#4242"], [bot])
    assert adapter._allowed_user_ids == {"777"}


def test_resolution_is_logged(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="plugins.platforms.discord.adapter"):
        _resolve(monkeypatch, ["juniperbevensee"], [_member(1001, "juniperbevensee")])
    hits = [r for r in caplog.records if "juniperbevensee" in r.getMessage() and "1001" in r.getMessage()]
    assert len(hits) == 1


def test_display_name_only_match_warns(monkeypatch, caplog):
    """Operators upgrading need to see which entries stopped matching."""
    impostor = _member(666, "stranger", display_name="Juniper")
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.discord.adapter"):
        _resolve(monkeypatch, ["juniper"], [impostor])
    assert any(
        r.levelno == logging.WARNING and "juniper" in r.getMessage().lower()
        for r in caplog.records
    )
