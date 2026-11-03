"""Security regression tests: R7 moderation → authorization propagation.

A user banned (or, in "remove" mode, kicked/leaving) from a guild must lose
Hermes authorization EVERYWHERE — pairing grant, DISCORD_ALLOWED_USERS
(in-memory + env), GATEWAY_ALLOWED_USERS — and land in a persistent,
adapter-owned deny store. The deny store exists because the HSM console's
stale whole-document save can re-write DISCORD_ALLOWED_USERS and silently
re-grant a banned user (audit SB-5/#2): deny must beat every allow branch —
pairing, allowlists, ALLOW_ALL flags, and the component-click auth union —
and must survive any console re-write.

These tests pin: revocation pruning across all layers, deny-beats-everything
semantics on both auth surfaces, handler registration per
DISCORD_MODERATION_SYNC mode (off | ban | remove), the privileged-intent
request being confined to "remove" mode, exemption and guild filters, and
idempotency under the ban/remove double-fire.
"""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig

# Importing the production modules triggers the shared discord mock from
# tests/gateway/conftest.py.
import plugins.platforms.discord.adapter as discord_platform
from plugins.platforms.discord import moderation_sync
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord.adapter_component_auth import (
    _component_check_auth,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOWED_ROLES",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "DISCORD_MODERATION_SYNC",
        "DISCORD_MODERATION_SYNC_GUILDS",
        "DISCORD_MODERATION_SYNC_EXEMPT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def pairing_store_mock():
    """Mock PairingStore at its source module (the adapter and pairing code
    import it at function level) so tests never touch real pairing files."""
    store = MagicMock()
    store.is_approved.return_value = False
    store.revoke.return_value = True
    with patch("gateway.pairing.PairingStore", return_value=store):
        yield store


@pytest.fixture
def env_persistence_mock():
    """Mock the .env persistence layer (save/remove) to observe writes."""
    with patch("hermes_cli.config.save_env_value") as save_mock, patch(
        "hermes_cli.config.remove_env_value"
    ) as remove_mock:
        yield SimpleNamespace(save=save_mock, remove=remove_mock)


def _bare_adapter(allowed_users=(), allowed_roles=()):
    """Adapter without __init__ (AGENTS.md pitfall #17 pattern)."""
    adapter = object.__new__(DiscordAdapter)
    adapter._allowed_user_ids = set(allowed_users)
    adapter._allowed_role_ids = set(allowed_roles)
    return adapter


BANNED = "769524422783664158"  # 18-digit snowflake
OTHER = "111111111111111111"


# ---------------------------------------------------------------------------
# revoke_user_authorization: every layer is pruned
# ---------------------------------------------------------------------------


class TestRevocation:
    def test_ban_prunes_all_authorization_layers(
        self, monkeypatch, pairing_store_mock, env_persistence_mock
    ):
        adapter = _bare_adapter(allowed_users={BANNED, OTHER})
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", f"{BANNED},{OTHER}")
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", f"{BANNED},999888777666555444")

        moderation_sync.revoke_user_authorization(adapter, BANNED, "ban", "g1")

        # In-memory allowlist + os.environ mirror pruned.
        assert BANNED not in adapter._allowed_user_ids
        assert OTHER in adapter._allowed_user_ids
        assert BANNED not in os.environ["DISCORD_ALLOWED_USERS"]
        # Pairing grant revoked.
        pairing_store_mock.revoke.assert_called_once_with("discord", BANNED)
        # Persisted .env allowlists pruned (DISCORD via pairing sync,
        # GATEWAY via the exact-match prune).
        saved = {call.args[0]: call.args[1] for call in env_persistence_mock.save.call_args_list}
        assert saved.get("DISCORD_ALLOWED_USERS") == OTHER
        assert saved.get("GATEWAY_ALLOWED_USERS") == "999888777666555444"
        assert BANNED not in os.environ["GATEWAY_ALLOWED_USERS"]
        # Deny-store backstop recorded.
        assert moderation_sync.deny_store().is_denied(BANNED)

    def test_never_paired_allowlisted_user_still_gets_env_prune(
        self, monkeypatch, pairing_store_mock, env_persistence_mock
    ):
        """PairingStore.revoke returns False for a never-paired user — its
        internal allowlist sync never fires. The public helper must prune the
        .env allowlist anyway."""
        pairing_store_mock.revoke.return_value = False
        adapter = _bare_adapter(allowed_users={BANNED})
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", BANNED)

        moderation_sync.revoke_user_authorization(adapter, BANNED, "ban", "g1")

        # Empty remainder → remove_env_value path.
        removed = [call.args[0] for call in env_persistence_mock.remove.call_args_list]
        assert "DISCORD_ALLOWED_USERS" in removed

    def test_revocation_is_idempotent(
        self, monkeypatch, pairing_store_mock, env_persistence_mock
    ):
        """Ban fires both on_member_ban and (in remove mode)
        on_raw_member_remove — double revocation must be harmless."""
        adapter = _bare_adapter(allowed_users={BANNED, OTHER})
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", f"{BANNED},{OTHER}")

        moderation_sync.revoke_user_authorization(adapter, BANNED, "ban", "g1")
        moderation_sync.revoke_user_authorization(adapter, BANNED, "remove", "g1")

        assert moderation_sync.deny_store().is_denied(BANNED)
        assert OTHER in adapter._allowed_user_ids

    def test_pruning_last_user_logs_critical_and_stays_denied(
        self, monkeypatch, pairing_store_mock, env_persistence_mock, caplog
    ):
        """Emptying the allowlist changes the auth-gate shape (channel
        bypass / ALLOW_ALL branch) — operators must be alerted, and the
        banned uid must stay blocked by the deny store regardless."""
        adapter = _bare_adapter(allowed_users={BANNED})
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", BANNED)

        with caplog.at_level("CRITICAL"):
            moderation_sync.revoke_user_authorization(adapter, BANNED, "ban", "g1")

        assert any(r.levelname == "CRITICAL" for r in caplog.records)
        # Even with the gate shape changed to allow-all, THIS uid stays out.
        monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
        assert adapter._is_allowed_user(BANNED) is False


# ---------------------------------------------------------------------------
# Deny beats every allow branch
# ---------------------------------------------------------------------------


class TestDenyBeatsAllow:
    def test_hsm_readd_scenario_still_denied(self, pairing_store_mock):
        """The HSM console's stale whole-document save re-writes
        DISCORD_ALLOWED_USERS with the banned uid back in. The deny store
        must still block them."""
        moderation_sync.deny_store().add(BANNED, "ban", "g1")
        adapter = _bare_adapter(allowed_users={BANNED})  # console re-granted

        assert adapter._is_allowed_user(BANNED) is False
        assert adapter._is_allowed_user(OTHER) is False  # not in list either

    def test_deny_beats_pairing(self, pairing_store_mock):
        pairing_store_mock.is_approved.return_value = True
        adapter = _bare_adapter()
        # Sanity: pairing alone authorizes.
        assert adapter._is_allowed_user(BANNED) is True

        moderation_sync.deny_store().add(BANNED, "ban", "g1")
        assert adapter._is_allowed_user(BANNED) is False

    def test_deny_beats_allow_all(self, monkeypatch, pairing_store_mock):
        monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
        adapter = _bare_adapter()
        assert adapter._is_allowed_user(BANNED) is True  # sanity

        moderation_sync.deny_store().add(BANNED, "ban", "g1")
        assert adapter._is_allowed_user(BANNED) is False
        assert adapter._is_allowed_user(OTHER) is True  # others unaffected

    @staticmethod
    def _interaction(uid):
        return SimpleNamespace(user=SimpleNamespace(id=int(uid), roles=[]))

    def test_component_deny_beats_allow_all(self, monkeypatch, pairing_store_mock):
        monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
        moderation_sync.deny_store().add(BANNED, "ban", "g1")

        assert _component_check_auth(self._interaction(BANNED), set(), set()) is False
        assert _component_check_auth(self._interaction(OTHER), set(), set()) is True

    def test_component_deny_beats_gateway_allowed_users_union(
        self, monkeypatch, pairing_store_mock
    ):
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", BANNED)
        assert _component_check_auth(self._interaction(BANNED), set(), set()) is True  # sanity

        moderation_sync.deny_store().add(BANNED, "ban", "g1")
        assert _component_check_auth(self._interaction(BANNED), set(), set()) is False

    def test_component_deny_beats_explicit_user_allowlist(
        self, pairing_store_mock
    ):
        moderation_sync.deny_store().add(BANNED, "ban", "g1")
        assert (
            _component_check_auth(self._interaction(BANNED), {BANNED}, set()) is False
        )

    def test_deny_store_remove_restores_access(self, monkeypatch, pairing_store_mock):
        """Operator un-ban path: DenyStore.remove lifts the block."""
        monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
        adapter = _bare_adapter()
        moderation_sync.deny_store().add(BANNED, "ban", "g1")
        assert adapter._is_allowed_user(BANNED) is False

        assert moderation_sync.deny_store().remove(BANNED) is True
        assert adapter._is_allowed_user(BANNED) is True


# ---------------------------------------------------------------------------
# Handler registration + intents per mode (full connect with a fake bot)
# ---------------------------------------------------------------------------


class _FakeTree:
    def __init__(self):
        self.sync = AsyncMock(return_value=[])
        self.fetch_commands = AsyncMock(return_value=[])

    def command(self, *args, **kwargs):
        return lambda fn: fn

    def get_commands(self, *args, **kwargs):
        return []


class _FakeBot:
    def __init__(self, *, intents, allowed_mentions=None, **_):
        self.intents = intents
        self.allowed_mentions = allowed_mentions
        self.application_id = 999
        self.user = SimpleNamespace(id=999, name="Hermes")
        self._events = {}
        self.tree = _FakeTree()

    def event(self, fn):
        self._events[fn.__name__] = fn
        return fn

    async def start(self, token):
        if "on_ready" in self._events:
            await self._events["on_ready"]()

    async def close(self):
        return None

    def is_closed(self):
        return False


async def _connect_adapter(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr("gateway.status_scoped_locks.release_scoped_lock", lambda scope, identity: None)

    intents = SimpleNamespace(
        message_content=False,
        dm_messages=False,
        guild_messages=False,
        members=False,
        voice_states=False,
    )
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)

    created = {}

    def fake_bot_factory(*, command_prefix, intents, proxy=None, allowed_mentions=None, **_):
        created["bot"] = _FakeBot(intents=intents, allowed_mentions=allowed_mentions)
        return created["bot"]

    monkeypatch.setattr(discord_platform.commands, "Bot", fake_bot_factory)
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())

    ok = await adapter.connect()
    assert ok is True
    return adapter, created["bot"]


class TestHandlerRegistration:
    @pytest.mark.asyncio
    async def test_mode_off_registers_no_moderation_handlers(self, monkeypatch):
        monkeypatch.setenv("DISCORD_MODERATION_SYNC", "off")
        adapter, bot = await _connect_adapter(monkeypatch)
        try:
            assert "on_member_ban" not in bot._events
            assert "on_raw_member_remove" not in bot._events
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_default_mode_ban_registers_ban_only(self, monkeypatch):
        """Default (env unset) is 'ban': bans propagate with default intents;
        GUILD_MEMBER_REMOVE is NOT subscribed (kick/leave don't revoke)."""
        adapter, bot = await _connect_adapter(monkeypatch)
        try:
            assert "on_member_ban" in bot._events
            assert "on_raw_member_remove" not in bot._events
            assert bot.intents.members is False, (
                "'ban' mode must not request the privileged members intent "
                "(portal-dependent boot risk)"
            )
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_mode_remove_registers_both_and_requests_members_intent(
        self, monkeypatch
    ):
        monkeypatch.setenv("DISCORD_MODERATION_SYNC", "remove")
        adapter, bot = await _connect_adapter(monkeypatch)
        try:
            assert "on_member_ban" in bot._events
            assert "on_raw_member_remove" in bot._events
            assert bot.intents.members is True, (
                "'remove' mode needs the Server Members intent for "
                "GUILD_MEMBER_REMOVE"
            )
        finally:
            await adapter.disconnect()


class TestHandlerBehavior:
    @pytest.mark.asyncio
    async def test_ban_event_revokes(self, monkeypatch):
        adapter, bot = await _connect_adapter(monkeypatch)
        calls = []
        monkeypatch.setattr(
            discord_platform._moderation_sync,
            "revoke_user_authorization",
            lambda a, uid, reason, gid: calls.append((uid, reason, gid)),
        )
        try:
            await bot._events["on_member_ban"](
                SimpleNamespace(id=int("123456789012345678")),
                SimpleNamespace(id=int(BANNED)),
            )
            assert calls == [(BANNED, "ban", "123456789012345678")]
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_exempt_uid_never_revoked(self, monkeypatch):
        monkeypatch.setenv("DISCORD_MODERATION_SYNC_EXEMPT", BANNED)
        adapter, bot = await _connect_adapter(monkeypatch)
        calls = []
        monkeypatch.setattr(
            discord_platform._moderation_sync,
            "revoke_user_authorization",
            lambda a, uid, reason, gid: calls.append(uid),
        )
        try:
            await bot._events["on_member_ban"](
                SimpleNamespace(id=1), SimpleNamespace(id=int(BANNED))
            )
            assert calls == []
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_sync_guilds_filter_honored(self, monkeypatch):
        monkeypatch.setenv("DISCORD_MODERATION_SYNC_GUILDS", "123")
        adapter, bot = await _connect_adapter(monkeypatch)
        calls = []
        monkeypatch.setattr(
            discord_platform._moderation_sync,
            "revoke_user_authorization",
            lambda a, uid, reason, gid: calls.append(gid),
        )
        try:
            # Ban in a non-synced guild: ignored.
            await bot._events["on_member_ban"](
                SimpleNamespace(id=999), SimpleNamespace(id=int(BANNED))
            )
            assert calls == []
            # Ban in the synced guild: propagates.
            await bot._events["on_member_ban"](
                SimpleNamespace(id=123), SimpleNamespace(id=int(BANNED))
            )
            assert calls == ["123"]
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_ban_and_remove_double_fire_is_idempotent(
        self, monkeypatch, pairing_store_mock, env_persistence_mock
    ):
        """A ban in 'remove' mode fires BOTH events; the second revocation
        must be a harmless no-op on top of the first."""
        monkeypatch.setenv("DISCORD_MODERATION_SYNC", "remove")
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", f"{BANNED},{OTHER}")
        adapter, bot = await _connect_adapter(monkeypatch)
        adapter._allowed_user_ids = {BANNED, OTHER}
        try:
            await bot._events["on_member_ban"](
                SimpleNamespace(id=123), SimpleNamespace(id=int(BANNED))
            )
            await bot._events["on_raw_member_remove"](
                SimpleNamespace(user=SimpleNamespace(id=int(BANNED)), guild_id=123)
            )
            assert moderation_sync.deny_store().is_denied(BANNED)
            assert adapter._is_allowed_user(BANNED) is False
            assert OTHER in adapter._allowed_user_ids
        finally:
            await adapter.disconnect()
