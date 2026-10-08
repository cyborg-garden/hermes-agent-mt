"""DISCORD_ALLOWED_BOT_ROLES / DISCORD_ALLOWED_BOT_USERS: gate BOT senders.

Before this gate, a bot admitted by DISCORD_ALLOW_BOTS=mentions|all skipped the
human user/role allowlists entirely, so any bot in an allowed channel (a
third-party community bot, say) could drive the agent by @mentioning it.

When either variable is set, a bot sender must ALSO hold a listed role in the
originating guild, or be a listed bot user id. Unset = unchanged. DMs never
pass (no guild, no roles). A role lookup that cannot be performed fails closed.

The on_message tests drive the REAL closure registered by ``connect()``, not
a copy of its logic.
"""

import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.session import Platform, SessionSource

# Reuse the discord stub + FakeBot from the connect tests (also installs the
# AllowedMentions shim when the real library is present).
from tests.gateway.test_discord_connect import FakeBot  # noqa: E402
import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402

GUILD_ID = 1531068097719570432
TRUSTED_ROLE = 1600000000000000001
OTHER_ROLE = 1600000000000000002
FLEET_BOT_ID = 1531117849433866412
STRANGER_BOT_ID = 1700000000000000009
HUMAN_ID = 100200300
SELF_ID = 999

_ENV_VARS = (
    "DISCORD_ALLOW_BOTS",
    "DISCORD_ALLOWED_USERS",
    "DISCORD_ALLOWED_ROLES",
    "DISCORD_ALLOWED_BOT_ROLES",
    "DISCORD_ALLOWED_BOT_USERS",
    "DISCORD_ALLOW_ALL_USERS",
    "DISCORD_ALLOWED_CHANNELS",
    "DISCORD_IGNORED_CHANNELS",
    "DISCORD_CHANNEL_SCOPED_ACCESS",
    "DISCORD_BOTS_REQUIRE_INLINE_MENTION",
    "DISCORD_REQUIRE_MENTION",
    "DISCORD_IGNORE_NO_MENTION",
    "GATEWAY_ALLOW_ALL_USERS",
    "GATEWAY_ALLOWED_USERS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


_ids = itertools.count(5_000_000_000_000_000)


def _guild():
    return SimpleNamespace(
        id=GUILD_ID,
        name="garden",
        roles=[
            SimpleNamespace(id=GUILD_ID, name="@everyone"),
            SimpleNamespace(id=TRUSTED_ROLE, name="trusted-bot"),
            SimpleNamespace(id=OTHER_ROLE, name="community"),
        ],
        self_role=None,
    )


def _member(uid, *, bot, role_ids=(), has_roles=True):
    m = SimpleNamespace(
        id=uid,
        bot=bot,
        name=f"u{uid}",
        discriminator="0",
        display_name=f"u{uid}",
        global_name=None,
    )
    if has_roles:
        m.roles = [SimpleNamespace(id=GUILD_ID)] + [SimpleNamespace(id=r) for r in role_ids]
    return m


def _message(author, *, dm=False, mention_self=True, client_user=None):
    guild = None if dm else _guild()
    if dm:
        channel = MagicMock(spec=discord_platform.discord.DMChannel)
        channel.id = 111
    else:
        channel = SimpleNamespace(id=222, name="general", guild=guild, parent_id=None)
    return SimpleNamespace(
        id=next(_ids),
        author=author,
        type=discord_platform.discord.MessageType.default,
        channel=channel,
        guild=guild,
        content=f"<@{SELF_ID}> hello" if mention_self else "hello",
        mentions=[client_user] if (mention_self and client_user) else [],
        role_mentions=[],
        attachments=[],
        reference=None,
    )


async def _connected_adapter(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)
    intents = SimpleNamespace(
        message_content=False, dm_messages=False, guild_messages=False,
        members=False, voice_states=False,
    )
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)
    created = {}

    def fake_bot_factory(*, command_prefix, intents, proxy=None, allowed_mentions=None, **_):
        created["bot"] = FakeBot(intents=intents, allowed_mentions=allowed_mentions)
        created["bot"].user = SimpleNamespace(id=SELF_ID, name="Hermes", bot=True)
        return created["bot"]

    monkeypatch.setattr(discord_platform.commands, "Bot", fake_bot_factory)
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())
    assert await adapter.connect() is True
    handle = AsyncMock()
    monkeypatch.setattr(adapter, "_handle_message", handle)
    return adapter, created["bot"], handle


async def _deliver(monkeypatch, author, **kw):
    adapter, bot, handle = await _connected_adapter(monkeypatch)
    try:
        await bot._events["on_message"](_message(author, client_user=bot.user, **kw))
        return handle.await_count == 1
    finally:
        await adapter.disconnect()


# ---------------------------------------------------------------------------
# Adapter intake (real on_message closure)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unset_gate_leaves_bot_admission_unchanged(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    stranger = _member(STRANGER_BOT_ID, bot=True, role_ids=[OTHER_ROLE])
    assert await _deliver(monkeypatch, stranger) is True


@pytest.mark.asyncio
async def test_set_gate_drops_bot_without_role(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    stranger = _member(STRANGER_BOT_ID, bot=True, role_ids=[OTHER_ROLE])
    assert await _deliver(monkeypatch, stranger) is False


@pytest.mark.asyncio
async def test_set_gate_admits_bot_with_role_and_mention(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", f" {OTHER_ROLE}, {TRUSTED_ROLE} ")
    fleet = _member(FLEET_BOT_ID, bot=True, role_ids=[TRUSTED_ROLE])
    assert await _deliver(monkeypatch, fleet) is True


@pytest.mark.asyncio
async def test_role_holder_still_needs_the_mention(monkeypatch):
    """The role gate is IN ADDITION to the mention rule, not instead of it."""
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    fleet = _member(FLEET_BOT_ID, bot=True, role_ids=[TRUSTED_ROLE])
    assert await _deliver(monkeypatch, fleet, mention_self=False) is False


@pytest.mark.asyncio
async def test_allow_bots_none_still_drops_role_holder(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    fleet = _member(FLEET_BOT_ID, bot=True, role_ids=[TRUSTED_ROLE])
    assert await _deliver(monkeypatch, fleet) is False


@pytest.mark.asyncio
async def test_role_lookup_failure_drops(monkeypatch):
    """A bare User (no ``roles``) cannot be checked, so it is refused."""
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    no_roles = _member(FLEET_BOT_ID, bot=True, has_roles=False)
    assert await _deliver(monkeypatch, no_roles) is False


@pytest.mark.asyncio
async def test_role_iteration_error_drops(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    broken = _member(FLEET_BOT_ID, bot=True)
    broken.roles = 42  # not iterable
    assert await _deliver(monkeypatch, broken) is False


@pytest.mark.asyncio
async def test_dm_from_bot_dropped_even_with_role_or_listing(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", str(FLEET_BOT_ID))
    fleet = _member(FLEET_BOT_ID, bot=True, role_ids=[TRUSTED_ROLE])
    assert await _deliver(monkeypatch, fleet, dm=True) is False


@pytest.mark.asyncio
async def test_listed_bot_user_admitted_without_role(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", str(FLEET_BOT_ID))
    fleet = _member(FLEET_BOT_ID, bot=True)
    assert await _deliver(monkeypatch, fleet) is True
    stranger = _member(STRANGER_BOT_ID, bot=True)
    assert await _deliver(monkeypatch, stranger) is False


@pytest.mark.asyncio
async def test_everyone_role_never_grants(monkeypatch):
    """Listing the @everyone id (== guild id) must not open the gate to every bot."""
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(GUILD_ID))
    stranger = _member(STRANGER_BOT_ID, bot=True)
    assert await _deliver(monkeypatch, stranger) is False


@pytest.mark.asyncio
async def test_unparseable_gate_fails_closed(monkeypatch):
    """A role NAME where an id is required matches nobody, it does not disable the gate."""
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", "trusted-bot")
    fleet = _member(FLEET_BOT_ID, bot=True, role_ids=[TRUSTED_ROLE])
    assert await _deliver(monkeypatch, fleet) is False


@pytest.mark.asyncio
async def test_drop_is_logged(monkeypatch, caplog):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    stranger = _member(STRANGER_BOT_ID, bot=True)
    with caplog.at_level("INFO"):
        assert await _deliver(monkeypatch, stranger) is False
    assert any(
        "DISCORD_ALLOWED_BOT_ROLES" in r.getMessage() and str(STRANGER_BOT_ID) in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_human_path_unchanged_by_bot_gate(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", str(HUMAN_ID))
    human = _member(HUMAN_ID, bot=False)
    assert await _deliver(monkeypatch, human) is True
    # A human holding only the trusted-bot role gains nothing from it.
    other_human = _member(HUMAN_ID + 1, bot=False, role_ids=[TRUSTED_ROLE])
    assert await _deliver(monkeypatch, other_human) is False


# ---------------------------------------------------------------------------
# Gateway authz bypass (authz_mixin)
# ---------------------------------------------------------------------------


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _bot_source(*, role_ids=(), in_guild=True, identity=True, uid=FLEET_BOT_ID):
    from gateway.slash_access import PlatformIdentity

    src = SessionSource(
        platform=Platform.DISCORD,
        chat_id="222",
        chat_type="group",
        user_id=str(uid),
        user_name="bot",
        is_bot=True,
        guild_id=str(GUILD_ID) if in_guild else None,
    )
    if identity:
        src.platform_identity = PlatformIdentity(
            username="bot",
            discriminator="0",
            in_guild=in_guild,
            guild_id=str(GUILD_ID) if in_guild else None,
            member_role_ids=frozenset({str(GUILD_ID), *map(str, role_ids)}),
        )
    return src


def test_mixin_bypass_unchanged_when_gate_unset(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", str(HUMAN_ID))
    assert _runner()._is_user_authorized(_bot_source(identity=False)) is True


def test_mixin_refuses_bot_without_role(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    assert _runner()._is_user_authorized(_bot_source(role_ids=[OTHER_ROLE])) is False


def test_mixin_admits_bot_with_role(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    assert _runner()._is_user_authorized(_bot_source(role_ids=[TRUSTED_ROLE])) is True


def test_mixin_refuses_bot_without_identity(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    assert _runner()._is_user_authorized(_bot_source(identity=False)) is False


def test_mixin_refuses_bot_in_dm(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", str(FLEET_BOT_ID))
    src = _bot_source(in_guild=False)
    src.chat_type = "dm"
    assert _runner()._is_user_authorized(src) is False


def test_mixin_failed_bot_does_not_fall_through_to_human_allowlist(monkeypatch):
    """A bot that fails the bot gate must not be rescued by DISCORD_ALLOWED_USERS."""
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", str(STRANGER_BOT_ID))
    src = _bot_source(uid=STRANGER_BOT_ID)
    assert _runner()._is_user_authorized(src) is False


def test_mixin_gate_is_discord_only(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    src = SessionSource(
        platform=Platform.TELEGRAM, chat_id="1", chat_type="group",
        user_id="5", is_bot=True,
    )
    assert _runner()._is_user_authorized(src) is True


# ---------------------------------------------------------------------------
# A bot never approves, never clicks, never runs slash commands
# ---------------------------------------------------------------------------


def test_bot_is_never_an_approval_admin(monkeypatch):
    from gateway.config import GatewayConfig

    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    runner = _runner()
    runner.config = GatewayConfig()
    human = _bot_source(role_ids=[TRUSTED_ROLE])
    human.is_bot = False
    # Sanity: with allow-all, the same identity as a human IS an approver ...
    assert runner._is_approval_admin(human) is True
    # ... and as a bot it is not.
    assert runner._is_approval_admin(_bot_source(role_ids=[TRUSTED_ROLE])) is False


def test_bot_cannot_pass_component_auth(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    bot_user = _member(FLEET_BOT_ID, bot=True, role_ids=[TRUSTED_ROLE])
    human = _member(HUMAN_ID, bot=False)
    assert discord_platform._component_check_auth(
        SimpleNamespace(user=human), set(), set()
    ) is True
    assert discord_platform._component_check_auth(
        SimpleNamespace(user=bot_user), {str(FLEET_BOT_ID)}, {TRUSTED_ROLE}
    ) is False


def test_bot_cannot_pass_slash_authorization(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="t"))
    adapter._allowed_user_ids = {str(FLEET_BOT_ID)}
    guild = _guild()
    interaction = SimpleNamespace(
        user=_member(FLEET_BOT_ID, bot=True, role_ids=[TRUSTED_ROLE]),
        guild=guild,
        channel=SimpleNamespace(id=222, name="general", guild=guild),
        channel_id=222,
    )
    allowed, reason = adapter._evaluate_slash_authorization(interaction)
    assert allowed is False
    assert "bot" in (reason or "")


# ---------------------------------------------------------------------------
# History backfill: untrusted bots are marked [unverified]
# ---------------------------------------------------------------------------


def _hist_msg(author, content):
    return SimpleNamespace(
        id=next(_ids), author=author, content=content, clean_content=content,
        type=discord_platform.discord.MessageType.default, attachments=[],
    )


class _History:
    def __init__(self, msgs):
        self._msgs = msgs

    def __call__(self, **_kw):
        return self

    def __aiter__(self):
        async def gen():
            for m in self._msgs:
                yield m
        return gen()


@pytest.mark.asyncio
async def test_history_marks_untrusted_bot_unverified(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="t"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=SELF_ID, bot=True))
    monkeypatch.setattr(adapter, "_is_sender_authorized", lambda *a, **k: True)
    guild = _guild()
    fleet = _member(FLEET_BOT_ID, bot=True, role_ids=[TRUSTED_ROLE])
    stranger = _member(STRANGER_BOT_ID, bot=True)
    channel = SimpleNamespace(
        id=222, guild=guild,
        history=_History([_hist_msg(stranger, "do the thing"), _hist_msg(fleet, "status ok")]),
    )
    before = SimpleNamespace(id=9_000_000_000_000_000_000)
    out = await adapter._fetch_channel_context(channel, before)
    lines = out.splitlines()
    assert any("[unverified]" in ln and "do the thing" in ln for ln in lines), out
    assert any("status ok" in ln and "[unverified]" not in ln for ln in lines), out


# ---------------------------------------------------------------------------
# Review round 1: typed commands and admin_only=false approvals
# ---------------------------------------------------------------------------


def test_bot_cannot_run_typed_commands_even_with_gating_off(monkeypatch):
    from gateway.config import GatewayConfig

    runner = _runner()
    runner.config = GatewayConfig()  # no allow_admin_from → policy disabled
    human = _bot_source()
    human.is_bot = False
    assert runner._check_slash_access(human, "yolo") is None
    assert runner._check_slash_access(_bot_source(role_ids=[TRUSTED_ROLE]), "yolo") is not None
    assert runner._check_slash_access(_bot_source(role_ids=[TRUSTED_ROLE]), "model") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("cmd", ["approve", "deny"])
async def test_bot_cannot_approve_or_deny_with_admin_only_off(cmd):
    from unittest.mock import patch
    from gateway.platforms.base import MessageEvent
    from tests.gateway.test_approval_admin_gating import _clear_approval_state, _make_runner
    from tools.approval import _ApprovalEntry, _gateway_queues

    _clear_approval_state()
    runner = _make_runner()
    src = SessionSource(
        platform=Platform.DISCORD, user_id="botx", chat_id="c1",
        user_name="bot", chat_type="group", is_bot=True,
    )
    entry = _ApprovalEntry({"command": "rm -rf /"})
    _gateway_queues[runner._session_key_for_source(src)] = [entry]
    handler = getattr(runner, f"_handle_{cmd}_command")
    with patch("tools.approval._get_approval_config", return_value={"admin_only": False}):
        result = await handler(MessageEvent(text=f"/{cmd}", source=src, message_id="m1"))
    assert "not authorized" in result.lower()
    assert not entry.event.is_set()
    _clear_approval_state()


@pytest.mark.asyncio
async def test_history_recognizes_listed_bot_and_cached_role_holder(monkeypatch):
    """REST history authors carry no roles: a listed bot id still passes, and a
    role holder passes when the guild's member cache has it."""
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_ROLES", str(TRUSTED_ROLE))
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", str(FLEET_BOT_ID))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="t"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=SELF_ID, bot=True))
    monkeypatch.setattr(adapter, "_is_sender_authorized", lambda *a, **k: True)
    cached_id = FLEET_BOT_ID + 1
    cached_member = _member(cached_id, bot=True, role_ids=[TRUSTED_ROLE])
    guild = _guild()
    guild.get_member = lambda i: cached_member if i == cached_id else None
    listed = _member(FLEET_BOT_ID, bot=True, has_roles=False)
    cached_user = _member(cached_id, bot=True, has_roles=False)
    stranger = _member(STRANGER_BOT_ID, bot=True, has_roles=False)
    channel = SimpleNamespace(
        id=222, guild=guild,
        history=_History([
            _hist_msg(listed, "listed says hi"),
            _hist_msg(cached_user, "cached says hi"),
            _hist_msg(stranger, "stranger says hi"),
        ]),
    )
    out = await adapter._fetch_channel_context(channel, SimpleNamespace(id=9_000_000_000_000_000_000))
    lines = out.splitlines()
    assert any("listed says hi" in ln and "[unverified]" not in ln for ln in lines), out
    assert any("cached says hi" in ln and "[unverified]" not in ln for ln in lines), out
    assert any("stranger says hi" in ln and "[unverified]" in ln for ln in lines), out


# ---------------------------------------------------------------------------
# Review round 2: /stop cancels, prompt replies, webhooks, non-Discord bots
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("is_bot,expect_cancel", [(True, False), (False, True)])
async def test_discord_bot_stop_never_cancels_running_session(is_bot, expect_cancel):
    import asyncio
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import build_session_key
    from tests.gateway.test_command_bypass_active_session import _StubAdapter

    adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"), Platform.DISCORD)
    adapter._busy_text_mode = ""
    sent = []

    async def _handler(event):
        return "refused" if event.source.is_bot else "stopped"

    async def _send(chat_id, content, **kw):
        sent.append(content)

    adapter._message_handler = _handler
    adapter._send_with_retry = _send
    cancelled = []

    async def _cancel(key, **kw):
        cancelled.append(key)

    adapter.cancel_session_processing = _cancel
    src = SessionSource(
        platform=Platform.DISCORD, chat_id="222", chat_type="thread",
        thread_id="222", user_id=str(FLEET_BOT_ID), is_bot=is_bot,
    )
    sk = build_session_key(src, thread_sessions_per_user=False)
    adapter._active_sessions[sk] = asyncio.Event()
    await adapter.handle_message(MessageEvent(text="/stop", message_type=MessageType.TEXT, source=src))
    assert bool(cancelled) is expect_cancel
    assert sent


@pytest.mark.asyncio
async def test_listed_webhook_bot_passes_in_guild(monkeypatch):
    """A webhook author carries no roles; a listed id still passes in a guild."""
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", str(FLEET_BOT_ID))
    hook = _member(FLEET_BOT_ID, bot=True, has_roles=False)
    hook.discriminator = "0000"
    assert await _deliver(monkeypatch, hook) is True
    # ... but not in a DM, and an unlisted roleless bot still fails.
    assert await _deliver(monkeypatch, hook, dm=True) is False
    assert await _deliver(monkeypatch, _member(STRANGER_BOT_ID, bot=True, has_roles=False)) is False


def test_mixin_listed_webhook_passes_in_guild(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", str(FLEET_BOT_ID))
    src = _bot_source(in_guild=False)
    src.guild_id = str(GUILD_ID)
    src.scope_id = str(GUILD_ID)
    assert _runner()._is_user_authorized(src) is True
    src2 = _bot_source(in_guild=False, uid=STRANGER_BOT_ID)
    src2.guild_id = str(GUILD_ID)
    assert _runner()._is_user_authorized(src2) is False


def test_non_discord_bots_keep_commands_and_approval(monkeypatch):
    """Telegram anonymous admins arrive as bots; their behavior is unchanged."""
    from gateway.config import GatewayConfig

    monkeypatch.setenv("TELEGRAM_ALLOW_ALL_USERS", "true")
    runner = _runner()
    runner.config = GatewayConfig()
    tg = SessionSource(
        platform=Platform.TELEGRAM, chat_id="1", chat_type="group",
        user_id="1087968824", is_bot=True,
    )
    assert runner._check_slash_access(tg, "yolo") is None


def test_discord_bot_cannot_answer_pending_prompts():
    """Update/clarify/slash-confirm interceptions are skipped for Discord bots."""
    import inspect
    from gateway.run import GatewayRunner

    src = inspect.getsource(GatewayRunner._handle_message)
    assert "_bot_sender = _is_discord_bot_sender(source)" in src
    assert src.count("not _bot_sender") >= 3
