"""Discord approvers by username and by role.

``allow_admin_from`` / ``group_allow_admin_from`` accept Discord usernames
(the unique handle) as well as numeric IDs, and ``approver_roles`` lets any
member holding a listed role approve dangerous commands. Both the exec-approval
buttons and the gateway's /approve, /deny and slash-command gate go through the
same matcher in ``gateway.slash_access``.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.slash_access import (
    PlatformIdentity,
    approver_role_matches,
    policy_from_extra,
)

# Trigger the shared discord mock (tests/gateway/conftest.py) before import.
from plugins.platforms.discord.adapter import (  # noqa: E402
    ExecApprovalView,
    _discord_platform_identity,
    _resolve_exec_approval_admin_gate,
)

JUNIPER = "1519531373352849518"
GUILD = "900000000000000001"
ROLE_APPROVER = "700000000000000001"
ROLE_OTHER = "700000000000000002"
GUILD_ROLES = (
    (GUILD, "@everyone"),  # the default role shares the guild's id
    (ROLE_APPROVER, "DCG Approver"),
    (ROLE_OTHER, "Moderators"),
)


def _ident(username="juniperbevensee", role_ids=(), in_guild=True,
           guild_roles=GUILD_ROLES, discriminator="0"):
    return PlatformIdentity(
        username=username,
        discriminator=discriminator,
        in_guild=in_guild,
        guild_id=GUILD if in_guild else None,
        member_role_ids=frozenset(role_ids) if in_guild else frozenset(),
        guild_roles=tuple(guild_roles) if in_guild else (),
    )


# ---------------------------------------------------------------------------
# allow_admin_from: IDs and usernames
# ---------------------------------------------------------------------------


def test_numeric_id_matches_without_identity():
    policy = policy_from_extra({"allow_admin_from": [JUNIPER]}, "dm")
    assert policy.is_admin(JUNIPER) is True
    assert policy.is_admin("123") is False


def test_username_matches_unique_handle():
    policy = policy_from_extra({"group_allow_admin_from": "@JuniperBevensee"}, "group")
    assert policy.is_admin(JUNIPER, _ident()) is True


def test_username_does_not_match_without_identity():
    """Non-Discord sources (no identity) never match a username entry."""
    policy = policy_from_extra({"allow_admin_from": "juniperbevensee"}, "dm")
    assert policy.is_admin(JUNIPER) is False


def test_display_name_never_matches():
    # policy below must actually be enabled, or is_admin() is vacuously True
    """Only the unique ``name`` counts; a display name/nickname equal to an
    admin's handle must not grant anything (spoofable)."""
    policy = policy_from_extra({"group_allow_admin_from": "juniperbevensee"}, "group")
    impostor = _ident(username="someone_else")
    # Even if the impostor's object carries a matching display name, the
    # identity is built from ``name`` only.
    user = SimpleNamespace(
        id=42, name="someone_else", display_name="juniperbevensee",
        global_name="juniperbevensee", nick="juniperbevensee", discriminator="0",
        roles=[],
    )
    guild = SimpleNamespace(id=int(GUILD), roles=[])
    built = _discord_platform_identity(user, guild)
    assert built.username == "someone_else"
    assert policy.enabled is True
    assert policy.is_admin("42", impostor) is False
    assert policy.is_admin("42", built) is False


def test_legacy_discriminator_user_needs_full_tag():
    policy = policy_from_extra({"group_allow_admin_from": "oldbot"}, "group")
    assert policy.is_admin("5", _ident(username="oldbot", discriminator="1234")) is False
    policy = policy_from_extra({"group_allow_admin_from": "oldbot#1234"}, "group")
    assert policy.is_admin("5", _ident(username="oldbot", discriminator="1234")) is True


def test_username_match_is_logged(caplog):
    policy = policy_from_extra({"group_allow_admin_from": "juniperbevensee"}, "group")
    with caplog.at_level(logging.INFO, logger="gateway.slash_access"):
        policy.is_admin("777000777", _ident())
    assert any("juniperbevensee" in r.message and "777000777" in r.message
               for r in caplog.records)


def test_empty_config_unchanged():
    policy = policy_from_extra({}, "group")
    assert policy.enabled is False
    assert policy.is_admin("anyone", _ident()) is True  # gating off = everyone
    assert approver_role_matches({}, _ident(role_ids={ROLE_APPROVER})) is False


def test_approver_roles_alone_do_not_enable_slash_gating():
    policy = policy_from_extra({"approver_roles": ROLE_APPROVER}, "group")
    assert policy.enabled is False


# ---------------------------------------------------------------------------
# approver_roles
# ---------------------------------------------------------------------------


def test_role_id_matches():
    extra = {"approver_roles": ROLE_APPROVER}
    assert approver_role_matches(extra, _ident(role_ids={ROLE_APPROVER})) is True
    assert approver_role_matches(extra, _ident(role_ids={ROLE_OTHER})) is False


def test_role_name_matches_case_insensitive():
    extra = {"approver_roles": ["dcg approver"]}
    assert approver_role_matches(extra, _ident(role_ids={ROLE_APPROVER})) is True
    assert approver_role_matches(extra, _ident(role_ids={ROLE_OTHER})) is False


def test_ambiguous_role_name_denied(caplog):
    roles = GUILD_ROLES + (("700000000000000003", "dcg approver"),)
    extra = {"approver_roles": "DCG Approver"}
    with caplog.at_level(logging.WARNING, logger="gateway.slash_access"):
        ok = approver_role_matches(
            extra, _ident(role_ids={ROLE_APPROVER}, guild_roles=roles)
        )
    assert ok is False
    assert any(r.levelno == logging.WARNING and "DCG Approver" in r.message
               for r in caplog.records)


def test_missing_role_name_denied(caplog):
    extra = {"approver_roles": "No Such Role"}
    with caplog.at_level(logging.WARNING, logger="gateway.slash_access"):
        ok = approver_role_matches(extra, _ident(role_ids={ROLE_APPROVER}))
    assert ok is False
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_everyone_role_never_grants():
    extra = {"approver_roles": [GUILD, "@everyone"]}
    assert approver_role_matches(extra, _ident(role_ids={GUILD})) is False


def test_roles_never_grant_in_dm():
    extra = {"approver_roles": ROLE_APPROVER}
    assert approver_role_matches(extra, _ident(in_guild=False)) is False
    assert approver_role_matches(extra, None) is False


def test_discord_env_var_feeds_approver_roles(monkeypatch):
    from gateway.config import load_gateway_config

    monkeypatch.setenv("DISCORD_BOT_TOKEN", "x")
    monkeypatch.setenv("DISCORD_APPROVER_ROLES", f"{ROLE_APPROVER},DCG Approver")
    cfg = load_gateway_config()
    extra = cfg.platforms[Platform.DISCORD].extra
    assert extra.get("approver_roles") == f"{ROLE_APPROVER},DCG Approver"


# ---------------------------------------------------------------------------
# Exec-approval buttons
# ---------------------------------------------------------------------------


@pytest.fixture
def _no_pairing():
    store = MagicMock()
    store.is_approved.return_value = False
    with patch("gateway.pairing.PairingStore", return_value=store):
        yield


def _interaction(uid, name, role_ids=(), in_guild=True, display_name=None):
    roles = [SimpleNamespace(id=int(r), name="") for r in role_ids]
    user = SimpleNamespace(
        id=int(uid), name=name, display_name=display_name or name,
        global_name=display_name, discriminator="0", roles=roles,
    )
    guild = None
    if in_guild:
        guild = SimpleNamespace(
            id=int(GUILD),
            roles=[SimpleNamespace(id=int(i), name=n) for i, n in GUILD_ROLES],
        )
    return SimpleNamespace(user=user, guild=guild)


def test_resolver_returns_approver_roles():
    gate = _resolve_exec_approval_admin_gate(
        {"require_admin_for_exec_approval": True, "allow_admin_from": JUNIPER,
         "approver_roles": "DCG Approver"}
    )
    assert gate[0] is True
    assert gate[1] == {JUNIPER}


def _view(**kw):
    return ExecApprovalView(
        session_key="s", allowed_user_ids={"*"}, require_admin=True, **kw
    )


def test_button_username_entry(_no_pairing):
    view = _view(admin_user_ids={"juniperbevensee"})
    assert view._check_auth(_interaction(JUNIPER, "juniperbevensee")) is True
    assert view._check_auth(
        _interaction("42", "someone_else", display_name="juniperbevensee")
    ) is False


def test_button_role_name(_no_pairing):
    view = _view(admin_user_ids=set(), approver_roles={"DCG Approver"})
    assert view._check_auth(_interaction("42", "x", role_ids={ROLE_APPROVER})) is True
    assert view._check_auth(_interaction("43", "y", role_ids={ROLE_OTHER})) is False


def test_button_role_id_not_honored_in_dm(_no_pairing):
    view = _view(admin_user_ids=set(), approver_roles={ROLE_APPROVER})
    assert view._check_auth(
        _interaction("42", "x", role_ids={ROLE_APPROVER}, in_guild=False)
    ) is False


def test_button_roles_only_is_not_misconfigured(_no_pairing, caplog):
    """approver_roles with no allow_admin_from must not log the
    'no admins configured' warning."""
    view = _view(admin_user_ids=set(), approver_roles={ROLE_APPROVER})
    with caplog.at_level(logging.WARNING):
        view._check_auth(_interaction("43", "y", role_ids={ROLE_OTHER}))
    assert not any("no admins are configured" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Gateway: /approve admin check and the slash gate share the matcher
# ---------------------------------------------------------------------------


def _runner(extra):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="t", extra=extra)}
    )
    runner._is_individual_allowlisted = lambda _s: False
    return runner


def _source(uid, ident):
    src = SessionSource(
        platform=Platform.DISCORD, chat_id="c", chat_type="group",
        user_id=uid, user_name="whatever", scope_id=GUILD,
    )
    src.platform_identity = ident
    return src


def test_gateway_approval_admin_by_role():
    runner = _runner({"approver_roles": "DCG Approver"})
    assert runner._is_approval_admin(_source("42", _ident("x", {ROLE_APPROVER}))) is True
    assert runner._is_approval_admin(_source("43", _ident("y", {ROLE_OTHER}))) is False


def test_gateway_approval_admin_by_username():
    runner = _runner({"group_allow_admin_from": "juniperbevensee"})
    assert runner._is_approval_admin(_source(JUNIPER, _ident())) is True
    assert runner._is_approval_admin(_source("42", _ident("someone_else"))) is False


def test_slash_gate_lets_role_approver_run_approve_only():
    runner = _runner({"group_allow_admin_from": JUNIPER, "approver_roles": ROLE_APPROVER})
    src = _source("42", _ident("x", {ROLE_APPROVER}))
    assert runner._check_slash_access(src, "approve") is None
    assert runner._check_slash_access(src, "deny") is None
    assert runner._check_slash_access(src, "restart") is not None


def test_slash_gate_username_admin_runs_everything():
    runner = _runner({"group_allow_admin_from": "@juniperbevensee"})
    assert runner._check_slash_access(_source(JUNIPER, _ident()), "restart") is None


def test_platform_identity_not_serialized():
    src = _source(JUNIPER, _ident())
    assert "platform_identity" not in src.to_dict()


def test_non_numeric_platform_ids_still_match_exactly():
    """Slack/Matrix/phone ids aren't digits; they must keep matching as ids."""
    for uid in ("U0123ABC", "@alice:example.org", "+6421000000", "admin1"):
        policy = policy_from_extra({"allow_admin_from": uid}, "dm")
        assert policy.is_admin(uid) is True
        assert policy.is_admin("someone") is False


def test_webhook_author_never_matches_username():
    """Webhook messages carry a sender-chosen ``name`` and discriminator
    ``0000`` (bridges, PluralKit, anyone with Manage Webhooks). That name is
    not a unique handle, so it must never match a username admin entry —
    neither the bare name nor an explicit ``#0000`` tag."""
    webhook_author = SimpleNamespace(
        id=555, name="juniperbevensee", discriminator="0000", bot=True,
    )
    built = _discord_platform_identity(webhook_author, None)
    for entry in ("juniperbevensee", "@juniperbevensee", "juniperbevensee#0000"):
        policy = policy_from_extra({"group_allow_admin_from": entry}, "group")
        assert policy.enabled is True
        assert policy.is_admin("555", built) is False, entry


def test_roles_never_grant_outside_guild_even_if_roles_present():
    """``in_guild`` is the DM guard on its own, not just the empty role set."""
    forged = PlatformIdentity(
        username="x", discriminator="0", in_guild=False, guild_id=GUILD,
        member_role_ids=frozenset({ROLE_APPROVER}), guild_roles=GUILD_ROLES,
    )
    assert approver_role_matches({"approver_roles": ROLE_APPROVER}, forged) is False
    assert approver_role_matches({"approver_roles": "DCG Approver"}, forged) is False
