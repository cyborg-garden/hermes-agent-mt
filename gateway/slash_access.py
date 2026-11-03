"""Per-platform slash command access control.

A second axis beside ``allow_from``: of the users allowed to talk to the gateway, which may run
which slash commands. Two lists per scope (DM vs group, mirroring ``allow_from`` /
``group_allow_from``): ``allow_admin_from`` (user IDs, or on Discord unique usernames, that get every registered command, built-in
and plugin) and ``user_allowed_commands`` (names non-admins may run; empty/unset -> only the
``_ALWAYS_ALLOWED_FOR_USERS`` floor). No ``allow_admin_from`` for a scope => gating disabled there,
so existing installs are unaffected until an operator lists an admin. Applied at the dispatch
site in ``gateway/run.py`` via the live registry; never affects plain chat.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

# Read-only floor every allowed user keeps under gating, so a non-admin can still discover what
# they can do. ``user_allowed_commands`` only adds to this set, never restricts it.
_ALWAYS_ALLOWED_FOR_USERS: frozenset[str] = frozenset({"help", "whoami"})

_DM_CHAT_TYPES = frozenset({"dm", "direct", "private", ""})

# scope -> (admin list key, user command list key)
_SCOPE_KEYS = {
    "dm": ("allow_admin_from", "user_allowed_commands"),
    "group": ("group_allow_admin_from", "group_user_allowed_commands"),
}


@dataclass(frozen=True)
class PlatformIdentity:
    """Live identity facts an adapter attaches for name/role admin matching.

    Built by the Discord adapter from the live ``Member``/``User`` object at
    ingest or button-click time (``_discord_platform_identity``). Platforms
    that don't build one pass ``None`` and keep the numeric-ID-only behavior.

    ``username`` is the platform's *unique* handle (Discord ``user.name``) —
    never a display name, global name or nickname, which anyone can set to
    anything. ``in_guild`` is False in DMs, and roles never grant there.
    """

    username: Optional[str] = None
    discriminator: Optional[str] = None
    in_guild: bool = False
    guild_id: Optional[str] = None
    member_role_ids: frozenset[str] = frozenset()
    guild_roles: tuple[tuple[str, str], ...] = ()  # (role id, role name)


# One log line per (entry, user) / (guild, role entry) so operators can verify
# what a name resolved to without the log repeating on every message.
_logged_resolutions: set = set()


def _log_once(key: tuple, level: int, msg: str, *args: Any) -> None:
    if key in _logged_resolutions:
        return
    _logged_resolutions.add(key)
    logger.log(level, msg, *args)


def _username_entry_matches(entry: str, identity: PlatformIdentity) -> bool:
    """Match one non-numeric admin entry against the unique handle.

    ``juniperbevensee`` / ``@juniperbevensee`` match a user whose ``name`` is
    that (case-insensitive) and who is on the unique-username system
    (discriminator ``0``/absent). Legacy accounts — mostly bots — must be
    written as ``name#1234``.
    """
    name = (identity.username or "").strip().lower()
    if not name:
        return False
    want = entry.strip().lstrip("@").strip().lower()
    if not want:
        return False
    disc = str(identity.discriminator or "0").strip()
    if disc == "0000":
        # Webhook authors: the sender picks ``name`` per message (bridges,
        # PluralKit, anyone with Manage Webhooks). Not a unique handle.
        return False
    if "#" in want:
        want_name, _, want_disc = want.rpartition("#")
        return want_name == name and want_disc == disc
    return want == name and disc in {"0", ""}


def admin_entries_match(
    entries: Iterable[str],
    user_id: Optional[str],
    identity: Optional[PlatformIdentity] = None,
) -> bool:
    """True if *user_id* / *identity* matches any admin entry.

    Every entry is first compared exactly against the user id (fast path, no
    identity needed) — that keeps non-numeric platform ids (Slack ``U…``,
    Matrix ``@a:b``, phone numbers) working as before. Non-numeric entries are
    then tried as usernames, only when the adapter supplied an identity. Shared by slash gating, /approve and /deny, and the
    Discord exec-approval buttons so all three agree.
    """
    uid = str(user_id).strip() if user_id is not None else ""
    names = []
    for raw in entries or ():
        entry = str(raw).strip()
        if not entry:
            continue
        if uid and entry == uid:
            return True
        if not entry.isdigit() and entry != "*":
            names.append(entry)
    if not names or identity is None or not uid:
        return False
    for entry in names:
        if _username_entry_matches(entry, identity):
            _log_once(
                ("user", entry.lower(), uid), logging.INFO,
                "Admin entry %r matched Discord username %r -> user id %s "
                "(usernames can be changed; list the id to be rename-proof)",
                entry, identity.username, uid,
            )
            return True
    return False


def approver_roles_match(
    role_entries: Iterable[str],
    identity: Optional[PlatformIdentity],
) -> bool:
    """True if the member holds a listed approver role in this guild.

    Entries are role IDs or role names. Names resolve against the guild's
    current roles, case-insensitive exact match; a name matching zero or
    several roles grants nothing and logs a WARNING. The @everyone role
    (id == guild id) never grants. Never grants outside a guild (DMs).
    """
    if identity is None or not identity.in_guild:
        return False
    entries = [str(e).strip() for e in (role_entries or ()) if str(e).strip()]
    if not entries:
        return False
    guild_id = str(identity.guild_id or "")
    held = {str(r) for r in identity.member_role_ids} - {guild_id}
    if not held:
        return False
    for entry in entries:
        if entry.isdigit():
            if entry != guild_id and entry in held:
                return True
            continue
        want = entry.lstrip("@").strip().lower() if entry.lower() != "@everyone" else ""
        matches = [
            str(rid) for rid, rname in identity.guild_roles
            if want and str(rname).strip().lower() == want and str(rid) != guild_id
        ]
        if len(matches) != 1:
            _log_once(
                ("role", guild_id, entry.lower(), len(matches)), logging.WARNING,
                "approver_roles entry %r matches %d roles in guild %s — not "
                "granting anything for it. Use the role id to be exact.",
                entry, len(matches), guild_id or "?",
            )
            continue
        _log_once(
            ("role", guild_id, entry.lower(), matches[0]), logging.INFO,
            "approver_roles entry %r resolved to role id %s in guild %s",
            entry, matches[0], guild_id or "?",
        )
        if matches[0] in held:
            return True
    return False


def approver_role_matches(extra: Any, identity: Optional[PlatformIdentity]) -> bool:
    """``approver_roles`` from a platform ``extra`` dict, checked for *identity*."""
    if not isinstance(extra, dict):
        return False
    return approver_roles_match(_coerce_id_list(extra.get("approver_roles")), identity)


_BOT_ROLES_ENV = "DISCORD_ALLOWED_BOT_ROLES"
_BOT_USERS_ENV = "DISCORD_ALLOWED_BOT_USERS"


def _snowflakes_from_env(var: str) -> tuple[bool, frozenset[str]]:
    """(configured, numeric ids) for a comma-separated id list in *var*.

    ``configured`` is True whenever the variable holds anything non-blank,
    even if no entry parses: a role NAME where an id belongs must match
    nobody, not switch the gate off.
    """
    raw = os.getenv(var, "") or ""
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    ids = frozenset(e for e in entries if e.isdigit())
    bad = [e for e in entries if not e.isdigit()]
    if bad:
        _log_once(
            ("bot-gate-parse", var, raw), logging.ERROR,
            "%s=%r has non-numeric entries %r — they match nobody. Use the "
            "numeric Discord id (Developer Mode > Copy ID).",
            var, raw, bad,
        )
    return bool(entries), ids


def discord_bot_gate_decision(
    user_id: Optional[str], identity: Optional[PlatformIdentity],
) -> Optional[bool]:
    """Whether a Discord BOT sender passes ``DISCORD_ALLOWED_BOT_ROLES/_USERS``.

    Returns ``None`` when neither variable is set — the gate is off and the
    caller keeps its old behavior (``DISCORD_ALLOW_BOTS`` alone decides).
    Otherwise ``True``/``False``. This is an extra requirement on top of
    ``DISCORD_ALLOW_BOTS``, never a substitute for it.

    Fails closed: no identity, no guild (DMs), or no readable role list all
    return False — a listed bot user id does not pass in a DM either. The
    @everyone role (id == guild id) never grants.
    """
    roles_set, role_ids = _snowflakes_from_env(_BOT_ROLES_ENV)
    users_set, user_ids = _snowflakes_from_env(_BOT_USERS_ENV)
    if not roles_set and not users_set:
        return None
    if identity is None or not identity.in_guild:
        return False
    uid = str(user_id).strip() if user_id is not None else ""
    if uid and uid in user_ids:
        return True
    guild_id = str(identity.guild_id or "")
    held = {str(r) for r in identity.member_role_ids} - {guild_id, ""}
    return bool(held & (set(role_ids) - {guild_id}))


@dataclass(frozen=True)
class SlashAccessPolicy:
    """Resolved access policy for one (platform, scope) pair; scope is ``"dm"`` or ``"group"``."""

    enabled: bool  # gating active for this scope?
    admin_user_ids: frozenset[str]
    user_allowed_commands: frozenset[str]

    def is_admin(
        self, user_id: Optional[str], identity: Optional[PlatformIdentity] = None,
    ) -> bool:
        # Gating disabled -> everyone is admin so callers can use is_admin/can_run uniformly.
        if not self.enabled:
            return True
        if not user_id:
            return False
        # mt (#175): Discord unique usernames match via the live platform identity.
        return admin_entries_match(self.admin_user_ids, user_id, identity)

    def can_run(
        self, user_id: Optional[str], canonical_cmd: str,
        identity: Optional[PlatformIdentity] = None,
    ) -> bool:
        if self.is_admin(user_id, identity):
            return True
        return bool(canonical_cmd) and (
            canonical_cmd in _ALWAYS_ALLOWED_FOR_USERS or canonical_cmd in self.user_allowed_commands
        )


_DISABLED_POLICY = SlashAccessPolicy(
    enabled=False, admin_user_ids=frozenset(), user_allowed_commands=frozenset()
)
# Gated with nobody on the admin list: only the ``_ALWAYS_ALLOWED_FOR_USERS`` floor runs.
_FAIL_CLOSED_POLICY = SlashAccessPolicy(
    enabled=True, admin_user_ids=frozenset(), user_allowed_commands=frozenset()
)


def _coerce_list(raw: Any, normalize: Callable[[str], str] = str) -> frozenset[str]:
    """Normalize a YAML-loaded value (None, list/tuple/set, comma string, or scalar) into a
    frozenset of stripped, non-empty strings, applying ``normalize`` to each."""
    if raw is None:
        return frozenset()
    if isinstance(raw, (list, tuple, set, frozenset)):
        items: Iterable[Any] = raw
    elif isinstance(raw, str):
        items = (s for s in raw.split(",") if s.strip())
    else:
        items = (raw,)  # single scalar (int user id, etc.)
    return frozenset(s for s in (normalize(str(it).strip()) for it in items) if s)


def _coerce_id_list(raw: Any) -> frozenset[str]:
    """Normalize an admin/user id list into a frozenset of strings."""
    return _coerce_list(raw)


def _coerce_command_list(raw: Any) -> frozenset[str]:
    """Command allowlist: strip leading slashes (``/help`` or ``help``) and lowercase to match
    how ``resolve_command()`` stores names."""
    return _coerce_list(raw, lambda s: s.lstrip("/").lower())


def policy_from_extra(extra: dict, scope: str) -> SlashAccessPolicy:
    """Build a policy from a platform's ``extra`` dict for one scope.

    DM scope falls back to ``group_user_allowed_commands`` ONLY for the command list, and only
    when DM didn't set its own. Admin lists are NOT cross-scope: a DM admin is not a group admin.
    """
    admin_key, cmd_key = _SCOPE_KEYS.get(scope, _SCOPE_KEYS["dm"])
    admin_ids = _coerce_id_list(extra.get(admin_key))
    cmds = _coerce_command_list(extra.get(cmd_key))
    if scope == "dm" and not cmds:
        cmds = _coerce_command_list(extra.get("group_user_allowed_commands"))
    return SlashAccessPolicy(enabled=bool(admin_ids), admin_user_ids=admin_ids, user_allowed_commands=cmds)


def policy_for_source(gateway_config: Any, source: Any) -> SlashAccessPolicy:
    """Resolve the slash-gating policy for a SessionSource.

    Disabled (allow-everything) when gateway_config/source is None, the platform has no
    PlatformConfig, or no admin list is set for the scope. Gates slash commands only, never chat.
    """
    if gateway_config is None or source is None:
        return _DISABLED_POLICY
    platforms = getattr(gateway_config, "platforms", None)
    platform_config = None
    if platforms is not None:
        try:
            platform_config = platforms.get(source.platform)
        except Exception:
            platform_config = None
    # ``extra`` from a PlatformConfig-like object, or a bare dict as some test harnesses pass.
    extra = getattr(platform_config, "extra", None)
    if not isinstance(extra, dict):
        extra = platform_config if isinstance(platform_config, dict) else {}
    chat_type = getattr(source, "chat_type", None)
    normalized = str(chat_type).strip().lower() if chat_type is not None else ""
    if normalized:
        scope = "dm" if normalized in _DM_CHAT_TYPES else "group"
        return policy_from_extra(extra, scope)
    # Blank/None chat_type is ambiguous (relay frames may send "" / null; restored rows keep a
    # stored empty value verbatim). _DM_CHAT_TYPES lists "" but a truthiness guard made it
    # unreachable: resolve to whichever scope is gated so an ambiguous source never lands in an
    # ungated allow-everything scope; keep the historical group scope on a tie.
    dm_policy = policy_from_extra(extra, "dm")
    group_policy = policy_from_extra(extra, "group")
    return dm_policy if dm_policy.enabled and not group_policy.enabled else group_policy


__all__ = [
    "PlatformIdentity",
    "SlashAccessPolicy",
    "admin_entries_match",
    "approver_role_matches",
    "approver_roles_match",
    "discord_bot_gate_decision",
    "policy_for_runner_source",
    "policy_for_source",
    "policy_from_extra",
]



def policy_for_runner_source(runner: Any, source: Any) -> SlashAccessPolicy:
    """Slash-gating policy from the profile whose bot received ``source``.

    Under ``multiplex_profiles`` the runner's ``config`` is the launch profile's alone, so a platform
    configured only in a secondary profile had no PlatformConfig there and every caller came back
    unrestricted (#121705). Authorization follows the transport (bot-owning) profile, the owner
    ``RoutingIdentity.authorization_home`` already names; a served profile whose config is not cached
    fails closed instead of inheriting the launch profile's open policy.
    """
    config = getattr(runner, "config", None)
    if not getattr(config, "multiplex_profiles", False) or source is None:
        return policy_for_source(config, source)
    from gateway.session_identity import identity_of

    identity = identity_of(source)
    owner = identity.transport_profile if identity is not None else getattr(source, "profile", None)
    primary = getattr(runner, "_primary_profile_name", None) or "default"
    if owner and owner != primary:
        config = (getattr(runner, "_profile_configs", None) or {}).get(owner)
        if config is None:
            return _FAIL_CLOSED_POLICY
    return policy_for_source(config, source)
