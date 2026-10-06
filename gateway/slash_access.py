"""Per-platform slash command access control.

This module sits beside the existing per-platform allowlist (``allow_from``)
and adds a second axis: of the users who are *allowed to talk to the
gateway*, which ones can run *which slash commands*.

Two lists per platform scope (DM vs group, mirroring ``allow_from`` vs
``group_allow_from``):

  - ``allow_admin_from``      — user IDs (or, on Discord, unique usernames)
                                that get every registered slash command
                                (built-in + plugin-registered).
  - ``user_allowed_commands`` — slash command names non-admin users may
                                run. Empty / unset → non-admins get no
                                slash commands.

Backward compatibility:

  If ``allow_admin_from`` is not set for a scope, slash command gating
  is disabled entirely for that scope. Every allowed user can run every
  slash command, exactly like before. This means existing installs are
  unaffected until an operator opts in by listing at least one admin.

The gate is applied at the slash command dispatch site in
``gateway/run.py`` so it covers BOTH built-in and plugin-registered
commands via the live registry. Gating slash commands does not affect
plain chat — non-admin users can still talk to the agent normally,
they just can't trigger commands outside ``user_allowed_commands``.

Authored as a slimmed-down salvage of PR #4443's permission tiers
(co-authored by @ReqX). The full tier system, audit log, usage
tracking, rate limiting, and tool filtering from that PR are not
included here — only the slash-command access split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)


# Slash commands that MUST stay reachable for any allowed user, even when
# slash gating is enabled and the user has no commands listed. Without this
# carve-out, a non-admin user has no way to discover what they can or
# can't do (``/help``, ``/whoami``) and no way to see what state the agent
# is in (``/status``). These mirror the smallest set of read-only commands
# we'd hand to a guest. Operators can still narrow this further by writing
# their own ``user_allowed_commands`` (this set is only the implicit
# fallback floor — anything in ``user_allowed_commands`` overrides it
# additively, never restrictively).
_ALWAYS_ALLOWED_FOR_USERS: FrozenSet[str] = frozenset({
    "help",
    "whoami",
})


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
    member_role_ids: FrozenSet[str] = frozenset()
    guild_roles: Tuple[Tuple[str, str], ...] = ()  # (role id, role name)


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
    if "#" in want:
        want_name, _, want_disc = want.rpartition("#")
        return want_name == name and want_disc == disc
    return want == name and disc in {"0", "0000", ""}


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


@dataclass(frozen=True)
class SlashAccessPolicy:
    """Resolved access policy for a single (platform, scope) pair.

    ``scope`` is ``"dm"`` for direct messages and ``"group"`` for groups,
    channels, threads, and any other multi-user context. The mapping from
    SessionSource.chat_type → scope happens in ``policy_for_source``.
    """

    enabled: bool                      # gating active for this scope?
    admin_user_ids: FrozenSet[str]
    user_allowed_commands: FrozenSet[str]

    def is_admin(
        self, user_id: Optional[str], identity: Optional[PlatformIdentity] = None,
    ) -> bool:
        if not self.enabled:
            # Gating disabled → treat every allowed user as admin so
            # downstream code can keep using ``is_admin`` / ``can_run``
            # uniformly.
            return True
        if not user_id:
            return False
        return admin_entries_match(self.admin_user_ids, user_id, identity)

    def can_run(
        self, user_id: Optional[str], canonical_cmd: str,
        identity: Optional[PlatformIdentity] = None,
    ) -> bool:
        if not self.enabled:
            return True
        if self.is_admin(user_id, identity):
            return True
        if not canonical_cmd:
            return False
        if canonical_cmd in _ALWAYS_ALLOWED_FOR_USERS:
            return True
        return canonical_cmd in self.user_allowed_commands


_DM_CHAT_TYPES = frozenset({"dm", "direct", "private", ""})


def _coerce_id_list(raw: Any) -> FrozenSet[str]:
    """Normalize a YAML-loaded admin/user list into a frozenset of strings.

    Accepts ``None``, list, tuple, or comma-separated string. Stringifies
    each entry and strips whitespace; empty entries are dropped.
    """
    if raw is None:
        return frozenset()
    if isinstance(raw, (list, tuple, set, frozenset)):
        items: Iterable[Any] = raw
    elif isinstance(raw, str):
        items = (s for s in raw.split(",") if s.strip())
    else:
        # single scalar (int user id, etc.)
        items = (raw,)
    out: list[str] = []
    for it in items:
        s = str(it).strip()
        if s:
            out.append(s)
    return frozenset(out)


def _coerce_command_list(raw: Any) -> FrozenSet[str]:
    """Normalize a slash command allowlist.

    Strips leading slashes so YAML can read either ``["help", "status"]``
    or ``["/help", "/status"]``. Lowercase canonicalization matches how
    ``resolve_command()`` stores names.
    """
    if raw is None:
        return frozenset()
    if isinstance(raw, (list, tuple, set, frozenset)):
        items: Iterable[Any] = raw
    elif isinstance(raw, str):
        items = (s for s in raw.split(",") if s.strip())
    else:
        items = (raw,)
    out: list[str] = []
    for it in items:
        s = str(it).strip().lstrip("/").lower()
        if s:
            out.append(s)
    return frozenset(out)


def _scope_for_chat_type(chat_type: Optional[str]) -> str:
    if chat_type and chat_type.lower() in _DM_CHAT_TYPES:
        return "dm"
    return "group"


def _platform_extra(platform_config: Any) -> dict:
    """Return the ``extra`` dict from a PlatformConfig-like object.

    Defensively handles None and non-PlatformConfig shapes so calling
    code can stay simple.
    """
    if platform_config is None:
        return {}
    extra = getattr(platform_config, "extra", None)
    if isinstance(extra, dict):
        return extra
    if isinstance(platform_config, dict):
        # Some test harnesses pass dicts directly.
        return platform_config
    return {}


def _keys_for_scope(scope: str) -> Tuple[str, str]:
    """Return (admin_key, user_cmd_key) names for a scope."""
    if scope == "group":
        return ("group_allow_admin_from", "group_user_allowed_commands")
    return ("allow_admin_from", "user_allowed_commands")


def policy_from_extra(extra: dict, scope: str) -> SlashAccessPolicy:
    """Build a policy from a platform's ``extra`` dict for one scope.

    DM scope falls back to group scope keys ONLY for ``user_allowed_commands``
    when the DM scope didn't specify its own. This keeps the common case
    (operator wants the same command set DM and group) ergonomic without
    forcing duplication. Admin lists are NOT cross-scope: an admin in
    DMs is not implicitly an admin in a group.
    """
    admin_key, cmd_key = _keys_for_scope(scope)
    admin_ids = _coerce_id_list(extra.get(admin_key))
    cmds = _coerce_command_list(extra.get(cmd_key))

    if scope == "dm" and not cmds:
        # DM didn't specify — let group's user_allowed_commands fall through
        # so operators only need to list it once if it's the same.
        cmds = _coerce_command_list(extra.get("group_user_allowed_commands"))

    enabled = bool(admin_ids)
    return SlashAccessPolicy(
        enabled=enabled,
        admin_user_ids=admin_ids,
        user_allowed_commands=cmds,
    )


def policy_for_source(gateway_config: Any, source: Any) -> SlashAccessPolicy:
    """Resolve the access policy for a SessionSource.

    Returns a "disabled" policy (gating off, allow everything) when:
      - gateway_config is None
      - the platform has no PlatformConfig
      - the platform's PlatformConfig has no admin list set for the scope

    Callers should treat the returned policy as authoritative for slash
    command gating only. It does not gate plain chat messages.
    """
    if gateway_config is None or source is None:
        return SlashAccessPolicy(
            enabled=False,
            admin_user_ids=frozenset(),
            user_allowed_commands=frozenset(),
        )
    platforms = getattr(gateway_config, "platforms", None)
    platform_config = None
    if platforms is not None:
        try:
            platform_config = platforms.get(source.platform)
        except Exception:
            platform_config = None
    extra = _platform_extra(platform_config)
    scope = _scope_for_chat_type(getattr(source, "chat_type", None))
    return policy_from_extra(extra, scope)


__all__ = [
    "PlatformIdentity",
    "admin_entries_match",
    "approver_role_matches",
    "approver_roles_match",
    "SlashAccessPolicy",
    "policy_from_extra",
    "policy_for_source",
]
