"""Shared user-or-role authorization for Discord component (button) clicks."""
from __future__ import annotations

from typing import Any, Callable, Optional

from gateway.platforms._shared import platform_gate_env as _scoped_gate_env


def _component_check_auth(
    interaction, allowed_user_ids: Optional[set], allowed_role_ids: Optional[set],
    live_auth: Optional[Callable[[Any], Optional[bool]]] = None,
) -> bool:
    """Shared user-or-role OR authorization for component button clicks.
    Allow on: DISCORD/GATEWAY_ALLOW_ALL_USERS, user in DISCORD/GATEWAY_ALLOWED_USERS, a role in the
    role allowlist, or pairing-store approval. Role allowlist with no ``roles`` (DM) rejects (fail closed).
    ``allowed_user_ids`` is the adapter's connect-time snapshot: a user (or ``*``) admitted only there is confirmed
    with ``live_auth`` (the gateway's per-call check), and an explicit False falls through to the role
    and pairing grants.
    """
    user = getattr(interaction, "user", None)
    if user is None or getattr(user, "id", None) is None:
        return False
    # Bots never click approval/component buttons. Discord does not deliver
    # bot-originated interactions today; this keeps it true if that changes.
    if getattr(user, "bot", False) is True:
        return False
    # R7: the moderation deny store beats every allow branch below —
    # including the ALLOW_ALL flags and the GATEWAY_ALLOWED_USERS union.
    # A banned user must not be able to click approval buttons. Imported via
    # the adapter so both share one deny-store module object.
    from plugins.platforms.discord.adapter import _moderation_sync

    try:
        _denied_uid = str(user.id)
    except AttributeError:
        _denied_uid = ""
    if _denied_uid and _moderation_sync.deny_store().is_denied(_denied_uid):
        return False
    # Scope-aware reads: interaction tasks inherit the owning profile's secret-scope contextvar;
    # under multiplex a raw os.getenv could return ANOTHER profile's allow-all flag.
    # Scope-aware reads (issue #72348): component interactions are dispatched from discord.py tasks
    # descended from the task created inside the owning profile's runtime scope, so the profile's
    # secret-scope contextvar is inherited here.
    if _scoped_gate_env("DISCORD_ALLOW_ALL_USERS").strip().lower() in {"true", "1", "yes"}:
        return True
    if _scoped_gate_env("GATEWAY_ALLOW_ALL_USERS").strip().lower() in {"true", "1", "yes"}:
        return True
    user_set = {str(uid).strip() for uid in (allowed_user_ids or set()) if str(uid).strip()}
    global_allowed = {
        uid.strip()
        for uid in _scoped_gate_env("GATEWAY_ALLOWED_USERS").split(",")
        if uid.strip()
    }
    user_set.update(global_allowed)
    role_set = set(allowed_role_ids or set())
    has_users = bool(user_set)
    has_roles = bool(role_set)
    try:
        uid = str(user.id)
    except AttributeError:
        uid = ""
    if has_users:
        if ("*" in user_set or (uid and uid in user_set)) and (
                live_auth is None or live_auth(interaction) is not False):
            return True
    if has_roles:
        roles_attr = getattr(user, "roles", None)
        if roles_attr is None:
            # Role policy configured but no role data (DM Member, raw User): fail closed.
            return False
        try:
            user_role_ids = {getattr(r, "id", None) for r in roles_attr}
        except TypeError:
            return False
        if user_role_ids & role_set:
            return True
    # Pairing store (mirrors ``authz_mixin._check_authorization``): paired users click without allowlist.
    if uid:
        try:
            from gateway.pairing import PairingStore
            store = PairingStore()
            if store.is_approved("discord", uid):
                return True
        except Exception:
            pass
    return False
