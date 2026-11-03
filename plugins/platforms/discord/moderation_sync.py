"""Discord moderation → authorization propagation (R7).

When a member is banned (or, in ``remove`` mode, kicked / leaves) from a
guild, their Hermes authorization must die with the ban — otherwise a banned
user keeps DM command access through the pairing store or a stale allowlist.

Revocation removes every grant:

  1. pairing store approval + persisted ``.env`` allowlist entry
     (``gateway.pairing.revoke_platform_authorization``);
  2. the adapter's in-memory ``_allowed_user_ids`` + ``os.environ``
     mirror of ``DISCORD_ALLOWED_USERS``;
  3. an exact ``GATEWAY_ALLOWED_USERS`` entry (cross-platform var — Discord
     snowflakes are 17-19 digits so collision risk with other platforms'
     IDs is nil);
  4. a persistent, adapter-owned **deny store** at
     ``{HERMES_HOME}/platforms/discord-denied.json``.

The deny store is the backstop for a specific failure mode: the HSM console
saves its whole env document back (stale whole-document PUT, audit SB-5/#2),
which can re-write ``DISCORD_ALLOWED_USERS`` and silently re-grant a banned
user. The deny store is checked before EVERY allow branch (pairing,
allowlists, ALLOW_ALL, channel bypass, component clicks) and survives every
console save. Deny beats allow, always.

Config (read at call time):
  DISCORD_MODERATION_SYNC=ban       off | ban (default) | remove.
      "ban": bans propagate; works with default (non-privileged
             moderation) intents; zero boot risk.
      "remove": ban+kick+leave all revoke. GUILD_MEMBER_REMOVE cannot
             distinguish a kick from a voluntary leave without audit-log
             permissions, and it requires the privileged Server Members
             intent (must be enabled in the Developer Portal or the bot
             may fail to boot — see the adapter's intents warning).
  DISCORD_MODERATION_SYNC_GUILDS=   comma guild ids; empty = all guilds.
  DISCORD_MODERATION_SYNC_EXEMPT=   comma snowflakes never auto-revoked
      (seed with break-glass operator ids so a test-guild departure cannot
      strand the operators).

This module NEVER posts to Discord/Telegram — revocation is silent on the
platform and loud in the logs.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home
from utils import atomic_replace

logger = logging.getLogger(__name__)

_VALID_MODES = ("off", "ban", "remove")


def moderation_sync_mode() -> str:
    """Current sync mode: 'off' | 'ban' | 'remove' (malformed → 'ban')."""
    raw = os.getenv("DISCORD_MODERATION_SYNC", "ban").strip().lower()
    if raw not in _VALID_MODES:
        logger.warning(
            "DISCORD_MODERATION_SYNC=%r is not one of off|ban|remove — using 'ban'",
            raw,
        )
        return "ban"
    return raw


def sync_guild_ids() -> set[str]:
    """Guilds whose moderation events propagate (empty = all guilds)."""
    raw = os.getenv("DISCORD_MODERATION_SYNC_GUILDS", "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def exempt_user_ids() -> set[str]:
    """Snowflakes that are never auto-revoked (break-glass operators)."""
    raw = os.getenv("DISCORD_MODERATION_SYNC_EXEMPT", "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def _secure_write(path: Path, data: str) -> None:
    """Atomic 0600 write — same pattern as ``gateway.pairing._secure_write``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass  # Windows doesn't support chmod the same way
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class DenyStore:
    """Persistent, adapter-owned deny list for Discord user ids.

    ``{uid: {"reason": str, "at": ts, "guild_id": str}}`` at
    ``{HERMES_HOME}/platforms/discord-denied.json`` (atomic 0600 writes).

    The path is resolved per call (not cached) so per-test HERMES_HOME
    isolation works and container recreates pick up env changes.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._explicit_path = path
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        if self._explicit_path is not None:
            return self._explicit_path
        return get_hermes_home() / "platforms" / "discord-denied.json"

    def _load(self) -> dict:
        path = self.path
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict):
                raise ValueError("deny store is not a JSON object")
            return data
        except Exception as err:
            # The file is only ever written atomically by this class, so
            # corruption is a serious signal. We log loudly but treat the
            # store as empty rather than denying every user on the platform:
            # the deny store is a *backstop* on top of the allowlist prune —
            # a banned user is normally already out of every allowlist.
            logger.error(
                "Discord deny store unreadable at %s (%s) — treating as empty. "
                "Banned users pruned from allowlists stay revoked, but the "
                "console-save backstop is DOWN until this file is repaired.",
                path,
                err,
            )
            return {}

    def add(self, uid: str, reason: str, guild_id: str) -> None:
        with self._lock:
            data = self._load()
            data[str(uid)] = {
                "reason": reason,
                "at": time.time(),
                "guild_id": str(guild_id),
            }
            _secure_write(self.path, json.dumps(data, indent=2, ensure_ascii=False))

    def is_denied(self, uid) -> bool:
        if uid is None:
            return False
        with self._lock:
            return str(uid) in self._load()

    def remove(self, uid: str) -> bool:
        """Operator un-ban (CLI). Returns True if the uid was present."""
        with self._lock:
            data = self._load()
            if str(uid) not in data:
                return False
            del data[str(uid)]
            _secure_write(self.path, json.dumps(data, indent=2, ensure_ascii=False))
            return True


_store: Optional[DenyStore] = None
_store_lock = threading.Lock()


def deny_store() -> DenyStore:
    """Process-wide deny store (path resolved lazily on every operation)."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = DenyStore()
    return _store


def revoke_user_authorization(adapter, uid: str, reason: str, guild_id: str) -> None:
    """Remove every Hermes authorization grant for ``uid``. Idempotent.

    Order matters: the pairing-layer prune (step 1) reads ``os.getenv`` to
    compute the persisted ``.env`` remainder, so it must run BEFORE the
    in-process ``os.environ`` rewrite (step 2). Steps are individually
    guarded so a failure in one layer cannot skip the deny-store backstop.

    Runs blocking file IO — call via ``asyncio.to_thread`` from the event
    loop. NEVER posts to Discord/Telegram.
    """
    uid = str(uid)
    guild_id = str(guild_id)

    # 1. Pairing grant + persisted .env allowlist entry (reads os.getenv,
    #    so it runs while os.environ still holds the pre-prune list).
    try:
        from gateway.pairing import revoke_platform_authorization

        revoke_platform_authorization("discord", uid)
    except Exception:
        logger.error(
            "R7: pairing/env revocation failed for uid=%s — continuing to "
            "deny-store backstop",
            uid,
            exc_info=True,
        )

    # 2. Adapter in-memory allowlist + os.environ mirror (same rewrite the
    #    adapter's username resolution performs).
    try:
        allowed = getattr(adapter, "_allowed_user_ids", None)
        if allowed and uid in allowed:
            allowed.discard(uid)
            os.environ["DISCORD_ALLOWED_USERS"] = ",".join(sorted(allowed))
    except Exception:
        logger.error(
            "R7: in-memory allowlist prune failed for uid=%s", uid, exc_info=True
        )

    # 3. Exact match in the cross-platform GATEWAY_ALLOWED_USERS var.
    try:
        raw = os.getenv("GATEWAY_ALLOWED_USERS", "")
        ids = [part.strip() for part in raw.split(",") if part.strip()]
        if uid in ids:
            remaining = [part for part in ids if part != uid]
            logger.warning(
                "R7: pruning uid=%s from cross-platform GATEWAY_ALLOWED_USERS "
                "(Discord snowflakes are 17-19 digits; exact-match prune)",
                uid,
            )
            os.environ["GATEWAY_ALLOWED_USERS"] = ",".join(remaining)
            from hermes_cli.config import remove_env_value, save_env_value

            if remaining:
                save_env_value("GATEWAY_ALLOWED_USERS", ",".join(remaining))
            else:
                remove_env_value("GATEWAY_ALLOWED_USERS")
    except Exception:
        logger.error(
            "R7: GATEWAY_ALLOWED_USERS prune failed for uid=%s", uid, exc_info=True
        )

    # 4. Deny-store backstop — survives HSM console whole-document saves
    #    that re-write the allowlist env vars.
    deny_store().add(uid, reason, guild_id)

    # 5. Gate-shape alarm: if the pruned user allowlist is now empty and no
    #    role allowlist is configured, the adapter's auth gate falls through
    #    to the channel-bypass / ALLOW_ALL branch semantics. The deny store
    #    still blocks THIS uid, but operators must know the shape changed.
    remaining_users = getattr(adapter, "_allowed_user_ids", set()) or set()
    remaining_roles = getattr(adapter, "_allowed_role_ids", set()) or set()
    if not remaining_users and not remaining_roles:
        logger.critical(
            "R7: DISCORD_ALLOWED_USERS is now EMPTY and no roles are "
            "configured — the Discord auth gate has fallen through to "
            "channel-bypass/ALLOW_ALL branch semantics. The deny store still "
            "blocks uid=%s, but review the adapter's authorization config.",
            uid,
        )

    logger.info(
        "R7: revoked Discord authorization for uid=%s (reason=%s, guild=%s)",
        uid,
        reason,
        guild_id,
    )
