"""Discord env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from functools import partial

from gateway.config import GatewayConfig, Platform
from gateway.platform_env import EnvCred, getenv, home_step, reply_mode_step


def _env_approver_roles(config: GatewayConfig, platform: Platform, env: str) -> None:
    """mt (#175): approver roles (role IDs or names) for dangerous-command approval.

    config.yaml wins when both are set.
    """
    roles = (getenv(env) or "").strip()
    if roles and platform in config.platforms:
        config.platforms[platform].extra.setdefault("approver_roles", roles)

ENV_ENABLE_CREDENTIALS = ("DISCORD_BOT_TOKEN",)
# Order matters: the home channel is read BEFORE the reply mode may create the entry.
ENV_OVERRIDES = (
    EnvCred(Platform.DISCORD, ("DISCORD_BOT_TOKEN",), token="DISCORD_BOT_TOKEN"),
    home_step(Platform.DISCORD, "DISCORD_HOME_CHANNEL"),
    reply_mode_step(Platform.DISCORD, "DISCORD_REPLY_TO_MODE"),
    partial(_env_approver_roles, platform=Platform.DISCORD, env="DISCORD_APPROVER_ROLES"),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "DISCORD_ALLOWED_USERS",
    "allow_all_env": "DISCORD_ALLOW_ALL_USERS",
    "allow_bots_env": "DISCORD_ALLOW_BOTS",
    "allowlist_suppresses_pairing": True,
}
