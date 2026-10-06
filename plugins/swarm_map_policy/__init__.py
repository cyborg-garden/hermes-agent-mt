"""Swarm Map Policy Plugin — HSM-backed group access control.

Integrates with Swarm Map (formerly HSM) to enforce:
- Group allowlists: only groups registered in HSM can interact
- Admin checks: platform admin status from HSM settings
- Session context caching: platform/chat_id/user_id/is_admin from gateway events
- Admin-gated tools: tools named in SWARM_MAP_ADMIN_GATED_TOOLS are blocked
  for anyone HSM does not report as a platform admin

This plugin does NOT gate dangerous-command approval. Approval is the
gateway approval queue (/approve, Discord buttons), not a tool call, so a
pre_tool_call hook never sees it. That gate lives in gateway/run.py
(approvals.admin_only + allow_admin_from) and the Discord adapter
(platforms.discord.extra.require_admin_for_exec_approval).

Configuration via environment variables:
- HSM_URL: URL of the HSM API (e.g., http://localhost:3002)
- HERMES_AGENT_NAME: Agent identifier in HSM (e.g., hermes-personal)
- SWARM_MAP_ADMIN_GATED_TOOLS: comma-separated registered tool names that
  only HSM platform admins may call (default: none)

Security model:
- Group checks: FAIL-CLOSED (deny if HSM unreachable)
- Admin checks: FAIL-CLOSED (deny if HSM unreachable)
- Admin-gated tools: FAIL-CLOSED (deny if no session context)
- Tool checks: FAIL-OPEN (allow if not configured)
"""

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    requests = None

# Thread-local storage for session context (plugin hooks are synchronous)
_session_ctx = threading.local()

# Tools that always require admin privileges. Empty by default: which tools
# are admin-only is per-deployment policy, set with SWARM_MAP_ADMIN_GATED_TOOLS.
# Every name here must be a registered tool (tests enforce it); the previous
# {"approval", "pr_approval"} matched no tool, so the gate never fired.
ADMIN_GATED_TOOLS: frozenset = frozenset()

_ADMIN_GATED_TOOLS_ENV = "SWARM_MAP_ADMIN_GATED_TOOLS"


def admin_gated_tools() -> frozenset:
    """Built-in gated tools plus any named in SWARM_MAP_ADMIN_GATED_TOOLS."""
    raw = os.environ.get(_ADMIN_GATED_TOOLS_ENV, "")
    extra = {t.strip() for t in raw.split(",") if t.strip()}
    return ADMIN_GATED_TOOLS | extra


def unknown_gated_tools(gated, known_tool_names) -> list:
    """Gated names that match no registered tool (a gate that can never fire)."""
    known = set(known_tool_names)
    return sorted(t for t in gated if t not in known)


def _hsm_url() -> Optional[str]:
    """Get HSM API URL from environment."""
    return os.environ.get("HSM_URL") or None


def _harness_id() -> Optional[str]:
    """Get this agent's harness ID from environment."""
    return os.environ.get("HERMES_AGENT_NAME") or None


def get_session_context() -> Optional[dict]:
    """Get cached session context dict, or None if not set."""
    if not hasattr(_session_ctx, "platform"):
        return None
    return {
        "platform": _session_ctx.platform,
        "chat_id": _session_ctx.chat_id,
        "user_id": _session_ctx.user_id,
        "is_admin": getattr(_session_ctx, "is_admin", False),
    }


def clear_session_context() -> None:
    """Clear cached session context."""
    for attr in ("platform", "chat_id", "user_id", "is_admin"):
        if hasattr(_session_ctx, attr):
            delattr(_session_ctx, attr)


def is_group_allowed(group_id: str, platform: str) -> bool:
    """Check if a group is in the HSM allowlist. Fail-closed."""
    url = _hsm_url()
    harness = _harness_id()
    if not url or not harness:
        logger.warning("swarm-map-policy: HSM not configured, denying group")
        return False
    try:
        resp = requests.get(
            f"{url}/api/harnesses/{harness}/surfaces/{platform}/groups/{group_id}",
            timeout=5,
        )
        return resp.status_code == 200 and resp.json().get("allowed", False)
    except Exception as e:
        logger.warning("swarm-map-policy: HSM check failed (fail-closed): %s", e)
        return False


def approve_group_add(
    group_id: str, added_by_user_id: str, platform: str = "telegram"
) -> bool:
    """Request HSM auto-approval for a group the bot was just added to.

    Called when someone adds the bot to a new group. HSM verifies that the
    adder is a platform admin and, if so, adds the group to the allowlist.
    Fail-closed: returns True only on HTTP 200 with ``approved: true`` —
    network errors, non-200 responses, and missing fields all deny.
    """
    url = _hsm_url()
    harness = _harness_id()
    if not url or not harness:
        logger.warning("swarm-map-policy: HSM not configured, denying group add")
        return False
    try:
        resp = requests.post(
            f"{url}/api/harnesses/{harness}/surfaces/{platform}/groups/{group_id}",
            json={"addedByUserId": added_by_user_id},
            timeout=5,
        )
        if resp.status_code != 200:
            reason = ""
            try:
                reason = resp.json().get("error", "")
            except Exception:
                pass
            logger.info(
                "swarm-map-policy: group add denied for %s (HTTP %s%s)",
                group_id, resp.status_code, f": {reason}" if reason else "",
            )
            return False
        data = resp.json()
        if data.get("approved") is True:
            logger.info(
                "swarm-map-policy: group add approved for %s (already_allowed=%s restarted=%s)",
                group_id, data.get("already_allowed", False), data.get("restarted"),
            )
            return True
        logger.info(
            "swarm-map-policy: group add not approved for %s: %s",
            group_id, data.get("reason", "no reason given"),
        )
        return False
    except Exception as e:
        logger.warning(
            "swarm-map-policy: group add approval failed (fail-closed): %s", e
        )
        return False


def is_tool_allowed(tool_name: str, group_id: str) -> bool:
    """Check if a tool is allowed for a group. Fail-open."""
    url = _hsm_url()
    if not url:
        return True
    return True  # Future: check HSM tool gating API


def is_platform_admin(user_id: str, platform: str) -> bool:
    """Check if a user is a platform admin via HSM. Fail-closed."""
    url = _hsm_url()
    harness = _harness_id()
    if not url or not harness:
        return False
    try:
        resp = requests.get(
            f"{url}/api/harnesses/{harness}/surfaces/{platform}/admins/{user_id}",
            timeout=5,
        )
        return resp.status_code == 200 and resp.json().get("is_admin", False)
    except Exception:
        return False


def _pre_gateway_dispatch(event=None, **kwargs):
    """Cache session context from incoming message event and resolve admin status."""
    if event is None:
        return None
    source = event.source
    _session_ctx.platform = source.platform.value if source.platform else ""
    _session_ctx.chat_id = source.chat_id or ""
    _session_ctx.user_id = source.user_id or ""
    # Resolve admin status from HSM (fail-closed)
    _session_ctx.is_admin = False
    try:
        user_id = _session_ctx.user_id
        platform = _session_ctx.platform
        if user_id and platform:
            _session_ctx.is_admin = is_platform_admin(user_id, platform)
    except Exception as e:
        logger.warning("swarm-map-policy: admin resolution failed (fail-closed): %s", e)
        _session_ctx.is_admin = False
    return None  # Allow normal dispatch


def _on_session_start(session_id: str = None, **kwargs) -> None:
    """Log session start."""
    logger.debug("swarm-map-policy: session start %s", session_id)


def _pre_tool_call(tool_name: str = None, **kwargs):
    """Gate tool calls based on HSM policy. Returns None to allow, dict to block."""
    if tool_name and tool_name in admin_gated_tools():
        ctx = get_session_context()
        if not ctx or not ctx.get("is_admin"):
            return {
                "action": "block",
                "message": f"Admin privileges required to use {tool_name}.",
            }
    return None


def _warn_unknown_gated_tools() -> None:
    """Log gated names that match no registered tool, so a typo is visible."""
    gated = admin_gated_tools()
    if not gated:
        return
    try:
        # At gateway startup plugins are discovered before model_tools imports
        # the built-ins, so the registry holds only plugin tools. Load the
        # built-ins first (idempotent) or every real name looks unknown.
        from tools.registry import discover_builtin_tools, registry
        discover_builtin_tools()
        known = registry.get_all_tool_names()
    except Exception:
        return
    if not known:
        return  # nothing to compare against
    unknown = unknown_gated_tools(gated, known)
    if unknown:
        logger.warning(
            "swarm-map-policy: %s names tools that match no built-in or plugin "
            "tool, so they are not gated: %s (MCP tools register later and "
            "are not checked here)", _ADMIN_GATED_TOOLS_ENV, ", ".join(unknown),
        )


def register(ctx):
    """Register plugin hooks."""
    if not requests:
        logger.warning("swarm-map-policy: 'requests' not installed, plugin disabled")
        return
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("pre_gateway_dispatch", _pre_gateway_dispatch)
    _warn_unknown_gated_tools()
    logger.info(
        "swarm-map-policy: registered (HSM_URL=%s, admin-gated tools=%s)",
        _hsm_url() or "not set", ",".join(sorted(admin_gated_tools())) or "none",
    )
