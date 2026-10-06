# Swarm Map Policy Plugin

Integrates Hermes with [Swarm Map](https://github.com/NimbleCoOrg/swarm-map) for multi-tenant group access control.

## Configuration

Set these environment variables in your agent's `.env`:

```
HSM_URL=http://localhost:3002
HERMES_AGENT_NAME=hermes-personal
# Optional: registered tool names only HSM platform admins may call
SWARM_MAP_ADMIN_GATED_TOOLS=
```

This plugin does not gate dangerous-command approval. Approval is not a tool
call; it is gated by `approvals.admin_only` in `gateway/run.py` and, for
Discord buttons, by `platforms.discord.extra.require_admin_for_exec_approval`
+ `allow_admin_from`.

## Security Model

- **Group checks:** Fail-closed. If HSM is unreachable, group messages are denied.
- **Tool checks:** Fail-open. If HSM is not configured, all tools are allowed.

## Hooks Used

- `on_session_start` — validate group access and cache admin status
- `pre_tool_call` — block tools in `SWARM_MAP_ADMIN_GATED_TOOLS` for non-admins (fail-closed)
