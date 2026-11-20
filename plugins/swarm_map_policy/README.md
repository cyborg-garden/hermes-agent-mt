# Swarm Map Policy Plugin

Connects a Hermes agent to [Swarm Map](https://github.com/cyborg-garden/swarm-map), which decides which groups the agent may talk in and who counts as an admin.
Swarm Map was formerly called HSM, and the setting names still use that name.

## Configuration

Set these in the agent's `.env`:

```
HSM_URL=http://localhost:3002
HERMES_AGENT_NAME=my-agent
# Optional: comma-separated tool names that only Swarm Map platform admins may call
SWARM_MAP_ADMIN_GATED_TOOLS=
```

- `HSM_URL` is the Swarm Map API.
- `HERMES_AGENT_NAME` is this agent's name in Swarm Map.
- `SWARM_MAP_ADMIN_GATED_TOOLS` is empty by default, so no tool is limited to admins until you name one. At startup the plugin logs any name that matches no registered tool. MCP tools register later and are not checked.

The plugin needs the `requests` package. Without it, the plugin logs a warning and does nothing.

## What it checks

- **Groups.** When the agent is added to a Telegram group, Swarm Map must approve it. If Swarm Map isn't configured or can't be reached, the group is denied.
- **Admins.** Each incoming message's sender is looked up in Swarm Map. If the lookup fails, the sender is treated as a non-admin.
- **Admin-only tools.** A tool listed in `SWARM_MAP_ADMIN_GATED_TOOLS` is blocked for non-admins, and blocked when the plugin has no session context.

This plugin does not gate approval of dangerous commands.
Approval is not a tool call. It is gated by `approvals.admin_only`, read in `gateway/slash_commands.py`, and for Discord buttons by `platforms.discord.extra.require_admin_for_exec_approval` and `allow_admin_from`.

## Hooks used

- `pre_gateway_dispatch` stores the platform, chat and sender for the message, and looks up whether the sender is an admin.
- `pre_tool_call` blocks admin-only tools for non-admins.
- `on_session_start` logs the session start.

Group-add approval (`approve_group_add`) is called by the Telegram adapter when the bot is added to a group. `is_group_allowed` is a helper for adapters to use; nothing calls it yet.
