# Multi-Tenant Fork Notice

This is **hermes-agent-mt**, a fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent).
It is for running one Hermes agent across many groups and people, usually managed by [Swarm Map](https://github.com/cyborg-garden/swarm-map).
Each group or DM keeps its own memory, and group access is checked against a policy service.

## Use it

```bash
# Docker
docker pull ghcr.io/cyborg-garden/hermes-agent-mt:latest

# From source
git clone https://github.com/cyborg-garden/hermes-agent-mt.git
cd hermes-agent-mt
pip install -e ".[all]"
```

To manage several agents, use [Swarm Map](https://github.com/cyborg-garden/swarm-map).

## What differs from upstream

### Core changes

- **Memory scoped per conversation.** Memory writes go to a separate store for each group or DM, keyed by a `context_id`. Reads merge the global entries with that conversation's entries. With no `context_id`, behaviour matches upstream.
- **Safe context IDs.** Platform chat IDs are normalized before they become file paths.
- **Letta-backed agent mode.** The gateway can hand each turn to a [Letta](https://github.com/letta-ai/letta) agent. See [Letta Brain Mode](website/docs/user-guide/features/letta-brain-mode.md).
- **Per-turn model override for plugins.** A `pre_llm_call` hook can return a `route` that switches the model for one turn (`agent/routing_override.py`). The `intelligent_routing` plugin uses it.

### Platform adapters

- **Signal:** allowlisting by UUID, a group invite policy, voice memo detection, and setting the profile name.
- **Mattermost:** channel join and leave gating, a per-channel allowlist, and mention gating.
- **Telegram:** separate sessions per group, and admin lookup.

### Plugins and hooks

| Name | What it does | Docs |
|------|--------------|------|
| `swarm_map_policy` | Checks group access and admin status against Swarm Map. Can limit chosen tools to admins. | [README](plugins/swarm_map_policy/README.md) |
| `intelligent_routing` | Sends simple, mechanical turns to a cheaper model. Off by default. | [README](plugins/intelligent_routing/README.md) |
| `observability/home_log_router` | Forwards important warnings to the agent's home channel. | [README](plugins/observability/home_log_router/README.md) |
| `boot_md` | Runs `~/.hermes/BOOT.md` as a prompt at the first session after startup. | `plugins/boot_md/plugin.yaml` |
| `gardener_approvals` | Routes a Signal approval reply, with the sender's verified UUID, to an approval gate. | `plugins/gardener_approvals/plugin.yaml` |
| `lifecycle-notify` (hook) | Posts a startup notice to the home channel. | `hooks/lifecycle-notify/HOOK.yaml` |

Enable a plugin with `hermes plugins enable <name>`.

The complete list of changes is the diff between this repo's `main` and the upstream release tag it last merged.

## Keeping up with upstream

- A [weekly workflow](.github/workflows/upstream-sync.yml) runs on Mondays at 06:00 UTC.
- It **merges** the latest upstream release tag. It does not rebase. You can also run it by hand against any upstream ref.
- Conflicts in fork-owned areas go to a person. Conflicts under `apps/`, `website/`, `nix/` and `docs/` are resolved in upstream's favour automatically.
- Changes that would help upstream are offered back as PRs.
- [`docs/rebase-journal.md`](docs/rebase-journal.md) records the first rebase in 2026-05, before the switch to merging.
- [`patches/`](patches/README.md) holds a patch series from that time. It is not kept up to date.

## Container image

CI publishes to **`ghcr.io/cyborg-garden/hermes-agent-mt`**.
The repo has moved twice: from `NimbleCoAI` to `NimbleCoOrg`, then to `cyborg-garden`.
Git URLs redirect, so existing clones keep working.
Container images do not redirect:

- `ghcr.io/nimblecoorg/hermes-agent-mt` no longer exists. Pulls fail.
- `ghcr.io/nimblecoai/hermes-agent-mt` (last build 2026-08-06) and `ghcr.io/nimblecoai/hermes-agent` (last build 2026-07-21) still pull, with no error, but return old code.

Repoint anything that uses an old path.

## License

MIT, the same as upstream.
