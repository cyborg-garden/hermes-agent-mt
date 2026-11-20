# Fork patch series (2026-05 snapshot)

These 27 patches are the fork's changes as they stood in May 2026, exported with `git format-patch`.
They are **not** kept up to date. The fork now follows upstream by merging release tags, so it no longer produces a clean patch series.
For the current changes, diff this repo's `main` against the upstream release tag it last merged. [FORK-NOTICE.md](../FORK-NOTICE.md) summarises them.

## What the series covers

- **Memory scoping by `context_id` (0001, 0002).** Memory writes go to a separate folder per channel. Reads merge global and scoped entries. With `context_id=None`, behaviour is unchanged.
- **Swarm Map policy plugin (0003, 0006, 0009).** Group access control through [Swarm Map](https://github.com/cyborg-garden/swarm-map). See [its README](../plugins/swarm_map_policy/README.md).
- **Startup hooks (0007, 0008, 0010, 0021).** The `boot_md` plugin and the `lifecycle-notify` hook.
- **Mattermost (0011, 0012, 0014, 0018).** Mention gating, system posts, channel join and leave gating, and passive observation of unmentioned messages.
- **Signal (0013, 0017, 0020, 0022, 0024 to 0027).** UUID allowlisting, group invite policy, passive observation, voice memos and stream reconnects.
- **Passive observation (0015, 0016).** An `observe_only` flag on incoming messages.
- **Build and deploy (0005, 0019, 0023).** CI workflows, the overlay configs in [`docker/deploy/`](../docker/deploy/README.md), and faster-whisper in the Docker image.
- **This README (0004).**

## Applying them to upstream

They apply to upstream as it was in May 2026. Expect conflicts on a newer upstream.

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
git am /path/to/patches/*.patch
```
