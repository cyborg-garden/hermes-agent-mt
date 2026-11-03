---
title: Letta-Backed Agent Mode (experimental)
sidebar_label: Letta Brain Mode
---

# Letta-Backed Agent Mode

Hermes can hand an entire conversational turn to a [Letta](https://github.com/letta-ai/letta) agent instead of running its own native agent loop. The Hermes gateway stays the **door** — it owns the platform adapters, pairing and authorization, group approval, streaming delivery, and transcript persistence — but the thinking for that turn happens inside a Letta agent you point it at over REST.

This is the "Hermes-front bridge": one messaging front end, a different runtime behind it.

:::caution Experimental — read the [budget accounting caveat](#budget-and-usage-accounting-the-important-caveat) before you rely on this
This is a productionized prototype, not a mature runtime. It is opt-in, off by default, and there are real gaps — most importantly, **token usage is reported as `0` on the default streaming path**, so per-turn usage accounting is only trustworthy when streaming is turned off. Details below.
:::

**Source files:** `gateway/letta_brain.py`, `gateway/run.py` (`_get_letta_brain`, `_run_agent_via_letta`, dispatch in `_run_agent_inner`)
**Tests:** `tests/gateway/test_letta_brain.py`, `tests/gateway/test_letta_audit_parity.py`

---

## When you'd want this

- You already have a Letta agent with memory blocks, tools, and history you care about, and you want to talk to it from Telegram / Discord / Slack / Signal / Matrix without rebuilding any of that in Hermes.
- You want Hermes' door policy — pairing, allowlists, group approval, audit transcripts — in front of a runtime that isn't Hermes.
- You want two heterogeneous runtimes reachable through one messaging surface (native Hermes on one gateway, a Letta brain on another).

If you just want a different *model*, you don't want this — use `/model` or [provider routing](provider-routing.md). This mode replaces the whole agent loop.

---

## Enabling it

Letta brain mode activates as soon as **both** a base URL and an agent ID are configured. There is no separate on/off flag.

### Via environment variables

```bash
LETTA_BRAIN_URL="http://localhost:8283"          # required
LETTA_BRAIN_AGENT_ID="agent-0000-1111-2222"      # required
LETTA_BRAIN_API_KEY="..."                        # optional
```

### Via `config.yaml`

```yaml
gateway:
  letta_brain:
    base_url: "http://localhost:8283"
    agent_id: "agent-0000-1111-2222"
    api_key: ""          # omit or leave empty for an unsecured self-hosted server
```

Environment variables are checked first (convenient for Docker), then `config.yaml`. If `LETTA_BRAIN_URL` and `LETTA_BRAIN_AGENT_ID` are both set in the environment, the `config.yaml` block is not consulted at all — but `LETTA_BRAIN_API_KEY` is still used as a fallback when the config block omits `api_key`.

### Environment variable reference

| Variable | Required | Description |
|----------|----------|-------------|
| `LETTA_BRAIN_URL` | yes | Base URL of the Letta server (e.g. `http://localhost:8283`). Trailing slashes are stripped. Also `gateway.letta_brain.base_url`. |
| `LETTA_BRAIN_AGENT_ID` | yes | ID of the Letta agent this gateway is bound to. Also `gateway.letta_brain.agent_id`. |
| `LETTA_BRAIN_API_KEY` | no | Sent as `Authorization: Bearer …`. Needed for Letta Cloud or a secured self-hosted server; leave unset for an unsecured local server. Also `gateway.letta_brain.api_key`. |
| `LETTA_BRAIN_STREAMING` | no | Set to `off`, `0`, `false`, or `no` to force the non-streaming (blocking) path. **Anything else — including unset — leaves streaming enabled.** See the caveat below. |

:::warning Partial configuration fails silently to the native loop
Both the URL and the agent ID are required. If **either** is missing or misspelled, `_get_letta_brain()` returns `None`, the dispatch branch in `_run_agent_inner` is skipped entirely, and the turn proceeds to proxy mode or the **native Hermes loop** — with no warning, no log line, and a perfectly normal-looking reply.

So a typo in `LETTA_BRAIN_AGENT_ID` does not surface an error; it surfaces native Hermes wearing the same door. That is the more dangerous failure mode, because none of the caveats on this page apply to that turn while the native toolset — which you may not have intended to expose — does. Confirm the binding is live (check the `letta brain response: url=… agent=…` log line on the first turn) before assuming traffic reaches Letta.

The string `⚠️ Letta brain not configured (LETTA_BRAIN_URL + LETTA_BRAIN_AGENT_ID)` does exist as a defensive guard at the top of `_run_agent_via_letta`, but because dispatch is gated on the *same* `_get_letta_brain()` check, it is effectively unreachable through the normal gateway path.
:::

---

## Where the bridge sits in the pipeline

The dispatch is a single branch at the top of `_run_agent_inner` in `gateway/run.py`:

```
inbound platform event
  → adapter (Telegram / Discord / Slack / Signal / Matrix / …)
  → _is_user_authorized  (pairing, allowlists, group membership)
  → _handle_message_with_agent
       → inbound preprocessing (attachments, vision, timestamps)
       → agent:start hook
       → _run_agent → _run_agent_inner
            ├── Letta brain bound?  → _run_agent_via_letta      ← this feature
            ├── GATEWAY_PROXY_URL?  → _run_agent_via_proxy
            └── otherwise           → native AIAgent loop
       → reply delivery, transcript persistence, agent:end hook
```

Two consequences worth internalizing:

1. **Everything upstream of `_run_agent` still runs.** Authorization, pairing, and group approval are enforced before the message ever reaches the bridge — a Letta-brained gateway is not an authorization bypass.
2. **Letta brain mode is checked before proxy mode.** If both `LETTA_BRAIN_URL`/`LETTA_BRAIN_AGENT_ID` and `GATEWAY_PROXY_URL` are configured, the Letta binding wins and proxy mode never runs.

Downstream, the bridge returns the same result-dict shape proxy mode returns, so reply delivery, streaming finalization, and session bookkeeping are unchanged.

---

## What reaches the Letta agent (and what doesn't)

Letta owns its own conversation state server-side, so the bridge sends **only the new message** — no history replay. Replaying Hermes' transcript would duplicate context the Letta agent already has.

### Door context: the sender tag

A shared Letta agent serving many people in many groups sees only message text over REST — none of the rich per-turn context the native loop assembles. Without help it cannot tell who is speaking or where. So the bridge prepends a one-line structured tag (`build_sender_tag` / `apply_sender_tag` in `gateway/letta_brain.py`):

```
[from Alice in #family]
what did we decide about the trip?
```

- **DMs** get the sender only: `[from Alice]`
- **Groups, channels, and threads** also get a group label — the human-readable `chat_name` when the platform provides one, otherwise `{chat_type}:{chat_id}` (e.g. `group:-100999`)
- If neither sender nor group is known, tagging is a no-op and the raw text is sent

The tag is **current-turn identity only** — deliberately not a group transcript. Letta's own history then accumulates identity naturally, turn over turn.

### Not forwarded

The bridge passes the message text and the sender tag. It does **not** forward:

- Hermes' assembled context prompt, channel prompt, personality, `MEMORY.md` / `USER.md` / `SOUL.md`, context files, or skills
- The Hermes toolset — the Letta agent uses whatever tools are configured *inside Letta*
- Mixture-of-Agents config

Whatever persona, memory, and tools you want the brain to have must be configured on the Letta agent itself. `/personality`, `/skills`, and memory features on the Hermes side do not shape a Letta-brained turn.

---

## Streaming

When the platform supports streaming, the reply streams from Letta's SSE endpoint (`POST /v1/agents/{id}/messages/stream` with `stream_tokens: true`) through the **same `GatewayStreamConsumer`** proxy mode uses — so Telegram typing-pause and fresh-final behavior, Matrix buffer-only mode, and edit-capability detection all work identically.

Streaming is attempted unless:

- `LETTA_BRAIN_STREAMING` is explicitly `off` / `0` / `false` / `no`, **or**
- streaming is disabled for that platform in your display config (`display.<platform>.streaming`) or globally (`streaming.enabled` / `streaming.transport: off`), **or**
- `aiohttp` is not installed, or stream-consumer setup fails

Fallback behavior is deliberately conservative, because Letta owns server-side history and a blind retry would make the brain process the same turn twice:

| Streaming outcome | What happens |
|---|---|
| Stream never connected, or the route returned 404/405, or `aiohttp` missing | Falls back to the blocking client (`send_message`). Provably pre-delivery, so a retry is safe. |
| Stream broke *after* deltas already reached the platform | The partial text is delivered as the final reply. No retry. |
| Any other stream failure (5xx, mid-stream drop with no text) | The turn fails with a `⚠️ …` message. No retry. |

The blocking client is stdlib-only (`urllib`) on purpose — `aiohttp` is an optional dependency for the gateway, so the fallback path must not require it. It is bridged onto the event loop with `asyncio.to_thread`.

The blocking endpoint was validated live against a self-hosted Letta on `:8283`. The SSE surface is less battle-tested; `parse_stream_line` is intentionally forgiving and skips anything it doesn't recognize rather than failing the turn.

---

## Budget and usage accounting: the important caveat

:::danger Streamed Letta turns report 0 tokens
Streaming is **on by default**. The streaming client never captures Letta's `usage_statistics` event, so every streamed brain turn records `prompt_tokens = 0` and `completion_tokens = 0`. Token accounting is only correct — and only tested — with `LETTA_BRAIN_STREAMING=off`. Cost/credits accounting does not happen for brain turns on either path.
:::

**Read this before repeating "a Letta-brained door inherits Hermes' whole policy stack."** Most of it does. Usage accounting does not, on the default path.

### Streamed turns report zero tokens

`stream_message` parses **only assistant-text deltas** off the SSE feed. Letta's terminal `usage_statistics` event is not captured, so a streamed turn reports `prompt_tokens = 0` and `completion_tokens = 0`. `_run_agent_via_letta` then records `last_prompt_tokens: 0` for that turn.

Because **streaming is on unless you explicitly disable it**, this is the *default* behavior, not an edge case. What that breaks:

- Context-window tracking (`session_entry.last_prompt_tokens`) reads 0, so compression decisions and `/usage`-style context reporting are wrong for streamed brain turns.
- The runtime footer (`display.runtime_footer`) shows 0 context tokens.

Only the **blocking** path returns real counts, pulled from the Letta payload's `usage` block (`LettaUsageStatistics`). The audit-parity tests that assert real token recording run with `LETTA_BRAIN_STREAMING=off` — which is exactly the scope of the guarantee: **token accounting is verified on the non-streaming path only.**

If per-turn usage numbers matter to you, run the brain non-streamed:

```bash
LETTA_BRAIN_STREAMING=off
```

You lose incremental delivery (the reply arrives as one message) and gain trustworthy token counts.

### Cost/credits accounting doesn't happen at all

Independently of streaming: Hermes' credits tracker lives in the native agent loop (`run_agent.py`, which parses provider credit headers). A Letta-brained turn never enters that loop, so **no credits or spend accounting happens for brain turns on the Hermes side, ever.** The same is true of `tool_calls.log` — the Letta server executes its own tools and Hermes has no visibility into them. Both are explicit non-goals of the current implementation, not bugs to be worked around; billing for the brain's own model calls is the Letta server's business.

So the honest summary: **the door's authorization and audit story holds; its cost-control story does not.** If you need spend caps on a Letta-brained gateway, enforce them at the Letta server or the upstream provider.

---

## Audit and persistence

The Letta bridge never writes to the agent's own SessionDB, so it returns `agent_persisted: False` — the documented opt-in that tells the gateway's persistence block to write the transcript rows itself. Without it, on a gateway with a live `_session_db` every brain turn's DB write would silently become a no-op.

What is persisted for a Letta-brained turn:

- The user message and the assistant reply, durably, in `state.db`
- The first-turn `session_meta` row
- `last_prompt_tokens` — real on the blocking path, `0` when streamed (see above)
- The user message is persisted even when the Letta round-trip **fails**, so a failed turn still leaves an audit trail

What is not:

- Tool-call records (Letta executes tools; Hermes cannot see them)
- Credits/cost accounting

`tests/gateway/test_letta_audit_parity.py` runs the real delegation path — `_run_agent` is not mocked — against a fake `urlopen`, and asserts each of the persistence properties above.

---

## Concurrency

Letta processes one agent's messages sequentially. Concurrent group messages therefore **queue on a per-agent lock** (keyed `base_url|agent_id`) rather than overlapping in flight.

Because a turn can sit in that queue while a newer message supersedes it, the bridge re-checks run generation at three points — after acquiring the lock, mid-stream, and post-turn — and discards a stale turn with an empty reply instead of delivering an out-of-date answer. This is the same staleness contract proxy mode uses.

Note the limit of that guarantee: superseding a turn suppresses *delivery* on the Hermes side. It does not cancel work already in flight on the Letta server — the brain finishes the turn and writes it to its own history regardless. Interrupt-by-new-message therefore looks right to the user but is not a true cancellation.

---

## Behavior details worth knowing

- **Tool-only turns produce silence, not an error.** If the Letta agent runs tools and chooses not to speak, the response has no `assistant_message`; the bridge returns an empty reply (delivery is suppressed downstream) and logs the message types it did see.
- **Failures surface as user-visible text.** `LettaBrainError` messages are written to be readable — e.g. `⚠️ Letta brain unreachable at http://localhost:8283 (is the Letta server running?)`.
- **Default timeout is 120s** (`DEFAULT_TIMEOUT_SECONDS`), applied as a socket-read timeout with no overall cap on the streaming path. It is not currently configurable — the gateway call sites don't pass a `timeout` argument.
- **The messages URL has no trailing slash** on purpose: `/messages/` 307-redirects and the POST body is dropped on the redirect.
- **Every turn reports `api_calls: 1` and an empty `tools` list** to the gateway, regardless of what the brain actually did internally.

---

## Known limitations

A candid list of what this mode does not do yet:

- Streamed turns report zero token usage (see [above](#budget-and-usage-accounting-the-important-caveat)).
- No credits/spend accounting and no tool-call audit for brain turns.
- One gateway binds to exactly **one** Letta agent. There is no per-platform, per-group, or per-profile agent routing.
- Partial configuration (only one of URL / agent ID) silently falls through to the native loop instead of erroring — see [above](#environment-variable-reference).
- Hermes personality, memory, context files, skills, and toolset do not reach the brain.
- The brain is identified to Letta only by an inline sender tag — there is no per-sender isolation on the Letta side. Everyone in a bound group shares one agent and one history.
- The SSE path has unit coverage for line parsing only; there is no end-to-end streaming test. The blocking path is the live-validated one.
- Superseding a turn suppresses delivery but does not cancel the Letta server's in-flight work.
- The request timeout (120s) is not configurable.

---

## See also

- [Codex App-Server Runtime](codex-app-server-runtime.md) — the other alternate-runtime integration
- [Proxy mode](../messaging/matrix.md#proxy-mode-e2ee-on-macos) — delegating to a remote Hermes API server instead
- [Security](../security.md) — pairing, group approval, command approval
- [Environment Variables](../../reference/environment-variables.md)
