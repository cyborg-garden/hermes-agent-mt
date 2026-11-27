# intelligent_routing

Sends cheap-safe work to a cheaper model and keeps everything else on the agent's primary model.
It routes by what a task is, not by how long its message is, for the main chat turn and for each subagent.
It is off by default, because it changes which model answers.

It has four parts:

1. **Main-turn routing** (`pre_llm_call` hook, `registration.py` + `task_router.py`).
2. **Subagent routing**, a delegation backend named `task-routing` (`delegation.py`).
3. **Escalation**: a cheap turn or child that errors, comes back empty or refuses is retried on the primary (`agent/routing_override.py`).
4. **An optional LLM router** for turns the heuristic cannot decide (`llm_router.py`).

Background: upstream PR #43534 proposed task-based model routing as a skill. The maintainer asked for a plugin instead and offered to extend the plugin interface. This directory is that plugin; `agent/routing_override.py` is the extension. See [references/prior-art.md](references/prior-art.md).

## Enable

```bash
hermes plugins enable intelligent_routing
```

Per agent, in `config.yaml`:

```yaml
routing:
  intelligent: true                             # main-turn routing; default false
  mode: heuristic                               # heuristic (default) | llm-router | legacy
  cheap_model: deepseek/deepseek-v4-flash-0731  # default; always the dated snapshot
  cheap_provider: openrouter                    # default
  escalate: true                                # default

delegation:
  backend: task-routing                         # subagent routing; independent of routing.intelligent
```

With the plugin enabled and both settings off, nothing changes.

## What goes cheap

`task_router.classify_task` is premium unless both of these hold:

- **Cheap evidence** in the instruction: a read-only lookup (list, show, grep, check, fetch, status, how many), formatting or extraction of supplied text, a summary for the person asking, a board toggle or reminder, running the tests, or gathering raw material for a parent agent.
- **No premium signal** anywhere in the text, context included: code, design or diagnosis, judgment or advice, research or explanation, writing other people read, creative work, sensitive subjects (money, law, health, credentials, personal matters, te reo Māori and translation), side effects (delete, send, restart, edit, install, buy), a later step that needs more than a lookup ("list the cards and rank them"), or a follow-up that leans on earlier turns ("yes", "same for the others").

Protected content, attachments and public-facing turns are always premium.
Length is not a signal: a long cron prompt that only checks feeds goes cheap, and a short "should we ship?" stays premium.
The cron hint, run-time line, sender prefix, script-output block and job notepad are stripped before classifying.

```python
from plugins.intelligent_routing.registration import probe_route
probe_route("grep for TODO in the repo")                  # {'tier': 'cheap', 'classification': 'lookup', ...}
probe_route("grep the logs and tell me why it crashed")   # {'tier': 'premium', ...}
```

## Subagents

With `delegation.backend: task-routing`, each `delegate_task` child is classified from its goal and context:

- Cheap-safe child: runs on `routing.cheap_model`, labelled `cheap · <type>`, with `escalate: true` and a fallback chain that starts with the profile's primary (`model.default`, or `delegation.model` when pinned), followed by the usual chain.
- Anything else: the default route, exactly as `delegation.backend: default` would give it.

The backend never calls an LLM. The goal is written by the parent model and is explicit enough for the heuristic.

Leave `routing.cheap_reasoning_effort` unset for `z-ai/glm-5.3-flash`: that endpoint returns HTTP 400 when reasoning is turned off. The same applies to `routing.cheap_extra_body`: never put `reasoning_effort: none` or `reasoning.enabled: false` there for glm.

Delegated children and review forks are not re-routed by the main-turn hook; they are routed once, where they are spawned.

## Escalation

A cheap route carries `escalate: true`. During that turn:

- An API error: `try_activate_fallback` switches back to the pre-turn route before trying any fallback-chain entry. This includes errors that normally end the turn (the model's own malformed tool-call JSON), a content-filter stall, a 429 the credential pool might otherwise wait out, and an auth failure, even when the agent has no fallback chain.
- An empty reply: escalates at once, without spending empty-retries on the cheap model.
- A short refusal ("I can't help with that", "As an AI..."): the refusal is dropped and the request re-sent on the primary.

- Protected content arriving mid-turn (a tool result carrying a protected marker): before the next API call, the turn moves to the primary, so the cheap model never receives it.

It fires at most once per turn and logs `routing_escalation {json}` with the trigger and models. Cheap subagents do the same through their fallback chain, whose first entry is the primary.

User directives ("use fable") carry no `escalate` flag and are not escalated away from.

## LLM router (`mode: llm-router`)

Asked only when the heuristic has no signal either way. A confident heuristic decision never makes the call.

- Sends the current instruction only: no history, no system prompt, sender prefix and scaffolding stripped, cut to `router_max_chars` (800).
- Never logs the text sent or the reply.
- Skipped on protected content and on platforms in `router_exclude_platforms`.
- Skipped on interactive turns unless `router_on_interactive: true`, so a person waiting never pays for the extra round trip.
- Wall-clock timeout `router_timeout_s` (2.5 s), enforced by joining a worker thread.
- Cheap only for a cheap type at `router_min_confidence` (0.85) or higher. Timeout, error, unparseable reply or anything else means premium.

```yaml
routing:
  mode: llm-router
  router_model: deepseek/deepseek-v4-flash-0731   # default: cheap_model
  router_provider: openrouter                     # default: cheap_provider
  router_timeout_s: 2.5
  router_on_interactive: false
  router_exclude_platforms: []
```

## Protected content

The papareo_transcribe plugin does not mark turns as protected, so this plugin looks for the text itself. Any turn whose message or history contains the Papa Reo transcript footer ("transcribed by Papa Reo", "Kaitiakitanga License") stays on the primary and never reaches the LLM router. History counts because a cheap turn sends the whole conversation to the cheap provider. A subagent whose goal or context carries a marker stays on the primary. If a marker arrives later in a cheap turn or child, through a tool result, the runtime moves to the primary before the next call (see Escalation).

The router thread's records on the `openai`, `httpx` and `httpcore` loggers are dropped, so DEBUG logging cannot print the router text.

Add more markers per agent:

```yaml
routing:
  protected_markers: ["do-not-route"]
```

## Metrics

One `routing_decision {json}` line per decision, at INFO. Fields: `tier`, `model`, `provider`, `task_type`, `why`, `reason`, `mode`, `scope` (`turn` or `subagent`), `confident`, `router`, `router_outcome`, `router_type`, `router_latency_ms`, `task_index`, `platform`, `is_group`, `session_id`, `turn_id`. Never message text.

## Quality evidence

`eval/` holds three synthetic, hand-labelled sets. None of it comes from real chats.

| Set | Cases | How it was made |
|---|---|---|
| `cases.py` | 196 (127 turn, 69 subagent) | written with the router; tuned against |
| `cases_holdout.py` | 68 | written after tuning, scored once, then fixed |
| `cases_blind.py` | 100 | written by a separate agent that never saw the rules |
| `cases_audit.py` | 20 | premium-worthy tasks an independent audit found routed cheap; all fixed |

First scored run on each unseen set (premium-worthy tasks routed cheap):

- Holdout: 8 of 38 (21%). The fixes were general (later-step rule, plural forms, translation, judgment words), not per case.
- Blind: 1 of 50 (2%), a "fetch papers, then synthesise" child. Fixed by adding synthesis words.

Current numbers, all sets:

| | Premium-worthy sent cheap | Cheap-safe caught | Sent cheap |
|---|---|---|---|
| dev | 0 / 131 | 65 / 65 | 33% |
| holdout | 0 / 38 | 28 / 30 | 41% |
| blind | 0 / 50 | 29 / 50 | 29% |
| audit | 0 / 20 | (none) | 0% |
| legacy classifier, dev | 25 / 131 | 20 / 65 | 23% |

The legacy row is the length-gated classifier this replaces. It sent 25 premium-worthy dev tasks cheap and never routed a subagent cheap.

Run it: `python -m plugins.intelligent_routing.eval.run_eval`. `tests/plugins/intelligent_routing/test_task_router.py` fails the build if any set routes a premium-worthy task cheap, and checks that an always-cheap router and the legacy classifier fail the same gate.

### Heuristic or LLM router as the default

Recommendation: **heuristic**. Measured 2026-11-27 with the shipped router prompt on the 16 eval cases the heuristic cannot decide (the only ones the router ever sees):

| Router model | p50 | p90 | max | Cost per call | Premium sent cheap | Cheap recovered |
|---|---|---|---|---|---|---|
| deepseek-v4-flash-0731 | 1.6 s | 3.1 s | 3.4 s | $0.00006 | 0 / 14 | 2 / 2 |
| glm-5.3-flash | 1.6 s | 3.5 s | 13 s | $0.00005 | 0 / 14 | 2 / 2 |

The router was accurate on that small sample, and each call costs almost nothing. But only 16 of 364 cases (4%) reach it, so at best it moves about 0.5% more traffic to the cheap tier, and it adds over a second and a half to each of those turns. At the 2.5 s timeout, 2 of 16 deepseek calls and 4 of 16 glm calls would have timed out and fallen back to premium anyway. It is worth turning on only for background-heavy agents.

## The interface change

`agent/routing_override.py` and a few call sites are the core changes:

- `extract_routing_override` / `apply_routing_override`: a `pre_llm_call` result may carry `{"route": {"model": ..., "provider": ..., "extra_body": ..., "escalate": true}}`. The swap uses `switch_model` and is scoped to one turn with a dedicated flag; `restore_primary_runtime` reverts it at the start of the next turn, before the fallback and cooldown checks.
- `escalate_routed_turn`, called first by `try_activate_fallback`, and by the empty-response ladder and the final-text check.
- `delegation_provider.ROUTE_KEYS` gains `escalate`; `delegate_tool._build_children` arms it on the child.

Any `pre_llm_call` plugin or delegation backend can use these, not only this one.

## Local model as the cheap tier

Turn thinking off for that model with the per-model reasoning setting. `switch_model` re-reads it for the routed model:

```yaml
routing:
  cheap_model: "qwen3.5:9b"
  cheap_provider: custom
agent:
  reasoning_overrides:
    "qwen3.5:9b": none
```

On ollama's OpenAI-compatible endpoint, `think: false` and `/no_think` have no effect (ollama#14820). Without the override a thinking model at a small `max_tokens` can return an empty reply; escalation then retries on the primary.

`routing.cheap_extra_body` is for provider-specific body fields such as `{"options": {"num_ctx": 8192}}`. The SDK merges it last, one level deep, so it overrides typed parameters of the same name.

## Model directives

A user can name a model; the directive runs before the classifier and lasts one turn.

| Directive | Routes to |
|---|---|
| `use kimi` / `use kimi3` / `use k3` | `moonshotai/kimi-k3` / openrouter |
| `use glm` | `z-ai/glm-5.2` / openrouter |
| `use deepseek` | `deepseek/deepseek-v4-flash-0731` / openrouter |
| `use fable` | `claude-fable-5` / anthropic |
| `use opus` | `claude-opus-4-8` / anthropic |
| `use sonnet` | `claude-sonnet-4-6` / anthropic |
| `use haiku` | `claude-haiku-4-5-20251001` / anthropic |
| `use claude` | no override; stays on the primary |

## Not covered

- Token spend measured before and after on a live agent.
- Tool-call history as a signal.
- A refusal that was already streamed to the chat is still shown before the primary's answer.
- `mode: legacy` keeps the old classifier (`classifier.py`) for comparison only.
