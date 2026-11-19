# intelligent_routing

Sends simple, mechanical turns to a cheaper model before the main chat turn runs.
Everything else stays on the agent's configured primary model.
It is off by default, because it changes which model answers a user.

It has two parts:

1. **The plugin.** It classifies each turn. If the turn is clearly mechanical, such as a cron job or a short single-action request, it routes the turn to a cheap model. Anything uncertain stays on the primary.
2. **A small change to the plugin interface** in `agent/routing_override.py`, with one call site in `agent/turn_context.py`. A `pre_llm_call` hook can now return a `route` that switches the model for that turn. Before this, `pre_llm_call` could add context but not change the model.

Background: upstream PR #43534 proposed task-based model routing as a skill. The maintainer closed it and asked for a plugin instead, offering to extend the plugin interface. This directory is that plugin and that extension. See [references/prior-art.md](references/prior-art.md).

## Enable

```bash
hermes plugins enable intelligent_routing
```

Then turn it on per agent in `config.yaml`:

```yaml
routing:
  intelligent: true                       # default false
  mode: heuristic                         # default; "llm-router" is reserved and not built
  cheap_model: deepseek/deepseek-v3.2     # default
  cheap_provider: openrouter              # default
```

With the plugin enabled but `routing.intelligent` off, the hook returns nothing and changes nothing.

### Using a local model as the cheap tier

Turn thinking off for that model.
A thinking model with a small `max_tokens` can spend the whole budget reasoning and return an empty reply. Measured: `qwen3.5:9b` at `max_tokens: 200` used all 200 tokens and returned nothing, with `finish_reason: length`.

Use the per-model reasoning setting:

```yaml
routing:
  intelligent: true
  cheap_model: "qwen3.5:9b"
  cheap_provider: custom            # aliases: ollama, local, vllm, llamacpp

agent:
  reasoning_overrides:
    "qwen3.5:9b": none              # sends reasoning_effort: "none"
```

`switch_model` reads the reasoning setting for the routed model, so this applies to cheap turns even though the primary is a different model.
With it, the same test used 2 tokens and answered correctly.

On ollama's OpenAI-compatible `/v1/chat/completions`, `think: false`, `chat_template_kwargs` and an in-prompt `/no_think` have no effect (ollama#14820). `reasoning_effort: "none"` is the only one that works.

### `routing.cheap_extra_body`

Use this only for provider-specific request body fields on a routed turn, such as ollama's `{"options": {"num_ctx": 8192}}` or an aggregator's `provider` preferences. It is empty by default.

- The OpenAI SDK merges `extra_body` into the request last, so a key here overrides a typed parameter of the same name.
- The merge is one level deep. A nested object you set replaces the assembled one. `{"reasoning": {"effort": "none"}}` drops the assembled `enabled` key, so write nested values out in full.

## How a turn flows

```
turn arrives
  └─ pre_llm_call hook (registration.py)
       ├─ user directive such as "use kimi"? → route to that model
       ├─ decide_tier(sig) (classifier.py): cheap only if classify_turn == MECHANICAL,
       │     otherwise premium
       ├─ add a note to the context: [Current model: … routed: mechanical → cheap]
       └─ if cheap: return {"route": {"model": …, "provider": …}}
             └─ turn_context applies it (agent/routing_override.py)
                  └─ agent.switch_model(...), reverted at the start of the next turn
```

To test the decision on its own:

```python
from plugins.intelligent_routing.registration import probe_route
probe_route("Write a Python function that merges two sorted linked lists.")
#   -> {"tier": "premium", "classification": "uncertain"}
probe_route("grep for TODO in the repo")
#   -> {"tier": "cheap", "classification": "mechanical"}
probe_route("run the nightly digest", platform="cron")
#   -> {"tier": "cheap", "classification": "mechanical"}
```

## The routing decision

`decide_tier` sends a turn to the cheap model only when `classify_turn` returns `MECHANICAL`.
That means a cron or background job, a kanban triage, or a short direct request for one action.
Architecture questions, code generation, open-ended reasoning, hedged requests, long messages and public-facing replies stay on the primary.

An earlier version routed on the five-way `classify_task_type`, whose default was `orchestration → cheap`.
In real use, hard architecture questions, code generation and hedged requests all fell through to that default and went to the cheap model.
The default is now premium.
`tests/plugins/intelligent_routing/test_realistic_distribution.py` runs a representative batch through the full hook to catch this.
`classify_task_type` still exists and is tested, but routing does not use it.

The cheap default is `deepseek/deepseek-v3.2`. The reasons are in [references/prior-art.md](references/prior-art.md).

## Tiers and model directives

Turns the classifier keeps on premium carry no override, so they use whatever the agent's primary model is.

A user can name a model in their message. The directive is checked before the classifier, so a bare `use kimi` is not mistaken for a mechanical request.

| Directive | Routes to |
|---|---|
| `use kimi` / `use kimi3` / `use k3` | `moonshotai/kimi-k3` / openrouter |
| `use glm` | `z-ai/glm-5.2` / openrouter |
| `use deepseek` | `deepseek/deepseek-v3.2` / openrouter |
| `use fable` | `claude-fable-5` / anthropic |
| `use opus` | `claude-opus-4-8` / anthropic |
| `use sonnet` | `claude-sonnet-4-6` / anthropic |
| `use haiku` | `claude-haiku-4-5-20251001` / anthropic |
| `use claude` | no override; stays on the primary |

Directives last for one turn, like any other route override.

## The interface change

`agent/routing_override.py`:

- `extract_routing_override(results)` takes the first well-formed `{"route": {...}}` from the `pre_llm_call` results. `model` is required.
- `apply_routing_override(agent, override)` calls the agent's existing `switch_model`, which rebuilds the client and rolls back on failure. It then sets a dedicated flag, `_routing_override_active`, and saves the primary runtime in `_routing_override_saved_primary`.
- During the routed turn, `_primary_runtime` points at the cheap runtime, and `_fallback_activated` is left alone.
- `restore_primary_runtime` undoes the switch at the start of the next turn. It does this before the fallback and rate-limit cooldown checks, so the cheap model never carries into the next turn.
- Any error leaves the agent on its primary and is never raised into the turn loop.

An earlier design reused the fallback state (`_primary_runtime` and `_fallback_activated`) instead of a dedicated flag. Review found three bugs in it:

1. The cheap model could stay on past its turn when the rate-limit cooldown skipped the restore.
2. Recovery from a transport error rebuilt the routed turn on the premium model.
3. A 429 from the cheap model did not start its cooldown.

The dedicated flag fixes all three. Reactive fallback still works inside a routed turn, with the cheap runtime acting as its primary.

The call site is in `agent/turn_context.py`, right after the `pre_llm_call` results are collected and before the first API call. Those are the only core changes; the rest is inside the plugin.

Any `pre_llm_call` plugin can use the override, not only this one.

## Status

Done:

- Per-agent settings: `routing.intelligent` (default off), `routing.mode`, `routing.cheap_model`, `routing.cheap_provider`.
- The classifier: deterministic, and premium whenever it is unsure.
- The `pre_llm_call` hook: adds the routing note and returns the override for cheap turns. Inert when disabled. Never raises.
- The interface change, scoped to one turn.
- Tests, including one that goes from decision to actual model switch.

Not done:

- Measuring token spend before and after on a real agent.
- `mode: llm-router`, using a cheap model as the router. The setting is read but does nothing.
- A toggle in the Swarm Map UI.
- Richer signals, such as tool-call history or real public-facing detection.
- An end-to-end test against a live agent calling OpenRouter. The switch is tested against fake agents and the real `restore_primary_runtime`.

## Notes for reviewers

The only core changes are `agent/routing_override.py` and the call site in `agent/turn_context.py`. They change which model answers a turn, so users can see the effect.

- Turn scoping: `tests/agent/test_routing_override_revert.py` tests against the real `restore_primary_runtime`, including the earlier leak.
- Fallback inside a routed turn: covered by `test_reactive_fallback_engages_under_a_routed_turn`.
- Failure handling: a bad or unreachable cheap provider should fall back to the primary without breaking the turn.
- Run it against a live agent before turning it on widely.
