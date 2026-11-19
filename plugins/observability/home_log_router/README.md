# home_log_router

Forwards important operational warnings to an agent's **home channel**.

## Why

Hermes keeps operational log records out of chat: model fallbacks, provider errors, repeated reconnects.
Only startup, shutdown and cron messages reach the home channel.
So the person running the agent can't see these problems from chat.

This plugin forwards a chosen, rate-limited set of those records.
It adds a `logging.Handler` to the root logger and sends through the existing `send_message` tool. A bare platform target, such as `signal`, means that platform's home channel.
It changes no code that writes logs, so it is unaffected by upstream merges.

## Enable

```
hermes plugins enable observability/home_log_router
```

It starts working straight away.
If no home channel is set (`SIGNAL_HOME_CHANNEL` or `/sethome`), it forwards nothing until one is.

## What it forwards

Records at **WARNING** or above from these loggers (matched by prefix):

- `plugins.platforms.signal`: reconnects and health. Routine churn logs at DEBUG, so only real problems show.
- `agent.conversation_loop`: model fallbacks.
- `agent.degradation`
- `model_tools`: provider errors during model calls.
- `agent.auxiliary_client`

## Storm safety

- Identical messages are sent once, then again once per dedup window.
- Sends are capped per window.
- The first send after any were held back starts with a short "N suppressed" line.
- Capturing a record never waits on sending. Records go into a bounded queue, and are dropped if it is full.

## Configuration

Enabling the plugin is usually enough. These optional overrides exist:

| Env | Default | Meaning |
|-----|---------|---------|
| `HERMES_HOME_LOG_ENABLED` | (on) | Kill switch. Any value other than `1`, `true`, `yes` or `on` turns forwarding off while the plugin stays enabled. |
| `HERMES_HOME_LOG_PLATFORM` | `signal` | Platform whose home channel receives forwards |
| `HERMES_HOME_LOG_LEVEL` | `WARNING` | Minimum level to forward |
| `HERMES_HOME_LOG_LOGGERS` | (the five above) | Comma-separated logger-name prefixes |
| `HERMES_HOME_LOG_RATE` | `20` | Max sends per window |
| `HERMES_HOME_LOG_WINDOW` | `60` | Rate window (seconds) |
| `HERMES_HOME_LOG_DEDUP_WINDOW` | `300` | Identical-message dedup window (seconds) |
| `HERMES_HOME_LOG_QUEUE` | `1000` | Bounded capture queue size |

## Design

Five small units, each tested on its own:

- `RoutePolicy` decides whether a record is forwarded: its logger is on the list and its level is high enough. No side effects.
- `Throttle` handles dedup, the rate cap and the suppression summary. No side effects; the clock can be injected.
- `HomeLogHandler.emit()` checks the policy, formats the record and queues it without blocking.
- `HomeLogWorker` is a background thread that drains the queue, applies the throttle and sends.
- `ReentrancyGuard` stops capture while the worker is sending. It is process-wide, because the platform adapter logs from another thread that a per-thread guard would miss.

The handler and worker live as long as the process. They are torn down at `atexit`, not `on_session_end`, which fires every turn. Installing twice is harmless.

Tests: `tests/plugins/home_log_router/`.

### Known tradeoff

The logger list uses internal module names. If upstream renames one of those loggers, its records stop being forwarded, with no error.
A structured alert sent through `get_home_channel()` would survive renames, but needs core changes this plugin avoids.
You can update the list with `HERMES_HOME_LOG_LOGGERS`.
