# disappearing_messages

When a Signal chat has disappearing messages on, this plugin deletes what the
agent keeps from that chat on the same timer.

Someone can be in a private group with the bot, and what they post there does
not pile up on the host. Signal deletes it from their phones; this deletes our
copy.

## Turn it on

Add it to the agent's `config.yaml`:

```yaml
plugins:
  enabled:
    - disappearing_messages
```

Then turn on disappearing messages in the Signal chat. Nothing else to set.
The sweep runs every 30 seconds (`DISAPPEARING_SWEEP_SECONDS` to change it).

## When things are deleted

Each message's clock starts when the bot stored it, measured on the host's own
clock. That is how Signal times a received message on a device too: the timer
starts when the device reads it. Timestamps a sender supplies are never used,
because they can be wrong or forged, and a forged one could otherwise keep a
message for ever.

Deletion can be early, never late: by up to one sweep (30 seconds) plus a small
margin (10% of the timer, at most 30 seconds). A message arriving after a quiet
spell is still counted from when it arrived.

Changing the timer works like Signal: the new timer applies to new messages.
Turning it off stops new messages being deleted; older ones still go on time.
A "timer off" report only counts from someone whose client has sent a timer
before, because an old or linked client always reports "off" while the chat
still has a timer. It takes two such reports in a row, and only messages after
the second are kept. When a busy bot merges several queued messages into one
turn (text, photos or other media), the merged turn keeps the shortest timer
among them.

## What gets deleted

- The person's messages and the bot's replies in `state.db`, including tool
  results and rows archived by compaction, and their search-index entries
  (FTS secure-delete is switched on, so terms leave the index straight away).
- Copies. `/retry`, `/branch`, compaction and forks write old content again as
  new rows. The sweep doesn't need to know each path: when a session loses rows
  it didn't delete and gains new ones, or a new session branches from a tracked
  one, the new rows get the age of the oldest content their source still held.
  For in-place compaction, once an archived original is due, every later row in
  that session goes too, and so does any session branched from it. Rows from a
  period when the timer was off are left alone. This can reset the bot's working
  context for the chat early, which is allowed; it never keeps a copy late.
- Delegated-task records for the chat's sessions.
- Attachments the bot downloaded: voice notes, photos, documents.
- signal-cli's own copy of each received attachment. signal-cli keeps every
  attachment it receives in its data folder and never removes it (it keeps no
  message history, so it has nothing to expire). The agent's container can't
  see that folder, so the plugin records each attachment id and its deadline,
  and `host/sweep_signal_cli.py`, run on the host every minute, deletes the
  file when it's due. They talk through two plain files in the agent's
  `disappearing/` folder (`signal_cli_due.json`, `signal_cli_done.json`), not
  the ledger: the ledger is SQLite in WAL mode inside the Docker VM, which a
  host process can't safely read. A row is only forgotten once the host has
  confirmed it; if nothing confirms for a day the agent logs a warning, and
  after 90 days it gives up. Files the bot sends aren't copied by signal-cli.
  Install: copy `host/sweep_signal_cli.py` to
  `~/.local/share/disappearing-messages/`, copy
  `host/ai.cyborg-garden.disappearing-signal-cli.plist` to
  `~/Library/LaunchAgents/`, set `__SCRIPT__` to the copied script and
  `__HOME__` to your home folder, and `launchctl load` it. Don't point
  `__SCRIPT__` at the repo checkout: macOS blocks launchd jobs from reading
  `~/Documents`, `~/Desktop` and `~/Downloads`, so the sweeper fails every
  minute ("Operation not permitted" in the log) and nothing is deleted.
  Copy the script again after you update it.
- Session snapshot files and request dumps for the chat.
- The session title, which is written from the conversation.

Deleted pages are overwritten (`secure_delete`), and after each sweep that
deletes something the write-ahead log (`state.db-wal`) is copied in and emptied.
That needs a moment when nothing is reading the database, so if the agent is
busy it is retried every sweep until it succeeds; until then the deleted text
can still be in the `-wal` file. After a delete, the bot's
cached agent for that chat is dropped, so nothing deleted stays in memory. In a
chat with a short timer that means the agent is rebuilt often, which costs a
little prompt caching.

Log lines about the chat keep who, where and when, but the message preview is
replaced with `<disappearing>`.

In a disappearing chat the bot will not write to long-term memory, save notes
about a person (`remember_person`, from the person_memory plugin), or create or
edit skills (including from the background skill review). Both are shared
across all chats and cannot expire with one of them. If the check itself fails,
the write is refused.

## Copies made by other tools

`scrub.py` blanks a disappearing chat's text in a *copy* of `state.db`, by the
same rules as deletion, so tools that copy the whole database don't carry it
past the timer. The Failure Atlas nightly export calls it on its snapshot
inside the container. Roles, tool names, timings and finish reasons stay; every
other chat keeps its text. Usage: `python3 scrub.py <copy.db> [<agent home>]`.

## What it cannot do

- It deletes copies on this host only. It cannot reach people's phones,
  Signal's servers, model providers, or host backups.
- External memory providers (anything wired through the memory manager's
  `sync_turn`) still receive each turn. The built-in memory tool is blocked;
  if an agent uses an external provider, turn it off for that agent or accept
  that it keeps a copy.
- Files a tool copies somewhere else (for example the papareo-transcribe
  long-job folder) are that tool's job to clean up.
- It does not delete anything stored before it was turned on, or log lines
  written before then. It only blanks previews on log lines that name the chat
  or one of its sessions, and a brand-new session is only known after the next
  sweep.
- Copies of history from before the timer was turned on (made by `/retry`,
  `/branch` or compaction after the timer is on) expire with the timer, early.
  `/branch` and compaction keep the originals; `/retry` replaces them, so after
  a `/retry` that earlier history is gone once the timer runs out.
- With `busy_input_mode: steer`, a message steered into a running turn never
  reaches the plugin's hook, so its timer isn't seen. The default interrupt mode
  and the queue mode are covered.
- Turning on FTS secure-delete changes the search index's on-disk format, so
  `state.db` can't be read by SQLite older than 3.42. Production runs 3.46.
- If reading the database is stalled at the moment a timer is first turned on,
  up to one sweep of earlier rows can be counted under the new timer.
- The person_memory plugin still records who someone is (display name, ids,
  first and last seen) on its person card. That is metadata, not what they said.
- Attachments in a group message the bot ignores entirely are never seen by
  the plugin, so signal-cli's copy stays. With `SIGNAL_OBSERVE_UNMENTIONED=true`
  (the default) every message in an approved group reaches the plugin; with it
  off, unmentioned messages are dropped before the plugin sees them.
- signal-cli also saves quote thumbnails and link-preview images in its
  attachments folder; those ids aren't recorded, so those copies stay.
- It doesn't clean up attachments signal-cli saved before the plugin was on.
- If a Signal client forwards a file by reusing the same attachment into an
  ordinary chat, that chat's copy goes on the disappearing chat's timer too
  (they are one file in signal-cli's folder). Not confirmed to happen.
- One profile per gateway process: it sweeps the `HERMES_HOME` it starts in.
- If the host clock jumps backwards, deadlines for rows stored after the jump
  are late by the size of the jump. A warning is logged.
- A message that is merged into an existing queued message keeps the shortest
  timer of the two, which may delete it early.

## How it works

The Signal adapter puts the timer on each incoming message
(`event.metadata["signal"]["expires_in_seconds"]`). The plugin keeps a ledger in
`$HERMES_HOME/disappearing/ledger.db` with chat ids, timer values, row ids,
attachment paths and times, never content:

- When a chat's timer changes, it records the change with the highest
  `messages.id` at that moment. Ids only grow, so a row's id says which timer
  period it belongs to.
- Each sweep records a watermark: the time and the highest id. A row above one
  watermark was stored after that watermark's time, and is counted from there.

A background sweep deletes what is due. It runs once at startup too, so anything
that fell due while the agent was down goes first.
