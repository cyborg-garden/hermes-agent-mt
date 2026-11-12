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
margin (10% of the timer, at most 30 seconds).

Changing the timer works like Signal: the new timer applies to new messages.
Turning it off stops new messages being deleted; older ones still go on time.
A single "timer off" report doesn't stop deletion, since a member on an old or
linked client can send one while the chat still has a timer. It takes two in a
row, and only messages after the second are kept.

## What gets deleted

- The person's messages and the bot's replies in `state.db`, including tool
  results and rows archived by compaction, and their search-index entries
  (FTS secure-delete is switched on, so terms leave the index straight away).
- Conversation summaries and other copies made by context compaction. Once any
  archived content in a session is due, every later row in that session goes
  too, and so does any session branched from it. This resets the bot's working
  context for that chat early, which is allowed; it never keeps a copy late.
  Rows from a period when the timer was off are left alone.
- Delegated-task records for the chat's sessions.
- Attachments the bot downloaded: voice notes, photos, documents.
- Session snapshot files and request dumps for the chat.
- The session title, which is written from the conversation.

Deleted pages are overwritten (`secure_delete`) and the write-ahead log is
checkpointed after each sweep that deletes something. After a delete, the bot's
cached agent for that chat is dropped, so nothing deleted stays in memory. In a
chat with a short timer that means the agent is rebuilt often, which costs a
little prompt caching.

Log lines about the chat keep who, where and when, but the message preview is
replaced with `<disappearing>`.

In a disappearing chat the bot will not write to long-term memory. That memory
is shared across all chats and cannot expire with one of them.

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
- One profile per gateway process: it sweeps the `HERMES_HOME` it starts in.

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
