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

## What gets deleted, and when

Each message's clock starts when Signal says it was sent. It is deleted a
little before Signal's own deadline, never after (up to 10% of the timer early,
at most 30 seconds).

- The person's messages and the bot's replies, in `state.db`, including the
  search index.
- Attachments the bot downloaded: voice notes, photos, documents.
- Session snapshot files and request dumps for the chat.
- The session title, which is written from the conversation.

Log lines about the chat keep who, where and when, but the message preview is
replaced with `<disappearing>`.

In a disappearing chat the bot will not write to long-term memory. That memory
is shared across all chats and cannot expire with one of them.

Changing the timer works like Signal: the new timer applies to new messages.
Turning it off stops new messages being deleted; older ones still go on time.

Conversation summaries (written when a long chat is compressed) are deleted as
soon as anything they summarise is due, and the bot's in-memory copy is dropped
too. That can be earlier than the summary's own timer, never later.

A single "timer off" report doesn't stop deletion, since a member on an old or
linked client can send one while the chat still has a timer. Two in a row do,
counted from the first.

Deleted pages are overwritten (`secure_delete`), the search index is optimised
so old words leave it, and the write-ahead log is checkpointed.

## What it cannot do

- It deletes copies on this host only. It cannot reach people's phones,
  Signal's servers, model providers, or host backups.
- External memory providers (anything wired through the memory manager's
  `sync_turn`) still receive each turn. The built-in memory tool is blocked;
  if an agent uses an external provider, turn it off for that agent or accept
  that it keeps a copy.
- Files a tool copies somewhere else (for example the papareo-transcribe
  long-job folder) are that tool's job to clean up.
- If compaction runs in the middle of a turn, that turn's newest messages are
  rewritten with a fresh timestamp, so in a chat with a very short timer they
  can last up to about twice the timer.
- It does not delete log lines written before it was turned on, and only
  blanks previews on lines that name the chat or session.
- One profile per gateway process: it sweeps the `HERMES_HOME` it starts in.

## How it works

The Signal adapter puts the timer on each incoming message
(`event.metadata["signal"]["expires_in_seconds"]`). The plugin records, per chat,
which timer was in force from when, and the attachment paths with their
deadlines, in `$HERMES_HOME/disappearing/ledger.db`. That ledger holds chat ids
and times only, never content. A background sweep deletes what is due. It runs
once at startup too, so anything that fell due while the agent was down goes
first.
