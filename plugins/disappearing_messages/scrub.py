"""Blank disappearing-chat text in a *copy* of state.db, before it leaves the host.

Tools that copy an agent's whole database elsewhere (the Failure Atlas export,
backups) would otherwise carry disappearing messages past their timer. Call
``scrub(snapshot, home)`` on the copy, never on the live database.

Which rows: the rows the retention sweep would delete when their timer runs
out: rows of a disappearing chat's sessions (and sessions branched from them)
that belong to a period when its timer was on, plus rows the sweep has marked
as copies of such content, plus the chat's delegated-task records from those
periods. Rows from before the timer was turned on, or from a period when it was
off, keep their text, and so does every other chat.

What is blanked: message text, reasoning, the arguments inside tool calls,
delegated-task payloads, and session titles. Roles, tool names, tool-call ids,
timings, token counts and finish reasons stay, so failure analysis still sees
the shape of the conversation. Search indexes are dropped from the copy (they
hold the text too), and the file is switched out of WAL and vacuumed so
blanked text isn't left in a log or free pages.

Known gap: a copy made by /retry or /branch in the half-minute since the
sweep last ran isn't marked as a copy yet, so if it sits in a timer-off period
its text survives this scrub (the sweep catches it on its next run).

Stdlib only, so it can run with any Python inside the agent's container.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

INF_ID = 2 ** 62
TEXT_COLUMNS = (
    "content", "reasoning", "reasoning_content", "reasoning_details",
    "codex_reasoning_items", "codex_message_items",
)
DELEGATION_TEXT = ("task_json", "event_json", "result_json")


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _ledger_rows(ledger: Path):
    conn = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True, timeout=10)
    try:
        timers = conn.execute(
            "SELECT platform, chat_id, since, from_id, ttl_s FROM timers ORDER BY platform, chat_id, since"
        ).fetchall()
        try:
            copies = conn.execute("SELECT session_id, lo_id, hi_id FROM copies").fetchall()
        except sqlite3.Error:
            copies = []
        try:
            top = conn.execute("SELECT MAX(hi_id) FROM watermarks").fetchone()[0]
        except sqlite3.Error:
            top = None
    finally:
        conn.close()
    return timers, copies, int(top or 0)


def _on_periods(timers, top_id: int) -> Dict[Tuple[str, str], List[Tuple[int, int, float, float]]]:
    """(platform, chat) -> [(lo_id, hi_id, since, until)] for periods with the timer on."""
    by_chat: Dict[Tuple[str, str], list] = {}
    for platform, chat_id, since, from_id, ttl in timers:
        by_chat.setdefault((platform, chat_id), []).append((float(since), from_id, int(ttl)))
    out: Dict[Tuple[str, str], List[Tuple[int, int, float, float]]] = {}
    for key, spans in by_chat.items():
        resolved: List[int] = []
        for i, (since, from_id, ttl) in enumerate(spans):
            if from_id is not None:
                resolved.append(int(from_id))
            else:
                # Not filled in yet (the sweep does that on its next run). The
                # change happened after the last sweep, so start from its id;
                # if the period before was also on, it simply continues.
                prev_on = i > 0 and spans[i - 1][2] > 0
                resolved.append(resolved[-1] if prev_on and resolved else top_id)
        periods = []
        for i, (since, _, ttl) in enumerate(spans):
            if ttl <= 0:
                continue
            nxt = i + 1 < len(spans)
            periods.append((
                resolved[i],
                resolved[i + 1] if nxt else INF_ID,
                since,
                spans[i + 1][0] if nxt else float("inf"),
            ))
        if periods:
            out[key] = periods
    return out


def _chat_sessions(conn: sqlite3.Connection, platform: str, chat_id: str) -> List[str]:
    bare = chat_id[len("group:"):] if chat_id.startswith("group:") else chat_id
    found = {r[0] for r in conn.execute(
        "SELECT id FROM sessions WHERE source=? AND (chat_id=? OR session_key LIKE ? OR session_key LIKE ?)",
        (platform, chat_id, f"agent:%:{platform}:group:{bare}%", f"agent:%:{platform}:dm:{bare}"),
    )}
    frontier = list(found)
    while frontier:
        marks = ",".join("?" * len(frontier))
        kids = [r[0] for r in conn.execute(
            f"SELECT id FROM sessions WHERE parent_session_id IN ({marks})", frontier
        )]
        frontier = [k for k in kids if k not in found]
        found.update(frontier)
    return sorted(found)


def _strip_arguments(raw):
    """Keep a tool call's shape (id, name) and drop what was passed to it."""
    if raw is None:
        return None
    try:
        calls = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(calls, list):
        return None
    for call in calls:
        if isinstance(call, dict):
            fn = call.get("function")
            if isinstance(fn, dict) and "arguments" in fn:
                fn["arguments"] = ""
            if "arguments" in call:
                call["arguments"] = ""
            if "input" in call:
                call["input"] = {}
    return json.dumps(calls)


def scrub(snapshot, home, required: bool = False) -> Dict[str, int]:
    """Blank disappearing-chat text in ``snapshot``. Returns counts only.

    Without a ledger there is nothing to do, unless ``required`` (the agent has
    the plugin on), in which case a missing or unreadable ledger raises so the
    caller can refuse to export the copy.
    """
    folder = Path(home) / "disappearing"
    ledger = folder / "ledger.db"
    if not ledger.exists():
        if required or folder.exists():
            raise RuntimeError("disappearing ledger expected but not readable; refusing to scrub blind")
        return {"chats": 0, "rows": 0, "delegations": 0}
    timers, copies, top_id = _ledger_rows(ledger)
    periods = _on_periods(timers, top_id)
    snap = str(snapshot)
    conn = sqlite3.connect(snap, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")  # no -wal left holding the old text
        conn.execute("PRAGMA secure_delete=ON")
        columns = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
        if "content" not in columns:
            raise RuntimeError("snapshot has no messages.content column; refusing to guess")
        blank = [c for c in TEXT_COLUMNS if c in columns]
        sets = ", ".join(f"{_q(c)}=NULL" if c != "content" else "content=''" for c in blank)
        has_calls = "tool_calls" in columns
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        dcols = set()
        if "async_delegations" in tables:
            dcols = {r[1] for r in conn.execute("PRAGMA table_info(async_delegations)")}
        dsets = ", ".join(f"{_q(c)}=NULL" for c in DELEGATION_TEXT if c in dcols)

        rows = delegations = 0
        touched: set = set()
        conn.execute("BEGIN")
        # Search indexes hold the text too, and their triggers would need it.
        for (trig,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='messages'"
        ).fetchall():
            conn.execute(f"DROP TRIGGER IF EXISTS {_q(trig)}")
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE name IN "
            "('messages_fts', 'messages_fts_trigram', 'messages_fts_cjk')"
        ).fetchall():
            conn.execute(f"DROP TABLE IF EXISTS {_q(name)}")

        def blank_rows(where: str, args: tuple) -> int:
            n = conn.execute(f"UPDATE messages SET {sets} WHERE {where}", args).rowcount or 0
            if has_calls:
                for mid, calls in conn.execute(
                    f"SELECT id, tool_calls FROM messages WHERE tool_calls IS NOT NULL AND {where}", args
                ).fetchall():
                    conn.execute("UPDATE messages SET tool_calls=? WHERE id=?", (_strip_arguments(calls), mid))
            return n

        for (platform, chat_id), spans in periods.items():
            sids = _chat_sessions(conn, platform, chat_id)
            if not sids:
                continue
            marks = ",".join("?" * len(sids))
            for lo, hi, since, until in spans:
                n = blank_rows(f"session_id IN ({marks}) AND id > ? AND id <= ?", (*sids, lo, hi))
                if n:
                    rows += n
                    touched.update(sids)
                if dsets:
                    delegations += conn.execute(
                        f"UPDATE async_delegations SET {dsets} WHERE (origin_session IN ({marks}) "
                        f"OR parent_session_id IN ({marks})) AND dispatched_at >= ? AND dispatched_at < ?",
                        (*sids, *sids, since, until if until != float("inf") else 1e18),
                    ).rowcount or 0
        for sid, lo, hi in copies:
            n = blank_rows("session_id=? AND id > ? AND id <= ?", (sid, lo, hi))
            if n:
                rows += n
                touched.add(sid)
        if touched:
            marks = ",".join("?" * len(touched))
            conn.execute(f"UPDATE sessions SET title=NULL WHERE id IN ({marks})", sorted(touched))
        conn.execute("COMMIT")
        conn.execute("VACUUM")
    finally:
        conn.close()
    for leftover in (snap + "-wal", snap + "-journal"):
        if os.path.exists(leftover) and os.path.getsize(leftover) > 0:
            raise RuntimeError("scrubbed copy still has a log file next to it; refusing it")
    return {"chats": len(periods), "rows": rows, "delegations": delegations}


if __name__ == "__main__":
    import sys
    print(json.dumps(scrub(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "/opt/data", required=True)))
