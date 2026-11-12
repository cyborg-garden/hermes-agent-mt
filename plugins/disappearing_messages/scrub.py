"""Blank disappearing-chat text in a *copy* of state.db, before it leaves the host.

Tools that copy an agent's whole database elsewhere (the Failure Atlas export,
backups) would otherwise carry disappearing messages past their timer. Call
``scrub(snapshot, home)`` on the copy, never on the live database.

Which rows: exactly the rows the retention sweep would delete when their timer
runs out: rows of a disappearing chat's sessions that belong to a period when
its timer was on, plus rows the sweep has marked as copies of such content.
Rows from before the timer was turned on, or from a period when it was off,
keep their text, and so does every other chat.

What is blanked: the message text, tool-call arguments and reasoning. Roles,
tool names, timings, token counts and finish reasons stay, so failure analysis
still sees the shape of the conversation. Search indexes are dropped from the
copy (they hold the text too), and the file is vacuumed so blanked text isn't
left in free pages.

Stdlib only, so it can run with any Python inside the agent's container.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

INF_ID = 2 ** 62
TEXT_COLUMNS = (
    "content", "tool_calls", "reasoning", "reasoning_content", "reasoning_details",
    "codex_reasoning_items", "codex_message_items",
)


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
    finally:
        conn.close()
    return timers, copies


def _on_ranges(timers) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    """(platform, chat) -> id ranges (lo, hi] when the timer was on."""
    by_chat: Dict[Tuple[str, str], list] = {}
    for platform, chat_id, since, from_id, ttl in timers:
        by_chat.setdefault((platform, chat_id), []).append((from_id, ttl))
    out: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for key, spans in by_chat.items():
        ranges = []
        for i, (from_id, ttl) in enumerate(spans):
            if ttl <= 0:
                continue
            lo = int(from_id) if from_id is not None else 0  # unresolved: assume the earliest
            nxt = spans[i + 1][0] if i + 1 < len(spans) else None
            hi = INF_ID if nxt is None else int(nxt)
            ranges.append((lo, hi))
        if ranges:
            out[key] = ranges
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


def scrub(snapshot: str | Path, home: str | Path) -> Dict[str, int]:
    """Blank disappearing-chat text in ``snapshot``. Returns counts only.

    Does nothing if this agent has no disappearing ledger. Raises if there is a
    ledger but the copy can't be scrubbed, so callers can refuse to export it.
    """
    ledger = Path(home) / "disappearing" / "ledger.db"
    if not ledger.exists():
        return {"chats": 0, "rows": 0}
    timers, copies = _ledger_rows(ledger)
    ranges = _on_ranges(timers)
    conn = sqlite3.connect(str(snapshot), isolation_level=None)
    try:
        conn.execute("PRAGMA secure_delete=ON")
        columns = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
        blank = [c for c in TEXT_COLUMNS if c in columns]
        if "content" not in blank:
            raise RuntimeError("snapshot has no messages.content column; refusing to guess")
        sets = ", ".join(f"{c}=NULL" if c != "content" else "content=''" for c in blank)
        rows = 0
        touched: set = set()
        conn.execute("BEGIN")
        # Drop the search indexes first: their delete/update triggers would
        # otherwise need the old text, and they hold copies of it.
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'messages_fts%'"
        ).fetchall():
            if "_" in name[len("messages_fts"):] and not name.endswith(("_fts", "_trigram", "_cjk")):
                continue  # shadow tables go with their parent
            conn.execute(f"DROP TABLE IF EXISTS {name}")
        for (trig,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='messages'"
        ).fetchall():
            conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
        for (platform, chat_id), spans in ranges.items():
            sids = _chat_sessions(conn, platform, chat_id)
            if not sids:
                continue
            marks = ",".join("?" * len(sids))
            for lo, hi in spans:
                cur = conn.execute(
                    f"UPDATE messages SET {sets} WHERE session_id IN ({marks}) AND id > ? AND id <= ?",
                    (*sids, lo, hi),
                )
                if cur.rowcount:
                    rows += cur.rowcount
                    touched.update(sids)
        for sid, lo, hi in copies:
            cur = conn.execute(
                f"UPDATE messages SET {sets} WHERE session_id=? AND id > ? AND id <= ?", (sid, lo, hi)
            )
            if cur.rowcount:
                rows += cur.rowcount
                touched.add(sid)
        if touched:
            marks = ",".join("?" * len(touched))
            conn.execute(f"UPDATE sessions SET title=NULL WHERE id IN ({marks})", sorted(touched))
        conn.execute("COMMIT")
        conn.execute("VACUUM")
    finally:
        conn.close()
    return {"chats": len(ranges), "rows": rows}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(scrub(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "/opt/data")))
