"""Retention engine for the disappearing_messages plugin.

Keeps a small content-free ledger next to Hermes's state and deletes what the
agent stored from a chat on that chat's disappearing-message timer.

What it records (no message content, ever):
  * ``timers``  — per chat, which timer was in force from which moment.
  * ``files``   — attachment paths received in a disappearing chat, with the
                  moment each one must be gone.

What a sweep deletes, for every chat that has ever had a timer:
  * rows in ``state.db`` ``messages`` (user, assistant and tool rows) whose
    timer has run out. The FTS tables follow through their delete triggers.
  * the attachment files whose deadline has passed.
  * session snapshot files and request dumps for the chat's sessions, which
    hold copies of the same conversation.
  * generated session titles, which are derived from content.

Deadlines are never later than Signal's: a message's clock starts at the
moment Signal says it was sent, not when we got round to storing it.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# Rows are stored a moment after Signal's send time. Deleting a little early
# keeps us on the safe side of Signal's own deadline; the cap keeps a short
# timer from deleting a message before the agent has answered it.
MAX_EARLY_S = 30.0
EARLY_FRACTION = 0.1

# A Signal send timestamp further than this from our clock is not trusted.
MAX_CLOCK_SKEW_S = 24 * 60 * 60.0

_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS timers (
    platform   TEXT NOT NULL,
    chat_id    TEXT NOT NULL,
    since      REAL NOT NULL,
    ttl_s      INTEGER NOT NULL,
    PRIMARY KEY (platform, chat_id, since)
);
CREATE TABLE IF NOT EXISTS files (
    id         TEXT PRIMARY KEY,
    platform   TEXT NOT NULL,
    chat_id    TEXT NOT NULL,
    path       TEXT NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS files_expiry ON files (expires_at);
"""


def early_margin(ttl_s: float) -> float:
    return min(MAX_EARLY_S, ttl_s * EARLY_FRACTION)


def sent_at(timestamp_ms: Optional[int], now: float) -> float:
    """When the message was sent, clamped to our own clock."""
    if isinstance(timestamp_ms, int) and timestamp_ms > 0:
        ts = timestamp_ms / 1000.0
        if abs(now - ts) <= MAX_CLOCK_SKEW_S:
            return min(ts, now)
    return now


@dataclass(frozen=True)
class TimerSpan:
    since: float
    until: Optional[float]   # None = still in force
    ttl_s: int


@dataclass
class SweepResult:
    messages_deleted: int = 0
    files_deleted: int = 0
    session_files_deleted: int = 0
    failures: int = 0


class Retention:
    """Ledger plus sweeper. Safe to call from several threads."""

    def __init__(
        self,
        home: Path,
        state_db: Path,
        extra_dump_dirs: Sequence[Path] = (),
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.home = Path(home)
        self.state_db = Path(state_db)
        self.dump_dirs = [self.home / "sessions", self.home / "logs", *map(Path, extra_dump_dirs)]
        self.clock = clock
        self._lock = threading.RLock()
        ledger_dir = self.home / "disappearing"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(ledger_dir, 0o700)
        except OSError:
            pass
        self.ledger_path = ledger_dir / "ledger.db"
        with self._ledger() as conn:
            conn.executescript(_LEDGER_SCHEMA)

    # ------------------------------------------------------------------ ledger

    def _ledger(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.ledger_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def current_ttl(self, platform: str, chat_id: str) -> Optional[int]:
        with self._lock, self._ledger() as conn:
            row = conn.execute(
                "SELECT ttl_s FROM timers WHERE platform=? AND chat_id=? "
                "ORDER BY since DESC LIMIT 1",
                (platform, chat_id),
            ).fetchone()
        return None if row is None else int(row[0])

    def is_disappearing(self, platform: str, chat_id: str) -> bool:
        ttl = self.current_ttl(platform, chat_id)
        return bool(ttl and ttl > 0)

    def spans(self, platform: str, chat_id: str) -> List[TimerSpan]:
        with self._lock, self._ledger() as conn:
            rows = conn.execute(
                "SELECT since, ttl_s FROM timers WHERE platform=? AND chat_id=? ORDER BY since",
                (platform, chat_id),
            ).fetchall()
        out: List[TimerSpan] = []
        for i, (since, ttl) in enumerate(rows):
            until = rows[i + 1][0] if i + 1 < len(rows) else None
            out.append(TimerSpan(float(since), None if until is None else float(until), int(ttl)))
        return out

    def chats(self) -> List[Tuple[str, str]]:
        with self._lock, self._ledger() as conn:
            return [tuple(r) for r in conn.execute(
                "SELECT DISTINCT platform, chat_id FROM timers"
            ).fetchall()]

    def observe(
        self,
        platform: str,
        chat_id: str,
        expires_in_s: Optional[int],
        timestamp_ms: Optional[int],
        media_paths: Iterable[str] = (),
    ) -> Optional[int]:
        """Record one inbound message. Returns the timer now in force.

        ``expires_in_s`` is 0 for "not disappearing" and None when the
        platform sent something unreadable. Unreadable never loosens an
        existing timer: the chat keeps the one it had.
        """
        now = self.clock()
        start = sent_at(timestamp_ms, now)
        with self._lock:
            previous = self.current_ttl(platform, chat_id)
            if expires_in_s is None:
                if previous is None:
                    logger.warning(
                        "disappearing: unreadable timer on a chat with no history; not tracking it"
                    )
                    return None
                ttl = previous
            else:
                ttl = int(expires_in_s)
            with self._ledger() as conn:
                if previous is None and ttl == 0:
                    return 0  # never disappearing, nothing to remember
                if previous != ttl:
                    # Spans must stay in order and never overwrite each other,
                    # even when two changes share a timestamp.
                    last = conn.execute(
                        "SELECT MAX(since) FROM timers WHERE platform=? AND chat_id=?",
                        (platform, chat_id),
                    ).fetchone()[0]
                    since = start if last is None else max(start, float(last) + 0.001)
                    conn.execute(
                        "INSERT INTO timers (platform, chat_id, since, ttl_s) VALUES (?,?,?,?)",
                        (platform, chat_id, since, ttl),
                    )
                if ttl > 0:
                    for p in media_paths:
                        if p:
                            conn.execute(
                                "INSERT INTO files (id, platform, chat_id, path, expires_at) "
                                "VALUES (?,?,?,?,?)",
                                (uuid.uuid4().hex, platform, chat_id, str(p),
                                 start + ttl - early_margin(ttl)),
                            )
        return ttl

    # ------------------------------------------------------------------ state.db

    def _state(self) -> Optional[sqlite3.Connection]:
        if not self.state_db.exists():
            return None
        conn = sqlite3.connect(self.state_db, timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @staticmethod
    def _chat_session_ids(conn: sqlite3.Connection, platform: str, chat_id: str) -> List[str]:
        """Every session for the chat, plus branch/compression descendants."""
        bare = chat_id[len("group:"):] if chat_id.startswith("group:") else chat_id
        rows = conn.execute(
            "SELECT id FROM sessions WHERE source=? AND (chat_id=? OR session_key LIKE ? OR session_key LIKE ?)",
            (platform, chat_id, f"agent:%:{platform}:group:{bare}%", f"agent:%:{platform}:dm:{bare}"),
        ).fetchall()
        found: Set[str] = {r[0] for r in rows}
        frontier = list(found)
        while frontier:
            marks = ",".join("?" * len(frontier))
            kids = conn.execute(
                f"SELECT id FROM sessions WHERE parent_session_id IN ({marks})", frontier
            ).fetchall()
            frontier = [k[0] for k in kids if k[0] not in found]
            found.update(frontier)
        return sorted(found)

    def session_chat(self, session_id: str) -> Optional[Tuple[str, str]]:
        """(platform, chat_id) for a session id, following parents."""
        conn = self._state()
        if conn is None or not session_id:
            return None
        try:
            sid, hops = session_id, 0
            while sid and hops < 20:
                row = conn.execute(
                    "SELECT source, chat_id, parent_session_id FROM sessions WHERE id=?", (sid,)
                ).fetchone()
                if row is None:
                    return None
                if row[1]:
                    return (row[0], row[1])
                sid, hops = row[2], hops + 1
            return None
        finally:
            conn.close()

    # ------------------------------------------------------------------ sweep

    def sweep(self) -> SweepResult:
        result = SweepResult()
        now = self.clock()
        with self._lock:
            self._sweep_files(now, result)
            conn = self._state()
            if conn is None:
                return result
            try:
                for platform, chat_id in self.chats():
                    self._sweep_chat(conn, platform, chat_id, now, result)
            finally:
                conn.close()
        if result.messages_deleted or result.files_deleted:
            logger.info(
                "disappearing: deleted %d messages, %d attachments, %d session files",
                result.messages_deleted, result.files_deleted, result.session_files_deleted,
            )
        if result.failures:
            logger.warning("disappearing: %d deletions failed and will be retried", result.failures)
        return result

    def _sweep_files(self, now: float, result: SweepResult) -> None:
        with self._ledger() as conn:
            due = conn.execute(
                "SELECT id, path FROM files WHERE expires_at <= ?", (now,)
            ).fetchall()
            for fid, path in due:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    result.failures += 1
                    continue
                conn.execute("DELETE FROM files WHERE id=?", (fid,))
                result.files_deleted += 1

    def _sweep_chat(
        self, conn: sqlite3.Connection, platform: str, chat_id: str, now: float, result: SweepResult
    ) -> None:
        sids = self._chat_session_ids(conn, platform, chat_id)
        if not sids:
            return
        marks = ",".join("?" * len(sids))
        deleted = 0
        for span in self.spans(platform, chat_id):
            if span.ttl_s <= 0:
                continue
            cutoff = now - span.ttl_s + early_margin(span.ttl_s)
            upper = cutoff if span.until is None else min(cutoff, span.until)
            if upper <= span.since:
                continue
            cur = conn.execute(
                f"DELETE FROM messages WHERE session_id IN ({marks}) "
                f"AND timestamp >= ? AND timestamp < ?",
                (*sids, span.since, upper),
            )
            deleted += cur.rowcount or 0
        if deleted:
            # Titles are generated from content; counts would be stale.
            conn.execute(
                f"UPDATE sessions SET title=NULL, message_count=("
                f"SELECT COUNT(*) FROM messages m WHERE m.session_id=sessions.id) "
                f"WHERE id IN ({marks})",
                sids,
            )
            conn.commit()
            result.messages_deleted += deleted
            # Snapshots and dumps copy the whole conversation; state.db is canonical.
            for sid in sids:
                result.session_files_deleted += self._remove_session_copies(sid)

    def _remove_session_copies(self, session_id: str) -> int:
        n = 0
        for d in self.dump_dirs:
            if not d.is_dir():
                continue
            patterns = (
                f"{session_id}.json", f"{session_id}.jsonl",
                f"session_{session_id}.json", f"request_dump_{session_id}_*.json",
            )
            for pat in patterns:
                for p in d.glob(pat):
                    try:
                        p.unlink()
                        n += 1
                    except OSError:
                        pass
        return n

    # ------------------------------------------------------------------ status

    def pending(self, platform: str, chat_id: str) -> int:
        with self._lock, self._ledger() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM files WHERE platform=? AND chat_id=?", (platform, chat_id)
            ).fetchone()[0])
