"""Retention engine for the disappearing_messages plugin.

Keeps a small content-free ledger next to Hermes's state and deletes what the
agent stored from a chat on that chat's disappearing-message timer.

What it records (no message content, ever):
  * ``timers`` — per chat, which timer was in force from which moment.
  * ``chats``  — per chat, how many "timer off" reports in a row it has seen.
  * ``files``  — attachment paths received in a disappearing chat, with the
                 moment each one must be gone.

What a sweep deletes, for every chat that has ever had a timer:
  * rows in ``state.db`` ``messages`` (user, assistant, tool, and rows archived
    by compaction) whose timer has run out. The FTS tables follow through
    their delete triggers, and are then optimised so old terms leave the index.
  * conversation summaries in a session as soon as anything they summarise is
    deleted. A summary is a new row written later, so its own timestamp would
    otherwise keep old content alive.
  * delegated-task records for the chat's sessions.
  * the attachment files whose deadline has passed.
  * session snapshot files and request dumps for the chat's sessions.
  * generated session titles, which are derived from content.

Deadlines are never later than Signal's: a message's clock starts at the
moment Signal says it was sent, not when we got round to storing it.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# Rows are stored a moment after Signal's send time. Deleting a little early
# keeps us on the safe side of Signal's own deadline; the cap keeps a short
# timer from deleting a message before the agent has answered it.
MAX_EARLY_S = 30.0
EARLY_FRACTION = 0.1

# Send times from the future are clamped to now. Send times from the past are
# trusted this far back, so a backlog delivered after downtime still counts
# from when it was really sent.
MAX_PAST_S = 30 * 24 * 60 * 60.0

# One "timer off" report is not enough to stop deleting: a member on a stale
# or linked client can send 0 while the chat still has a timer.
ZERO_REPORTS_TO_TURN_OFF = 2

STATE_BUSY_TIMEOUT_MS = 5000

# Conversation summaries written by context compaction start with one of these.
SUMMARY_PREFIXES = ("[CONTEXT COMPACTION", "[CONTEXT SUMMARY]")

_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS timers (
    platform   TEXT NOT NULL,
    chat_id    TEXT NOT NULL,
    since      REAL NOT NULL,
    ttl_s      INTEGER NOT NULL,
    PRIMARY KEY (platform, chat_id, since)
);
CREATE TABLE IF NOT EXISTS chats (
    platform     TEXT NOT NULL,
    chat_id      TEXT NOT NULL,
    zero_streak  INTEGER NOT NULL DEFAULT 0,
    first_zero   REAL,
    PRIMARY KEY (platform, chat_id)
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
    """When the message was sent: Signal's time, never in the future."""
    if isinstance(timestamp_ms, int) and not isinstance(timestamp_ms, bool) and timestamp_ms > 0:
        ts = timestamp_ms / 1000.0
        if now - ts <= MAX_PAST_S:
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
    summaries_deleted: int = 0
    delegations_deleted: int = 0
    files_deleted: int = 0
    session_files_deleted: int = 0
    failures: int = 0
    touched_session_keys: Set[str] = field(default_factory=set)


class Retention:
    """Ledger plus sweeper.

    ``observe`` is called on the gateway's event loop, so it never waits on a
    sweep or on state.db: it only touches the small ledger file.
    """

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
        self._sweep_lock = threading.Lock()
        ledger_dir = self.home / "disappearing"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(ledger_dir, 0o700)
        except OSError:
            pass
        self.ledger_path = ledger_dir / "ledger.db"
        conn = self._ledger()
        try:
            conn.executescript(_LEDGER_SCHEMA)
        finally:
            conn.close()

    # ------------------------------------------------------------------ ledger

    def _ledger(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.ledger_path, timeout=10, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _current(conn: sqlite3.Connection, platform: str, chat_id: str) -> Optional[Tuple[float, int]]:
        row = conn.execute(
            "SELECT since, ttl_s FROM timers WHERE platform=? AND chat_id=? "
            "ORDER BY since DESC LIMIT 1",
            (platform, chat_id),
        ).fetchone()
        return None if row is None else (float(row[0]), int(row[1]))

    def current_ttl(self, platform: str, chat_id: str) -> Optional[int]:
        conn = self._ledger()
        try:
            cur = self._current(conn, platform, chat_id)
        finally:
            conn.close()
        return None if cur is None else cur[1]

    def is_disappearing(self, platform: str, chat_id: str) -> bool:
        ttl = self.current_ttl(platform, chat_id)
        return bool(ttl and ttl > 0)

    def spans(self, platform: str, chat_id: str) -> List[TimerSpan]:
        conn = self._ledger()
        try:
            rows = conn.execute(
                "SELECT since, ttl_s FROM timers WHERE platform=? AND chat_id=? ORDER BY since",
                (platform, chat_id),
            ).fetchall()
        finally:
            conn.close()
        out: List[TimerSpan] = []
        for i, (since, ttl) in enumerate(rows):
            until = rows[i + 1][0] if i + 1 < len(rows) else None
            out.append(TimerSpan(float(since), None if until is None else float(until), int(ttl)))
        return out

    def chats(self) -> List[Tuple[str, str]]:
        conn = self._ledger()
        try:
            return [tuple(r) for r in conn.execute(
                "SELECT DISTINCT platform, chat_id FROM timers"
            ).fetchall()]
        finally:
            conn.close()

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
        existing timer, and a single 0 doesn't either.
        """
        now = self.clock()
        start = sent_at(timestamp_ms, now)
        conn = self._ledger()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = self._current(conn, platform, chat_id)
            previous = None if cur is None else cur[1]
            if expires_in_s is None:
                if previous is None:
                    conn.execute("ROLLBACK")
                    logger.warning(
                        "disappearing: unreadable timer on a chat with no history; not tracking it"
                    )
                    return None
                ttl = previous
            else:
                ttl = int(expires_in_s)

            if previous is None and ttl == 0:
                conn.execute("ROLLBACK")
                return 0  # never disappearing, nothing to remember

            span_start = start
            if ttl == 0 and previous and previous > 0:
                row = conn.execute(
                    "SELECT zero_streak, first_zero FROM chats WHERE platform=? AND chat_id=?",
                    (platform, chat_id),
                ).fetchone()
                streak = (row[0] if row else 0) + 1
                first_zero = (row[1] if row and row[1] is not None else start)
                conn.execute(
                    "INSERT INTO chats (platform, chat_id, zero_streak, first_zero) VALUES (?,?,?,?) "
                    "ON CONFLICT(platform, chat_id) DO UPDATE SET zero_streak=excluded.zero_streak, "
                    "first_zero=excluded.first_zero",
                    (platform, chat_id, streak, first_zero),
                )
                if streak < ZERO_REPORTS_TO_TURN_OFF:
                    ttl = previous  # keep deleting until the off is confirmed
                else:
                    span_start = min(first_zero, start)
            else:
                conn.execute(
                    "UPDATE chats SET zero_streak=0, first_zero=NULL WHERE platform=? AND chat_id=?",
                    (platform, chat_id),
                )

            if previous != ttl:
                # Spans stay ordered and never overwrite each other, even when
                # two changes share a timestamp.
                since = span_start if cur is None else max(span_start, cur[0] + 0.001)
                conn.execute(
                    "INSERT INTO timers (platform, chat_id, since, ttl_s) VALUES (?,?,?,?)",
                    (platform, chat_id, since, ttl),
                )
            elif ttl > 0 and start < cur[0]:
                # A message sent before the current span began arrived late
                # (out of order, or a backlog). Stretch the span back to cover
                # it, but never into the previous span.
                prev = conn.execute(
                    "SELECT MAX(since) FROM timers WHERE platform=? AND chat_id=? AND since < ?",
                    (platform, chat_id, cur[0]),
                ).fetchone()[0]
                new_since = start if prev is None else max(start, float(prev) + 0.001)
                if new_since < cur[0]:
                    conn.execute(
                        "UPDATE timers SET since=? WHERE platform=? AND chat_id=? AND since=?",
                        (new_since, platform, chat_id, cur[0]),
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
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()
        return ttl

    # ------------------------------------------------------------------ state.db

    def _state(self) -> Optional[sqlite3.Connection]:
        if not self.state_db.exists():
            return None
        conn = sqlite3.connect(self.state_db, timeout=STATE_BUSY_TIMEOUT_MS / 1000)
        conn.execute(f"PRAGMA busy_timeout={STATE_BUSY_TIMEOUT_MS}")
        # Overwrite freed pages so deleted text doesn't linger in the file.
        conn.execute("PRAGMA secure_delete=ON")
        return conn

    @staticmethod
    def _tables(conn: sqlite3.Connection) -> Set[str]:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    @staticmethod
    def _chat_sessions(conn: sqlite3.Connection, platform: str, chat_id: str) -> List[Tuple[str, str]]:
        """(session id, session key) for the chat, plus branch/compression descendants."""
        bare = chat_id[len("group:"):] if chat_id.startswith("group:") else chat_id
        rows = conn.execute(
            "SELECT id, COALESCE(session_key, '') FROM sessions WHERE source=? AND "
            "(chat_id=? OR session_key LIKE ? OR session_key LIKE ?)",
            (platform, chat_id, f"agent:%:{platform}:group:{bare}%", f"agent:%:{platform}:dm:{bare}"),
        ).fetchall()
        found = {r[0]: r[1] for r in rows}
        frontier = list(found)
        while frontier:
            marks = ",".join("?" * len(frontier))
            kids = conn.execute(
                f"SELECT id, COALESCE(session_key, '') FROM sessions WHERE parent_session_id IN ({marks})",
                frontier,
            ).fetchall()
            frontier = [k[0] for k in kids if k[0] not in found]
            for k in kids:
                found.setdefault(k[0], k[1])
        return sorted(found.items())

    def session_chat(self, session_id: str) -> Optional[Tuple[str, str]]:
        """(platform, chat_id) for a session id, following parents."""
        if not session_id:
            return None
        conn = self._state()
        if conn is None:
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
        if not self._sweep_lock.acquire(blocking=False):
            return result  # another sweep is running
        try:
            now = self.clock()
            self._sweep_files(now, result)
            conn = self._state()
            if conn is None:
                return result
            try:
                tables = self._tables(conn)
                for platform, chat_id in self.chats():
                    try:
                        self._sweep_chat(conn, tables, platform, chat_id, now, result)
                    except sqlite3.Error:
                        result.failures += 1
                        logger.warning("disappearing: could not sweep one chat; will retry", exc_info=True)
                if result.messages_deleted or result.summaries_deleted:
                    self._scrub(conn, tables)
            finally:
                conn.close()
        finally:
            self._sweep_lock.release()
        if result.messages_deleted or result.files_deleted or result.summaries_deleted:
            logger.info(
                "disappearing: deleted %d messages, %d summaries, %d delegations, "
                "%d attachments, %d session files",
                result.messages_deleted, result.summaries_deleted, result.delegations_deleted,
                result.files_deleted, result.session_files_deleted,
            )
        if result.failures:
            logger.warning("disappearing: %d deletions failed and will be retried", result.failures)
        return result

    def _sweep_files(self, now: float, result: SweepResult) -> None:
        conn = self._ledger()
        try:
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
        finally:
            conn.close()

    def _sweep_chat(
        self, conn: sqlite3.Connection, tables: Set[str], platform: str, chat_id: str,
        now: float, result: SweepResult,
    ) -> None:
        sessions = self._chat_sessions(conn, platform, chat_id)
        if not sessions:
            return
        sids = [s for s, _ in sessions]
        marks = ",".join("?" * len(sids))
        windows = []
        for span in self.spans(platform, chat_id):
            if span.ttl_s <= 0:
                continue
            cutoff = now - span.ttl_s + early_margin(span.ttl_s)
            upper = cutoff if span.until is None else min(cutoff, span.until)
            if upper > span.since:
                windows.append((span.since, upper))
        if not windows:
            return

        touched: Set[str] = set()
        deleted = 0
        conn.execute("BEGIN IMMEDIATE")
        try:
            for lo, hi in windows:
                hit = conn.execute(
                    f"SELECT DISTINCT session_id FROM messages WHERE session_id IN ({marks}) "
                    f"AND timestamp >= ? AND timestamp < ?",
                    (*sids, lo, hi),
                ).fetchall()
                if not hit:
                    continue
                touched.update(r[0] for r in hit)
                cur = conn.execute(
                    f"DELETE FROM messages WHERE session_id IN ({marks}) "
                    f"AND timestamp >= ? AND timestamp < ?",
                    (*sids, lo, hi),
                )
                deleted += cur.rowcount or 0
                if "async_delegations" in tables:
                    cur = conn.execute(
                        f"DELETE FROM async_delegations WHERE (origin_session IN ({marks}) "
                        f"OR parent_session_id IN ({marks})) AND dispatched_at >= ? AND dispatched_at < ?",
                        (*sids, *sids, lo, hi),
                    )
                    result.delegations_deleted += cur.rowcount or 0

            summaries = 0
            if touched:
                # A summary carries the content it replaced. Once any of that
                # is due, the summary goes too.
                tmarks = ",".join("?" * len(touched))
                like = " OR ".join("content LIKE ?" for _ in SUMMARY_PREFIXES)
                cur = conn.execute(
                    f"DELETE FROM messages WHERE session_id IN ({tmarks}) AND ({like})",
                    (*sorted(touched), *(p + "%" for p in SUMMARY_PREFIXES)),
                )
                summaries = cur.rowcount or 0
                conn.execute(
                    f"UPDATE sessions SET title=NULL, message_count=("
                    f"SELECT COUNT(*) FROM messages m WHERE m.session_id=sessions.id AND m.active=1) "
                    f"WHERE id IN ({tmarks})",
                    sorted(touched),
                )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

        if touched:
            result.messages_deleted += deleted
            result.summaries_deleted += summaries
            keys = dict(sessions)
            result.touched_session_keys.update(k for s in touched if (k := keys.get(s)))
            # Snapshots and dumps copy the whole conversation; state.db is canonical.
            for sid in touched:
                result.session_files_deleted += self._remove_session_copies(sid)

    def _scrub(self, conn: sqlite3.Connection, tables: Set[str]) -> None:
        """Push deleted text out of the FTS segments and the WAL."""
        for t in ("messages_fts", "messages_fts_trigram", "messages_fts_cjk"):
            if t in tables:
                try:
                    conn.execute(f"INSERT INTO {t}({t}) VALUES('optimize')")
                    conn.commit()
                except sqlite3.Error:
                    logger.debug("disappearing: could not optimise %s", t, exc_info=True)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            logger.debug("disappearing: WAL checkpoint deferred", exc_info=True)

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
        conn = self._ledger()
        try:
            return int(conn.execute(
                "SELECT COUNT(*) FROM files WHERE platform=? AND chat_id=?", (platform, chat_id)
            ).fetchone()[0])
        finally:
            conn.close()
