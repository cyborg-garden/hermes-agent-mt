"""Retention engine for the disappearing_messages plugin.

Deletes what the agent stored from a chat on that chat's disappearing-message
timer, using a small content-free ledger next to Hermes's state.

Clocks. Nothing here trusts a timestamp a sender supplied: those can be wrong
or forged, and a forged one could keep a message for ever. Every deadline is
measured on this host's clock from when the agent *stored* the row. That is
also how Signal times a received message on a device: its timer starts when
the device reads it, which for a bot is when it arrives.

How a row is timed without trusting its timestamp:
  * Timer changes are recorded when they arrive, together with the highest
    ``messages.id`` at that moment. ``messages.id`` only ever grows, so a row's
    id says which timer period it belongs to.
  * Each sweep records a watermark: (time, highest id). A row whose id is above
    one watermark was stored after that watermark's time. Using that time as
    the row's start is never later than the truth, so deletion may be up to one
    sweep early, never late.

What it records (no message content, ever): chat ids, timer values, row ids,
attachment paths, and times.

What a sweep deletes, for every chat that has had a timer on:
  * the chat's rows in ``state.db`` ``messages`` whose timer has run out. FTS
    tables follow through their delete triggers, with FTS secure-delete on.
  * copies. Several paths rewrite or branch a conversation under new, higher
    ids (/retry, /branch, rotation compaction, forks). Rather than knowing each
    one, a sweep notices when a session lost rows it didn't delete, or when a
    new session branched from a tracked one, and gives the new rows the age of
    the oldest content they could have copied. In-place compaction keeps the
    archived originals (compacted=1); once one of those is due, every later row
    in that session goes too.
  * delegated-task records, downloaded attachments, session snapshot files and
    request dumps for the chat's sessions, and generated session titles.
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
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# Rows are stored a moment after they arrive, and sweeps run on an interval.
# A small early margin keeps us on the safe side; the cap keeps a short timer
# from deleting a message before the agent has answered it.
MAX_EARLY_S = 30.0
EARLY_FRACTION = 0.1

# One "timer off" report is not enough to stop deleting: a member on a stale
# or linked client can send 0 while the chat still has a timer.
ZERO_REPORTS_TO_TURN_OFF = 2

STATE_BUSY_TIMEOUT_MS = 5000
READ_TIMEOUT_S = 0.25          # observe() runs on the event loop; never wait long

# Signal's longest timer is 4 weeks. Keep watermarks a little longer than that.
WATERMARK_KEEP_S = 35 * 24 * 60 * 60.0

INF_ID = 2 ** 62

_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS timers (
    platform   TEXT NOT NULL,
    chat_id    TEXT NOT NULL,
    since      REAL NOT NULL,     -- arrival time on this host
    from_id    INTEGER,           -- rows with a higher messages.id belong to this timer
    ttl_s      INTEGER NOT NULL,
    PRIMARY KEY (platform, chat_id, since)
);
CREATE TABLE IF NOT EXISTS chats (
    platform     TEXT NOT NULL,
    chat_id      TEXT NOT NULL,
    zero_streak  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (platform, chat_id)
);
CREATE TABLE IF NOT EXISTS watermarks (
    at     REAL PRIMARY KEY,
    hi_id  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS senders (
    platform   TEXT NOT NULL,
    chat_id    TEXT NOT NULL,
    sender     TEXT NOT NULL,     -- has sent a non-zero timer, so their client supports it
    PRIMARY KEY (platform, chat_id, sender)
);
CREATE TABLE IF NOT EXISTS sessions_seen (
    session_id   TEXT PRIMARY KEY,
    last_max_id  INTEGER NOT NULL,
    last_count   INTEGER NOT NULL,
    oldest_at    REAL,            -- store time of the oldest content still in it
    oldest_ttl   INTEGER
);
CREATE TABLE IF NOT EXISTS copies (
    session_id TEXT NOT NULL,
    lo_id      INTEGER NOT NULL,  -- rows with lo_id < id <= hi_id ...
    hi_id      INTEGER NOT NULL,
    at         REAL NOT NULL,     -- ... count from this time
    ttl_s      INTEGER NOT NULL,
    PRIMARY KEY (session_id, lo_id)
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


@dataclass(frozen=True)
class TimerSpan:
    since: float
    until: Optional[float]   # arrival time of the next change, None = in force
    from_id: int
    to_id: int               # INF_ID while in force
    ttl_s: int


@dataclass
class SweepResult:
    messages_deleted: int = 0
    delegations_deleted: int = 0
    files_deleted: int = 0
    session_files_deleted: int = 0
    failures: int = 0
    touched_session_keys: Set[str] = field(default_factory=set)


class Retention:
    """Ledger plus sweeper.

    ``observe`` runs on the gateway's event loop. It touches only the small
    ledger file, plus one bounded read of state.db's highest message id.
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
        self._fts_secure_set = False
        self._wal_pending = False
        # session id -> (platform, chat_id), for disappearing chats; filled by sweeps
        self.session_owner: Dict[str, Tuple[str, str]] = {}
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
            if conn.execute("SELECT COUNT(*) FROM watermarks").fetchone()[0] == 0:
                hi = self._global_max_id()
                conn.execute(
                    "INSERT INTO watermarks (at, hi_id) VALUES (?, ?)",
                    (self.clock(), hi if hi is not None else 0),
                )
        finally:
            conn.close()

    # ------------------------------------------------------------------ ledger

    def _ledger(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.ledger_path, timeout=10, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _global_max_id(self) -> Optional[int]:
        """Highest messages.id right now, or None if it can't be read quickly."""
        if not self.state_db.exists():
            return 0
        try:
            conn = sqlite3.connect(
                f"file:{self.state_db}?mode=ro", uri=True, timeout=READ_TIMEOUT_S
            )
            try:
                row = conn.execute("SELECT MAX(id) FROM messages").fetchone()
                return int(row[0] or 0)
            finally:
                conn.close()
        except sqlite3.Error:
            return None

    @staticmethod
    def _counts(conn: sqlite3.Connection, platform: str, chat_id: str, sender: Optional[str]) -> bool:
        if not sender:
            return False
        return conn.execute(
            "SELECT 1 FROM senders WHERE platform=? AND chat_id=? AND sender=?",
            (platform, chat_id, sender),
        ).fetchone() is not None

    @staticmethod
    def _current(conn: sqlite3.Connection, platform: str, chat_id: str):
        return conn.execute(
            "SELECT since, from_id, ttl_s FROM timers WHERE platform=? AND chat_id=? "
            "ORDER BY since DESC LIMIT 1",
            (platform, chat_id),
        ).fetchone()

    def current_ttl(self, platform: str, chat_id: str) -> Optional[int]:
        conn = self._ledger()
        try:
            cur = self._current(conn, platform, chat_id)
        finally:
            conn.close()
        return None if cur is None else int(cur[2])

    def is_disappearing(self, platform: str, chat_id: str) -> bool:
        ttl = self.current_ttl(platform, chat_id)
        return bool(ttl and ttl > 0)

    def spans(self, platform: str, chat_id: str) -> List[TimerSpan]:
        conn = self._ledger()
        try:
            rows = conn.execute(
                "SELECT since, from_id, ttl_s FROM timers WHERE platform=? AND chat_id=? "
                "ORDER BY since",
                (platform, chat_id),
            ).fetchall()
        finally:
            conn.close()
        out: List[TimerSpan] = []
        for i, (since, from_id, ttl) in enumerate(rows):
            nxt = rows[i + 1] if i + 1 < len(rows) else None
            out.append(TimerSpan(
                since=float(since),
                until=None if nxt is None else float(nxt[0]),
                from_id=int(from_id if from_id is not None else INF_ID),
                to_id=INF_ID if nxt is None else int(nxt[1] if nxt[1] is not None else INF_ID),
                ttl_s=int(ttl),
            ))
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
        media_paths: Iterable[str] = (),
        sender: Optional[str] = None,
    ) -> Optional[int]:
        """Record one inbound message. Returns the timer now in force.

        ``expires_in_s`` is 0 for "not disappearing" and None when the
        platform sent something unreadable. Unreadable never loosens an
        existing timer. A 0 only counts from a sender whose client has sent a
        timer before (an old client always sends 0), and it takes two in a row.
        """
        now = self.clock()
        max_id = self._global_max_id()
        conn = self._ledger()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = self._current(conn, platform, chat_id)
            previous = None if cur is None else int(cur[2])
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

            if ttl > 0 and sender:
                conn.execute(
                    "INSERT OR IGNORE INTO senders (platform, chat_id, sender) VALUES (?,?,?)",
                    (platform, chat_id, sender),
                )
            if ttl == 0 and previous and previous > 0 and not self._counts(conn, platform, chat_id, sender):
                ttl = previous  # a client that never shows timers; ignore its 0
            elif ttl == 0 and previous and previous > 0:
                row = conn.execute(
                    "SELECT zero_streak FROM chats WHERE platform=? AND chat_id=?",
                    (platform, chat_id),
                ).fetchone()
                streak = (row[0] if row else 0) + 1
                conn.execute(
                    "INSERT INTO chats (platform, chat_id, zero_streak) VALUES (?,?,?) "
                    "ON CONFLICT(platform, chat_id) DO UPDATE SET zero_streak=excluded.zero_streak",
                    (platform, chat_id, streak),
                )
                if streak < ZERO_REPORTS_TO_TURN_OFF:
                    ttl = previous  # keep deleting until the off is confirmed
            else:
                conn.execute(
                    "UPDATE chats SET zero_streak=0 WHERE platform=? AND chat_id=?",
                    (platform, chat_id),
                )

            if previous != ttl:
                since = now if cur is None else max(now, float(cur[0]) + 0.001)
                from_id = max_id
                if from_id is None:
                    stricter = ttl > 0 and (not previous or ttl < previous)
                    if stricter:
                        # On, or shorter: start from the last known id, so rows
                        # in doubt get the stricter timer (early, never late).
                        wm = conn.execute("SELECT MAX(hi_id) FROM watermarks").fetchone()[0]
                        from_id = int(wm or 0)
                    # Off, or longer: leave it NULL. The next sweep fills in its
                    # own highest id, so rows in doubt keep the stricter timer.
                conn.execute(
                    "INSERT INTO timers (platform, chat_id, since, from_id, ttl_s) VALUES (?,?,?,?,?)",
                    (platform, chat_id, since, from_id, ttl),
                )

            if ttl > 0:
                for p in media_paths:
                    if p:
                        conn.execute(
                            "INSERT INTO files (id, platform, chat_id, path, expires_at) "
                            "VALUES (?,?,?,?,?)",
                            (uuid.uuid4().hex, platform, chat_id, str(p),
                             now + ttl - early_margin(ttl)),
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
        conn.execute("PRAGMA journal_size_limit=0")
        return conn

    @staticmethod
    def _tables(conn: sqlite3.Connection) -> Set[str]:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def _enable_fts_secure_delete(self, conn: sqlite3.Connection, tables: Set[str]) -> None:
        """Make deletes remove terms from the FTS index straight away (SQLite >= 3.42).

        A persistent per-index setting, so it is set once rather than paying
        for a full index rebuild on every sweep.
        """
        if self._fts_secure_set:
            return
        for t in ("messages_fts", "messages_fts_trigram", "messages_fts_cjk"):
            if t in tables:
                try:
                    conn.execute(f"INSERT INTO {t}({t}, rank) VALUES('secure-delete', 1)")
                    conn.commit()
                except sqlite3.Error:
                    logger.debug("disappearing: FTS secure-delete unavailable for %s", t, exc_info=True)
        self._fts_secure_set = True

    @staticmethod
    def _chat_sessions(conn: sqlite3.Connection, platform: str, chat_id: str) -> Dict[str, Tuple[str, Optional[str]]]:
        """{session id: (session key, parent id)} for the chat and its descendants."""
        bare = chat_id[len("group:"):] if chat_id.startswith("group:") else chat_id
        rows = conn.execute(
            "SELECT id, COALESCE(session_key, ''), parent_session_id FROM sessions WHERE source=? AND "
            "(chat_id=? OR session_key LIKE ? OR session_key LIKE ?)",
            (platform, chat_id, f"agent:%:{platform}:group:{bare}%", f"agent:%:{platform}:dm:{bare}"),
        ).fetchall()
        found = {r[0]: (r[1], r[2]) for r in rows}
        frontier = list(found)
        while frontier:
            marks = ",".join("?" * len(frontier))
            kids = conn.execute(
                f"SELECT id, COALESCE(session_key, ''), parent_session_id FROM sessions "
                f"WHERE parent_session_id IN ({marks})",
                frontier,
            ).fetchall()
            frontier = [k[0] for k in kids if k[0] not in found]
            for k in kids:
                found.setdefault(k[0], (k[1], k[2]))
        return found

    def session_chat(self, session_id: str) -> Optional[Tuple[str, str]]:
        """(platform, chat_id) for a session id, following parents."""
        if not session_id:
            return None
        if session_id in self.session_owner:
            return self.session_owner[session_id]
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
                if "messages" not in tables:
                    return result
                self._enable_fts_secure_delete(conn, tables)
                hi = int(conn.execute("SELECT MAX(id) FROM messages").fetchone()[0] or 0)
                watermarks = self._advance_watermarks(now, hi)
                owners: Dict[str, Tuple[str, str]] = {}
                for platform, chat_id in self.chats():
                    try:
                        self._sweep_chat(conn, tables, platform, chat_id, now, watermarks, owners, result)
                    except sqlite3.Error:
                        result.failures += 1
                        logger.warning("disappearing: could not sweep one chat; will retry", exc_info=True)
                self.session_owner = owners
                if result.messages_deleted or self._wal_pending:
                    self._truncate_wal(conn)
            finally:
                conn.close()
        finally:
            self._sweep_lock.release()
        if result.messages_deleted or result.files_deleted:
            logger.info(
                "disappearing: deleted %d messages, %d delegations, %d attachments, %d session files",
                result.messages_deleted, result.delegations_deleted,
                result.files_deleted, result.session_files_deleted,
            )
        if result.failures:
            logger.warning("disappearing: %d deletions failed and will be retried", result.failures)
        return result

    def _truncate_wal(self, conn: sqlite3.Connection) -> None:
        """Copy the log into the database and empty it, so deleted text leaves
        the -wal file. Needs a moment with no readers; retried each sweep."""
        try:
            # PASSIVE never blocks. Only when it has copied the whole log is a
            # TRUNCATE tried, with a short wait so the agent's writes don't stall.
            busy, log, done = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            if busy or (log >= 0 and done < log):
                self._wal_pending = True
            else:
                conn.execute("PRAGMA busy_timeout=100")
                try:
                    busy = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                    self._wal_pending = bool(busy and busy[0])
                finally:
                    conn.execute(f"PRAGMA busy_timeout={STATE_BUSY_TIMEOUT_MS}")
        except sqlite3.Error:
            self._wal_pending = True
        if self._wal_pending:
            logger.debug("disappearing: WAL still in use; will truncate on a later sweep")

    def _advance_watermarks(self, now: float, hi: int) -> List[Tuple[float, int]]:
        conn = self._ledger()
        try:
            conn.execute("UPDATE timers SET from_id=? WHERE from_id IS NULL", (hi,))
            last = conn.execute("SELECT at, hi_id FROM watermarks ORDER BY at DESC LIMIT 1").fetchone()
            # Every sweep, even with nothing new: after a quiet spell the next
            # row must count from now, not from the last busy moment.
            if last is None or now > float(last[0]):
                hi = max(hi, int(last[1]) if last else 0)
                if last is not None and hi == int(last[1]):
                    # Only the newest time a given highest id was seen matters.
                    conn.execute("UPDATE watermarks SET at=? WHERE at=?", (now, float(last[0])))
                else:
                    conn.execute("INSERT INTO watermarks (at, hi_id) VALUES (?, ?)", (now, hi))
            elif now < float(last[0]) - 1:
                logger.warning("disappearing: the clock went backwards; deadlines may be late by the jump")
            # Drop watermarks no timer can still need, keeping one older anchor.
            conn.execute(
                "DELETE FROM watermarks WHERE at < (SELECT MAX(at) FROM watermarks WHERE at < ?)",
                (now - WATERMARK_KEEP_S,),
            )
            return [(float(a), int(h)) for a, h in conn.execute(
                "SELECT at, hi_id FROM watermarks ORDER BY at"
            ).fetchall()]
        finally:
            conn.close()

    @staticmethod
    def deletable_upto(watermarks: List[Tuple[float, int]], cutoff: float) -> int:
        """Highest messages.id that is due when rows stored before ``cutoff`` are.

        A row with id in (hi[k], hi[k+1]] was stored between at[k] and
        at[k+1]. Counting it from the earlier bound, at[k], means it is due
        once at[k] <= cutoff: never later than the truth, at most one sweep
        early.
        """
        j = -1
        for i, (at, _) in enumerate(watermarks):
            if at <= cutoff:
                j = i
            else:
                break
        if j < 0:
            return 0
        if j == len(watermarks) - 1:
            return INF_ID
        return watermarks[j + 1][1]

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

    @staticmethod
    def stored_after(watermarks: List[Tuple[float, int]], row_id: int) -> float:
        """The latest time we know row ``row_id`` did not exist yet."""
        at = watermarks[0][0] if watermarks else 0.0
        for a, h in watermarks:
            if h < row_id:
                at = a
            else:
                break
        return at

    def _sweep_chat(
        self, conn: sqlite3.Connection, tables: Set[str], platform: str, chat_id: str,
        now: float, watermarks: List[Tuple[float, int]], owners: Dict[str, Tuple[str, str]],
        result: SweepResult,
    ) -> None:
        all_spans = self.spans(platform, chat_id)
        spans = [s for s in all_spans if s.ttl_s > 0]
        if not spans:
            return
        # Rows stored while the timer was off are never deleted, not even by
        # the compaction cleanup below.
        keep = [(s.from_id, s.to_id) for s in all_spans if s.ttl_s <= 0]
        keep_sql = "".join(" AND NOT (id > ? AND id <= ?)" for _ in keep)
        keep_args = [x for pair in keep for x in pair]
        sessions = self._chat_sessions(conn, platform, chat_id)
        if not sessions:
            return
        if spans[-1].until is None:
            for sid in sessions:
                owners[sid] = (platform, chat_id)
        sids = sorted(sessions)
        marks = ",".join("?" * len(sids))
        floor_id = min(s.from_id for s in spans)

        self._detect_copies(conn, sessions)

        touched: Set[str] = set()
        deleted = 0
        conn.execute("BEGIN IMMEDIATE")
        try:
            for span in spans:
                cutoff = now - span.ttl_s + early_margin(span.ttl_s)
                if "async_delegations" in tables:
                    hi_t = cutoff if span.until is None else min(cutoff, span.until)
                    cur = conn.execute(
                        f"DELETE FROM async_delegations WHERE (origin_session IN ({marks}) "
                        f"OR parent_session_id IN ({marks})) AND dispatched_at >= ? AND dispatched_at < ?",
                        (*sids, *sids, span.since, hi_t),
                    )
                    result.delegations_deleted += cur.rowcount or 0
                upto = min(span.to_id, self.deletable_upto(watermarks, cutoff))
                if upto <= span.from_id:
                    continue
                due = conn.execute(
                    f"SELECT session_id, MAX(compacted) FROM messages WHERE session_id IN ({marks}) "
                    f"AND id > ? AND id <= ? GROUP BY session_id",
                    (*sids, span.from_id, upto),
                ).fetchall()
                if not due:
                    continue
                touched.update(r[0] for r in due)
                cur = conn.execute(
                    f"DELETE FROM messages WHERE session_id IN ({marks}) AND id > ? AND id <= ?",
                    (*sids, span.from_id, upto),
                )
                deleted += cur.rowcount or 0
                # In-place compaction wrote copies of these rows (summaries,
                # the kept tail) as new rows. Once archived content is due,
                # those go too, here and in any session branched from this one.
                compacted = [r[0] for r in due if r[1]]
                if compacted:
                    reset = set(compacted)
                    frontier = list(compacted)
                    while frontier:
                        kids = [s for s, (_, parent) in sessions.items()
                                if parent in frontier and s not in reset]
                        reset.update(kids)
                        frontier = kids
                    rmarks = ",".join("?" * len(reset))
                    cur = conn.execute(
                        f"DELETE FROM messages WHERE session_id IN ({rmarks}) AND id > ?{keep_sql}",
                        (*sorted(reset), floor_id, *keep_args),
                    )
                    deleted += cur.rowcount or 0
                    touched.update(reset)

            # Rows written by a rewrite or branch carry the age of what they copied.
            for sid, lo, hi, at, ttl in self._copies_for(sids):
                if at + ttl - early_margin(ttl) > now:
                    continue
                was_compacted = conn.execute(
                    "SELECT MAX(compacted) FROM messages WHERE session_id=? AND id > ? AND id <= ?",
                    (sid, lo, hi),
                ).fetchone()[0]
                cur = conn.execute(
                    "DELETE FROM messages WHERE session_id=? AND id > ? AND id <= ?", (sid, lo, hi)
                )
                if cur.rowcount:
                    deleted += cur.rowcount
                    touched.add(sid)
                if was_compacted:
                    # The copies were themselves compacted: the summary of
                    # them is a newer row and goes with them.
                    reset = {sid}
                    frontier = [sid]
                    while frontier:
                        kids = [s for s, (_, parent) in sessions.items()
                                if parent in frontier and s not in reset]
                        reset.update(kids)
                        frontier = kids
                    rmarks = ",".join("?" * len(reset))
                    cur = conn.execute(
                        f"DELETE FROM messages WHERE session_id IN ({rmarks}) AND id > ?{keep_sql}",
                        (*sorted(reset), min(floor_id, lo), *keep_args),
                    )
                    deleted += cur.rowcount or 0
                    touched.update(reset)
                self._drop_copy(sid, lo)

            if touched:
                tmarks = ",".join("?" * len(touched))
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

        self._remember_sessions(conn, sessions, spans, watermarks)

        if touched:
            result.messages_deleted += deleted
            result.touched_session_keys.update(
                sessions[s][0] for s in touched if s in sessions and sessions[s][0]
            )
            # Snapshots and dumps copy the whole conversation; state.db is canonical.
            for sid in touched:
                result.session_files_deleted += self._remove_session_copies(sid)

    # ------------------------------------------------------------------ copies

    def _seen(self, ledger: sqlite3.Connection, sid: str):
        return ledger.execute(
            "SELECT last_max_id, last_count, oldest_at, oldest_ttl FROM sessions_seen WHERE session_id=?",
            (sid,),
        ).fetchone()

    def _detect_copies(self, conn: sqlite3.Connection, sessions: Dict[str, Tuple[str, Optional[str]]]) -> None:
        """Spot rows that are copies of older content, whatever path made them.

        * A tracked session lost rows we didn't delete, and gained new ones:
          it was rewritten (/retry and the like). The new rows may hold any of
          the old content.
        * A session we haven't seen before branched from one we have (/branch,
          rotation compaction, forks). Its rows may hold the parent's content.
        Either way the new rows are given the age of the oldest content the
        source still held.
        """
        ledger = self._ledger()
        try:
            for sid, (_, parent) in sessions.items():
                cur_max = int(conn.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id=?", (sid,)
                ).fetchone()[0])
                seen = self._seen(ledger, sid)
                if seen is None or int(seen[0]) == 0:
                    # New, or seen only while still empty (rotation creates the
                    # child before writing to it). Inherit the parent's age.
                    pseen = self._seen(ledger, parent) if parent else None
                    inherit = None
                    if seen is not None and seen[2] is not None:
                        inherit = (float(seen[2]), int(seen[3]))
                    elif pseen is not None and pseen[2] is not None:
                        inherit = (float(pseen[2]), int(pseen[3]))
                    if inherit is None:
                        continue
                    if cur_max > 0:
                        self._add_copy(ledger, sid, 0, cur_max, *inherit)
                    else:
                        ledger.execute(
                            "INSERT OR IGNORE INTO sessions_seen (session_id, last_max_id, last_count, "
                            "oldest_at, oldest_ttl) VALUES (?, 0, 0, ?, ?)",
                            (sid, inherit[0], inherit[1]),
                        )
                    continue
                last_max, last_count, oldest_at, oldest_ttl = seen
                if oldest_at is None or cur_max <= last_max:
                    continue
                still = int(conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id=? AND id <= ?", (sid, last_max)
                ).fetchone()[0])
                if still < last_count:
                    self._add_copy(ledger, sid, last_max, cur_max, float(oldest_at), int(oldest_ttl))
        finally:
            ledger.close()

    @staticmethod
    def _add_copy(ledger: sqlite3.Connection, sid: str, lo: int, hi: int, at: float, ttl: int) -> None:
        ledger.execute(
            "INSERT INTO copies (session_id, lo_id, hi_id, at, ttl_s) VALUES (?,?,?,?,?) "
            "ON CONFLICT(session_id, lo_id) DO UPDATE SET hi_id=MAX(hi_id, excluded.hi_id), "
            "at=MIN(at, excluded.at), ttl_s=MIN(ttl_s, excluded.ttl_s)",
            (sid, lo, hi, at, ttl),
        )

    def _copies_for(self, sids: Sequence[str]) -> List[Tuple[str, int, int, float, int]]:
        if not sids:
            return []
        ledger = self._ledger()
        try:
            marks = ",".join("?" * len(sids))
            return [(r[0], int(r[1]), int(r[2]), float(r[3]), int(r[4])) for r in ledger.execute(
                f"SELECT session_id, lo_id, hi_id, at, ttl_s FROM copies WHERE session_id IN ({marks})",
                list(sids),
            ).fetchall()]
        finally:
            ledger.close()

    def _drop_copy(self, sid: str, lo: int) -> None:
        ledger = self._ledger()
        try:
            ledger.execute("DELETE FROM copies WHERE session_id=? AND lo_id=?", (sid, lo))
        finally:
            ledger.close()

    def _remember_sessions(
        self, conn: sqlite3.Connection, sessions: Dict[str, Tuple[str, Optional[str]]],
        spans: List[TimerSpan], watermarks: List[Tuple[float, int]],
    ) -> None:
        """Record each session's shape and the age of the oldest content it holds."""
        ledger = self._ledger()
        try:
            for sid in sessions:
                cur_max = int(conn.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id=?", (sid,)
                ).fetchone()[0])
                prev = self._seen(ledger, sid)
                last_max = max(cur_max, int(prev[0]) if prev else 0)
                count = int(conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id=? AND id <= ?", (sid, last_max)
                ).fetchone()[0])
                best: Optional[Tuple[float, int]] = None
                for span in spans:
                    row = conn.execute(
                        "SELECT MIN(id) FROM messages WHERE session_id=? AND id > ? AND id <= ?",
                        (sid, span.from_id, span.to_id),
                    ).fetchone()
                    if row and row[0] is not None:
                        cand = (self.stored_after(watermarks, int(row[0])), span.ttl_s)
                        if best is None or cand[0] + cand[1] < best[0] + best[1]:
                            best = cand
                for _, lo, hi, at, ttl in self._copies_for([sid]):
                    left = conn.execute(
                        "SELECT 1 FROM messages WHERE session_id=? AND id > ? AND id <= ? LIMIT 1",
                        (sid, lo, hi),
                    ).fetchone()
                    if left and (best is None or at + ttl < best[0] + best[1]):
                        best = (at, ttl)
                if cur_max == 0 and prev is not None and prev[2] is not None:
                    continue  # an empty child still waiting for its inherited rows
                ledger.execute(
                    "INSERT INTO sessions_seen (session_id, last_max_id, last_count, oldest_at, oldest_ttl) "
                    "VALUES (?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET "
                    "last_max_id=excluded.last_max_id, last_count=excluded.last_count, "
                    "oldest_at=excluded.oldest_at, oldest_ttl=excluded.oldest_ttl",
                    (sid, last_max, count, best[0] if best else None, best[1] if best else None),
                )
        finally:
            ledger.close()

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
