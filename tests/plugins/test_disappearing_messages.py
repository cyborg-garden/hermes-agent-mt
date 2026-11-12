"""disappearing_messages: deletes what we keep from a chat on Signal's timer.

Uses a real state.db (SessionDB creates the schema, FTS and triggers) and a
fake clock, so deadlines are exact and no test sleeps. Rows are timed by when
we stored them (row id plus sweep watermarks), never by their own timestamp,
so several tests store rows with forged timestamps on purpose.
"""

import logging
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from plugins.disappearing_messages.retention import INF_ID, Retention, early_margin

GROUP = "group:testgroupid"
OTHER = "group:othergroup"
KEY = "agent:main:signal:group:testgroupid"
T0 = 1_800_000_000.0
YEAR_2100 = 4_102_444_800.0


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def env(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    db = SessionDB(home / "state.db")
    clock = Clock()
    r = Retention(home=home, state_db=home / "state.db", clock=clock)
    return SimpleNamespace(home=home, db=db, clock=clock, r=r)


def _session(env, sid, chat_id=GROUP, parent=None):
    key = f"agent:main:signal:group:{chat_id.split(':', 1)[1]}" if chat_id else None
    env.db.create_session(sid, "signal", chat_id=chat_id or None, session_key=key)
    if parent:
        with sqlite3.connect(env.home / "state.db") as c:
            c.execute("UPDATE sessions SET parent_session_id=? WHERE id=?", (parent, sid))


def _msg(env, sid, text, role="user", stamp=None):
    """Store a row. ``stamp`` overrides its timestamp (a sender can forge it)."""
    env.db.append_message(sid, role, content=text)
    if stamp is not None:
        with sqlite3.connect(env.home / "state.db") as c:
            c.execute("UPDATE messages SET timestamp=? WHERE id=(SELECT MAX(id) FROM messages)", (stamp,))


def _texts(env, sid=None):
    with sqlite3.connect(env.home / "state.db") as c:
        q = "SELECT content FROM messages" + (" WHERE session_id=?" if sid else "") + " ORDER BY id"
        return [r[0] for r in c.execute(q, (sid,) if sid else ())]


def _fts_hits(env, word):
    with sqlite3.connect(env.home / "state.db") as c:
        return c.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?", (word,)
        ).fetchone()[0]


def _at(env, t, sweep=True):
    env.clock.t = t
    return env.r.sweep() if sweep else None


# ---------------------------------------------------------------- timing rules

def test_early_margin_is_small_and_never_eats_a_short_timer():
    assert early_margin(30) == 3
    assert early_margin(7 * 86400) == 30


def test_deletable_upto_counts_rows_from_the_earlier_watermark():
    wm = [(100.0, 10), (200.0, 20), (300.0, 30)]
    assert Retention.deletable_upto(wm, 99.0) == 0
    assert Retention.deletable_upto(wm, 100.0) == 20   # rows 11-20 stored after 100
    assert Retention.deletable_upto(wm, 250.0) == 30
    assert Retention.deletable_upto(wm, 300.0) == INF_ID


# ---------------------------------------------------------------- ledger

def test_never_disappearing_chat_is_not_recorded(env):
    assert env.r.observe("signal", GROUP, 0, sender="moana") == 0
    assert env.r.chats() == []


def test_unreadable_timer_keeps_the_previous_one(env):
    env.r.observe("signal", GROUP, 3600, sender="moana")
    assert env.r.observe("signal", GROUP, None, sender="moana") == 3600


def test_unreadable_timer_on_unknown_chat_is_not_guessed(env):
    assert env.r.observe("signal", GROUP, None, sender="moana") is None
    assert env.r.chats() == []


# ---------------------------------------------------------------- deletion

def test_messages_and_replies_go_when_the_timer_runs_out(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _msg(env, "s1", "kiwiwordone")
    _msg(env, "s1", "replywordone", role="assistant")
    _at(env, T0 + 30)                        # a sweep sees the rows
    assert _at(env, T0 + 3600 - 60).messages_deleted == 0
    assert len(_texts(env, "s1")) == 2
    assert _at(env, T0 + 3600).messages_deleted == 2
    assert _texts(env, "s1") == []
    assert _fts_hits(env, "kiwiwordone") == 0


def test_never_deleted_late_even_between_sweeps(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 600, sender="moana")
    _at(env, T0 + 30)
    _at(env, T0 + 60, sweep=False)
    _msg(env, "s1", "storedat60")           # stored after the T0+30 watermark
    # Due at most 600s after T0+30, the last time we know it wasn't there.
    _at(env, T0 + 30 + 600 - early_margin(600))
    assert _texts(env, "s1") == []


@pytest.mark.parametrize("stamp", [YEAR_2100, T0 + 7 * 86400, 0.0, T0 - 90 * 86400])
def test_forged_timestamps_do_not_change_the_deadline(env, stamp):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 60, sender="moana")
    _msg(env, "s1", "forged", stamp=stamp)
    _at(env, T0 + 30)
    _at(env, T0 + 1000)
    assert _texts(env, "s1") == []


def test_messages_before_the_timer_was_turned_on_are_kept(env):
    _session(env, "s1")
    _msg(env, "s1", "before")
    env.r.observe("signal", GROUP, 60, sender="moana")
    _msg(env, "s1", "after")
    _at(env, T0 + 30)
    _at(env, T0 + 1000)
    assert _texts(env, "s1") == ["before"]


def test_one_stray_zero_does_not_stop_deletion(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 60, sender="moana")
    assert env.r.observe("signal", GROUP, 0, sender="moana") == 60   # stale client
    _msg(env, "s1", "fromstale")
    env.r.observe("signal", GROUP, 60, sender="moana")
    _at(env, T0 + 30)
    _at(env, T0 + 1000)
    assert _texts(env, "s1") == []


def test_two_zeros_turn_it_off_only_from_the_second(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 60, sender="moana")
    _at(env, T0 + 5, sweep=False)
    env.r.observe("signal", GROUP, 0, sender="moana")
    _msg(env, "s1", "betweenzeros")          # still under the timer
    _at(env, T0 + 8, sweep=False)
    assert env.r.observe("signal", GROUP, 0, sender="moana") == 0
    _msg(env, "s1", "afteroff")
    _at(env, T0 + 30)
    _at(env, T0 + 10_000)
    assert _texts(env, "s1") == ["afteroff"]


def test_timer_change_applies_to_new_messages_only(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 86400, sender="moana")
    _msg(env, "s1", "longtimer")
    _at(env, T0 + 100)
    env.r.observe("signal", GROUP, 60, sender="moana")
    _msg(env, "s1", "shorttimer")
    _at(env, T0 + 130)
    _at(env, T0 + 1000)
    assert _texts(env, "s1") == ["longtimer"]
    _at(env, T0 + 86400)
    assert _texts(env, "s1") == []


def test_turning_it_on_again_does_not_touch_the_off_period(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 60, sender="moana")
    _msg(env, "s1", "on1")
    env.r.observe("signal", GROUP, 0, sender="moana")
    env.r.observe("signal", GROUP, 0, sender="moana")
    _msg(env, "s1", "offperiod")
    _at(env, T0 + 30)
    _at(env, T0 + 100, sweep=False)
    env.r.observe("signal", GROUP, 60, sender="moana")
    _msg(env, "s1", "on2")
    _at(env, T0 + 130)
    _at(env, T0 + 10_000)
    assert _texts(env, "s1") == ["offperiod"]


def test_other_chats_are_untouched(env):
    _session(env, "s1")
    _session(env, "s2", chat_id=OTHER)
    env.r.observe("signal", GROUP, 60, sender="moana")
    _msg(env, "s1", "mine")
    _msg(env, "s2", "theirs")
    _at(env, T0 + 30)
    _at(env, T0 + 1000)
    assert _texts(env, "s2") == ["theirs"]
    assert _texts(env, "s1") == []


def test_compaction_copies_go_with_the_content_they_copy(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _msg(env, "s1", "originalwords")
    _at(env, T0 + 30)
    # Activity elsewhere moves the watermark on, so the copies below are
    # clearly newer rows that the ordinary rule would keep for another hour.
    _session(env, "s2", chat_id=OTHER)
    _msg(env, "s2", "elsewhere")
    _at(env, T0 + 3000)
    # Compaction: archive the originals, write a summary and a copy.
    with sqlite3.connect(env.home / "state.db") as c:
        c.execute("UPDATE messages SET active=0, compacted=1 WHERE session_id='s1'")
    _msg(env, "s1", "[PRIOR CONTEXT — for reference only] they said originalwords")
    _msg(env, "s1", "kept tail copy of originalwords")
    _at(env, T0 + 3030)
    _at(env, T0 + 3600)                      # originals due; copies are newer rows
    assert _texts(env, "s1") == []
    assert _fts_hits(env, "originalwords") == 0


def test_compaction_cleanup_keeps_the_off_period(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _msg(env, "s1", "secret")
    env.r.observe("signal", GROUP, 0, sender="moana")
    env.r.observe("signal", GROUP, 0, sender="moana")
    _msg(env, "s1", "keptforever")
    with sqlite3.connect(env.home / "state.db") as c:
        c.execute("UPDATE messages SET active=0, compacted=1 WHERE content='secret'")
    _at(env, T0 + 30)
    _at(env, T0 + 10_000)
    assert _texts(env, "s1") == ["keptforever"]


def test_branched_sessions_are_reset_with_their_parent(env):
    _session(env, "s1")
    _session(env, "s2", chat_id=OTHER)
    env.r.observe("signal", GROUP, 60, sender="moana")
    _msg(env, "s1", "parentwords")
    _msg(env, "s2", "elsewhere1")
    _at(env, T0 + 5)                         # watermark after the parent row
    # Rotation-mode compaction: archive the parent, start a child holding a copy.
    with sqlite3.connect(env.home / "state.db") as c:
        c.execute("UPDATE messages SET active=0, compacted=1 WHERE session_id='s1'")
    _msg(env, "s2", "elsewhere2")
    _at(env, T0 + 40)                        # watermark before the child row
    _session(env, "s1c", chat_id="", parent="s1")
    _msg(env, "s1c", "copy of parentwords in a rotated child")
    _at(env, T0 + 60)                        # parent due; the child row alone is not
    assert _texts(env, "s1") == []
    assert _texts(env, "s1c") == []


def test_title_is_cleared_and_session_copies_removed(env):
    _session(env, "s1")
    with sqlite3.connect(env.home / "state.db") as c:
        c.execute("UPDATE sessions SET title='about kiwi' WHERE id='s1'")
    (env.home / "sessions").mkdir()
    (env.home / "logs").mkdir()
    snap = env.home / "sessions" / "s1.jsonl"
    dump = env.home / "logs" / "request_dump_s1_20260101.json"
    snap.write_text("x")
    dump.write_text("x")
    env.r.observe("signal", GROUP, 60, sender="moana")
    _msg(env, "s1", "hello")
    _at(env, T0 + 30)
    _at(env, T0 + 1000)
    assert not snap.exists() and not dump.exists()
    with sqlite3.connect(env.home / "state.db") as c:
        assert c.execute("SELECT title FROM sessions WHERE id='s1'").fetchone()[0] is None


def test_attachments_are_deleted_on_their_deadline(env, tmp_path):
    f = tmp_path / "voice.m4a"
    f.write_bytes(b"audio")
    env.r.observe("signal", GROUP, 120, [str(f)], sender="moana")
    _at(env, T0 + 120 - early_margin(120) - 1)
    assert f.exists()
    assert _at(env, T0 + 120).files_deleted == 1
    assert not f.exists()


def test_attachments_in_ordinary_chats_are_left_alone(env, tmp_path):
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"img")
    env.r.observe("signal", GROUP, 0, [str(f)], sender="moana")
    _at(env, T0 + 10 ** 6)
    assert f.exists()


def test_deletions_survive_a_restart(env, tmp_path):
    _session(env, "s1")
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"pdf")
    env.r.observe("signal", GROUP, 60, [str(f)], sender="moana")
    _msg(env, "s1", "hi")
    _at(env, T0 + 30)
    env.clock.t = T0 + 1000
    fresh = Retention(home=env.home, state_db=env.home / "state.db", clock=env.clock)
    fresh.sweep()
    assert _texts(env, "s1") == [] and not f.exists()


def test_delegation_records_are_swept(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 60, sender="moana")
    _msg(env, "s1", "x")
    with sqlite3.connect(env.home / "state.db") as c:
        cols = [r[1] for r in c.execute("PRAGMA table_info(async_delegations)")]
        if not cols:
            pytest.skip("no async_delegations table")
        row = {"delegation_id": "d1", "origin_session": "s1", "state": "done",
               "dispatched_at": T0 + 2, "updated_at": T0 + 2}
        c.execute(f"INSERT INTO async_delegations ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                  tuple(row.values()))
    _at(env, T0 + 30)
    assert _at(env, T0 + 1000).delegations_deleted == 1


def test_sweep_reports_sessions_to_drop_from_memory(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 60, sender="moana")
    _msg(env, "s1", "x")
    _at(env, T0 + 30)
    assert _at(env, T0 + 1000).touched_session_keys == {KEY}
    assert env.r.session_owner == {"s1": ("signal", GROUP)}


def test_observe_never_waits_on_a_running_sweep(env):
    env.r._sweep_lock.acquire()
    try:
        assert env.r.observe("signal", GROUP, 60, sender="moana") == 60
        assert env.r.sweep().messages_deleted == 0  # a second sweep just skips
    finally:
        env.r._sweep_lock.release()


def test_observe_returns_quickly_while_state_db_is_locked(env):
    import time as _t
    _session(env, "s1")
    blocker = sqlite3.connect(env.home / "state.db", isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        start = _t.monotonic()
        assert env.r.observe("signal", GROUP, 60, sender="moana") == 60
        assert _t.monotonic() - start < 2
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()


# ---------------------------------------------------------------- plugin hooks

@pytest.fixture
def plugin(env, monkeypatch):
    import plugins.disappearing_messages as dm
    monkeypatch.setattr(dm, "_retention", env.r)
    monkeypatch.setattr(dm, "_filter", dm.RedactFilter())
    return dm


def _event(chat_id=GROUP, ttl=3600, media=()):
    return SimpleNamespace(
        source=SimpleNamespace(platform=SimpleNamespace(value="signal"), chat_id=chat_id),
        metadata={"signal": {"expires_in_seconds": ttl, "timestamp_ms": int(YEAR_2100 * 1000)}},
        media_urls=list(media),
    )


def test_dispatch_records_the_chat_and_never_blocks(plugin, env):
    assert plugin._on_pre_gateway_dispatch(event=_event()) is None
    assert env.r.is_disappearing("signal", GROUP)


def test_dispatch_ignores_events_without_the_timer_field(plugin, env):
    ev = _event()
    ev.metadata = {}
    plugin._on_pre_gateway_dispatch(event=ev)
    assert env.r.chats() == []


def test_memory_writes_are_refused_in_a_disappearing_chat(plugin, env):
    _session(env, "s1")
    plugin._on_pre_gateway_dispatch(event=_event())
    res = plugin._on_pre_tool_call(tool_name="memory", args={"action": "add"}, session_id="s1")
    assert res and res["action"] == "block"
    assert plugin._on_pre_tool_call(tool_name="memory", args={"action": "read"}, session_id="s1") is None
    assert plugin._on_pre_tool_call(tool_name="web_search", args={}, session_id="s1") is None


def test_memory_writes_allowed_in_ordinary_chats(plugin, env):
    _session(env, "s2", chat_id=OTHER)
    assert plugin._on_pre_tool_call(tool_name="memory", args={"action": "add"}, session_id="s2") is None


def test_log_previews_are_blanked_for_disappearing_chats_only(plugin, env):
    _session(env, "s1")
    plugin._on_pre_gateway_dispatch(event=_event())
    env.r.sweep()                                  # fills the session map
    f = plugin._filter

    def fmt(msg, *args):
        rec = logging.LogRecord("gateway.run", logging.INFO, __file__, 1, msg, args, None)
        f.filter(rec)
        return rec.getMessage()

    line = fmt("inbound message: platform=%s user=%s chat=%s msg=%r reply_to_id=%s reply_to_text=%r",
               "signal", "Moana", GROUP, "secret words", None, "quoted secret")
    assert "secret" not in line and f"chat={GROUP}" in line
    turn = fmt("conversation turn: session=%s model=%s msg=%r", "s1", "kimi", "secret words")
    assert "secret" not in turn
    paren = fmt("delivery (chat=%s) msg=%r", GROUP, "secret words")
    assert "secret" not in paren
    other = fmt("inbound message: platform=%s chat=%s msg=%r", "signal", OTHER, "fine to keep")
    assert "fine to keep" in other


def test_log_filter_never_opens_state_db(plugin, env, monkeypatch):
    plugin._on_pre_gateway_dispatch(event=_event())

    def boom():
        raise AssertionError("log filter opened state.db")

    monkeypatch.setattr(env.r, "_state", boom)
    rec = logging.LogRecord("x", logging.INFO, __file__, 1,
                            "conversation turn: session=%s msg=%r", ("unknown-sid", "hi"), None)
    assert plugin._filter.filter(rec) is True


def test_only_one_sweeper_thread(plugin, monkeypatch):
    started = []
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self.name))
    real = threading.enumerate
    monkeypatch.setattr(threading, "enumerate", lambda: real())
    plugin._start_sweeper()
    fake = type("T", (), {"name": plugin.SWEEPER_THREAD_NAME, "is_alive": lambda self: True})()
    monkeypatch.setattr(threading, "enumerate", lambda: [fake])
    plugin._start_sweeper()
    assert started == [plugin.SWEEPER_THREAD_NAME]


# ---------------------------------------------------------------- adapter field

def test_adapter_parses_the_timer():
    from gateway.platforms.signal import parse_expires_in_seconds as p
    assert p(None) == 0
    assert p(0) == 0
    assert p(604800) == 604800
    assert p("3600") == 3600
    assert p(-1) is None
    assert p(True) is None
    assert p("soon") is None


# ---------------------------------------------------------------- copies and rewrites

def _replace_messages(env, sid):
    """What /retry does: delete every row in the session and re-insert them."""
    with sqlite3.connect(env.home / "state.db") as c:
        rows = c.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id", (sid,)).fetchall()
        c.execute("DELETE FROM messages WHERE session_id=?", (sid,))
    for role, content in rows:
        _msg(env, sid, content, role=role)


def test_retry_rewrite_does_not_reset_the_clock(env):
    _session(env, "s1")
    _session(env, "s2", chat_id=OTHER)
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _msg(env, "s1", "originalwords")
    _at(env, T0 + 30)
    _msg(env, "s2", "elsewhere")
    _at(env, T0 + 2970)                      # watermark moves on
    _at(env, T0 + 3000, sweep=False)
    _replace_messages(env, "s1")             # /retry: same words, new ids
    _at(env, T0 + 3030)                      # sweep notices the rewrite
    _at(env, T0 + 3600)
    assert _texts(env, "s1") == []


def test_repeated_retries_cannot_keep_content(env):
    _session(env, "s1")
    _session(env, "s2", chat_id=OTHER)
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _msg(env, "s1", "originalwords")
    _at(env, T0 + 30)
    for t in (1000, 2000, 3000):
        _msg(env, "s2", f"elsewhere{t}")
        _at(env, T0 + t)
        _replace_messages(env, "s1")
        _at(env, T0 + t + 30)
    _at(env, T0 + 3600)
    assert _texts(env, "s1") == []


def test_branch_copy_lasts_no_longer_than_its_source(env):
    _session(env, "s1")
    _session(env, "s2", chat_id=OTHER)
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _msg(env, "s1", "parentwords")
    _at(env, T0 + 30)
    _msg(env, "s2", "elsewhere")
    _at(env, T0 + 3000)
    # /branch: a child session holding a plain copy (compacted=0).
    _session(env, "s1b", chat_id="", parent="s1")
    _msg(env, "s1b", "copy of parentwords")
    _at(env, T0 + 3030)
    _at(env, T0 + 3600)
    assert _texts(env, "s1b") == []


def test_ordinary_new_messages_are_not_treated_as_copies(env):
    _session(env, "s1")
    _session(env, "s2", chat_id=OTHER)
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _msg(env, "s1", "first")
    _at(env, T0 + 30)
    _msg(env, "s2", "elsewhere")
    _at(env, T0 + 3000)
    _msg(env, "s1", "second")                # plain new turn, nothing removed
    _at(env, T0 + 3030)
    _at(env, T0 + 3600)
    assert _texts(env, "s1") == ["second"]


def test_quiet_spell_does_not_make_the_next_message_vanish(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _at(env, T0 + 30)
    for t in range(600, 7200, 600):         # two quiet hours of sweeps
        _at(env, T0 + t)
    _msg(env, "s1", "afterquiet")
    _at(env, T0 + 7230)
    _at(env, T0 + 7260)
    assert _texts(env, "s1") == ["afterquiet"]
    _at(env, T0 + 7200 + 3600)
    assert _texts(env, "s1") == []


def test_watermarks_stay_few_while_nothing_happens(env):
    for t in range(30, 30 * 200, 30):
        _at(env, T0 + t)
    with sqlite3.connect(env.r.ledger_path) as c:
        assert c.execute("SELECT COUNT(*) FROM watermarks").fetchone()[0] <= 2


def test_deleted_text_leaves_the_wal_file(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 60, sender="moana")
    _msg(env, "s1", "walsecretword")
    _at(env, T0 + 30)
    _at(env, T0 + 1000)
    wal = env.home / "state.db-wal"
    if wal.exists():
        assert b"walsecretword" not in wal.read_bytes()
    assert b"walsecretword" not in (env.home / "state.db").read_bytes()


def test_late_delegations_are_still_swept(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _msg(env, "s1", "x")
    _at(env, T0 + 30)
    _at(env, T0 + 3600)                       # the message goes
    with sqlite3.connect(env.home / "state.db") as c:
        if not [r for r in c.execute("PRAGMA table_info(async_delegations)")]:
            pytest.skip("no async_delegations table")
        row = {"delegation_id": "late", "origin_session": "s1", "state": "done",
               "dispatched_at": T0 + 3610, "updated_at": T0 + 3610}
        c.execute(f"INSERT INTO async_delegations ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                  tuple(row.values()))
    _at(env, T0 + 3610 + 3600)
    with sqlite3.connect(env.home / "state.db") as c:
        assert c.execute("SELECT COUNT(*) FROM async_delegations").fetchone()[0] == 0


def test_old_client_zeros_never_turn_it_off(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 60, sender="moana")
    for _ in range(5):
        assert env.r.observe("signal", GROUP, 0, sender="oldphone") == 60
    _msg(env, "s1", "fromoldphone")
    _at(env, T0 + 30)
    _at(env, T0 + 1000)
    assert _texts(env, "s1") == []


def test_merged_queued_events_keep_the_shortest_timer():
    from gateway.platforms.base import _merge_retention_metadata
    queued = SimpleNamespace(metadata={"signal": {"expires_in_seconds": 0, "timestamp_ms": 1}})
    newer = SimpleNamespace(metadata={"signal": {"expires_in_seconds": 3600, "timestamp_ms": 2}})
    _merge_retention_metadata(queued, newer)
    assert queued.metadata["signal"]["expires_in_seconds"] == 3600
    a = SimpleNamespace(metadata={"signal": {"expires_in_seconds": 86400}})
    _merge_retention_metadata(a, SimpleNamespace(metadata={"signal": {"expires_in_seconds": 60}}))
    assert a.metadata["signal"]["expires_in_seconds"] == 60
    b = SimpleNamespace(metadata={"signal": {"expires_in_seconds": 0}})
    _merge_retention_metadata(b, SimpleNamespace(metadata={"signal": {"expires_in_seconds": None}}))
    assert b.metadata["signal"]["expires_in_seconds"] is None


def test_skill_writes_are_refused_in_a_disappearing_chat(plugin, env):
    _session(env, "s1")
    plugin._on_pre_gateway_dispatch(event=_event())
    res = plugin._on_pre_tool_call(tool_name="skill_manage", args={"action": "create"}, session_id="s1")
    assert res and res["action"] == "block"


def test_memory_block_fails_closed(plugin, env, monkeypatch):
    def broken(_sid):
        raise RuntimeError("ledger unreadable")
    monkeypatch.setattr(env.r, "session_chat", broken)
    res = plugin._on_pre_tool_call(tool_name="memory", args={"action": "add"}, session_id="s1")
    assert res and res["action"] == "block"


def test_summary_of_copies_goes_with_them(env):
    _session(env, "s1")
    _session(env, "s2", chat_id=OTHER)
    env.r.observe("signal", GROUP, 86400, sender="moana")
    _msg(env, "s1", "SECRET-hour0")
    _at(env, T0 + 30)
    _msg(env, "s2", "e1")
    _at(env, T0 + 20 * 3600)
    _replace_messages(env, "s1")             # /retry at hour 20
    _at(env, T0 + 20 * 3600 + 30)
    _msg(env, "s2", "e2")
    _at(env, T0 + 22 * 3600)
    # in-place compaction at hour 22: archive the copies, write a summary
    with sqlite3.connect(env.home / "state.db") as c:
        c.execute("UPDATE messages SET active=0, compacted=1 WHERE session_id='s1'")
    _msg(env, "s1", "SUMMARY: SECRET-hour0")
    _at(env, T0 + 22 * 3600 + 30)
    _at(env, T0 + 24 * 3600)
    assert _texts(env, "s1") == []


def test_rotation_child_seen_while_empty_still_inherits(env):
    _session(env, "s1")
    _session(env, "s2", chat_id=OTHER)
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _msg(env, "s1", "parentwords")
    _at(env, T0 + 30)
    _msg(env, "s2", "e1")
    _at(env, T0 + 3000)
    _session(env, "s1c", chat_id="", parent="s1")   # created empty
    _at(env, T0 + 3030)                              # a sweep lands in the gap
    _msg(env, "s1c", "summary of parentwords")      # written at the next LLM call
    _at(env, T0 + 3060)
    _at(env, T0 + 3600)
    assert _texts(env, "s1c") == []


def test_text_merges_keep_the_shortest_timer():
    from gateway.platforms.base import merge_pending_message_event, MessageType
    first = SimpleNamespace(message_type=MessageType.TEXT, text="hi", media_urls=[], media_types=[],
                            source=SimpleNamespace(is_bot=False),
                            metadata={"signal": {"expires_in_seconds": 0}})
    second = SimpleNamespace(message_type=MessageType.TEXT, text="secret", media_urls=[], media_types=[],
                             source=SimpleNamespace(is_bot=False),
                             metadata={"signal": {"expires_in_seconds": 3600}})
    pending = {"k": first}
    merge_pending_message_event(pending, "k", second, merge_text=True)
    assert pending["k"].text.endswith("secret")
    assert pending["k"].metadata["signal"]["expires_in_seconds"] == 3600


def test_chat_that_fully_expired_keeps_new_messages(env):
    """Idle longer than the timer, everything expires, then people write again."""
    _session(env, "s1")
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _msg(env, "s1", "first")
    _msg(env, "s1", "reply", role="assistant")
    for t in range(180, 3 * 3600, 180):     # sweeps every 3 minutes, as live
        _at(env, T0 + t)
    assert _texts(env, "s1") == []
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _msg(env, "s1", "back again")
    _msg(env, "s1", "welcome back", role="assistant")
    for t in range(3 * 3600 + 30, 3 * 3600 + 1800, 180):
        _at(env, T0 + t)
    assert _texts(env, "s1") == ["back again", "welcome back"]


def test_person_notes_are_refused_in_a_disappearing_chat(plugin, env):
    _session(env, "s1")
    plugin._on_pre_gateway_dispatch(event=_event())
    res = plugin._on_pre_tool_call(tool_name="remember_person", args={"note": "likes x"}, session_id="s1")
    assert res and res["action"] == "block"


# ---------------------------------------------------------------- signal-cli's own copies

def test_signal_cli_attachment_ids_are_recorded_with_a_deadline(env):
    env.r.observe("signal", GROUP, 120, sender="moana", signal_attachment_ids=["Cu4DRu8FuI5cwprZyEeP"])
    with sqlite3.connect(env.r.ledger_path) as c:
        rows = c.execute("SELECT attachment_id, expires_at FROM signal_cli_files").fetchall()
    assert rows == [("Cu4DRu8FuI5cwprZyEeP", T0 + 120 - early_margin(120))]


def test_signal_cli_ids_from_ordinary_chats_are_not_recorded(env):
    env.r.observe("signal", GROUP, 0, sender="moana", signal_attachment_ids=["abcdefgh"])
    with sqlite3.connect(env.r.ledger_path) as c:
        assert c.execute("SELECT COUNT(*) FROM signal_cli_files").fetchone()[0] == 0


def test_the_same_attachment_keeps_its_earliest_deadline(env):
    env.r.observe("signal", GROUP, 3600, sender="moana", signal_attachment_ids=["sharedfile1"])
    env.r.observe("signal", OTHER, 60, sender="moana", signal_attachment_ids=["sharedfile1"])
    with sqlite3.connect(env.r.ledger_path) as c:
        assert c.execute("SELECT expires_at FROM signal_cli_files").fetchone()[0] == T0 + 60 - early_margin(60)


def test_dispatch_passes_attachment_ids_through(plugin, env):
    ev = _event(ttl=300)
    ev.metadata["signal"]["attachment_ids"] = ["voicenote123"]
    plugin._on_pre_gateway_dispatch(event=ev)
    with sqlite3.connect(env.r.ledger_path) as c:
        assert c.execute("SELECT attachment_id FROM signal_cli_files").fetchall() == [("voicenote123",)]


def test_merged_events_keep_every_attachment_id():
    from gateway.platforms.base import _merge_retention_metadata
    a = SimpleNamespace(metadata={"signal": {"expires_in_seconds": 60, "attachment_ids": ["one"]}})
    _merge_retention_metadata(a, SimpleNamespace(metadata={"signal": {"expires_in_seconds": 60, "attachment_ids": ["two"]}}))
    assert a.metadata["signal"]["attachment_ids"] == ["one", "two"]


def _host_sweeper():
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parents[2] / "plugins/disappearing_messages/host/sweep_signal_cli.py"
    spec = importlib.util.spec_from_file_location("sweep_signal_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _attachments(tmp_path, *names):
    data = tmp_path / "signal-data"
    (data / "attachments").mkdir(parents=True)
    for n in names:
        (data / "attachments" / n).write_bytes(b"x")
    return data


def test_host_sweeper_deletes_only_due_copies(env, tmp_path):
    mod = _host_sweeper()
    data = _attachments(tmp_path, "dueVoice01.m4a", "laterPhoto1.jpg", "ordinary001.png", "*")
    env.r.observe("signal", GROUP, 60, sender="moana", signal_attachment_ids=["dueVoice01"])
    env.r.observe("signal", GROUP, 86400, sender="moana", signal_attachment_ids=["laterPhoto1.jpg"])
    with sqlite3.connect(env.r.ledger_path) as c:
        c.execute("INSERT INTO signal_cli_files VALUES ('*', ?)", (T0,))   # never a valid id
    _at(env, T0 + 1000)                      # the plugin publishes what's due
    due, deleted = mod.sweep(str(data), str(env.home))
    left = sorted(p.name for p in (data / "attachments").iterdir())
    assert left == ["*", "laterPhoto1.jpg", "ordinary001.png"] and deleted == 1


def test_host_confirms_and_the_plugin_forgets(env, tmp_path):
    mod = _host_sweeper()
    data = _attachments(tmp_path, "dueVoice01.m4a")
    env.r.observe("signal", GROUP, 60, sender="moana", signal_attachment_ids=["dueVoice01"])
    _at(env, T0 + 1000)
    mod.sweep(str(data), str(env.home))
    _at(env, T0 + 1030)                      # plugin reads the confirmation
    with sqlite3.connect(env.r.ledger_path) as c:
        assert c.execute("SELECT COUNT(*) FROM signal_cli_files").fetchone()[0] == 0


def test_unconfirmed_copies_are_not_forgotten_after_a_week(env, tmp_path):
    env.r.observe("signal", GROUP, 60, sender="moana", signal_attachment_ids=["dueVoice01"])
    _at(env, T0 + 30 * 86400)                 # host sweeper never ran
    with sqlite3.connect(env.r.ledger_path) as c:
        assert c.execute("SELECT COUNT(*) FROM signal_cli_files").fetchone()[0] == 1
    import json as _json
    due = _json.loads((env.r.ledger_path.parent / "signal_cli_due.json").read_text())["ids"]
    assert due == ["dueVoice01"]


def test_a_missing_host_sweeper_is_warned_about(env, caplog):
    env.r.observe("signal", GROUP, 60, sender="moana", signal_attachment_ids=["dueVoice01"])
    with caplog.at_level(logging.WARNING):
        _at(env, T0 + 2 * 86400)
    assert "sweep_signal_cli" in caplog.text


def test_host_matches_ids_with_or_without_an_extension(env, tmp_path):
    mod = _host_sweeper()
    data = _attachments(tmp_path, "plainId0001", "withExt0001.m4a")
    env.r.observe("signal", GROUP, 60, sender="moana", signal_attachment_ids=["plainId0001.jpg", "withExt0001"])
    _at(env, T0 + 1000)
    assert mod.sweep(str(data), str(env.home))[1] == 2


def test_host_reads_every_agent_home(tmp_path):
    mod = _host_sweeper()
    import json as _json
    data = _attachments(tmp_path, "fromAgentA1.m4a", "fromAgentB1.jpg", "keepThis001.png")
    for name, aid in (("agent-a", "fromAgentA1"), ("agent-b", "fromAgentB1")):
        d = tmp_path / "homes" / f".hermes-{name}" / "disappearing"
        d.mkdir(parents=True)
        (d / "signal_cli_due.json").write_text(_json.dumps({"ids": [aid]}))
    mod.sweep(str(data), str(tmp_path / "homes" / ".hermes*"))
    assert sorted(p.name for p in (data / "attachments").iterdir()) == ["keepThis001.png"]


def test_host_sweeper_dry_run_deletes_nothing(env, tmp_path):
    mod = _host_sweeper()
    data = _attachments(tmp_path, "dueVoice01.m4a")
    env.r.observe("signal", GROUP, 60, sender="moana", signal_attachment_ids=["dueVoice01"])
    _at(env, T0 + 1000)
    assert mod.sweep(str(data), str(env.home), dry_run=True) == (1, 1)
    assert (data / "attachments" / "dueVoice01.m4a").exists()


def test_attachment_only_messages_are_recorded_without_a_turn(plugin, env, monkeypatch):
    import gateway.platforms.signal as sig
    monkeypatch.setattr(sig, "RETENTION_LISTENERS", [])
    plugin._listen_to_signal()
    plugin._listen_to_signal()                # a reload adds no second listener
    assert len(sig.RETENTION_LISTENERS) == 1
    sig._notify_retention(GROUP, "moana", {"expires_in_seconds": 60, "attachment_ids": ["bigVideo001"]})
    with sqlite3.connect(env.r.ledger_path) as c:
        assert c.execute("SELECT attachment_id FROM signal_cli_files").fetchall() == [("bigVideo001",)]



def test_an_old_confirmation_cannot_wipe_a_readded_row(env, tmp_path):
    mod = _host_sweeper()
    data = _attachments(tmp_path, "reusedId001.m4a")
    env.r.observe("signal", GROUP, 60, sender="moana", signal_attachment_ids=["reusedId001"])
    _at(env, T0 + 1000)
    mod.sweep(str(data), str(env.home))       # confirms reusedId001
    # The same id turns up again before the plugin reads the confirmation.
    (data / "attachments" / "reusedId001.m4a").write_bytes(b"x")
    env.r.observe("signal", GROUP, 3600, sender="moana", signal_attachment_ids=["reusedId001"])
    with sqlite3.connect(env.r.ledger_path) as c:
        c.execute("UPDATE signal_cli_files SET expires_at=? WHERE attachment_id='reusedId001'", (T0 + 1000 + 3600,))
    _at(env, T0 + 1030)                       # plugin reads the old confirmation
    with sqlite3.connect(env.r.ledger_path) as c:
        assert c.execute("SELECT COUNT(*) FROM signal_cli_files").fetchone()[0] == 1
    assert not (env.r.ledger_path.parent / "signal_cli_done.json").exists()
    _at(env, T0 + 1000 + 3600)
    mod.sweep(str(data), str(env.home))
    assert not (data / "attachments" / "reusedId001.m4a").exists()
