"""disappearing_messages: deletes what we keep from a chat on Signal's timer.

Uses a real state.db (SessionDB creates the schema, FTS and triggers) and a
fake clock, so deadlines are exact and no test sleeps.
"""

import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from plugins.disappearing_messages.retention import Retention, early_margin, sent_at

GROUP = "group:testgroupid"
OTHER = "group:othergroup"
T0 = 1_800_000_000.0


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


def _msg(env, sid, text, at, role="user"):
    env.db.append_message(sid, role, content=text)
    with sqlite3.connect(env.home / "state.db") as c:
        c.execute(
            "UPDATE messages SET timestamp=? WHERE id=(SELECT MAX(id) FROM messages)", (at,)
        )


def _texts(env, sid=None):
    with sqlite3.connect(env.home / "state.db") as c:
        q = "SELECT content FROM messages" + (" WHERE session_id=?" if sid else "")
        return [r[0] for r in c.execute(q, (sid,) if sid else ())]


def _fts_hits(env, word):
    with sqlite3.connect(env.home / "state.db") as c:
        return c.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?", (word,)
        ).fetchone()[0]


# ---------------------------------------------------------------- timing rules

def test_sent_at_uses_signal_time_but_never_the_future():
    assert sent_at(int((T0 - 5) * 1000), T0) == T0 - 5
    assert sent_at(int((T0 + 5) * 1000), T0) == T0
    assert sent_at(None, T0) == T0
    assert sent_at(int((T0 - 3 * 86400) * 1000), T0) == T0 - 3 * 86400  # backlog counts
    assert sent_at(int((T0 - 60 * 86400) * 1000), T0) == T0  # implausible, ignored


def test_early_margin_is_small_and_never_eats_a_short_timer():
    assert early_margin(30) == 3
    assert early_margin(7 * 86400) == 30


# ---------------------------------------------------------------- ledger

def test_never_disappearing_chat_is_not_recorded(env):
    assert env.r.observe("signal", GROUP, 0, None) == 0
    assert env.r.chats() == []


def test_unreadable_timer_keeps_the_previous_one(env):
    env.r.observe("signal", GROUP, 3600, None)
    assert env.r.observe("signal", GROUP, None, None) == 3600
    assert env.r.current_ttl("signal", GROUP) == 3600


def test_unreadable_timer_on_unknown_chat_is_not_guessed(env):
    assert env.r.observe("signal", GROUP, None, None) is None
    assert env.r.chats() == []


# ---------------------------------------------------------------- deletion

def test_messages_and_replies_go_when_the_timer_runs_out(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 3600, int(T0 * 1000))
    _msg(env, "s1", "kiwiwordone", T0 + 1)
    _msg(env, "s1", "replywordone", T0 + 2, role="assistant")

    env.clock.t = T0 + 3600 - 60
    assert env.r.sweep().messages_deleted == 0
    assert len(_texts(env, "s1")) == 2

    env.clock.t = T0 + 3600
    res = env.r.sweep()
    assert res.messages_deleted == 2
    assert _texts(env, "s1") == []
    assert _fts_hits(env, "kiwiwordone") == 0  # search index follows


def test_messages_before_the_timer_was_turned_on_are_kept(env):
    _session(env, "s1")
    _msg(env, "s1", "before", T0 - 100)
    env.r.observe("signal", GROUP, 60, int(T0 * 1000))
    _msg(env, "s1", "after", T0 + 1)
    env.clock.t = T0 + 600
    env.r.sweep()
    assert _texts(env, "s1") == ["before"]


def test_turning_the_timer_off_keeps_new_messages_but_old_ones_still_go(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 60, int(T0 * 1000))
    _msg(env, "s1", "during", T0 + 1)
    env.clock.t = T0 + 10
    env.r.observe("signal", GROUP, 0, int((T0 + 10) * 1000))
    env.clock.t = T0 + 11
    env.r.observe("signal", GROUP, 0, int((T0 + 11) * 1000))
    _msg(env, "s1", "afteroff", T0 + 12)
    env.clock.t = T0 + 10_000
    env.r.sweep()
    assert _texts(env, "s1") == ["afteroff"]


def test_timer_change_applies_to_new_messages_only(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 86400, int(T0 * 1000))
    _msg(env, "s1", "longtimer", T0 + 1)
    env.clock.t = T0 + 100
    env.r.observe("signal", GROUP, 60, int((T0 + 100) * 1000))
    _msg(env, "s1", "shorttimer", T0 + 101)
    env.clock.t = T0 + 1000
    env.r.sweep()
    assert _texts(env, "s1") == ["longtimer"]
    env.clock.t = T0 + 86400
    env.r.sweep()
    assert _texts(env, "s1") == []


def test_two_changes_at_the_same_instant_both_count(env):
    env.r.observe("signal", GROUP, 60, int(T0 * 1000))
    env.r.observe("signal", GROUP, 120, int(T0 * 1000))
    assert [sp.ttl_s for sp in env.r.spans("signal", GROUP)] == [60, 120]


def test_one_stray_zero_does_not_stop_deletion(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 60, int(T0 * 1000))
    env.clock.t = T0 + 5
    assert env.r.observe("signal", GROUP, 0, int((T0 + 5) * 1000)) == 60  # stale client
    _msg(env, "s1", "fromstale", T0 + 6)
    env.clock.t = T0 + 7
    env.r.observe("signal", GROUP, 60, int((T0 + 7) * 1000))
    env.clock.t = T0 + 1000
    env.r.sweep()
    assert _texts(env, "s1") == []


def test_two_zeros_turn_it_off_from_the_first_one(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 60, int(T0 * 1000))
    env.clock.t = T0 + 5
    env.r.observe("signal", GROUP, 0, int((T0 + 5) * 1000))
    _msg(env, "s1", "betweenzeros", T0 + 6)
    env.clock.t = T0 + 8
    assert env.r.observe("signal", GROUP, 0, int((T0 + 8) * 1000)) == 0
    env.clock.t = T0 + 1000
    env.r.sweep()
    assert _texts(env, "s1") == ["betweenzeros"]


def test_message_sent_earlier_but_delivered_late_is_still_deleted(env):
    _session(env, "s1")
    env.clock.t = T0 + 10
    env.r.observe("signal", GROUP, 60, int((T0 + 10) * 1000))  # first seen
    env.r.observe("signal", GROUP, 60, int((T0 + 2) * 1000))   # sent earlier, arrived later
    _msg(env, "s1", "lateone", T0 + 2)
    _msg(env, "s1", "firstseen", T0 + 10)
    env.clock.t = T0 + 1000
    env.r.sweep()
    assert _texts(env, "s1") == []


def test_backlog_after_downtime_counts_from_send_time(env):
    _session(env, "s1")
    old = T0 - 2 * 86400
    env.r.observe("signal", GROUP, 3600, int(old * 1000))
    _msg(env, "s1", "backlog", old)
    env.r.sweep()
    assert _texts(env, "s1") == []


def test_summary_goes_when_what_it_summarises_goes(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 60, int(T0 * 1000))
    _msg(env, "s1", "originalwords", T0 + 1)
    _msg(env, "s1", "[CONTEXT COMPACTION — REFERENCE ONLY] they said originalwords", T0 + 50)
    env.clock.t = T0 + 70  # original due, summary's own timestamp not yet
    env.r.sweep()
    assert _texts(env, "s1") == []
    assert _fts_hits(env, "originalwords") == 0


def test_delegation_records_are_swept(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 60, int(T0 * 1000))
    _msg(env, "s1", "x", T0 + 1)
    with sqlite3.connect(env.home / "state.db") as c:
        cols = [r[1] for r in c.execute("PRAGMA table_info(async_delegations)")]
        if not cols:
            pytest.skip("no async_delegations table")
        row = {"delegation_id": "d1", "origin_session": "s1", "state": "done",
               "dispatched_at": T0 + 2, "updated_at": T0 + 2}
        c.execute(f"INSERT INTO async_delegations ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                  tuple(row.values()))
    env.clock.t = T0 + 1000
    assert env.r.sweep().delegations_deleted == 1


def test_sweep_reports_sessions_to_drop_from_memory(env):
    _session(env, "s1")
    env.r.observe("signal", GROUP, 60, int(T0 * 1000))
    _msg(env, "s1", "x", T0 + 1)
    env.clock.t = T0 + 1000
    keys = env.r.sweep().touched_session_keys
    assert keys == {"agent:main:signal:group:testgroupid"}


def test_observe_never_waits_on_a_running_sweep(env):
    env.r._sweep_lock.acquire()
    try:
        assert env.r.observe("signal", GROUP, 60, int(T0 * 1000)) == 60
        assert env.r.sweep().messages_deleted == 0  # a second sweep just skips
    finally:
        env.r._sweep_lock.release()


def test_other_chats_are_untouched(env):
    _session(env, "s1")
    _session(env, "s2", chat_id=OTHER)
    env.r.observe("signal", GROUP, 60, int(T0 * 1000))
    _msg(env, "s1", "mine", T0 + 1)
    _msg(env, "s2", "theirs", T0 + 1)
    env.clock.t = T0 + 1000
    env.r.sweep()
    assert _texts(env, "s2") == ["theirs"]
    assert _texts(env, "s1") == []


def test_compression_and_branch_children_are_covered(env):
    _session(env, "s1")
    _session(env, "s1c", chat_id="", parent="s1")
    env.r.observe("signal", GROUP, 60, int(T0 * 1000))
    _msg(env, "s1c", "summarywords", T0 + 5)
    env.clock.t = T0 + 1000
    env.r.sweep()
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
    env.r.observe("signal", GROUP, 60, int(T0 * 1000))
    _msg(env, "s1", "hello", T0 + 1)
    env.clock.t = T0 + 1000
    env.r.sweep()
    assert not snap.exists() and not dump.exists()
    with sqlite3.connect(env.home / "state.db") as c:
        assert c.execute("SELECT title FROM sessions WHERE id='s1'").fetchone()[0] is None


def test_attachments_are_deleted_on_their_deadline(env, tmp_path):
    f = tmp_path / "voice.m4a"
    f.write_bytes(b"audio")
    env.r.observe("signal", GROUP, 120, int(T0 * 1000), [str(f)])
    env.clock.t = T0 + 120 - early_margin(120) - 1
    env.r.sweep()
    assert f.exists()
    env.clock.t = T0 + 120
    assert env.r.sweep().files_deleted == 1
    assert not f.exists()


def test_attachments_in_ordinary_chats_are_left_alone(env, tmp_path):
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"img")
    env.r.observe("signal", GROUP, 0, int(T0 * 1000), [str(f)])
    env.clock.t = T0 + 10 ** 6
    env.r.sweep()
    assert f.exists()


def test_deletions_survive_a_restart(env, tmp_path):
    _session(env, "s1")
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"pdf")
    env.r.observe("signal", GROUP, 60, int(T0 * 1000), [str(f)])
    _msg(env, "s1", "hi", T0 + 1)
    env.clock.t = T0 + 1000
    fresh = Retention(home=env.home, state_db=env.home / "state.db", clock=env.clock)
    fresh.sweep()
    assert _texts(env, "s1") == [] and not f.exists()


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
        metadata={"signal": {"expires_in_seconds": ttl, "timestamp_ms": int(T0 * 1000)}},
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
    other = fmt("inbound message: platform=%s chat=%s msg=%r", "signal", OTHER, "fine to keep")
    assert "fine to keep" in other
    paren = fmt("delivery (chat=%s) msg=%r", GROUP, "secret words")
    assert "secret" not in paren


def test_only_one_sweeper_thread(plugin, monkeypatch):
    import threading
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
