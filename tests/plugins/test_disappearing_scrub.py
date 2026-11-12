"""scrub(): blank disappearing-chat text in a copy of state.db, keep everything else."""
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from plugins.disappearing_messages.retention import Retention
from plugins.disappearing_messages.scrub import scrub

T0 = 1_800_000_000.0
GROUP, OTHER = "group:moanagroup", "group:ordinary"


@pytest.fixture
def env(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    db = SessionDB(home / "state.db")
    clock = SimpleNamespace(t=T0)
    r = Retention(home=home, state_db=home / "state.db", clock=lambda: clock.t)
    for sid, chat in (("s1", GROUP), ("s2", OTHER)):
        db.create_session(sid, "signal", chat_id=chat, session_key=f"agent:main:signal:group:{chat[6:]}")
    return SimpleNamespace(home=home, db=db, r=r, clock=clock)


def _msg(env, sid, text, role="user"):
    env.db.append_message(sid, role, content=text)


def _snapshot(env, tmp_path):
    snap = tmp_path / "snap.db"
    src = sqlite3.connect(env.home / "state.db")
    dst = sqlite3.connect(snap)
    src.backup(dst)
    dst.close(); src.close()
    return snap


def _contents(path, sid):
    with sqlite3.connect(path) as c:
        return [r[0] for r in c.execute("SELECT content FROM messages WHERE session_id=? ORDER BY id", (sid,))]


def test_only_disappearing_text_is_blanked(env, tmp_path):
    _msg(env, "s1", "before the timer")
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _msg(env, "s1", "kiwisecretword")
    _msg(env, "s1", "a reply", role="assistant")
    env.r.observe("signal", GROUP, 0, sender="moana")
    env.r.observe("signal", GROUP, 0, sender="moana")
    _msg(env, "s1", "after the timer went off")
    _msg(env, "s2", "an ordinary chat keeps its words")
    snap = _snapshot(env, tmp_path)
    out = scrub(snap, env.home)
    assert out["rows"] == 2
    assert _contents(snap, "s1") == ["before the timer", "", "", "after the timer went off"]
    assert _contents(snap, "s2") == ["an ordinary chat keeps its words"]
    with sqlite3.connect(snap) as c:
        roles = [r[0] for r in c.execute("SELECT role FROM messages WHERE session_id='s1' ORDER BY id")]
    assert roles == ["user", "user", "assistant", "user"]       # shape kept
    assert b"kiwisecretword" not in snap.read_bytes()            # not in free pages or an index


def test_live_database_is_never_touched(env, tmp_path):
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _msg(env, "s1", "still live")
    scrub(_snapshot(env, tmp_path), env.home)
    assert _contents(env.home / "state.db", "s1") == ["still live"]


def test_copies_are_blanked_too(env, tmp_path):
    env.r.observe("signal", GROUP, 3600, sender="moana")
    env.r.observe("signal", GROUP, 0, sender="moana")
    env.r.observe("signal", GROUP, 0, sender="moana")
    _msg(env, "s1", "copy of on-period words")   # in an off-period id, but a known copy
    with sqlite3.connect(env.r.ledger_path) as c:
        hi = sqlite3.connect(env.home / "state.db").execute("SELECT MAX(id) FROM messages").fetchone()[0]
        c.execute("INSERT INTO copies VALUES ('s1', 0, ?, ?, 3600)", (hi, T0))
    snap = _snapshot(env, tmp_path)
    scrub(snap, env.home)
    assert _contents(snap, "s1") == [""]


def test_no_ledger_means_nothing_changes(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", "signal", chat_id=GROUP)
    db.append_message("s", "user", content="kept")
    assert scrub(tmp_path / "state.db", tmp_path / "nohome")["rows"] == 0
    assert _contents(tmp_path / "state.db", "s") == ["kept"]


def test_a_copy_without_the_expected_schema_is_refused(env, tmp_path):
    env.r.observe("signal", GROUP, 3600, sender="moana")
    bad = tmp_path / "bad.db"
    with sqlite3.connect(bad) as c:
        c.execute("CREATE TABLE messages (id INTEGER, session_id TEXT, body TEXT)")
        c.execute("CREATE TABLE sessions (id TEXT, source TEXT, chat_id TEXT, session_key TEXT, parent_session_id TEXT, title TEXT)")
    with pytest.raises(RuntimeError):
        scrub(bad, env.home)


def _row(path, sid, col):
    with sqlite3.connect(path) as c:
        return [r[0] for r in c.execute(f"SELECT {col} FROM messages WHERE session_id=? ORDER BY id", (sid,))]


def test_tool_calls_keep_their_shape_but_lose_their_arguments(env, tmp_path):
    import json
    env.r.observe("signal", GROUP, 3600, sender="moana")
    env.db.append_message("s1", "assistant", content="", tool_calls=[
        {"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": '{"q": "secretquery"}'}}
    ])
    snap = _snapshot(env, tmp_path)
    scrub(snap, env.home)
    calls = json.loads(_row(snap, "s1", "tool_calls")[0])
    assert calls[0]["id"] == "call_1" and calls[0]["function"]["name"] == "web_search"
    assert calls[0]["function"]["arguments"] == ""
    assert b"secretquery" not in snap.read_bytes()


def test_delegated_task_text_is_blanked(env, tmp_path):
    env.r.observe("signal", GROUP, 3600, sender="moana")
    with sqlite3.connect(env.home / "state.db") as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(async_delegations)")}
        if not cols:
            pytest.skip("no async_delegations table")
        row = {"delegation_id": "d1", "origin_session": "s1", "state": "done", "dispatched_at": T0 + 5,
               "updated_at": T0 + 5, "task_json": '{"goal": "secrettask"}', "result_json": '{"r": "secretresult"}'}
        row = {k: v for k, v in row.items() if k in cols}
        c.execute(f"INSERT INTO async_delegations ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                  tuple(row.values()))
    snap = _snapshot(env, tmp_path)
    out = scrub(snap, env.home)
    assert out["delegations"] == 1
    data = snap.read_bytes()
    assert b"secrettask" not in data and b"secretresult" not in data


def test_session_found_by_key_alone_and_dms(env, tmp_path):
    env.db.create_session("k1", "signal", chat_id=None, session_key=f"agent:main:signal:group:{GROUP[6:]}:moana")
    env.db.create_session("dm1", "signal", chat_id="+15550001", session_key="agent:main:signal:dm:+15550001")
    env.r.observe("signal", GROUP, 3600, sender="moana")
    env.r.observe("signal", "+15550001", 3600, sender="+15550001")
    _msg(env, "k1", "groupsecret")
    _msg(env, "dm1", "dmsecret")
    snap = _snapshot(env, tmp_path)
    scrub(snap, env.home)
    assert _contents(snap, "k1") == [""] and _contents(snap, "dm1") == [""]


def test_branched_children_titles_and_reasoning(env, tmp_path):
    env.r.observe("signal", GROUP, 3600, sender="moana")
    env.db.create_session("child", "signal")
    with sqlite3.connect(env.home / "state.db") as c:
        c.execute("UPDATE sessions SET parent_session_id='s1' WHERE id='child'")
        c.execute("UPDATE sessions SET title='about secrettitle' WHERE id='s1'")
    env.db.append_message("child", "assistant", content="childsecret", reasoning="secretthought")
    _msg(env, "s1", "x")
    snap = _snapshot(env, tmp_path)
    scrub(snap, env.home)
    assert _contents(snap, "child") == [""] and _row(snap, "child", "reasoning") == [None]
    with sqlite3.connect(snap) as c:
        assert c.execute("SELECT title FROM sessions WHERE id='s1'").fetchone()[0] is None


def test_unresolved_timer_change_never_blanks_pre_timer_history(env, tmp_path):
    _msg(env, "s1", "pretimer")
    _msg(env, "s2", "elsewhere")
    env.clock.t = T0 + 0.5
    env.r.sweep()                                  # watermark covers both rows
    with sqlite3.connect(env.r.ledger_path) as c:  # the id read timed out at turn-on
        c.execute("INSERT INTO timers (platform, chat_id, since, from_id, ttl_s) VALUES ('signal', ?, ?, NULL, 3600)",
                  (GROUP, T0 + 1))
    _msg(env, "s1", "afterturnon")
    snap = _snapshot(env, tmp_path)
    scrub(snap, env.home)
    assert _contents(snap, "s1") == ["pretimer", ""]


def test_plugin_on_but_ledger_unreadable_is_refused(env, tmp_path):
    with pytest.raises(RuntimeError):
        scrub(_snapshot(env, tmp_path), tmp_path / "no-ledger-here", required=True)


def test_no_log_file_is_left_next_to_the_copy(env, tmp_path):
    env.r.observe("signal", GROUP, 3600, sender="moana")
    _msg(env, "s1", "walsecret")
    snap = _snapshot(env, tmp_path)
    scrub(snap, env.home)
    wal = Path(str(snap) + "-wal")
    assert not wal.exists() or wal.stat().st_size == 0
