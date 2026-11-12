"""scrub(): blank disappearing-chat text in a copy of state.db, keep everything else."""
import shutil
import sqlite3
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
    assert scrub(tmp_path / "state.db", tmp_path / "nohome") == {"chats": 0, "rows": 0}
    assert _contents(tmp_path / "state.db", "s") == ["kept"]


def test_a_copy_without_the_expected_schema_is_refused(env, tmp_path):
    env.r.observe("signal", GROUP, 3600, sender="moana")
    bad = tmp_path / "bad.db"
    with sqlite3.connect(bad) as c:
        c.execute("CREATE TABLE messages (id INTEGER, session_id TEXT, body TEXT)")
        c.execute("CREATE TABLE sessions (id TEXT, source TEXT, chat_id TEXT, session_key TEXT, parent_session_id TEXT, title TEXT)")
    with pytest.raises(RuntimeError):
        scrub(bad, env.home)
