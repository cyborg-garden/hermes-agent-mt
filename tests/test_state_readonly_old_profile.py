"""A read-only look at another profile's older state.db still works."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture
def old_profile_db(tmp_path):
    """A profile that has not run since before messages gained ``observed``."""
    current = SessionDB(tmp_path / "current.db")
    current.create_session("s1", source="cli")
    current.append_message("s1", "user", "owner words")
    current.append_message("s1", "assistant", "reply")
    current.close()

    old = sqlite3.connect(tmp_path / "current.db")
    old.execute("DROP TRIGGER IF EXISTS messages_fts_delete")
    old.execute("DROP TRIGGER IF EXISTS messages_fts_update")
    old.execute("DROP TRIGGER IF EXISTS messages_fts_trigram_delete")
    old.execute("DROP TRIGGER IF EXISTS messages_fts_trigram_update")
    old.execute("ALTER TABLE messages DROP COLUMN observed")
    old.commit()
    assert "observed" not in [r[1] for r in old.execute("PRAGMA table_info(messages)")]
    old.close()
    return tmp_path / "current.db"


def test_listing_works(old_profile_db):
    db = SessionDB(db_path=old_profile_db, read_only=True)
    assert [(s["id"], s["preview"]) for s in db.list_sessions_rich()] == [("s1", "owner words")]


def test_one_session_row_works(old_profile_db):
    db = SessionDB(db_path=old_profile_db, read_only=True)
    assert db._get_session_rich_row("s1")["preview"] == "owner words"


def test_replay_and_search_work(old_profile_db):
    db = SessionDB(db_path=old_profile_db, read_only=True)
    assert [m["content"] for m in db.get_messages_as_conversation("s1")] == ["owner words", "reply"]
    assert [m["content"] for m in db.get_messages("s1")] == ["owner words", "reply"]
    assert db.search_messages("owner") == []  # read-only skips the search index probe


def test_old_file_is_left_as_it_was(old_profile_db):
    SessionDB(db_path=old_profile_db, read_only=True).list_sessions_rich()
    check = sqlite3.connect(old_profile_db)
    assert "observed" not in [r[1] for r in check.execute("PRAGMA table_info(messages)")]
    assert check.execute("SELECT count(*) FROM sqlite_master WHERE type='view'").fetchone()[0] == 0
    check.close()


def test_a_current_file_reads_its_real_flag(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="discord")
    db.append_message("s1", "user", "[stranger]: chatter", observed=True)
    db.append_message("s1", "user", "owner words")
    db.close()
    ro = SessionDB(db_path=tmp_path / "state.db", read_only=True)
    assert ro.list_sessions_rich()[0]["preview"] == "owner words"
    assert [bool(m.get("observed")) for m in ro.get_messages_as_conversation("s1")] == [True, False]


def test_a_file_that_is_not_a_database_still_fails_only_on_read(tmp_path):
    bad = tmp_path / "state.db"
    bad.write_bytes(b"this is not a database at all, not even close......")
    db = SessionDB(db_path=bad, read_only=True)
    with pytest.raises(sqlite3.DatabaseError):
        db.list_sessions_rich()


@pytest.mark.parametrize("dropped", [("served_by",), ("served_by", "observed")])
def test_a_file_from_before_served_by_reads(tmp_path, dropped):
    path = tmp_path / "state.db"
    current = SessionDB(path)
    current.create_session("s1", source="cli")
    current.append_message("s1", "assistant", "reply", served_by="Fireworks")
    current.close()
    old = sqlite3.connect(path)
    for trig in ("messages_fts_delete", "messages_fts_update",
                 "messages_fts_trigram_delete", "messages_fts_trigram_update"):
        old.execute(f"DROP TRIGGER IF EXISTS {trig}")
    for col in dropped:
        old.execute(f"ALTER TABLE messages DROP COLUMN {col}")
    old.commit()
    old.close()
    db = SessionDB(db_path=path, read_only=True)
    conv = db.get_messages_as_conversation("s1")
    assert [m["content"] for m in conv] == ["reply"] and "_served_by" not in conv[0]
