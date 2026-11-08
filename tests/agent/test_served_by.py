"""Which upstream host served each assistant reply.

OpenRouter forwards a request to one of many hosts and names it in a top-level
``provider`` field. Dropped tool arguments were traced to some hosts under long
prompts, but state.db never recorded the host, so nothing could be pinned out.
These tests cover reading the field, carrying it through a streamed response,
stamping it on the assistant message, keeping it off the wire, and storing it.
"""

import sqlite3
from types import SimpleNamespace

from agent.chat_completion_helpers import build_assistant_message, served_by_from_response
from agent.transports.chat_completions import ChatCompletionsTransport
from hermes_state import SessionDB
from tests.agent.test_tool_call_arg_no_redaction import _FakeAgent, _FakeAssistantMsg


def test_served_by_reads_every_shape_the_sdk_and_raw_json_use():
    assert served_by_from_response(SimpleNamespace(provider="Fireworks")) == "Fireworks"
    assert served_by_from_response(SimpleNamespace(model_extra={"provider": "Novita"})) == "Novita"
    assert served_by_from_response({"provider": "Chutes"}) == "Chutes"
    for empty in (None, SimpleNamespace(), SimpleNamespace(provider=""), {"provider": 3}, {}):
        assert served_by_from_response(empty) is None


def test_the_assistant_message_carries_it_and_the_wire_does_not():
    agent = _FakeAgent()
    agent._last_served_by = "Fireworks"
    msg = build_assistant_message(agent, _FakeAssistantMsg("hi", None), "stop")
    assert msg["_served_by"] == "Fireworks"
    sent = ChatCompletionsTransport().convert_messages([{"role": "user", "content": "x"}, msg])
    assert all("_served_by" not in m and "served_by" not in m for m in sent)


def test_no_host_means_no_field():
    msg = build_assistant_message(_FakeAgent(), _FakeAssistantMsg("hi", None), "stop")
    assert "_served_by" not in msg


def test_state_db_stores_it_and_adds_the_column_to_an_old_database(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("s1", "cli")
    mid = db.append_message("s1", "assistant", content="hi", served_by="Fireworks")
    uid = db.append_message("s1", "user", content="hello")
    rows = dict(sqlite3.connect(db_path).execute("select id, served_by from messages").fetchall())
    assert rows[mid] == "Fireworks" and rows[uid] is None

    # A database from before this column opens, gains the column, and keeps working.
    con = sqlite3.connect(db_path)
    con.execute("alter table messages drop column served_by")
    con.commit()
    con.close()
    db2 = SessionDB(db_path=db_path)
    cols = [r[1] for r in sqlite3.connect(db_path).execute("pragma table_info(messages)")]
    assert "served_by" in cols
    db2.append_message("s1", "assistant", content="ok", served_by="Novita")


def test_bulk_rewrite_keeps_it(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", "cli")
    conn = sqlite3.connect(tmp_path / "state.db")
    n, _ = db._insert_message_rows(conn, "s1", [
        {"role": "assistant", "content": "a", "_served_by": "Fireworks"},
        {"role": "user", "content": "b", "_served_by": "ignored for users"},
    ])
    conn.commit()
    got = [r[0] for r in conn.execute("select served_by from messages order by id")]
    assert n == 2 and got == ["Fireworks", None]


def test_a_transcript_rewrite_keeps_the_host(tmp_path):
    """/retry, /undo and rewind reload the conversation and re-store it."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", "cli")
    db.append_message("s1", "user", content="hi")
    db.append_message("s1", "assistant", content="hello", served_by="Fireworks")
    conv = db.get_messages_as_conversation("s1")
    assert conv[-1]["_served_by"] == "Fireworks"
    db.replace_messages("s1", conv)
    got = [r[0] for r in sqlite3.connect(tmp_path / "state.db").execute(
        "select served_by from messages where role='assistant'")]
    assert got == ["Fireworks"]


def test_the_sdk_chunk_type_keeps_the_field():
    from openai.types.chat import ChatCompletionChunk
    chunk = ChatCompletionChunk.model_validate({
        "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
        "choices": [], "provider": "Fireworks",
    })
    assert served_by_from_response(chunk) == "Fireworks"
