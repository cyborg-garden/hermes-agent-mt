"""A saved vision row has no pixels, so it must not tell the model it can see them.

The gateway reloads history from the database each turn. Whatever the row says
goes to the next model as plain text, blind or not.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent
from tools.vision_tools import NATIVE_VISION_ACK, _build_native_vision_tool_result

OLD_SUMMARY = "Image attached natively for the main model (19.8 KB). Answer using built-in vision."
FALSE_CLAIMS = ("you can see it", "attached natively", "built-in vision")
TRUTH = "you cannot see it now"


def _assert_honest(text):
    assert isinstance(text, str)
    assert TRUTH in text
    for claim in FALSE_CLAIMS:
        assert claim not in text


@pytest.fixture
def db(tmp_path: Path):
    database = SessionDB(tmp_path / "state.db")
    database.create_session(session_id="s1", source="discord", model="m")
    yield database
    database.close()


@pytest.fixture
def agent(db):
    return AIAgent(
        api_key="x",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        model="moonshotai/kimi-k3.1-pro",
        quiet_mode=True,
        skip_memory=True,
        session_id="s1",
        session_db=db,
        enabled_toolsets=[],
    )


def _save_a_vision_turn(agent):
    """Run the turn on a model that sees: real result, real flush."""
    envelope = _build_native_vision_tool_result(
        "/tmp/sheet.png", "what is this?", "data:image/png;base64,AAAA", 2048
    )
    with patch.object(AIAgent, "_model_supports_vision", lambda self: True):
        content = agent._tool_result_content_for_active_model("vision_analyze", envelope)
    assert [part["type"] for part in content] == ["text", "image_url"]
    agent._flush_messages_to_session_db(
        [
            {"role": "user", "content": "look at this"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "vision_analyze", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "tool_name": "vision_analyze", "content": content},
            {"role": "assistant", "content": "ok"},
        ]
    )


def _tool_row(messages):
    return next(m for m in messages if m["role"] == "tool")


def test_the_saved_row_makes_no_false_claim(agent, db):
    _save_a_vision_turn(agent)
    _assert_honest(_tool_row(db.get_messages("s1"))["content"])


def test_a_blind_fallback_is_not_told_it_can_see_on_the_next_turn(agent, db):
    _save_a_vision_turn(agent)
    history = db.get_messages_as_conversation("s1")
    agent.model = "z-ai/glm-5.3"
    with patch.object(AIAgent, "_model_supports_vision", lambda self: False):
        prepared = agent._prepare_messages_for_non_vision_model(history)
    sent = _tool_row(prepared)["content"]
    _assert_honest(sent)
    assert "what is this?" in sent


def test_a_model_that_sees_is_not_told_it_can_see_a_missing_image(agent, db):
    _save_a_vision_turn(agent)
    history = db.get_messages_as_conversation("s1")
    with patch.object(AIAgent, "_model_supports_vision", lambda self: True):
        prepared = agent._prepare_messages_for_non_vision_model(history)
    _assert_honest(_tool_row(prepared)["content"])


@pytest.mark.parametrize("tool_name", ["vision_analyze", "browser_vision"])
@pytest.mark.parametrize(
    "stored",
    [
        pytest.param(OLD_SUMMARY, id="old-summary"),
        pytest.param(f"{NATIVE_VISION_ACK}\n\nQuestion: what is this?\n[screenshot]", id="old-ack"),
    ],
)
def test_rows_already_stored_are_corrected_on_load(db, tool_name, stored):
    db.append_message(session_id="s1", role="tool", content=stored, tool_name=tool_name, tool_call_id="c1")
    _assert_honest(_tool_row(db.get_messages_as_conversation("s1"))["content"])
    # The stored row is left as it was.
    assert _tool_row(db.get_messages("s1"))["content"] == stored


def test_the_question_survives_on_an_old_row(db):
    stored = f"{NATIVE_VISION_ACK}\n\nQuestion: what is this?\n[screenshot]"
    db.append_message(session_id="s1", role="tool", content=stored, tool_name="vision_analyze", tool_call_id="c1")
    assert "Question: what is this?" in _tool_row(db.get_messages_as_conversation("s1"))["content"]


# Controls: these pass without the fix and guard against over-correcting.


@pytest.mark.parametrize("role,tool_name", [("tool", "memory"), ("user", None), ("assistant", None)])
def test_other_rows_that_quote_the_sentence_are_left_alone(db, role, tool_name):
    """A memory note can record this sentence on purpose. It is not a claim about pixels."""
    stored = f"vision_analyze replying only '{OLD_SUMMARY}' means the image was dropped"
    db.append_message(session_id="s1", role=role, content=stored, tool_name=tool_name, tool_call_id="c1")
    assert stored in str(db.get_messages_as_conversation("s1")[0]["content"])


def test_the_live_result_still_carries_the_claim_beside_the_pixels(agent):
    envelope = _build_native_vision_tool_result(
        "/tmp/sheet.png", "what is this?", "data:image/png;base64,AAAA", 2048
    )
    with patch.object(AIAgent, "_model_supports_vision", lambda self: True):
        content = agent._tool_result_content_for_active_model("vision_analyze", envelope)
    assert NATIVE_VISION_ACK in content[0]["text"]
    assert content[1]["type"] == "image_url"
