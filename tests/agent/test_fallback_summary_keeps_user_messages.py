"""When no summary can be written, the fallback must not forget the user."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import SUMMARY_PREFIX, ContextCompressor

MUSIC_RULE = (
    "[Triggering message id: `1` — use as `message_id` for reply/react/pin via the discord "
    "tools.]\n\n[sam] aesthetic stuff: i am thinking maybe we have no music for the dream "
    "sequences. just rely on the scene and dialog to make it feel nice to the player."
)
SHEET = "/opt/data/image_cache/img_e59957575ebf.png"


def _compressor():
    # Each pass gets a fresh compressor, as gateway hygiene does.
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        return ContextCompressor(
            model="test/model", protect_first_n=1, protect_last_n=2, quiet_mode=True
        )


def _filler(tag, n=12):
    turns = []
    for i in range(n):
        turns.append({"role": "user", "content": f"[sam] request {tag}-{i}"})
        turns.append({"role": "assistant", "content": f"working on {tag}-{i}"})
    return turns


def _summary_text(messages):
    for msg in messages:
        if SUMMARY_PREFIX in str(msg.get("content")):
            return str(msg["content"])
    raise AssertionError("no summary in compacted messages")


def _compact_with_dead_summariser(messages):
    with patch(
        "agent.context_compressor.call_llm", side_effect=RuntimeError("Error code: 500 boom")
    ):
        return _compressor().compress(messages, current_tokens=90_000, force=True)


def test_user_rule_survives_many_failed_compactions():
    """A dropped rule is what made the agent propose music for the dream scenes."""
    messages = [
        {"role": "user", "content": "first message stays in the protected head"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": MUSIC_RULE},
    ]
    messages += _filler("a")
    for round_no in range(11):
        messages = _compact_with_dead_summariser(messages)
        assert "no music for the dream sequences" in _summary_text(messages), round_no
        messages += _filler(f"r{round_no}", n=6)


def test_every_user_message_and_attachment_is_listed():
    turns = []
    for i in range(30):
        turns.append({"role": "user", "content": f"[sam] ask number {i}"})
        turns.append({"role": "assistant", "content": "ok"})
    turns.insert(
        5, {"role": "user", "content": f"[sam] look\n\n[Image attached at: {SHEET}]\n[screenshot]"}
    )
    summary = _compressor()._build_static_fallback_summary(turns, reason="down")
    for i in range(30):
        assert f"[sam] ask number {i}" in summary
    assert "[sam] look" in summary
    assert SHEET in summary


def test_an_image_part_is_listed_as_an_attachment():
    turns = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see this"},
                {"type": "image_url", "image_url": {"url": "https://cdn.example/a.png"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
    summary = _compressor()._build_static_fallback_summary(turns, reason="down")
    assert "https://cdn.example/a.png" in summary
    assert "AAAA" not in summary


def test_an_earlier_real_summary_is_carried_forward():
    c = _compressor()
    c._previous_summary = "## Goal\nBuild the hospital level; the player must hide in lockers."
    summary = c._build_static_fallback_summary(
        [{"role": "user", "content": "go on"}], reason="down"
    )
    assert "the player must hide in lockers" in summary


def test_a_real_summary_with_headings_survives_several_failed_passes():
    c = _compressor()
    c._previous_summary = (
        "## Goal\nBuild the level.\n\n## Key Decisions\nThe player must hide in lockers."
    )
    for _ in range(3):
        summary = c._build_static_fallback_summary(
            [{"role": "user", "content": "go on"}], reason="down"
        )
        c._previous_summary = c._strip_summary_prefix(summary)
    assert "The player must hide in lockers." in summary
    assert summary.count("## Carried From Earlier Summary") == 1


def test_carry_forward_does_not_nest_or_repeat():
    messages = [
        {"role": "user", "content": "head"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "[sam] once only please"},
    ]
    messages += _filler("a")
    for round_no in range(4):
        messages = _compact_with_dead_summariser(messages)
        messages += _filler(f"r{round_no}", n=6)
    messages = _compact_with_dead_summariser(messages)
    text = _summary_text(messages)
    assert text.count("[sam] once only please") == 1
    assert text.count("## Every User Message So Far") == 1


def test_list_stays_bounded_and_still_names_every_message():
    turns = []
    for i in range(120):
        turns.append({"role": "user", "content": f"msg{i:03d} " + "x" * 600})
    summary = _compressor()._build_static_fallback_summary(turns, reason="down")
    for i in range(120):
        assert f"msg{i:03d}" in summary
    assert len(summary) < 45_000


def test_a_later_real_summary_is_told_about_the_list():
    """A summary written once the route recovers must build on the list."""
    messages = [
        {"role": "user", "content": "head"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": MUSIC_RULE},
    ]
    messages += _filler("a")
    messages = _compact_with_dead_summariser(messages)
    messages += _filler("b", n=6)

    seen = {}

    class _Resp:
        class _C:
            class message:
                content = "## Goal\nrecovered"

            finish_reason = "stop"

        choices = [_C()]

    def _call(**kwargs):
        seen["prompt"] = kwargs["messages"][0]["content"]
        return _Resp()

    with patch("agent.context_compressor.call_llm", side_effect=_call):
        _compressor().compress(messages, current_tokens=90_000, force=True)
    assert "no music for the dream sequences" in seen["prompt"]


def test_secrets_in_a_user_message_stay_redacted():
    turns = [{"role": "user", "content": "my key is ghp_abcdefghijklmnopqrstuvwxyz0123456789"}]
    summary = _compressor()._build_static_fallback_summary(turns, reason="down")
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in summary
