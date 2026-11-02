"""Once the pixels are gone, nothing may say the model can see them."""

from __future__ import annotations

import copy

from agent.context_compressor import _strip_historical_media, _strip_image_parts_from_parts
from agent.message_sanitization import _strip_images_from_messages
from tools.vision_tools import (
    NATIVE_VISION_ACK,
    NO_IMAGE_SEEN,
    _build_native_vision_tool_result,
)

QUESTION = "what colour is the door"


def _tool_message():
    envelope = _build_native_vision_tool_result(
        "/tmp/door.png", QUESTION, "data:image/png;base64,AAAA", 4
    )
    return {"role": "tool", "tool_call_id": "t1", "content": envelope["content"]}


def _conversation():
    return [
        {"role": "user", "content": "look"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "vision_analyze", "arguments": "{}"}}
            ],
        },
        _tool_message(),
    ]


def test_server_rejection_replaces_the_claim_with_the_truth():
    messages = _conversation()
    assert _strip_images_from_messages(messages) is True
    text = str(messages[2]["content"])
    assert NATIVE_VISION_ACK not in text
    assert NO_IMAGE_SEEN in text
    assert QUESTION in text
    assert "image_url" not in text


def test_server_rejection_fixes_the_payload_copy_as_well():
    """The loop strips the stored messages, then the copy it is about to send."""
    messages = _conversation()
    api_messages = [dict(m) for m in messages]
    _strip_images_from_messages(messages)
    _strip_images_from_messages(api_messages)
    assert NATIVE_VISION_ACK not in str(api_messages[2]["content"])
    assert NO_IMAGE_SEEN in str(api_messages[2]["content"])


def test_server_rejection_leaves_a_user_who_typed_the_words_alone():
    """Only a message that lost an image has its text changed."""
    messages = [{"role": "user", "content": [{"type": "text", "text": NATIVE_VISION_ACK}]}]
    assert _strip_images_from_messages(messages) is False
    assert messages[0]["content"][0]["text"] == NATIVE_VISION_ACK


def test_anthropic_image_cap_drops_the_claim_from_evicted_results():
    """Only the newest three tool images go to native Anthropic."""
    from agent.anthropic_adapter import convert_messages_to_anthropic

    messages = [{"role": "user", "content": "look"}]
    for index in range(5):
        call_id = f"t{index}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": call_id, "type": "function", "function": {"name": "vision_analyze", "arguments": "{}"}}
                ],
            }
        )
        envelope = _build_native_vision_tool_result(
            "/tmp/door.png", QUESTION, "data:image/png;base64,AAAA", 4
        )
        messages.append({"role": "tool", "tool_call_id": call_id, "content": envelope["content"]})
    messages.append({"role": "user", "content": "ok"})

    _, converted = convert_messages_to_anthropic(messages)

    results = [
        block
        for msg in converted
        if isinstance(msg["content"], list)
        for block in msg["content"]
        if block.get("type") == "tool_result"
    ]
    assert len(results) == 5
    for block in results[:2]:
        assert not any(part["type"] == "image" for part in block["content"])
        assert NATIVE_VISION_ACK not in str(block["content"])
        assert "[screenshot removed to save context]" in str(block["content"])
        assert QUESTION in str(block["content"])
    # A result that still holds its image keeps the sentence, which is true there.
    for block in results[2:]:
        assert any(part["type"] == "image" for part in block["content"])
        assert NATIVE_VISION_ACK in str(block["content"])


def test_old_screenshot_prune_drops_the_claim_and_does_not_mutate():
    original = _tool_message()["content"]
    before = copy.deepcopy(original)
    stripped = _strip_image_parts_from_parts(original)
    assert NATIVE_VISION_ACK not in str(stripped)
    assert "[screenshot removed to save context]" in str(stripped)
    assert QUESTION in str(stripped)
    assert original == before


def test_historical_media_strip_drops_the_claim_and_does_not_mutate():
    messages = _conversation()
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "another"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
            ],
        }
    )
    before = copy.deepcopy(messages)
    result = _strip_historical_media(messages)
    assert NATIVE_VISION_ACK not in str(result[2]["content"])
    assert "stripped after compression" in str(result[2]["content"])
    assert QUESTION in str(result[2]["content"])
    assert messages == before


def test_a_claim_with_no_question_leaves_no_empty_text_part():
    content = [
        {"type": "text", "text": NATIVE_VISION_ACK},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    stripped = _strip_image_parts_from_parts(content)
    assert stripped == [{"type": "text", "text": "[screenshot removed to save context]"}]


def test_full_turn_retry_after_a_server_rejection_has_no_false_claim():
    """Drive the real loop: the server refuses images, and the retry must tell the truth."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="moonshotai/kimi-k3.1-pro",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False

    payloads = []

    class _Rejected(Exception):
        status_code = 404

    def _create(**kwargs):
        payloads.append(copy.deepcopy(kwargs["messages"]))
        if len(payloads) == 1:
            raise _Rejected("No endpoints found that support image input")
        message = SimpleNamespace(
            content="I cannot see it.",
            tool_calls=None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            model="test/model",
            usage=None,
        )

    agent.client.chat.completions.create.side_effect = _create
    agent._ensure_primary_openai_client = lambda **_kw: agent.client

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.conversation_loop.time.sleep"),
        # The route says it takes images. Only the server's refusal says otherwise.
        patch.object(agent, "_model_supports_vision", return_value=True),
    ):
        result = agent.run_conversation("is the door red", conversation_history=_conversation())

    assert result["final_response"] == "I cannot see it."
    assert len(payloads) == 2
    assert "image_url" in str(payloads[0])
    assert "image_url" not in str(payloads[1])
    assert NATIVE_VISION_ACK not in str(payloads[1])
    assert NO_IMAGE_SEEN in str(payloads[1])
