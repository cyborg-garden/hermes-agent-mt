"""When no image reaches the model, the tool result must say so in plain words."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.vision_tools import _build_native_vision_tool_result

FALSE_CLAIMS = (
    "Image attached natively",
    "built-in vision",
    "you can see it",
)
TRUTH = "no image reached you, you have not seen it"


def _agent(provider="openrouter", model="z-ai/glm-5.3"):
    from run_agent import AIAgent

    agent = MagicMock(spec=AIAgent)
    agent.provider = provider
    agent.model = model
    agent._no_list_tool_content_models = set()
    agent._content_has_image_parts = AIAgent._content_has_image_parts
    agent._tool_result_content_for_active_model = (
        lambda name, result: AIAgent._tool_result_content_for_active_model(agent, name, result)
    )
    return agent


def _envelope():
    return _build_native_vision_tool_result(
        "/tmp/sheet.png", "what is this?", "data:image/png;base64,AAAA", 2048
    )


def _assert_truthful(text):
    assert isinstance(text, str)
    assert TRUTH in text
    for claim in FALSE_CLAIMS:
        assert claim not in text


def test_blind_model_gets_a_plain_failure():
    agent = _agent()
    agent._model_supports_vision = lambda: False
    _assert_truthful(agent._tool_result_content_for_active_model("vision_analyze", _envelope()))


def test_provider_refusing_image_tool_results_gets_a_plain_failure():
    agent = _agent(provider="xiaomi", model="mimo")
    agent._model_supports_vision = lambda: True
    agent._provider_supports_vision_tool_messages = lambda: False
    _assert_truthful(agent._tool_result_content_for_active_model("vision_analyze", _envelope()))


def test_model_remembered_as_refusing_list_content_gets_a_plain_failure():
    agent = _agent(provider="xiaomi", model="mimo")
    agent._model_supports_vision = lambda: True
    agent._provider_supports_vision_tool_messages = lambda: True
    agent._no_list_tool_content_models = {("xiaomi", "mimo")}
    _assert_truthful(agent._tool_result_content_for_active_model("vision_analyze", _envelope()))


def test_vision_model_still_gets_the_pixels():
    agent = _agent(provider="anthropic", model="claude-x")
    agent._model_supports_vision = lambda: True
    agent._provider_supports_vision_tool_messages = lambda: True
    content = agent._tool_result_content_for_active_model("vision_analyze", _envelope())
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)


def test_recovery_that_strips_images_does_not_keep_the_false_note():
    from run_agent import AIAgent

    agent = _agent(provider="xiaomi", model="mimo")
    messages = [{"role": "tool", "tool_call_id": "a", "content": _envelope()["content"]}]
    assert AIAgent._try_strip_image_parts_from_tool_messages(agent, messages) is True
    _assert_truthful(messages[0]["content"])


def test_replayed_tool_result_on_a_blind_fallback_drops_the_false_note():
    """A result stored while on a vision model is replayed after a fallback."""
    from run_agent import AIAgent

    agent = _agent()
    agent._content_has_image_parts = AIAgent._content_has_image_parts
    agent._describe_image_for_anthropic_fallback = lambda url, role: (
        "[The tool result attached an image. Here's what it contains:\nImage analysis failed.]"
    )
    text = AIAgent._preprocess_anthropic_content(agent, _envelope()["content"], "tool")
    for claim in FALSE_CLAIMS:
        assert claim not in text


@pytest.mark.parametrize(
    "tool_json",
    [
        '{"success": false, "analysis": "There was a problem with the request and the image could not be analyzed. Error: 404"}',
        "not json",
    ],
)
def test_failed_description_says_no_image_was_seen(tool_json):
    from run_agent import AIAgent

    agent = _agent()
    agent._anthropic_image_fallback_cache = {}

    async def _fake_tool(**_kw):
        return tool_json

    with patch("tools.vision_tools.vision_analyze_tool", _fake_tool):
        note = AIAgent._describe_image_for_anthropic_fallback(agent, "/tmp/sheet.png", "user")
    assert TRUTH in note
    assert "Here's what it contains" not in note
