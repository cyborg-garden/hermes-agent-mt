"""The person in the thread gets a plain sentence, not provider names."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

PRIMARY = "claude-opus-5-5"
BACKUP = "z-ai/glm-5.3"


def _activate(sees_images: bool):
    from run_agent import AIAgent

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.anthropic.com",
            provider="anthropic",
            model=PRIMARY,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{"provider": "openrouter", "model": BACKUP}],
        )
    buffered = []
    agent._buffer_status = buffered.append
    fb_client = MagicMock()
    fb_client.api_key = "test-key"
    fb_client.base_url = "https://openrouter.ai/api/v1"
    fb_client._custom_headers = None
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(fb_client, BACKUP),
    ), patch.object(AIAgent, "_model_supports_vision", lambda self: sees_images):
        assert agent._try_activate_fallback() is True
    return agent, buffered


JARGON = ("glm", "claude", "openrouter", "anthropic", "provider", "fallback", "primary", "via")


@pytest.mark.parametrize("sees_images", [True, False])
def test_notice_has_no_provider_detail(sees_images):
    agent, buffered = _activate(sees_images)
    for text in [agent._pending_fallback_notice, *buffered]:
        low = text.lower()
        for word in JARGON:
            assert word not in low, text
        assert "backup" in low


def test_notice_says_when_images_cannot_be_seen():
    agent, buffered = _activate(sees_images=False)
    for text in [agent._pending_fallback_notice, *buffered]:
        assert "can't see images" in text


def test_notice_stays_quiet_about_images_when_the_backup_can_see():
    agent, buffered = _activate(sees_images=True)
    for text in [agent._pending_fallback_notice, *buffered]:
        assert "image" not in text.lower()


def test_operators_still_get_the_detail_in_the_log(caplog):
    with caplog.at_level("WARNING", logger="agent.degradation"):
        _activate(sees_images=False)
    assert any(BACKUP in r.getMessage() for r in caplog.records)
