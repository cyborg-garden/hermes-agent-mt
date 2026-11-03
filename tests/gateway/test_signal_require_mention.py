"""Regression tests for the Signal group mention gate (``require_mention``).

Reproduces the production incident where ``hermes-cyborg`` — which had
``SIGNAL_REQUIRE_MENTION=true`` — still replied to a plain, unmentioned group
message ("Testing if I can message without agents replying"). With the gate
on, an unmentioned group message must be dispatched ``observe_only`` (recorded
for context, no agent response).
"""
import pytest

from gateway.config import PlatformConfig


def _make_signal_adapter(monkeypatch, account="+15551234567", **extra):
    """Create a SignalAdapter with test defaults, mirroring test_signal_enhancements."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", extra.pop("group_allowed", "*"))
    from plugins.platforms.signal.adapter import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "http_url": "http://localhost:8080",
        "account": account,
        **extra,
    }
    return SignalAdapter(config)


def _group_text_envelope(text, *, sender="+15559999999", group_id="grp-abc-123", mentions=None):
    data_message = {
        "message": text,
        "groupInfo": {"groupId": group_id},
    }
    if mentions is not None:
        data_message["mentions"] = mentions
    return {
        "envelope": {
            "sourceNumber": sender,
            "sourceUuid": "66666666-6666-6666-6666-666666666666",
            "sourceName": "Juni",
            "dataMessage": data_message,
        }
    }


@pytest.mark.asyncio
async def test_unmentioned_group_message_is_observe_only(monkeypatch):
    """require_mention=True + plain unmentioned group message → observe_only.

    This is the exact cyborg scenario.
    """
    adapter = _make_signal_adapter(monkeypatch, require_mention=True)
    # Precondition: the gate is actually enabled on this adapter.
    assert adapter.require_mention is True

    captured = []

    async def fake_handle_message(event):
        captured.append(event)

    adapter.handle_message = fake_handle_message

    envelope = _group_text_envelope(
        "Testing if I can message without agents replying"
    )
    await adapter._handle_envelope(envelope)

    assert len(captured) == 1, "message should be dispatched exactly once"
    assert captured[0].observe_only is True, (
        "an unmentioned group message must be observe_only (no agent response)"
    )


@pytest.mark.asyncio
async def test_mentioned_group_message_responds_and_strips_mention(monkeypatch):
    """require_mention=True + a message that @mentions the bot → full response,
    and the bot's own @mention is stripped from the text before dispatch.

    This is why the production log showed clean text ("Testing if I can
    message without agents replying") even though the bot responded: the
    inbound @mention was stripped at signal.py:694-700 before logging.
    """
    account = "+15551234567"
    adapter = _make_signal_adapter(monkeypatch, account=account, require_mention=True)

    captured = []

    async def fake_handle_message(event):
        captured.append(event)

    adapter.handle_message = fake_handle_message

    # Juni @mentions the bot by account number, then types the same text.
    envelope = _group_text_envelope(
        f"@{account} Testing if I can message without agents replying"
    )
    await adapter._handle_envelope(envelope)

    assert len(captured) == 1
    assert captured[0].observe_only is False, (
        "a message that @mentions the bot must trigger a full response"
    )
    assert captured[0].text == "Testing if I can message without agents replying", (
        "the bot's own @mention must be stripped from the dispatched text"
    )
