"""Regression tests for the Signal group @mention-by-ACI bug.

Production failure: in a Signal GROUP with require_mention=true, the bot did
NOT respond to genuine @mentions delivered by ACI/UUID. Modern Signal delivers
@mentions as a mention-metadata entry carrying only the mentioned party's ACI
(a ``uuid``, often with no ``number``). The mention-match code compared
``m.get("uuid") == account_norm`` where ``account_norm`` is the bot's *phone*
number — the bot's own ACI was never resolved, so a real @mention never matched
and the message fell through to observe_only (agent stayed silent).

Fix under test (additive):
- Resolve the bot's own ACI once at connect() and store it on ``_own_uuid``.
- A metadata mention matches if ``m.get("uuid")`` equals the bot's own ACI in
  addition to the existing ``== account_norm`` (phone) check.
- The reply-to-bot ``bot_uuid`` fallback falls back to ``_own_uuid``.

Conventions follow tests/gateway/test_signal_aci_reresolve.py.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig

BOT_PHONE = "+15551234567"
BOT_ACI = "5eca7c21-1111-2222-3333-000000000099"
GROUP_ID = "dGVzdC1ncm91cC1pZA=="  # opaque base64-ish group id
OTHER_ACI = "9f0d3e42-dddd-eeee-ffff-000000000002"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_signal_env(monkeypatch):
    for var in (
        "SIGNAL_ALLOWED_USERS",
        "SIGNAL_GROUP_ALLOWED_USERS",
        "SIGNAL_ALLOW_ALL_USERS",
        "SIGNAL_REQUIRE_MENTION",
        "SIGNAL_OBSERVE_UNMENTIONED",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_signal_adapter(monkeypatch, account=BOT_PHONE, **extra):
    """Create a real SignalAdapter with test defaults."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", extra.pop("group_allowed", "*"))
    from plugins.platforms.signal.adapter import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "http_url": "http://localhost:8080",
        "account": account,
        "require_mention": True,
        **extra,
    }
    return SignalAdapter(config)


def _group_envelope(mentions=None, text="hello", timestamp=1750000000000):
    """A group dataMessage envelope from an OTHER member, optionally @mentioning."""
    data_message = {
        "message": text,
        "timestamp": timestamp,
        "groupV2": {"id": GROUP_ID},
    }
    if mentions is not None:
        data_message["mentions"] = mentions
    return {
        "envelope": {
            "sourceUuid": OTHER_ACI,
            "source": OTHER_ACI,
            "sourceName": "Someone",
            "timestamp": timestamp,
            "dataMessage": data_message,
        }
    }


async def _dispatch(adapter, envelope):
    """Run _handle_envelope with handle_message mocked; return the event or None."""
    captured = {}

    async def _capture(event):
        captured["event"] = event

    adapter.handle_message = AsyncMock(side_effect=_capture)
    await adapter._handle_envelope(envelope)
    return captured.get("event")


# ---------------------------------------------------------------------------
# Test A — the fix: a metadata @mention by the bot's own ACI is honored
# ---------------------------------------------------------------------------

class TestGroupMentionByOwnAci:

    @pytest.mark.asyncio
    async def test_aci_mention_is_dispatched_not_observe_only(self, monkeypatch):
        """mentions=[{"uuid": <bot ACI>}] (no number) ⇒ NOT observe_only."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._own_uuid = BOT_ACI  # resolved at connect() in production

        event = await _dispatch(
            adapter, _group_envelope(mentions=[{"uuid": BOT_ACI}])
        )

        assert event is not None
        assert event.observe_only is False


# ---------------------------------------------------------------------------
# Test B — no regression: a phone-based mention still matches
# ---------------------------------------------------------------------------

class TestGroupMentionByPhoneStillWorks:

    @pytest.mark.asyncio
    async def test_phone_mention_still_matches(self, monkeypatch):
        """mentions=[{"number": <bot phone>}] ⇒ NOT observe_only (unchanged)."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._own_uuid = BOT_ACI

        event = await _dispatch(
            adapter, _group_envelope(mentions=[{"number": BOT_PHONE}])
        )

        assert event is not None
        assert event.observe_only is False

    @pytest.mark.asyncio
    async def test_phone_mention_matches_even_without_own_uuid(self, monkeypatch):
        """Phone-mention path must not depend on ACI resolution succeeding."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._own_uuid = None  # ACI resolution failed at connect()

        event = await _dispatch(
            adapter, _group_envelope(mentions=[{"number": BOT_PHONE}])
        )

        assert event is not None
        assert event.observe_only is False


# ---------------------------------------------------------------------------
# Test C — no regression: an un-mentioned group message stays observe_only
# ---------------------------------------------------------------------------

class TestGroupNoMentionStaysSilent:

    @pytest.mark.asyncio
    async def test_no_mention_is_observe_only(self, monkeypatch):
        """No mention at all ⇒ observe_only (agent stays silent)."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._own_uuid = BOT_ACI

        event = await _dispatch(adapter, _group_envelope(mentions=None))

        assert event is not None
        assert event.observe_only is True

    @pytest.mark.asyncio
    async def test_someone_elses_aci_mention_stays_silent(self, monkeypatch):
        """A mention of a DIFFERENT ACI must not wake the bot."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._own_uuid = BOT_ACI

        event = await _dispatch(
            adapter, _group_envelope(mentions=[{"uuid": OTHER_ACI}])
        )

        assert event is not None
        assert event.observe_only is True

    @pytest.mark.asyncio
    async def test_aci_mention_without_own_uuid_stays_silent(self, monkeypatch):
        """If own ACI is unresolved, an ACI-only mention cannot match ⇒ silent.

        This documents the pre-fix behavior for the fail-soft path: with
        _own_uuid=None the ACI mention still won't match (no false positive),
        matching the constraint 'do NOT change default behavior when own ACI
        can't be resolved'.
        """
        adapter = _make_signal_adapter(monkeypatch)
        adapter._own_uuid = None

        event = await _dispatch(
            adapter, _group_envelope(mentions=[{"uuid": BOT_ACI}])
        )

        assert event is not None
        assert event.observe_only is True
