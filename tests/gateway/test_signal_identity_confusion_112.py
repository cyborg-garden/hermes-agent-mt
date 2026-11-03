"""Regression tests for hermes-agent-mt#112 — Signal adapter layers.

Incident: in a multi-agent Signal group, a task @-mentioned to agent A was
also executed by agent B. Two of the four defect layers live in the Signal
adapter:

1. Own-ACI resolution trusted ``getUserStatus`` on the bot's own number.
   signal-cli (verified 0.14.5) resolves a self-lookup through the recipient
   store and returns the account's **PNI** as a bare ``uuid`` — the ``PNI:``
   prefix is stripped in the RPC response, so the old prefix guard accepted
   it. Every agent then believed its PNI was its ACI, and genuine @mentions
   (which carry the ACI) could never match. Fix: resolve via
   ``listIdentities`` on the own number (the identity store keys by ACI);
   never accept a getUserStatus self-lookup result.

2. Reply-quote addressing was ignored: an unmentioned group message that
   quotes ANOTHER author's message is unambiguously addressed elsewhere, but
   the command/voice-memo bypasses could still grant a full agent turn. Fix:
   a quote targeting someone other than this bot forces observe-only.

Conventions follow tests/gateway/test_signal_mention_own_aci.py.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig

BOT_PHONE = "+15551234567"
BOT_ACI = "5eca7c21-1111-2222-3333-000000000099"
BOT_PNI = "aaaa1111-bbbb-cccc-dddd-000000000001"  # what getUserStatus returns for self
GROUP_ID = "dGVzdC1ncm91cC1pZA=="
OTHER_ACI = "9f0d3e42-dddd-eeee-ffff-000000000002"
OTHER_AGENT_ACI = "bbbb2222-cccc-dddd-eeee-000000000002"


@pytest.fixture(autouse=True)
def _isolate_signal_env(monkeypatch):
    for var in (
        "SIGNAL_ALLOWED_USERS",
        "SIGNAL_GROUP_ALLOWED_USERS",
        "SIGNAL_ALLOW_ALL_USERS",
        "SIGNAL_REQUIRE_MENTION",
        "SIGNAL_OBSERVE_UNMENTIONED",
        "SIGNAL_PROFILE_NAME",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_signal_adapter(monkeypatch, account=BOT_PHONE, **extra):
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


def _group_envelope(mentions=None, text="hello", quote=None, attachments=None,
                    timestamp=1750000000000):
    data_message = {
        "message": text,
        "timestamp": timestamp,
        "groupV2": {"id": GROUP_ID},
    }
    if mentions is not None:
        data_message["mentions"] = mentions
    if quote is not None:
        data_message["quote"] = quote
    if attachments is not None:
        data_message["attachments"] = attachments
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
    captured = {}

    async def _capture(event):
        captured["event"] = event

    adapter.handle_message = AsyncMock(side_effect=_capture)
    await adapter._handle_envelope(envelope)
    return captured.get("event")


# ---------------------------------------------------------------------------
# Layer 1 — own-ACI resolution
# ---------------------------------------------------------------------------

class TestResolveOwnUuid:

    @pytest.mark.asyncio
    async def test_uses_list_identities_not_get_user_status(self, monkeypatch):
        """listIdentities ACI wins even when getUserStatus returns the PNI."""
        adapter = _make_signal_adapter(monkeypatch)

        async def _rpc(method, params=None):
            if method == "listIdentities":
                assert params["number"] == BOT_PHONE
                return [{"number": BOT_PHONE, "uuid": BOT_ACI}]
            if method == "getUserStatus":
                # signal-cli self-lookup: PNI as a bare uuid, no "PNI:" prefix
                return [{"recipient": BOT_PHONE, "number": BOT_PHONE,
                         "uuid": BOT_PNI, "isRegistered": True}]
            raise AssertionError(f"unexpected RPC {method}")

        adapter._rpc = AsyncMock(side_effect=_rpc)
        await adapter._resolve_own_uuid()

        assert adapter._own_uuid == BOT_ACI
        # Cached for reply-to-bot / self-mention stripping too.
        assert adapter._recipient_uuid_by_number.get(BOT_PHONE) == BOT_ACI

    @pytest.mark.asyncio
    async def test_never_falls_back_to_get_user_status_pni(self, monkeypatch):
        """If listIdentities fails, the PNI from getUserStatus must NOT be
        cached as the own ACI — unresolved (fail-closed) is the safe state."""
        adapter = _make_signal_adapter(monkeypatch)

        async def _rpc(method, params=None):
            if method == "listIdentities":
                raise RuntimeError("method not available")
            if method == "getUserStatus":
                return [{"uuid": BOT_PNI, "isRegistered": True}]
            raise AssertionError(f"unexpected RPC {method}")

        adapter._rpc = AsyncMock(side_effect=_rpc)
        await adapter._resolve_own_uuid()

        assert adapter._own_uuid is None

    @pytest.mark.asyncio
    async def test_pni_prefixed_identity_rejected(self, monkeypatch):
        """A PNI:-prefixed service id from listIdentities is not an ACI."""
        adapter = _make_signal_adapter(monkeypatch)

        async def _rpc(method, params=None):
            if method == "listIdentities":
                return [{"number": BOT_PHONE, "uuid": f"PNI:{BOT_PNI}"}]
            return []

        adapter._rpc = AsyncMock(side_effect=_rpc)
        await adapter._resolve_own_uuid()

        assert adapter._own_uuid is None


# ---------------------------------------------------------------------------
# Layer 4 — reply-quote addressing
# ---------------------------------------------------------------------------

class TestReplyQuoteAddressing:

    @pytest.mark.asyncio
    async def test_reply_to_other_agent_is_observe_only(self, monkeypatch):
        """An unmentioned reply quoting ANOTHER agent's message is addressed
        to that agent — observe only."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._own_uuid = BOT_ACI

        event = await _dispatch(adapter, _group_envelope(
            text="yes update the URL in your skill please",
            quote={"id": 1749999990000, "authorUuid": OTHER_AGENT_ACI,
                   "text": "done — references updated"},
        ))

        assert event is not None
        assert event.observe_only is True

    @pytest.mark.asyncio
    async def test_command_reply_to_other_agent_is_observe_only(self, monkeypatch):
        """The slash-command bypass must not override reply-quote addressing."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._own_uuid = BOT_ACI

        event = await _dispatch(adapter, _group_envelope(
            text="/status",
            quote={"id": 1749999990000, "authorUuid": OTHER_AGENT_ACI},
        ))

        assert event is not None
        assert event.observe_only is True

    @pytest.mark.asyncio
    async def test_voice_memo_reply_to_other_agent_is_observe_only(self, monkeypatch):
        """The voice-memo bypass must not override reply-quote addressing."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._own_uuid = BOT_ACI

        # Text present so the contentless-envelope skip doesn't drop the
        # event before the mention filter (attachment downloads need a live
        # RPC client); the voice-memo bypass keys on attachment metadata.
        event = await _dispatch(adapter, _group_envelope(
            text="voice note follow-up",
            quote={"id": 1749999990000, "authorUuid": OTHER_AGENT_ACI},
            attachments=[{"contentType": "audio/aac", "id": "a1"}],
        ))

        assert event is not None
        assert event.observe_only is True

    @pytest.mark.asyncio
    async def test_reply_to_bot_still_gets_full_turn(self, monkeypatch):
        """Replies quoting THIS bot's message keep their full agent turn."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._own_uuid = BOT_ACI

        event = await _dispatch(adapter, _group_envelope(
            text="yes please do that",
            quote={"id": 1749999990000, "authorUuid": BOT_ACI,
                   "text": "want me to finish that step?"},
        ))

        assert event is not None
        assert event.observe_only is False

    @pytest.mark.asyncio
    async def test_reply_quoting_bot_by_uuid_with_unresolved_identity_fails_closed(
        self, monkeypatch,
    ):
        """With own ACI unresolved, a uuid-only quote of the bot's own message
        cannot be confirmed as self — fail closed to observe (never answer on
        uncertain identity)."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._own_uuid = None

        event = await _dispatch(adapter, _group_envelope(
            text="yes please do that",
            quote={"id": 1749999990000, "authorUuid": BOT_ACI},
        ))

        assert event is not None
        assert event.observe_only is True


# ---------------------------------------------------------------------------
# Layer 3 support — group channel prompt (identity + observed-context marker)
# ---------------------------------------------------------------------------

class TestGroupChannelPrompt:

    @pytest.mark.asyncio
    async def test_group_event_carries_observed_context_marker(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._own_uuid = BOT_ACI

        event = await _dispatch(
            adapter, _group_envelope(mentions=[{"uuid": BOT_ACI}])
        )

        assert event is not None
        assert event.channel_prompt is not None
        assert "observed group context" in event.channel_prompt
        assert BOT_ACI in event.channel_prompt
        assert BOT_PHONE in event.channel_prompt

    @pytest.mark.asyncio
    async def test_dm_event_has_no_group_channel_prompt(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._own_uuid = BOT_ACI
        envelope = {
            "envelope": {
                "sourceUuid": OTHER_ACI,
                "source": OTHER_ACI,
                "sourceName": "Someone",
                "timestamp": 1750000000000,
                "dataMessage": {"message": "hi", "timestamp": 1750000000000},
            }
        }
        # DMs are allowlist-gated in run.py, not here; adapter-level dispatch
        # only needs the DM not to carry the group prompt.
        event = await _dispatch(adapter, envelope)

        assert event is not None
        assert event.channel_prompt is None

    def test_marker_matches_run_path_helper(self, monkeypatch):
        """The adapter's marker must be the one _build_gateway_agent_history
        keys on — otherwise observed rows replay as ordinary user turns."""
        from gateway.run import _uses_telegram_observed_group_context

        adapter = _make_signal_adapter(monkeypatch)
        adapter._own_uuid = BOT_ACI
        assert _uses_telegram_observed_group_context(
            adapter._group_observe_channel_prompt()
        )
