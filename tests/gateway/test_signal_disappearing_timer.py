"""The Signal adapter passes the disappearing-message timer on to plugins.

signal-cli puts ``expiresInSeconds`` on every dataMessage. The adapter used to
drop it, so nothing downstream could match a chat's timer.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig

BOT_PHONE = "+15551234567"
GROUP_ID = "dGVzdC1ncm91cC1pZA=="
SENDER = "9f0d3e42-dddd-eeee-ffff-000000000002"
TS = 1750000000000


@pytest.fixture(autouse=True)
def _isolate_signal_env(monkeypatch):
    for var in ("SIGNAL_ALLOWED_USERS", "SIGNAL_GROUP_ALLOWED_USERS", "SIGNAL_ALLOW_ALL_USERS",
                "SIGNAL_REQUIRE_MENTION", "SIGNAL_OBSERVE_UNMENTIONED", "SIGNAL_PROFILE_NAME"):
        monkeypatch.delenv(var, raising=False)


def _adapter(monkeypatch):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "*")
    from gateway.platforms.signal import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {"http_url": "http://localhost:8080", "account": BOT_PHONE,
                    "require_mention": False}
    return SignalAdapter(config)


def _envelope(expires=None, group=True, text="kia ora"):
    dm = {"message": text, "timestamp": TS}
    if group:
        dm["groupV2"] = {"id": GROUP_ID}
    if expires is not None:
        dm["expiresInSeconds"] = expires
    return {"envelope": {"sourceUuid": SENDER, "source": SENDER, "sourceName": "Moana",
                         "timestamp": TS, "dataMessage": dm}}


async def _dispatch(adapter, envelope):
    got = {}

    async def _capture(event):
        got["event"] = event

    adapter.handle_message = AsyncMock(side_effect=_capture)
    await adapter._handle_envelope(envelope)
    return got.get("event")


@pytest.mark.asyncio
@pytest.mark.parametrize("value,expected", [(604800, 604800), (30, 30), (0, 0), (None, 0)])
async def test_timer_reaches_the_event(monkeypatch, value, expected):
    ev = await _dispatch(_adapter(monkeypatch), _envelope(expires=value))
    assert ev is not None
    assert ev.metadata["signal"]["expires_in_seconds"] == expected
    assert ev.metadata["signal"]["timestamp_ms"] == TS


@pytest.mark.asyncio
async def test_dm_carries_the_timer_too(monkeypatch):
    ev = await _dispatch(_adapter(monkeypatch), _envelope(expires=3600, group=False))
    assert ev is not None and ev.metadata["signal"]["expires_in_seconds"] == 3600


@pytest.mark.asyncio
async def test_unreadable_timer_is_reported_as_unknown(monkeypatch):
    ev = await _dispatch(_adapter(monkeypatch), _envelope(expires="never"))
    assert ev is not None and ev.metadata["signal"]["expires_in_seconds"] is None


@pytest.mark.asyncio
async def test_timer_change_alone_does_not_start_a_turn(monkeypatch):
    env = _envelope(expires=3600, text="")
    env["envelope"]["dataMessage"]["isExpirationUpdate"] = True
    assert await _dispatch(_adapter(monkeypatch), env) is None


@pytest.mark.asyncio
async def test_metadata_holds_no_message_content(monkeypatch):
    ev = await _dispatch(_adapter(monkeypatch), _envelope(expires=60, text="a private sentence"))
    assert "private" not in repr(ev.metadata)
