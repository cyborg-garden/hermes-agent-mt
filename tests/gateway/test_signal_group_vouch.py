"""swarm-map#102: a group the bot joined at creation gets approved when an
approved user @mentions the bot there.

Groups created with the bot already in them never send an invite envelope,
so under SIGNAL_GROUP_INVITE_POLICY=approved-only they stayed unapproved and
someone had to add the group id in Swarm Map by hand. Approval must be an act
by an approved user (#56), not mere co-membership.
"""

import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig

BOT_PHONE = "+15551234567"
BOT_ACI = "5eca7c21-1111-2222-3333-000000000099"
ADMIN_ACI = "aaaa0000-1111-2222-3333-000000000001"
STRANGER_ACI = "bbbb0000-1111-2222-3333-000000000002"
GROUP_ID = "bmV3LWdyb3VwLWlk"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for var in ("SIGNAL_ALLOWED_USERS", "SIGNAL_GROUP_ALLOWED_USERS", "SIGNAL_ALLOW_ALL_USERS",
                "SIGNAL_REQUIRE_MENTION", "SIGNAL_OBSERVE_UNMENTIONED", "SIGNAL_PROFILE_NAME",
                "SIGNAL_GROUP_INVITE_POLICY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _adapter(monkeypatch, allowed=ADMIN_ACI, groups=""):
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", allowed)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", groups)
    from gateway.platforms.signal import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {"http_url": "http://localhost:8080", "account": BOT_PHONE,
                    "require_mention": True}
    a = SignalAdapter(config)
    a._own_uuid = BOT_ACI
    return a


def _envelope(sender=ADMIN_ACI, mention=True, text="kia ora"):
    dm = {"message": text, "timestamp": 1750000000000, "groupV2": {"id": GROUP_ID}}
    if mention:
        dm["mentions"] = [{"uuid": BOT_ACI, "start": 0, "length": 1}]
        dm["message"] = "￼ " + text
    return {"envelope": {"sourceUuid": sender, "source": sender, "sourceName": "Juniper",
                         "timestamp": 1750000000000, "dataMessage": dm}}


async def _dispatch(adapter, envelope):
    got = {}

    async def _capture(event):
        got["event"] = event

    adapter.handle_message = AsyncMock(side_effect=_capture)
    await adapter._handle_envelope(envelope)
    return got.get("event")


@pytest.mark.asyncio
async def test_approved_user_mentioning_the_bot_approves_the_group(monkeypatch, _env):
    a = _adapter(monkeypatch)
    ev = await _dispatch(a, _envelope())
    assert ev is not None
    assert GROUP_ID in a.group_allow_from
    saved = json.loads((_env / "signal_approved_groups.json").read_text())
    assert saved[GROUP_ID]["added_by_uuid"] == ADMIN_ACI


@pytest.mark.asyncio
async def test_after_approval_other_members_can_talk_to_the_bot(monkeypatch):
    a = _adapter(monkeypatch)
    await _dispatch(a, _envelope())
    ev = await _dispatch(a, _envelope(sender=STRANGER_ACI))
    assert ev is not None and ev.source.chat_id == f"group:{GROUP_ID}"


@pytest.mark.asyncio
async def test_stranger_cannot_approve_a_group(monkeypatch):
    a = _adapter(monkeypatch)
    assert await _dispatch(a, _envelope(sender=STRANGER_ACI)) is None
    assert GROUP_ID not in a.group_allow_from


@pytest.mark.asyncio
async def test_approved_user_chatting_without_a_mention_does_not_approve(monkeypatch):
    a = _adapter(monkeypatch)
    assert await _dispatch(a, _envelope(mention=False)) is None
    assert GROUP_ID not in a.group_allow_from


@pytest.mark.asyncio
async def test_open_allowlist_vouches_for_nobody(monkeypatch):
    a = _adapter(monkeypatch, allowed="*")
    assert await _dispatch(a, _envelope()) is None
    assert GROUP_ID not in a.group_allow_from


@pytest.mark.asyncio
async def test_already_allowed_groups_are_unchanged(monkeypatch, _env):
    a = _adapter(monkeypatch, groups=GROUP_ID)
    ev = await _dispatch(a, _envelope(sender=STRANGER_ACI))
    assert ev is not None
    assert not (_env / "signal_approved_groups.json").exists()
