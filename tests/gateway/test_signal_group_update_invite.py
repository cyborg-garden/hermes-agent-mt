"""Tests for treating a dataMessage groupInfo.type == "UPDATE" as a group invite.

Current signal-cli (JSON-RPC daemon) delivers "you were added to a group"
(including remove + re-add) as a dataMessage whose groupInfo has
type "UPDATE", not as a top-level groupV2 envelope. Without handling it the
group never enters group_allow_from and every later message is dropped.

Covers:
  - approved sender UPDATE -> approved + persisted (UUID / sealed-sender too)
  - unapproved sender UPDATE -> not approved, not persisted
  - UPDATE for an already-allowed group -> no-op (no RPC, no re-persist)
  - allow-all policy -> any sender approved
  - already a full member -> no joinGroup/updateGroup RPC
  - not a member and accept RPCs fail -> not approved
  - UPDATE never forwarded to the agent as a message
  - classic groupV2 invite path unchanged
  - Note-to-Self sync with UPDATE groupInfo is not treated as an invite
"""
import json

import pytest
from unittest.mock import AsyncMock

from gateway.platforms.signal import _APPROVED_GROUPS_FILE
from tests.gateway.test_signal_group_persistence import (
    _make_signal_adapter,
    _send_invite_envelope,
)

APPROVED = "+15559999999"
APPROVED_UUID = "66666666-6666-6666-6666-666666666666"
STRANGER = "+15550000000"
STRANGER_UUID = "77777777-7777-7777-7777-777777777777"
GID = "groceries-group-id-0001"


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    from gateway.platforms.signal_rate_limit import _reset_scheduler
    _reset_scheduler()
    yield
    _reset_scheduler()


def _update_envelope(group_id=GID, sender=APPROVED, sender_uuid=APPROVED_UUID,
                     revision=2, message=None):
    data_message = {
        "timestamp": 1760000000000,
        "groupInfo": {"groupId": group_id, "type": "UPDATE", "revision": revision},
    }
    if message is not None:
        data_message["message"] = message
    env = {"sourceUuid": sender_uuid, "sourceName": "Someone", "dataMessage": data_message}
    if sender:
        env["sourceNumber"] = sender
    return {"envelope": env}


def _wire(adapter, *, is_member=True, list_groups_ok=True, accept_ok=True):
    calls = []

    async def mock_rpc(method, params, rpc_id=None, **kwargs):
        calls.append(method)
        if method == "listGroups":
            if not list_groups_ok:
                raise RuntimeError("listGroups failed")
            return [{"id": GID, "isMember": is_member, "isBlocked": False}]
        if method in ("joinGroup", "updateGroup") and not accept_ok:
            raise RuntimeError(f"{method} failed")
        return {"success": True}

    adapter._rpc = mock_rpc
    adapter.handle_message = AsyncMock()
    return calls


def _persisted(tmp_path):
    path = tmp_path / _APPROVED_GROUPS_FILE
    return json.loads(path.read_text()) if path.exists() else {}


class TestUpdateFromApprovedSender:

    @pytest.mark.asyncio
    async def test_approved_sender_update_approves_and_persists(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users=APPROVED)
        _wire(adapter)

        await adapter._handle_envelope(_update_envelope())

        assert GID in adapter.group_allow_from
        entry = _persisted(tmp_path)[GID]
        assert entry["added_by"] == APPROVED
        assert entry["added_by_uuid"] == APPROVED_UUID

    @pytest.mark.asyncio
    async def test_already_member_skips_join_rpc(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users=APPROVED)
        calls = _wire(adapter, is_member=True)

        await adapter._handle_envelope(_update_envelope())

        assert "joinGroup" not in calls and "updateGroup" not in calls
        assert GID in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_pending_member_runs_accept_rpc(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users=APPROVED)
        calls = _wire(adapter, is_member=False)

        await adapter._handle_envelope(_update_envelope())

        assert "joinGroup" in calls
        assert GID in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_not_member_and_accept_fails_does_not_approve(self, monkeypatch, tmp_path):
        """e.g. the UPDATE is the bot being *removed* — never approve or rejoin."""
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users=APPROVED)
        _wire(adapter, is_member=False, accept_ok=False)

        await adapter._handle_envelope(_update_envelope())

        assert GID not in adapter.group_allow_from
        assert GID not in _persisted(tmp_path)

    @pytest.mark.asyncio
    async def test_sealed_sender_uuid_only_approved_via_resolved_uuid(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users=APPROVED)
        adapter.dm_allow_from_uuids.add(APPROVED_UUID)
        _wire(adapter)

        await adapter._handle_envelope(_update_envelope(sender=None))

        assert GID in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_approved_by_uuid_in_allowlist(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users=APPROVED_UUID)
        _wire(adapter)

        await adapter._handle_envelope(_update_envelope(sender=None))

        assert GID in adapter.group_allow_from


class TestUpdateRejected:

    @pytest.mark.asyncio
    async def test_unapproved_sender_update_not_approved(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users=APPROVED)
        calls = _wire(adapter)

        await adapter._handle_envelope(
            _update_envelope(sender=STRANGER, sender_uuid=STRANGER_UUID)
        )

        assert GID not in adapter.group_allow_from
        assert GID not in _persisted(tmp_path)
        assert calls == []
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_unapproved_sealed_sender_not_approved(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users=APPROVED)
        _wire(adapter)

        await adapter._handle_envelope(
            _update_envelope(sender=None, sender_uuid=STRANGER_UUID)
        )

        assert GID not in adapter.group_allow_from


class TestUpdateAlreadyAllowed:

    @pytest.mark.asyncio
    async def test_already_allowed_group_is_noop(self, monkeypatch, tmp_path):
        """A title change by a non-approved member of an approved group does nothing."""
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users=APPROVED, group_allowed=GID,
        )
        calls = _wire(adapter)

        await adapter._handle_envelope(
            _update_envelope(sender=STRANGER, sender_uuid=STRANGER_UUID, revision=7)
        )

        assert calls == []
        assert GID not in _persisted(tmp_path)  # not re-persisted
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_wildcard_groups_is_noop(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users=APPROVED, group_allowed="*",
        )
        calls = _wire(adapter)

        await adapter._handle_envelope(_update_envelope())

        assert calls == []
        assert _persisted(tmp_path) == {}
        adapter.handle_message.assert_not_called()


class TestUpdateAllowAll:

    @pytest.mark.asyncio
    async def test_allow_all_policy_accepts_any_sender(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users=APPROVED, group_invite_policy="allow-all",
        )
        _wire(adapter)

        await adapter._handle_envelope(
            _update_envelope(sender=STRANGER, sender_uuid=STRANGER_UUID)
        )

        assert GID in adapter.group_allow_from
        assert GID in _persisted(tmp_path)


class TestUpdateNotForwarded:

    @pytest.mark.asyncio
    async def test_approving_update_not_forwarded(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users=APPROVED)
        _wire(adapter)

        await adapter._handle_envelope(_update_envelope())

        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_with_text_in_allowed_group_not_forwarded(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users=APPROVED, group_allowed=GID,
        )
        _wire(adapter)

        await adapter._handle_envelope(_update_envelope(message="injected text"))

        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_deliver_message_still_forwarded(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users=APPROVED, group_allowed=GID,
        )
        adapter.require_mention = False
        _wire(adapter)
        env = _update_envelope(message="milk please")
        env["envelope"]["dataMessage"]["groupInfo"]["type"] = "DELIVER"

        await adapter._handle_envelope(env)

        adapter.handle_message.assert_called_once()


class TestExistingPathsUnchanged:

    @pytest.mark.asyncio
    async def test_classic_groupv2_invite_still_works(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users=APPROVED)

        calls = await _send_invite_envelope(adapter, "classic-group-001")

        assert "joinGroup" in calls
        assert "listGroups" not in calls
        assert "classic-group-001" in adapter.group_allow_from
        assert "classic-group-001" in _persisted(tmp_path)

    @pytest.mark.asyncio
    async def test_classic_groupv2_invite_unapproved_rejected(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users=APPROVED)

        calls = await _send_invite_envelope(
            adapter, "classic-group-002", sender=STRANGER, sender_uuid=STRANGER_UUID,
        )

        assert calls == []
        assert "classic-group-002" not in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_note_to_self_sync_update_not_treated_as_invite(self, monkeypatch, tmp_path):
        account = "+15551234567"
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, account=account, allowed_users=account,
        )
        calls = _wire(adapter)
        env = {
            "envelope": {
                "sourceNumber": account,
                "sourceUuid": "88888888-8888-8888-8888-888888888888",
                "syncMessage": {
                    "sentMessage": {
                        "timestamp": 1760000000001,
                        "groupInfo": {"groupId": GID, "type": "UPDATE"},
                    }
                },
            }
        }

        await adapter._handle_envelope(env)

        assert "listGroups" not in calls
        assert GID not in adapter.group_allow_from
