"""Tests for explicit group approval via "@bot /approve-group".

Current signal-cli delivers "bot added to group" as a dataMessage with
groupInfo.type == "UPDATE", which carries no editor / added-member detail, so
approval cannot be safely inferred from it (issue #56). Instead an approved
user approves a group explicitly by sending /approve-group in it.

Covers:
  - approved sender -> approved + persisted + exactly one reply
  - unapproved sender -> nothing, no reply
  - spacing / case / @mention variants; other text is not the command
  - SIGNAL_REQUIRE_MENTION=true requires the mention
  - plain messages in unapproved groups still dropped
  - already-approved group -> no-op, not forwarded to the agent
  - sealed-sender (UUID-only) approved sender works
  - unset / empty SIGNAL_ALLOWED_USERS never approves (fail-closed)
  - explicit '*' / allow-all flags approve, mirroring run.py
  - command in a DM approves nothing
  - UPDATE envelopes approve nothing
  - legacy groupV2 invite path unchanged
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
BOT = "+15551112222"
BOT_UUID = "99999999-9999-9999-9999-999999999999"


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    from gateway.platforms.signal_rate_limit import _reset_scheduler
    _reset_scheduler()
    yield
    _reset_scheduler()


def _adapter(monkeypatch, tmp_path, allowed_users=APPROVED, **kw):
    """Build an adapter with a hermetic environment.

    ``allowed_users=None`` leaves SIGNAL_ALLOWED_USERS unset. Env vars other
    tests may leak into os.environ (test_config writes SIGNAL_REQUIRE_MENTION)
    are cleared, and require_mention is always passed explicitly so the result
    never depends on test order.
    """
    for var in (
        "SIGNAL_REQUIRE_MENTION",
        "SIGNAL_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "SIGNAL_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "SIGNAL_GROUP_INVITE_POLICY",
    ):
        monkeypatch.delenv(var, raising=False)
    kw.setdefault("require_mention", False)
    if allowed_users is not None:
        kw["allowed_users"] = allowed_users
    adapter = _make_signal_adapter(monkeypatch, tmp_path, account=BOT, **kw)
    adapter._own_uuid = BOT_UUID
    adapter._account_normalized = BOT
    calls = []

    async def mock_rpc(method, params, rpc_id=None, **kwargs):
        calls.append(method)
        return {"success": True}

    adapter._rpc = mock_rpc
    adapter.send = AsyncMock()
    adapter.handle_message = AsyncMock()
    adapter._rpc_calls = calls
    return adapter


def _group_msg(text, *, group_id=GID, sender=APPROVED, sender_uuid=APPROVED_UUID,
               mention_bot=False, group_type="DELIVER"):
    data_message = {
        "timestamp": 1760000000000,
        "message": text,
        "groupInfo": {"groupId": group_id, "type": group_type},
    }
    if mention_bot:
        data_message["message"] = "￼ " + text
        data_message["mentions"] = [{"start": 0, "length": 1, "uuid": BOT_UUID}]
    env = {"sourceUuid": sender_uuid, "sourceName": "Someone", "dataMessage": data_message}
    if sender:
        env["sourceNumber"] = sender
    return {"envelope": env}


def _persisted(tmp_path):
    path = tmp_path / _APPROVED_GROUPS_FILE
    return json.loads(path.read_text()) if path.exists() else {}


class TestApprovedSender:

    @pytest.mark.asyncio
    async def test_approves_persists_and_replies_once(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path)

        await adapter._handle_envelope(_group_msg("/approve-group", mention_bot=True))

        assert GID in adapter.group_allow_from
        entry = _persisted(tmp_path)[GID]
        assert entry["added_by"] == APPROVED
        assert entry["added_by_uuid"] == APPROVED_UUID
        adapter.send.assert_awaited_once_with(f"group:{GID}", "This group is approved.")
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_next_message_after_approval_is_processed(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path)
        await adapter._handle_envelope(_group_msg("/approve-group", mention_bot=True))

        await adapter._handle_envelope(_group_msg("what's for dinner?", mention_bot=True))

        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", [
        "/approve-group",
        "  /approve-group  ",
        "/APPROVE-GROUP",
        "/Approve-Group\n",
        f"@{BOT} /approve-group",
        f"/approve-group @{BOT}",
        f"@{BOT.lstrip('+')} /approve-group",
        f"@{BOT_UUID} /approve-group",
    ])
    async def test_variants_approve(self, monkeypatch, tmp_path, text):
        adapter = _adapter(monkeypatch, tmp_path)

        await adapter._handle_envelope(_group_msg(text))

        assert GID in adapter.group_allow_from
        adapter.send.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", [
        "/approve-group please",
        "please /approve-group",
        "/approve-groups",
        "/approve group",
        f"@{STRANGER} /approve-group",
        "/approve-group /approve-group",
    ])
    async def test_other_text_is_not_the_command(self, monkeypatch, tmp_path, text):
        adapter = _adapter(monkeypatch, tmp_path)

        await adapter._handle_envelope(_group_msg(text))

        assert GID not in adapter.group_allow_from
        assert _persisted(tmp_path) == {}
        adapter.send.assert_not_called()
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_sealed_sender_uuid_only_in_allowlist(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path, allowed_users=APPROVED_UUID)

        await adapter._handle_envelope(_group_msg("/approve-group", sender=None))

        assert GID in adapter.group_allow_from
        assert _persisted(tmp_path)[GID]["added_by_uuid"] == APPROVED_UUID
        adapter.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sealed_sender_matched_via_resolved_uuid(self, monkeypatch, tmp_path):
        # Allowlist holds the phone; startup resolution mapped it to the ACI.
        adapter = _adapter(monkeypatch, tmp_path, allowed_users=APPROVED)
        adapter.dm_allow_from_uuids.add(APPROVED_UUID)

        await adapter._handle_envelope(_group_msg("/approve-group", sender=None))

        assert GID in adapter.group_allow_from
        adapter.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_persist_failure_still_approves(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path)

        def boom(*a, **k):
            raise RuntimeError("disk full")

        adapter._persist_approved_group = boom
        await adapter._handle_envelope(_group_msg("/approve-group"))

        assert GID in adapter.group_allow_from
        adapter.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reply_failure_is_non_fatal(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path)
        adapter.send = AsyncMock(side_effect=RuntimeError("send failed"))

        await adapter._handle_envelope(_group_msg("/approve-group"))

        assert GID in adapter.group_allow_from


class TestUnapprovedSender:

    @pytest.mark.asyncio
    async def test_stranger_command_is_silently_ignored(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path)

        await adapter._handle_envelope(_group_msg(
            "/approve-group", sender=STRANGER, sender_uuid=STRANGER_UUID, mention_bot=True,
        ))

        assert GID not in adapter.group_allow_from
        assert _persisted(tmp_path) == {}
        adapter.send.assert_not_called()
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_stranger_uuid_only_is_ignored(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path)

        await adapter._handle_envelope(_group_msg(
            "/approve-group", sender=None, sender_uuid=STRANGER_UUID,
        ))

        assert GID not in adapter.group_allow_from
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_allow_all_policy_does_not_open_the_command(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path, group_invite_policy="allow-all")

        await adapter._handle_envelope(_group_msg(
            "/approve-group", sender=STRANGER, sender_uuid=STRANGER_UUID,
        ))

        assert GID not in adapter.group_allow_from
        adapter.send.assert_not_called()


class TestRequireMention:

    @pytest.mark.asyncio
    async def test_require_mention_rejects_bare_command(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path, require_mention=True)

        await adapter._handle_envelope(_group_msg("/approve-group"))

        assert GID not in adapter.group_allow_from
        adapter.send.assert_not_called()
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_require_mention_accepts_metadata_mention(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path, require_mention=True)

        await adapter._handle_envelope(_group_msg("/approve-group", mention_bot=True))

        assert GID in adapter.group_allow_from
        adapter.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_require_mention_rejects_typed_text_mention(self, monkeypatch, tmp_path):
        # "@+1555..." typed as plain text is not a Signal mention: no metadata.
        adapter = _adapter(monkeypatch, tmp_path, require_mention=True)

        await adapter._handle_envelope(_group_msg(f"@{BOT} /approve-group"))

        assert GID not in adapter.group_allow_from
        adapter.send.assert_not_called()


class TestExplicitAllowlistRequired:
    """Round-3 audit: an unset SIGNAL_ALLOWED_USERS must never approve.

    dm_allow_from defaults to {"*"} when the variable is unset; the approval
    gate mirrors run.py's fail-closed _is_individual_allowlisted instead.
    """

    @pytest.mark.asyncio
    async def test_unset_allowlist_stranger_cannot_approve(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path, allowed_users=None)
        assert "*" in adapter.dm_allow_from  # the fail-open default being guarded

        await adapter._handle_envelope(_group_msg(
            "/approve-group", sender=STRANGER, sender_uuid=STRANGER_UUID, mention_bot=True,
        ))

        assert GID not in adapter.group_allow_from
        assert _persisted(tmp_path) == {}
        adapter.send.assert_not_called()
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_allowlist_cannot_approve(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path, allowed_users="")

        await adapter._handle_envelope(_group_msg("/approve-group", mention_bot=True))

        assert GID not in adapter.group_allow_from
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_listed_phone_approves(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path, allowed_users=f"+15553333333,{APPROVED}")

        await adapter._handle_envelope(_group_msg("/approve-group", mention_bot=True))

        assert GID in adapter.group_allow_from
        adapter.send.assert_awaited_once_with(f"group:{GID}", "This group is approved.")

    @pytest.mark.asyncio
    async def test_listed_uuid_sealed_sender_approves(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path, allowed_users=APPROVED_UUID)

        await adapter._handle_envelope(_group_msg(
            "/approve-group", sender=None, sender_uuid=APPROVED_UUID, mention_bot=True,
        ))

        assert GID in adapter.group_allow_from
        adapter.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_explicit_wildcard_approves_like_run_py(self, monkeypatch, tmp_path):
        # run.py's _is_individual_allowlisted treats an explicit "*" as allow-all.
        adapter = _adapter(monkeypatch, tmp_path, allowed_users="*")

        await adapter._handle_envelope(_group_msg(
            "/approve-group", sender=STRANGER, sender_uuid=STRANGER_UUID, mention_bot=True,
        ))

        assert GID in adapter.group_allow_from
        adapter.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gateway_allowed_users_listed_approves(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path, allowed_users=None)
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", APPROVED)

        await adapter._handle_envelope(_group_msg("/approve-group", mention_bot=True))

        assert GID in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_signal_allow_all_users_approves(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path, allowed_users=None)
        monkeypatch.setenv("SIGNAL_ALLOW_ALL_USERS", "true")

        await adapter._handle_envelope(_group_msg(
            "/approve-group", sender=STRANGER, sender_uuid=STRANGER_UUID, mention_bot=True,
        ))

        assert GID in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_resolved_uuid_without_explicit_list_cannot_approve(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path, allowed_users=None)
        adapter.dm_allow_from_uuids.add(STRANGER_UUID)

        await adapter._handle_envelope(_group_msg(
            "/approve-group", sender=None, sender_uuid=STRANGER_UUID, mention_bot=True,
        ))

        assert GID not in adapter.group_allow_from
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_legacy_invite_unset_allowlist_unchanged(self, monkeypatch, tmp_path):
        # Legacy groupV2 invite path keeps its old semantics (dm_allow_from).
        adapter = _adapter(monkeypatch, tmp_path, allowed_users=None)

        calls = await _send_invite_envelope(
            adapter, GID, sender=STRANGER, sender_uuid=STRANGER_UUID,
        )

        assert "joinGroup" in calls
        assert GID in adapter.group_allow_from


class TestEverythingElseUnchanged:

    @pytest.mark.asyncio
    async def test_plain_message_in_unapproved_group_dropped(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path)

        await adapter._handle_envelope(_group_msg("hello bot", mention_bot=True))

        assert GID not in adapter.group_allow_from
        adapter.send.assert_not_called()
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_plain_message_with_other_groups_allowed_dropped(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path, group_allowed="other-group")

        await adapter._handle_envelope(_group_msg("hello bot", mention_bot=True))

        assert GID not in adapter.group_allow_from
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_already_allowed_group_is_noop(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path, group_allowed=GID)

        await adapter._handle_envelope(_group_msg("/approve-group", mention_bot=True))

        assert _persisted(tmp_path) == {}
        adapter.send.assert_not_called()
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_wildcard_allowed_is_noop(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path, group_allowed="*")

        await adapter._handle_envelope(_group_msg("/approve-group"))

        assert GID not in adapter.group_allow_from
        assert _persisted(tmp_path) == {}
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_command_in_dm_approves_nothing(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path)
        before = set(adapter.group_allow_from)
        env = {"envelope": {
            "sourceNumber": APPROVED, "sourceUuid": APPROVED_UUID, "sourceName": "A",
            "dataMessage": {"timestamp": 1760000000000, "message": "/approve-group"},
        }}

        await adapter._handle_envelope(env)

        assert adapter.group_allow_from == before
        assert _persisted(tmp_path) == {}
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sender,uuid", [(APPROVED, APPROVED_UUID), (STRANGER, STRANGER_UUID)])
    async def test_update_envelope_approves_nothing(self, monkeypatch, tmp_path, sender, uuid):
        adapter = _adapter(monkeypatch, tmp_path)
        env = {"envelope": {
            "sourceNumber": sender, "sourceUuid": uuid, "sourceName": "A",
            "dataMessage": {
                "timestamp": 1760000000000,
                "groupInfo": {"groupId": GID, "type": "UPDATE", "revision": 2},
            },
        }}

        await adapter._handle_envelope(env)

        assert GID not in adapter.group_allow_from
        assert _persisted(tmp_path) == {}
        assert adapter._rpc_calls == []
        adapter.send.assert_not_called()
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_envelope_with_command_text_approves_nothing(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path)

        await adapter._handle_envelope(_group_msg("/approve-group", group_type="UPDATE"))

        assert GID not in adapter.group_allow_from
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_legacy_groupv2_invite_still_approves(self, monkeypatch, tmp_path):
        adapter = _adapter(monkeypatch, tmp_path)

        calls = await _send_invite_envelope(adapter, GID, sender=APPROVED, sender_uuid=APPROVED_UUID)

        assert "joinGroup" in calls
        assert GID in adapter.group_allow_from
