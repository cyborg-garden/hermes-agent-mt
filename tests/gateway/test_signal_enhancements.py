"""Tests for Signal adapter enhancements: UUID allowlisting, group invite policy, profile name."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

def _make_signal_adapter(monkeypatch, account="+15551234567", **extra):
    """Create a SignalAdapter with sensible test defaults."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", extra.pop("group_allowed", ""))
    if "allowed_users" in extra:
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", extra.pop("allowed_users"))
    if "group_invite_policy" in extra:
        monkeypatch.setenv("SIGNAL_GROUP_INVITE_POLICY", extra.pop("group_invite_policy"))
    if "profile_name" in extra:
        monkeypatch.setenv("SIGNAL_PROFILE_NAME", extra.pop("profile_name"))
    from plugins.platforms.signal.adapter import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "http_url": "http://localhost:8080",
        "account": account,
        **extra,
    }
    return SignalAdapter(config)


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    from plugins.platforms.signal.signal_rate_limit import _reset_scheduler
    _reset_scheduler()
    yield
    _reset_scheduler()


# ---------------------------------------------------------------------------
# 2.1 UUID-Based Allowlisting
# ---------------------------------------------------------------------------

class TestUUIDAallowlisting:
    """UUID resolution for DM and group allowlists."""

    def test_init_creates_empty_uuid_sets(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        assert isinstance(adapter.dm_allow_from_uuids, set)
        assert isinstance(adapter.group_allow_from_uuids, set)
        assert len(adapter.dm_allow_from_uuids) == 0
        assert len(adapter.group_allow_from_uuids) == 0

    def test_uuid_entries_in_allowlist_added_directly(self, monkeypatch):
        """UUIDs already in the allowlist should be added to uuid sets on resolve."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,+15559999999",
        )
        # UUID is in dm_allow_from
        assert "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" in adapter.dm_allow_from

    @pytest.mark.asyncio
    async def test_resolve_allowlist_uuids_from_contacts(self, monkeypatch):
        """Phone numbers in allowlists should be resolved via listContacts."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="+15559999999",
        )

        contacts_response = [
            {
                "number": "+15559999999",
                "uuid": "11111111-1111-1111-1111-111111111111",
            }
        ]

        async def mock_rpc(method, params, **kwargs):
            if method == "listContacts":
                return contacts_response
            if method == "getUserStatus":
                return []
            return None

        adapter._rpc = mock_rpc
        await adapter._resolve_allowlist_uuids()

        assert "11111111-1111-1111-1111-111111111111" in adapter.dm_allow_from_uuids
        assert "+15559999999" in adapter._recipient_uuid_by_number
        assert adapter._recipient_uuid_by_number["+15559999999"] == "11111111-1111-1111-1111-111111111111"

    @pytest.mark.asyncio
    async def test_resolve_allowlist_uuids_fallback_to_getUserStatus(self, monkeypatch):
        """Numbers not in contacts should fall back to getUserStatus."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="+15558888888",
        )

        async def mock_rpc(method, params, **kwargs):
            if method == "listContacts":
                return []  # no contacts match
            if method == "getUserStatus":
                return [{"uuid": "22222222-2222-2222-2222-222222222222"}]
            return None

        adapter._rpc = mock_rpc
        await adapter._resolve_allowlist_uuids()

        assert "22222222-2222-2222-2222-222222222222" in adapter.dm_allow_from_uuids

    @pytest.mark.asyncio
    async def test_resolve_allowlist_skips_wildcards(self, monkeypatch):
        """Wildcard '*' entries should not be resolved."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="*",
        )

        call_count = 0

        async def mock_rpc(method, params, **kwargs):
            nonlocal call_count
            call_count += 1
            return []

        adapter._rpc = mock_rpc
        await adapter._resolve_allowlist_uuids()

        # Should not have called listContacts — nothing to resolve
        assert call_count == 0

    def test_lazy_resolution_on_remember_identifiers(self, monkeypatch):
        """_remember_recipient_identifiers should lazily add UUIDs to allowlist sets."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="+15557777777",
        )

        assert len(adapter.dm_allow_from_uuids) == 0

        adapter._remember_recipient_identifiers("+15557777777", "33333333-3333-3333-3333-333333333333")

        assert "33333333-3333-3333-3333-333333333333" in adapter.dm_allow_from_uuids

    def test_lazy_resolution_group_allowlist(self, monkeypatch):
        """Lazy resolution should also work for group allowlists."""
        adapter = _make_signal_adapter(
            monkeypatch,
            group_allowed="+15556666666",
        )

        assert len(adapter.group_allow_from_uuids) == 0

        adapter._remember_recipient_identifiers("+15556666666", "44444444-4444-4444-4444-444444444444")

        assert "44444444-4444-4444-4444-444444444444" in adapter.group_allow_from_uuids

    def test_lazy_resolution_skips_non_allowlist_numbers(self, monkeypatch):
        """Numbers not in any allowlist should not be added to uuid sets."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="+15551111111",
        )

        adapter._remember_recipient_identifiers("+15552222222", "55555555-5555-5555-5555-555555555555")

        assert "55555555-5555-5555-5555-555555555555" not in adapter.dm_allow_from_uuids


# ---------------------------------------------------------------------------
# 2.4 Group Invite Policy
# ---------------------------------------------------------------------------

class TestGroupInvitePolicy:
    """SIGNAL_GROUP_INVITE_POLICY controls auto-accept of group invites."""

    def test_default_policy_is_approved_only(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        assert adapter.group_invite_policy == "approved-only"

    def test_allow_all_policy(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            group_invite_policy="allow-all",
        )
        assert adapter.group_invite_policy == "allow-all"

    def test_policy_strips_whitespace(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            group_invite_policy="  Allow-All  ",
        )
        assert adapter.group_invite_policy == "allow-all"

    @pytest.mark.asyncio
    async def test_group_invite_accepted_from_approved_user(self, monkeypatch):
        """Group invite from approved user should be auto-accepted."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="+15559999999",
        )

        rpc_calls = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            rpc_calls.append(method)
            if method == "joinGroup":
                return {"success": True}
            return None

        adapter._rpc = mock_rpc

        envelope = {
            "envelope": {
                "sourceNumber": "+15559999999",
                "sourceUuid": "66666666-6666-6666-6666-666666666666",
                "sourceName": "Approved User",
                "groupV2": {"groupId": "test-group-id-123"},
            }
        }

        await adapter._handle_envelope(envelope)

        assert "joinGroup" in rpc_calls

    @pytest.mark.asyncio
    async def test_group_invite_rejected_from_unapproved_user(self, monkeypatch):
        """Group invite from unapproved user should be rejected (approved-only policy)."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="+15551111111",
        )

        rpc_calls = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            rpc_calls.append(method)
            return {"success": True}

        adapter._rpc = mock_rpc

        envelope = {
            "envelope": {
                "sourceNumber": "+15559999999",
                "sourceUuid": "77777777-7777-7777-7777-777777777777",
                "sourceName": "Unknown User",
                "groupV2": {"groupId": "test-group-id-456"},
            }
        }

        await adapter._handle_envelope(envelope)

        # Should NOT have called joinGroup
        assert "joinGroup" not in rpc_calls

    @pytest.mark.asyncio
    async def test_group_invite_accepted_with_allow_all_policy(self, monkeypatch):
        """Group invite from anyone should be accepted with allow-all policy."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="+15551111111",
            group_invite_policy="allow-all",
        )

        rpc_calls = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            rpc_calls.append(method)
            return {"success": True}

        adapter._rpc = mock_rpc

        envelope = {
            "envelope": {
                "sourceNumber": "+15559999999",
                "sourceUuid": "88888888-8888-8888-8888-888888888888",
                "sourceName": "Random User",
                "groupV2": {"groupId": "test-group-id-789"},
            }
        }

        await adapter._handle_envelope(envelope)

        assert "joinGroup" in rpc_calls

    @pytest.mark.asyncio
    async def test_group_invite_accepted_with_wildcard_allowlist(self, monkeypatch):
        """Group invite should be accepted when dm_allow_from contains '*'."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="*",
        )

        rpc_calls = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            rpc_calls.append(method)
            return {"success": True}

        adapter._rpc = mock_rpc

        envelope = {
            "envelope": {
                "sourceNumber": "+15559999999",
                "sourceUuid": "99999999-9999-9999-9999-999999999999",
                "sourceName": "Any User",
                "groupV2": {"groupId": "test-group-wildcard"},
            }
        }

        await adapter._handle_envelope(envelope)

        assert "joinGroup" in rpc_calls

    @pytest.mark.asyncio
    async def test_group_invite_fallback_to_updateGroup(self, monkeypatch):
        """If joinGroup fails, should fall back to updateGroup."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="+15559999999",
        )

        rpc_calls = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            rpc_calls.append(method)
            if method == "joinGroup":
                raise Exception("joinGroup not supported")
            if method == "updateGroup":
                return {"success": True}
            return None

        adapter._rpc = mock_rpc

        envelope = {
            "envelope": {
                "sourceNumber": "+15559999999",
                "sourceUuid": "66666666-6666-6666-6666-666666666666",
                "sourceName": "Approved User",
                "groupV2": {"groupId": "test-group-fallback"},
            }
        }

        await adapter._handle_envelope(envelope)

        assert "joinGroup" in rpc_calls
        assert "updateGroup" in rpc_calls

    @pytest.mark.asyncio
    async def test_group_invite_uuid_match(self, monkeypatch):
        """Group invite should be accepted when inviter's UUID is in the resolved allowlist."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="+15559999999",
        )
        # Pre-populate UUID resolution
        adapter.dm_allow_from_uuids.add("aabbccdd-aabb-ccdd-eeff-aabbccddeeff")

        rpc_calls = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            rpc_calls.append(method)
            return {"success": True}

        adapter._rpc = mock_rpc

        # Envelope where sender is identified only by UUID (no number match)
        envelope = {
            "envelope": {
                "sourceNumber": "+15550000000",  # not in allowlist
                "sourceUuid": "aabbccdd-aabb-ccdd-eeff-aabbccddeeff",  # IS in uuid set
                "sourceName": "UUID User",
                "groupV2": {"groupId": "test-group-uuid"},
            }
        }

        await adapter._handle_envelope(envelope)

        assert "joinGroup" in rpc_calls

    @pytest.mark.asyncio
    async def test_accepted_group_added_to_runtime_allowlist(self, monkeypatch):
        """After accepting a group invite, the group ID should be added to
        the runtime allowlist so subsequent messages are not dropped."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="+15559999999",
        )

        # Initially no groups in allowlist
        assert len(adapter.group_allow_from) == 0

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            return {"success": True}

        adapter._rpc = mock_rpc

        envelope = {
            "envelope": {
                "sourceNumber": "+15559999999",
                "sourceUuid": "66666666-6666-6666-6666-666666666666",
                "sourceName": "Approved User",
                "groupV2": {"groupId": "new-group-abc123"},
            }
        }

        await adapter._handle_envelope(envelope)

        # Group should now be in the runtime allowlist
        assert "new-group-abc123" in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_failed_join_does_not_add_to_allowlist(self, monkeypatch):
        """If both joinGroup and updateGroup fail, group should NOT be added
        to the allowlist."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="+15559999999",
        )

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            raise Exception(f"{method} failed")

        adapter._rpc = mock_rpc

        envelope = {
            "envelope": {
                "sourceNumber": "+15559999999",
                "sourceUuid": "66666666-6666-6666-6666-666666666666",
                "sourceName": "Approved User",
                "groupV2": {"groupId": "failed-group-xyz"},
            }
        }

        await adapter._handle_envelope(envelope)

        # Group should NOT be in the allowlist since join failed
        assert "failed-group-xyz" not in adapter.group_allow_from


# ---------------------------------------------------------------------------
# 2.7 Profile Name Setting
# ---------------------------------------------------------------------------

class TestProfileNameSetting:
    """SIGNAL_PROFILE_NAME sets bot profile via updateProfile on connect."""

    @pytest.mark.asyncio
    async def test_set_profile_name_calls_updateProfile(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            profile_name="TestBot",
        )

        rpc_calls = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            rpc_calls.append({"method": method, "params": dict(params)})
            return {"success": True}

        adapter._rpc = mock_rpc
        await adapter._set_profile_name()

        assert len(rpc_calls) == 1
        assert rpc_calls[0]["method"] == "updateProfile"
        assert rpc_calls[0]["params"]["givenName"] == "TestBot"
        assert rpc_calls[0]["params"]["account"] == "+15551234567"

    @pytest.mark.asyncio
    async def test_set_profile_name_noop_when_not_set(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)

        rpc_calls = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            rpc_calls.append(method)
            return {"success": True}

        adapter._rpc = mock_rpc
        await adapter._set_profile_name()

        # Should not have called updateProfile
        assert "updateProfile" not in rpc_calls

    @pytest.mark.asyncio
    async def test_set_profile_name_handles_failure_gracefully(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            profile_name="FailBot",
        )

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            if method == "updateProfile":
                raise Exception("RPC timeout")

        adapter._rpc = mock_rpc

        # Should not raise — failures are logged but non-fatal
        await adapter._set_profile_name()

    @pytest.mark.asyncio
    async def test_set_profile_name_handles_null_result(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            profile_name="NullBot",
        )

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            return None  # updateProfile returned no result

        adapter._rpc = mock_rpc

        # Should not raise
        await adapter._set_profile_name()
