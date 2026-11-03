"""Tests for Signal adapter group reconciliation (Gap G1 + issue #56 hardening).

When an admin *creates* a group with the agent already in it, no invite
envelope carrying the *adder's* identity is delivered — empirically, add-at-
creation surfaces only at connect-time ``listGroups``, with no record of who
added the bot. PR #53 closed the "messages silently dropped" gap by auto-
approving such groups on connect when an approved user was merely *present* as a
group admin.

Issue #56: that passive-presence predicate is a privilege-escalation path — a
non-approved actor can stand up a group, drop in (or co-opt) an approved admin,
and get the bot responding without that admin ever acting. Because nothing at
reconcile time records who added the bot, "added *by* an approved admin" cannot
be verified there. So under ``approved-only`` reconciliation no longer auto-
approves on admin presence; secure add-at-creation is recovered only when an
editor-bearing envelope is available (then via the invite handler, which already
checks the *inviter*).

What reconciliation still does:
* ``allow-all`` invite policy → approve any member group (operator opted in).
* open mode / wildcard allowlist → approve (no DM allowlist configured).
* persisted-reload → previously-approved groups stay approved across restarts.
* ``approved-only`` → never auto-approve on passive admin presence.
"""
import json
import pytest

from gateway.config import PlatformConfig


def _make_signal_adapter(monkeypatch, tmp_path, account="+15551234567", **extra):
    """Create a SignalAdapter with a tmp HERMES_HOME so tests never touch the real one.

    Sets HERMES_HOME (read live by ``get_hermes_home``) rather than monkeypatching
    the function object. Patching ``plugins.platforms.signal.adapter.get_hermes_home`` via a
    dotted string leaks across tests when the module is imported for the first time
    *during* the patch (monkeypatch records the already-patched lambda as the
    "original" and never restores the real function). See the matching helper in
    ``test_signal_group_persistence.py`` for the full root-cause writeup.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", extra.pop("group_allowed", ""))
    if "allowed_users" in extra:
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", extra.pop("allowed_users"))
    if "group_invite_policy" in extra:
        monkeypatch.setenv("SIGNAL_GROUP_INVITE_POLICY", extra.pop("group_invite_policy"))

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


def _group(gid, *, admin_number=None, admin_uuid=None, is_member=True,
           is_blocked=False, extra_members=None):
    """Build a listGroups-shaped group dict."""
    admins = []
    members = list(extra_members or [])
    if admin_number or admin_uuid:
        admin = {"number": admin_number, "uuid": admin_uuid}
        admins.append(admin)
        members.append({"number": admin_number, "uuid": admin_uuid, "isAdmin": True})
    return {
        "id": gid,
        "name": f"group {gid}",
        "isMember": is_member,
        "isBlocked": is_blocked,
        "members": members,
        "admins": admins,
    }


def _mock_list_groups(adapter, groups):
    """Patch adapter._rpc so listGroups returns `groups`; record calls."""
    calls = []

    async def mock_rpc(method, params=None, rpc_id=None, **kwargs):
        calls.append((method, params))
        if method == "listGroups":
            return groups
        return {"success": True}

    adapter._rpc = mock_rpc
    return calls


# ---------------------------------------------------------------------------
# Issue #56: approved-only must NOT auto-approve on passive admin presence
# ---------------------------------------------------------------------------

class TestReconcileDoesNotApproveOnAdminPresence:

    @pytest.mark.asyncio
    async def test_admin_presence_by_number_not_approved(self, monkeypatch, tmp_path):
        # An approved user is merely *present* as an admin — nobody recorded that
        # they *added* the bot. Under approved-only this must NOT auto-approve.
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")
        _mock_list_groups(adapter, [_group("grp-1", admin_number="+15559999999")])

        await adapter._reconcile_groups()

        assert "grp-1" not in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_admin_presence_by_uuid_not_approved(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")
        adapter.dm_allow_from_uuids.add("66666666-6666-6666-6666-666666666666")
        _mock_list_groups(
            adapter,
            [_group("grp-2", admin_uuid="66666666-6666-6666-6666-666666666666")],
        )

        await adapter._reconcile_groups()

        assert "grp-2" not in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_admin_presence_does_not_persist(self, monkeypatch, tmp_path):
        # The escalation path must leave no persisted trace either.
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")
        _mock_list_groups(adapter, [_group("grp-3", admin_number="+15559999999")])

        await adapter._reconcile_groups()

        from plugins.platforms.signal.adapter import _APPROVED_GROUPS_FILE
        path = tmp_path / _APPROVED_GROUPS_FILE
        if path.exists():
            assert "grp-3" not in json.loads(path.read_text())

    @pytest.mark.asyncio
    async def test_approved_only_skips_every_member_group(self, monkeypatch, tmp_path):
        # No approved admin, approved admin present — under approved-only neither
        # is auto-approved by reconciliation.
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")
        _mock_list_groups(
            adapter,
            [_group("grp-none", admin_number="+15550000000"),
             _group("grp-present", admin_number="+15559999999")],
        )

        await adapter._reconcile_groups()

        assert "grp-none" not in adapter.group_allow_from
        assert "grp-present" not in adapter.group_allow_from


# ---------------------------------------------------------------------------
# Ineligible groups are skipped even under policies that would otherwise approve
# ---------------------------------------------------------------------------

class TestReconcileSkipsIneligible:

    @pytest.mark.asyncio
    async def test_skips_non_member_group(self, monkeypatch, tmp_path):
        # open-mode would approve a member group; the is_member gate must still
        # skip a non-member one.
        adapter = _make_signal_adapter(monkeypatch, tmp_path)
        _mock_list_groups(adapter, [_group("grp-nm", admin_number="+15550000000", is_member=False)])

        await adapter._reconcile_groups()

        assert "grp-nm" not in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_skips_blocked_group(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path)
        _mock_list_groups(adapter, [_group("grp-bl", admin_number="+15550000000", is_blocked=True)])

        await adapter._reconcile_groups()

        assert "grp-bl" not in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_already_approved_group_not_repersisted(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users="+15559999999", group_allowed="grp-known"
        )
        _mock_list_groups(adapter, [_group("grp-known", admin_number="+15559999999")])

        await adapter._reconcile_groups()

        from plugins.platforms.signal.adapter import _APPROVED_GROUPS_FILE
        # Idempotent: an already-approved group should not create a persistence entry.
        path = tmp_path / _APPROVED_GROUPS_FILE
        if path.exists():
            assert "grp-known" not in json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Policy parity with the invite handler: allow-all / open-mode still approve
# ---------------------------------------------------------------------------

class TestReconcilePolicyParity:

    @pytest.mark.asyncio
    async def test_allow_all_policy_approves_any_member_group(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users="+15559999999",
            group_invite_policy="allow-all",
        )
        # No approved admin in the group, but policy is allow-all.
        _mock_list_groups(adapter, [_group("grp-aa", admin_number="+15550000000")])

        await adapter._reconcile_groups()

        assert "grp-aa" in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_open_mode_no_allowlist_approves_member_group(self, monkeypatch, tmp_path):
        # No SIGNAL_ALLOWED_USERS configured = open mode (mirrors invite handler).
        adapter = _make_signal_adapter(monkeypatch, tmp_path)
        _mock_list_groups(adapter, [_group("grp-open", admin_number="+15550000000")])

        await adapter._reconcile_groups()

        assert "grp-open" in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_open_mode_persists_reconciled_group(self, monkeypatch, tmp_path):
        # A policy-level approval is persisted so it survives a restart.
        adapter = _make_signal_adapter(monkeypatch, tmp_path)
        _mock_list_groups(adapter, [_group("grp-persist", admin_number="+15550000000")])

        await adapter._reconcile_groups()

        from plugins.platforms.signal.adapter import _APPROVED_GROUPS_FILE
        data = json.loads((tmp_path / _APPROVED_GROUPS_FILE).read_text())
        assert "grp-persist" in data


# ---------------------------------------------------------------------------
# Robustness: a malformed listGroups payload must never break startup
# ---------------------------------------------------------------------------

class TestReconcileRobustness:

    @pytest.mark.asyncio
    async def test_malformed_group_does_not_raise(self, monkeypatch, tmp_path):
        # A group with admins in an unexpected shape must not crash startup, and
        # must not be approved under approved-only.
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")
        bad = {
            "id": "grp-bad",
            "isMember": True,
            "isBlocked": False,
            "admins": ["+15559999999"],  # strings, not dicts
            "members": [],
        }
        _mock_list_groups(adapter, [bad])

        await adapter._reconcile_groups()  # must not raise

        assert "grp-bad" not in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_non_dict_group_entry_does_not_raise(self, monkeypatch, tmp_path):
        # open-mode would approve the well-formed group; a non-dict entry must be
        # skipped without breaking the loop.
        adapter = _make_signal_adapter(monkeypatch, tmp_path)
        _mock_list_groups(
            adapter,
            ["i am not a dict", _group("grp-ok", admin_number="+15550000000")],
        )

        await adapter._reconcile_groups()  # must not raise

        assert "grp-ok" in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_wildcard_groups_does_not_repersist_every_group(self, monkeypatch, tmp_path):
        # SIGNAL_GROUP_ALLOWED_USERS="*" already allows all groups — reconciliation
        # must not individually approve+persist each member group on every connect.
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users="+15559999999", group_allowed="*"
        )
        _mock_list_groups(
            adapter,
            [_group("grp-a", admin_number="+15559999999"),
             _group("grp-b", admin_number="+15559999999")],
        )

        await adapter._reconcile_groups()

        from plugins.platforms.signal.adapter import _APPROVED_GROUPS_FILE
        path = tmp_path / _APPROVED_GROUPS_FILE
        assert not path.exists(), "wildcard allow-all should not persist per-group entries"


# ---------------------------------------------------------------------------
# Wiring: connect() runs reconciliation after a successful health check
# ---------------------------------------------------------------------------

class TestReconcileWiredIntoConnect:

    @pytest.mark.asyncio
    async def test_connect_invokes_reconcile_groups(self, monkeypatch, tmp_path):
        from unittest.mock import AsyncMock, MagicMock, patch

        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client.aclose = AsyncMock()

        # Replace the network-touching collaborators so connect() stays local.
        monkeypatch.setattr(adapter, "_resolve_allowlist_uuids", AsyncMock())
        monkeypatch.setattr(adapter, "_resolve_own_uuid", AsyncMock())
        monkeypatch.setattr(adapter, "_set_profile_name", AsyncMock())
        monkeypatch.setattr(adapter, "_sse_listener", AsyncMock())
        monkeypatch.setattr(adapter, "_health_monitor", AsyncMock())
        reconcile = AsyncMock()
        monkeypatch.setattr(adapter, "_reconcile_groups", reconcile)
        monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *a, **k: True)
        monkeypatch.setattr(adapter, "_release_platform_lock", lambda *a, **k: None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            ok = await adapter.connect()

        await adapter.disconnect()

        assert ok is True
        reconcile.assert_awaited_once()
