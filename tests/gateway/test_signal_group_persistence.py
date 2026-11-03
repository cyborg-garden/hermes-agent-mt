"""Tests for Signal adapter group allowlist persistence across restarts.

Covers:
  - accepted invite writes the JSON file with correct groupId + metadata keys
  - fresh adapter init loads persisted groups into group_allow_from
  - persisted groups UNION with env-var groups (both sources present)
  - corrupted JSON file → init still succeeds, group_allow_from = env groups only
  - persistence write failure → invite acceptance still completes + group in runtime allowlist
  - group already in env allowlist re-invited → no duplicate issues
"""
import asyncio
import json
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_signal_adapter(monkeypatch, tmp_path, account="+15551234567", **extra):
    """Create a SignalAdapter with a tmp HERMES_HOME so tests never touch the real one.

    We point HERMES_HOME (the env var ``get_hermes_home`` reads) at ``tmp_path``
    rather than monkeypatching the ``get_hermes_home`` function object. Patching
    the function via the dotted string ``plugins.platforms.signal.adapter.get_hermes_home``
    is a cross-test-pollution hazard: if the signal module has not been imported
    yet, the *first* patch of ``hermes_constants.get_hermes_home`` runs before the
    module is imported, so when the second patch triggers the import, signal.py's
    ``from hermes_constants import get_hermes_home`` binds the already-patched
    lambda into the module namespace. monkeypatch then records that lambda as the
    "original" value and restores it (not the real function) on teardown, leaking
    the lambda — and its captured tmp_path — into every later test. Setting the
    env var exercises the real ``get_hermes_home`` code path and tears down cleanly.
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


# ---------------------------------------------------------------------------
# Helper: send an invite envelope and verify join was called
# ---------------------------------------------------------------------------

async def _send_invite_envelope(adapter, group_id, sender="+15559999999",
                                sender_uuid="66666666-6666-6666-6666-666666666666",
                                sender_name="Approved User"):
    rpc_calls = []

    async def mock_rpc(method, params, rpc_id=None, **kwargs):
        rpc_calls.append(method)
        return {"success": True}

    adapter._rpc = mock_rpc
    envelope = {
        "envelope": {
            "sourceNumber": sender,
            "sourceUuid": sender_uuid,
            "sourceName": sender_name,
            "groupV2": {"groupId": group_id},
        }
    }
    await adapter._handle_envelope(envelope)
    return rpc_calls


# ---------------------------------------------------------------------------
# 1. Accepted invite writes the JSON file with the right groupId + metadata keys
# ---------------------------------------------------------------------------

class TestInviteWritesPersistenceFile:

    @pytest.mark.asyncio
    async def test_accepted_invite_creates_json_file(self, monkeypatch, tmp_path):
        """Accepting a group invite must write signal_approved_groups.json."""
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users="+15559999999"
        )

        await _send_invite_envelope(adapter, "group-aaa-111")

        from plugins.platforms.signal.adapter import _APPROVED_GROUPS_FILE
        persistence_file = tmp_path / _APPROVED_GROUPS_FILE
        assert persistence_file.exists(), "Persistence file was not created after invite"

    @pytest.mark.asyncio
    async def test_accepted_invite_writes_correct_group_id(self, monkeypatch, tmp_path):
        """The JSON file must contain the correct groupId as a top-level key."""
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users="+15559999999"
        )

        await _send_invite_envelope(adapter, "group-bbb-222")

        from plugins.platforms.signal.adapter import _APPROVED_GROUPS_FILE
        data = json.loads((tmp_path / _APPROVED_GROUPS_FILE).read_text())
        assert "group-bbb-222" in data

    @pytest.mark.asyncio
    async def test_accepted_invite_writes_metadata_keys(self, monkeypatch, tmp_path):
        """Each entry must contain added_by, added_by_uuid, added_by_name, approved_at."""
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users="+15559999999"
        )

        await _send_invite_envelope(
            adapter, "group-ccc-333",
            sender="+15559999999",
            sender_uuid="66666666-6666-6666-6666-666666666666",
            sender_name="Alice",
        )

        from plugins.platforms.signal.adapter import _APPROVED_GROUPS_FILE
        data = json.loads((tmp_path / _APPROVED_GROUPS_FILE).read_text())
        entry = data["group-ccc-333"]

        assert "added_by" in entry
        assert "added_by_uuid" in entry
        assert "added_by_name" in entry
        assert "approved_at" in entry

        assert entry["added_by"] == "+15559999999"
        assert entry["added_by_uuid"] == "66666666-6666-6666-6666-666666666666"
        assert entry["added_by_name"] == "Alice"
        # approved_at must be ISO-8601 formatted (contains 'T' separator)
        assert "T" in entry["approved_at"]


# ---------------------------------------------------------------------------
# 2. Fresh adapter loads persisted groups
# ---------------------------------------------------------------------------

class TestAdapterLoadsPersistedGroups:

    def test_init_loads_persisted_groups_from_file(self, monkeypatch, tmp_path):
        """A new adapter must load group IDs from the persistence file into group_allow_from."""
        from plugins.platforms.signal.adapter import _APPROVED_GROUPS_FILE
        existing = {
            "group-persisted-001": {
                "added_by": "+15559999999",
                "added_by_uuid": "aaaa-bbbb",
                "added_by_name": "Bob",
                "approved_at": "2024-01-01T00:00:00+00:00",
            }
        }
        (tmp_path / _APPROVED_GROUPS_FILE).write_text(json.dumps(existing))

        # No env-var groups set
        adapter = _make_signal_adapter(monkeypatch, tmp_path)

        assert "group-persisted-001" in adapter.group_allow_from

    def test_init_with_no_persistence_file_does_not_raise(self, monkeypatch, tmp_path):
        """If the persistence file doesn't exist, init must succeed normally."""
        adapter = _make_signal_adapter(monkeypatch, tmp_path)
        assert isinstance(adapter.group_allow_from, set)


# ---------------------------------------------------------------------------
# 3. Persisted groups UNION with env-var groups
# ---------------------------------------------------------------------------

class TestPersistenceUnionWithEnvGroups:

    def test_persisted_and_env_groups_both_present(self, monkeypatch, tmp_path):
        """group_allow_from must contain groups from both the env var and the file."""
        from plugins.platforms.signal.adapter import _APPROVED_GROUPS_FILE
        existing = {
            "group-from-file": {
                "added_by": "+15558888888",
                "added_by_uuid": "",
                "added_by_name": "",
                "approved_at": "2024-01-01T00:00:00+00:00",
            }
        }
        (tmp_path / _APPROVED_GROUPS_FILE).write_text(json.dumps(existing))

        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, group_allowed="group-from-env"
        )

        assert "group-from-env" in adapter.group_allow_from
        assert "group-from-file" in adapter.group_allow_from


# ---------------------------------------------------------------------------
# 4. Corrupted JSON file → init still succeeds
# ---------------------------------------------------------------------------

class TestCorruptedFileHandling:

    def test_corrupted_json_file_init_succeeds(self, monkeypatch, tmp_path):
        """Corrupted persistence file must not crash init; env groups are preserved."""
        from plugins.platforms.signal.adapter import _APPROVED_GROUPS_FILE
        (tmp_path / _APPROVED_GROUPS_FILE).write_text("{invalid json!!!}")

        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, group_allowed="env-group-safe"
        )

        # Init must succeed
        assert isinstance(adapter.group_allow_from, set)
        # Env-var group must still be present
        assert "env-group-safe" in adapter.group_allow_from

    def test_corrupted_json_file_does_not_load_bad_groups(self, monkeypatch, tmp_path):
        """Corrupted persistence file must not add any groups (no partial state)."""
        from plugins.platforms.signal.adapter import _APPROVED_GROUPS_FILE
        (tmp_path / _APPROVED_GROUPS_FILE).write_text("{invalid json!!!")

        adapter = _make_signal_adapter(monkeypatch, tmp_path)

        # Only the empty set from the empty env var
        assert len(adapter.group_allow_from) == 0


# ---------------------------------------------------------------------------
# 5. Persistence write failure → invite acceptance still completes
# ---------------------------------------------------------------------------

class TestPersistenceWriteFailure:

    @pytest.mark.asyncio
    async def test_write_failure_does_not_break_invite_flow(self, monkeypatch, tmp_path):
        """If persisting the group fails, the group must still be in the runtime allowlist."""
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users="+15559999999"
        )

        # Make _persist_approved_group always raise
        def _failing_persist(group_id, sender, sender_uuid, sender_name):
            raise OSError("Disk full")

        adapter._persist_approved_group = _failing_persist

        await _send_invite_envelope(adapter, "group-write-fail-xyz")

        # Group must be in runtime allowlist despite write failure
        assert "group-write-fail-xyz" in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_write_failure_unwritable_dir(self, monkeypatch, tmp_path):
        """Unwritable HERMES_HOME directory must not crash the invite flow."""
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users="+15559999999"
        )

        # Make HERMES_HOME a file (not a dir) so writes will fail
        hermes_home_fake = tmp_path / "blocked_home"
        hermes_home_fake.write_text("not a directory")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home_fake))

        await _send_invite_envelope(adapter, "group-unwritable-dir")

        # Still in runtime allowlist
        assert "group-unwritable-dir" in adapter.group_allow_from


# ---------------------------------------------------------------------------
# 6. Group already in env allowlist re-invited → no duplicate issues
# ---------------------------------------------------------------------------

class TestReInviteNoduplicates:

    @pytest.mark.asyncio
    async def test_reinvite_of_env_group_no_duplicates(self, monkeypatch, tmp_path):
        """Re-inviting a group already in env allowlist must not cause errors."""
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path,
            allowed_users="+15559999999",
            group_allowed="already-in-env-group",
        )

        assert "already-in-env-group" in adapter.group_allow_from
        original_count = len(adapter.group_allow_from)

        # Invite for the same group
        await _send_invite_envelope(adapter, "already-in-env-group")

        # group_allow_from is a set — no duplicates possible
        assert "already-in-env-group" in adapter.group_allow_from
        # Set cardinality stays the same or smaller (add is idempotent)
        assert len(adapter.group_allow_from) >= original_count

    @pytest.mark.asyncio
    async def test_reinvite_of_env_group_writes_file_without_error(self, monkeypatch, tmp_path):
        """Re-invite of a group that's in the env allowlist should write to file cleanly."""
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path,
            allowed_users="+15559999999",
            group_allowed="env-group-reinvited",
        )

        await _send_invite_envelope(adapter, "env-group-reinvited")

        from plugins.platforms.signal.adapter import _APPROVED_GROUPS_FILE
        persistence_file = tmp_path / _APPROVED_GROUPS_FILE
        assert persistence_file.exists()
        data = json.loads(persistence_file.read_text())
        assert "env-group-reinvited" in data


# ---------------------------------------------------------------------------
# 7. Test-isolation regression: the helper must NOT leak get_hermes_home
# ---------------------------------------------------------------------------

class TestHelperDoesNotLeakHermesHome:
    """Regression for a cross-test pollution bug.

    The old helper monkeypatched ``plugins.platforms.signal.adapter.get_hermes_home``
    via a dotted string. When the signal module was imported for the first
    time *during* that patch (because ``hermes_constants.get_hermes_home`` was
    patched first, then the signal-module patch triggered the import),
    monkeypatch recorded the already-patched lambda as the "original" value and
    restored that lambda — not the real function — on teardown. The lambda,
    closing over the test's ``tmp_path``, then leaked into every later test and
    made unrelated tests load this test's persisted groups.

    These tests assert the helper leaves the signal module's ``get_hermes_home``
    binding identical to ``hermes_constants.get_hermes_home`` after a test runs,
    and that an adapter built afterwards reads the *current* HERMES_HOME.
    """

    def test_helper_leaves_signal_get_hermes_home_unpatched(
        self, monkeypatch, tmp_path
    ):
        import hermes_constants
        import plugins.platforms.signal.adapter as signal_mod

        # Persist a group under this test's home so a leak would be observable.
        from plugins.platforms.signal.adapter import _APPROVED_GROUPS_FILE
        (tmp_path / _APPROVED_GROUPS_FILE).write_text('{"leaked-group": {}}')

        _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")

        # The signal module must still resolve get_hermes_home to the real
        # function object — not a lambda left behind by the helper.
        assert signal_mod.get_hermes_home is hermes_constants.get_hermes_home

    def test_fresh_adapter_reads_current_hermes_home(self, monkeypatch, tmp_path):
        """After the helper runs for one home, a plain adapter built against a
        different HERMES_HOME must see that new home, proving no stale binding."""
        from plugins.platforms.signal.adapter import SignalAdapter, _APPROVED_GROUPS_FILE

        # First: run the helper pointed at tmp_path with a persisted group.
        (tmp_path / _APPROVED_GROUPS_FILE).write_text('{"old-home-group": {}}')
        _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")

        # Now point HERMES_HOME somewhere clean and build an adapter directly.
        clean_home = tmp_path / "clean"
        clean_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(clean_home))
        monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")

        config = PlatformConfig()
        config.enabled = True
        config.extra = {"http_url": "http://localhost:8080", "account": "+15551234567"}
        adapter = SignalAdapter(config)

        # Must NOT have loaded the group from the previous home.
        assert "old-home-group" not in adapter.group_allow_from
        assert len(adapter.group_allow_from) == 0
