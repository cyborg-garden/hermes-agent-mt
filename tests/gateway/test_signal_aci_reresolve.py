"""Regression tests for the Signal sealed-sender ACI authorization bug.

Production failure: SIGNAL_ALLOWED_USERS held a phone number, the admin's
first-ever sealed-sender DM arrived carrying only their ACI UUID, and:

1. run.py::_is_user_authorized compared the ACI against the raw env strings
   (a phone entry can never match an ACI) and never consulted the Signal
   adapter's resolved sets (dm_allow_from_uuids / _recipient_uuid_by_number).
2. The adapter's startup _resolve_allowlist_uuids() ran pre-first-contact,
   when getUserStatus can only return the PNI (a different UUID) — the ACI
   is unknowable until the first envelope arrives.

Fix under test:
- Part 1 (run.py): Signal phone↔UUID alias expansion in _is_user_authorized,
  mirroring the WhatsApp phone↔LID expansion precedent.
- Part 2 (signal.py): rate-limited allowlist re-resolution when an unknown
  sender's envelope arrives and the allowlist contains phone entries.

Conventions follow tests/gateway/test_signal_enhancements.py.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.session import Platform, SessionSource

ADMIN_PHONE = "+15550001111"
ADMIN_ACI = "5eca7c21-aaaa-bbbb-cccc-000000000001"
UNKNOWN_ACI = "9f0d3e42-dddd-eeee-ffff-000000000002"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_signal_env(monkeypatch):
    for var in (
        "SIGNAL_ALLOWED_USERS",
        "SIGNAL_GROUP_ALLOWED_USERS",
        "SIGNAL_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    from plugins.platforms.signal.signal_rate_limit import _reset_scheduler
    _reset_scheduler()
    yield
    _reset_scheduler()


def _make_bare_runner(signal_adapter=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    runner.adapters = {}
    if signal_adapter is not None:
        runner.adapters[Platform.SIGNAL] = signal_adapter
    return runner


def _make_adapter_state(
    uuid_by_number=None,
    number_by_uuid=None,
    dm_allow_from_uuids=None,
):
    """Duck-typed Signal adapter carrying only the auth-relevant caches."""
    return SimpleNamespace(
        _recipient_uuid_by_number=dict(uuid_by_number or {}),
        _recipient_number_by_uuid=dict(number_by_uuid or {}),
        dm_allow_from_uuids=set(dm_allow_from_uuids or set()),
        group_allow_from=set(),
    )


def _make_signal_dm_source(user_id):
    return SessionSource(
        platform=Platform.SIGNAL,
        chat_id=user_id,
        chat_type="dm",
        user_id=user_id,
        user_name="Admin",
    )


def _make_signal_adapter(monkeypatch, account="+15551234567", **extra):
    """Create a real SignalAdapter with test defaults (mirrors test_signal_enhancements)."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", extra.pop("group_allowed", ""))
    if "allowed_users" in extra:
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", extra.pop("allowed_users"))
    from plugins.platforms.signal.adapter import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "http_url": "http://localhost:8080",
        "account": account,
        **extra,
    }
    return SignalAdapter(config)


def _dm_envelope(source_uuid, text="hello", timestamp=1750000000000):
    """Sealed-sender-style DM envelope: only sourceUuid, no sourceNumber."""
    return {
        "envelope": {
            "sourceUuid": source_uuid,
            "source": source_uuid,
            "sourceName": "Admin",
            "timestamp": timestamp,
            "dataMessage": {"message": text, "timestamp": timestamp},
        }
    }


# ---------------------------------------------------------------------------
# Test A — run.py: Signal alias expansion in _is_user_authorized
# ---------------------------------------------------------------------------

class TestRunPySignalAliasExpansion:

    def test_aci_authorized_via_adapter_phone_to_uuid_cache(self, monkeypatch):
        """Phone in env + adapter knows phone→ACI ⇒ ACI sender authorized."""
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN_PHONE)
        adapter = _make_adapter_state(uuid_by_number={ADMIN_PHONE: ADMIN_ACI})
        runner = _make_bare_runner(adapter)

        assert runner._is_user_authorized(_make_signal_dm_source(ADMIN_ACI)) is True

    def test_aci_authorized_via_dm_allow_from_uuids(self, monkeypatch):
        """Phone in env + adapter resolved allowlist UUIDs ⇒ ACI authorized."""
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN_PHONE)
        adapter = _make_adapter_state(dm_allow_from_uuids={ADMIN_ACI})
        runner = _make_bare_runner(adapter)

        assert runner._is_user_authorized(_make_signal_dm_source(ADMIN_ACI)) is True

    def test_aci_authorized_via_uuid_to_number_reverse_cache(self, monkeypatch):
        """Sender UUID whose cached number is allowlisted ⇒ authorized."""
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN_PHONE)
        adapter = _make_adapter_state(number_by_uuid={ADMIN_ACI: ADMIN_PHONE})
        runner = _make_bare_runner(adapter)

        assert runner._is_user_authorized(_make_signal_dm_source(ADMIN_ACI)) is True

    def test_aci_denied_without_adapter_knowledge(self, monkeypatch):
        """No adapter mapping ⇒ deny is preserved (no accidental widening)."""
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN_PHONE)
        adapter = _make_adapter_state()  # empty caches
        runner = _make_bare_runner(adapter)

        assert runner._is_user_authorized(_make_signal_dm_source(ADMIN_ACI)) is False

    def test_unrelated_aci_denied_even_with_adapter_knowledge(self, monkeypatch):
        """Adapter caches for the admin must not authorize a different UUID."""
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN_PHONE)
        adapter = _make_adapter_state(
            uuid_by_number={ADMIN_PHONE: ADMIN_ACI},
            number_by_uuid={ADMIN_ACI: ADMIN_PHONE},
            dm_allow_from_uuids={ADMIN_ACI},
        )
        runner = _make_bare_runner(adapter)

        assert runner._is_user_authorized(_make_signal_dm_source(UNKNOWN_ACI)) is False

    def test_missing_adapter_fails_toward_existing_behavior(self, monkeypatch):
        """No signal adapter registered ⇒ same result as before the fix."""
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN_PHONE)
        runner = _make_bare_runner()  # no adapter

        assert runner._is_user_authorized(_make_signal_dm_source(ADMIN_ACI)) is False
        assert runner._is_user_authorized(_make_signal_dm_source(ADMIN_PHONE)) is True

    def test_broken_adapter_does_not_crash_auth(self, monkeypatch):
        """A duck-typed adapter with hostile attrs must not raise or allow-all."""
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN_PHONE)

        class _Hostile:
            @property
            def _recipient_uuid_by_number(self):
                raise RuntimeError("boom")

            @property
            def _recipient_number_by_uuid(self):
                raise RuntimeError("boom")

            @property
            def dm_allow_from_uuids(self):
                raise RuntimeError("boom")

        runner = _make_bare_runner(_Hostile())

        assert runner._is_user_authorized(_make_signal_dm_source(ADMIN_ACI)) is False
        assert runner._is_user_authorized(_make_signal_dm_source(ADMIN_PHONE)) is True


# ---------------------------------------------------------------------------
# Test C — regression: wildcard / UUID-entry / empty allowlists unchanged
# ---------------------------------------------------------------------------

class TestRunPySignalAuthRegression:

    def test_wildcard_allowlist_still_allows_everyone(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "*")
        runner = _make_bare_runner(_make_adapter_state())

        assert runner._is_user_authorized(_make_signal_dm_source(UNKNOWN_ACI)) is True

    def test_uuid_entry_allowlist_still_matches_directly(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN_ACI)
        runner = _make_bare_runner(_make_adapter_state())

        assert runner._is_user_authorized(_make_signal_dm_source(ADMIN_ACI)) is True
        assert runner._is_user_authorized(_make_signal_dm_source(UNKNOWN_ACI)) is False

    def test_empty_allowlist_stays_deny(self, monkeypatch):
        # No SIGNAL_ALLOWED_USERS at all; runner has no adapter-policy hooks,
        # so provide the bits _is_user_authorized touches on that path.
        runner = _make_bare_runner(_make_adapter_state())
        runner._adapter_enforces_own_access_policy = lambda *_a, **_kw: False

        assert runner._is_user_authorized(_make_signal_dm_source(UNKNOWN_ACI)) is False

    def test_phone_sender_still_matches_phone_entry(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN_PHONE)
        runner = _make_bare_runner(_make_adapter_state())

        assert runner._is_user_authorized(_make_signal_dm_source(ADMIN_PHONE)) is True


# ---------------------------------------------------------------------------
# Test B — signal.py: rate-limited re-resolution on unknown inbound sender
# ---------------------------------------------------------------------------

class TestSignalAdapterReResolve:

    @pytest.mark.asyncio
    async def test_unknown_aci_triggers_reresolve_before_dispatch(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, allowed_users=ADMIN_PHONE)
        assert adapter.dm_allow_from == {ADMIN_PHONE}
        assert adapter.dm_allow_from_uuids == set()

        adapter._resolve_allowlist_uuids = AsyncMock()
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(_dm_envelope(ADMIN_ACI))

        adapter._resolve_allowlist_uuids.assert_awaited_once()
        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_second_envelope_within_rate_window_does_not_retrigger(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, allowed_users=ADMIN_PHONE)
        adapter._resolve_allowlist_uuids = AsyncMock()
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(_dm_envelope(ADMIN_ACI, text="one"))
        await adapter._handle_envelope(_dm_envelope(ADMIN_ACI, text="two"))

        assert adapter._resolve_allowlist_uuids.await_count == 1
        assert adapter.handle_message.await_count == 2

    @pytest.mark.asyncio
    async def test_known_uuid_sender_does_not_trigger_reresolve(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, allowed_users=ADMIN_PHONE)
        adapter.dm_allow_from_uuids.add(ADMIN_ACI)
        adapter._resolve_allowlist_uuids = AsyncMock()
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(_dm_envelope(ADMIN_ACI))

        adapter._resolve_allowlist_uuids.assert_not_awaited()
        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wildcard_allowlist_does_not_trigger_reresolve(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, allowed_users="*")
        adapter._resolve_allowlist_uuids = AsyncMock()
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(_dm_envelope(UNKNOWN_ACI))

        adapter._resolve_allowlist_uuids.assert_not_awaited()
        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_uuid_only_allowlist_does_not_trigger_reresolve(self, monkeypatch):
        """No phone entries ⇒ nothing a re-resolve could learn; skip it."""
        adapter = _make_signal_adapter(monkeypatch, allowed_users=ADMIN_ACI)
        adapter._resolve_allowlist_uuids = AsyncMock()
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(_dm_envelope(UNKNOWN_ACI))

        adapter._resolve_allowlist_uuids.assert_not_awaited()
        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reresolve_failure_does_not_break_message_handling(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, allowed_users=ADMIN_PHONE)
        adapter._resolve_allowlist_uuids = AsyncMock(side_effect=RuntimeError("daemon down"))
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(_dm_envelope(ADMIN_ACI))

        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reresolve_allowed_again_after_rate_window(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, allowed_users=ADMIN_PHONE)
        adapter._resolve_allowlist_uuids = AsyncMock()
        adapter.handle_message = AsyncMock()

        await adapter._handle_envelope(_dm_envelope(ADMIN_ACI, text="one"))
        # Simulate the rate window elapsing.
        adapter._last_allowlist_reresolve -= 31.0
        await adapter._handle_envelope(_dm_envelope(ADMIN_ACI, text="two"))

        assert adapter._resolve_allowlist_uuids.await_count == 2
