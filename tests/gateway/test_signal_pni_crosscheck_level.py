"""The PNI/ACI self-lookup cross-check must not log at WARNING.

`_resolve_own_uuid` resolves the account's own ACI from `listIdentities` and
cross-checks it against `getUserStatus`. Per that method's docstring, signal-cli
*always* returns the account **PNI** for a self `getUserStatus` (verified on
0.14.5), so the two values always differ — the mismatch branch is the expected
steady state on every gateway start, not a fault.

It used to log at WARNING. Because `plugins.platforms.signal.adapter` is in
home_log_router's `DEFAULT_LOGGERS` and that router forwards at WARNING, every
agent on the fleet relayed this line to the operator's home channel on each
restart. That is the lifecycle spam home_log_router's own module docstring
forbids, and it made a correct code path read as a recurring failure.

These tests pin:
- the cross-check mismatch does NOT emit a WARNING (or worse), which is
  what home_log_router's floor relays
- it IS still recorded at INFO, so it survives in gateway.log (whose
  handler is INFO+); DEBUG would have dropped it from the component log
- the ACI from listIdentities wins regardless of what getUserStatus returns
"""
from __future__ import annotations

import logging

import pytest


ACI = "a615f34c-8daa-4bbb-cccc-000000000001"
PNI = "288ecc5e-30ea-4bbb-cccc-000000000002"


@pytest.fixture
def adapter():
    """A SignalAdapter with _rpc stubbed to the real-world PNI/ACI split."""
    from plugins.platforms.signal.adapter import SignalAdapter

    a = SignalAdapter.__new__(SignalAdapter)
    a.account = "+15550001111"
    a._account_normalized = "+15550001111"
    a._own_uuid = None

    async def _rpc(method, params=None):
        if method == "listIdentities":
            return [{"uuid": ACI}]
        if method == "getUserStatus":
            # signal-cli returns the PNI here, prefix already stripped.
            return [{"uuid": PNI}]
        raise AssertionError(f"unexpected rpc {method}")

    a._rpc = _rpc
    a._remember_recipient_identifiers = lambda *args, **kwargs: None
    return a


async def _resolve(adapter):
    await adapter._resolve_own_uuid()


class TestPniCrossCheckLevel:
    @pytest.mark.asyncio
    async def test_mismatch_does_not_warn(self, adapter, caplog):
        caplog.set_level(logging.DEBUG, logger="plugins.platforms.signal.adapter")
        await _resolve(adapter)

        offending = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "getUserStatus" in r.getMessage()
        ]
        assert not offending, (
            "the PNI/ACI cross-check must not log at WARNING — it is the "
            "expected steady state and home_log_router relays WARNING to the "
            f"operator's home channel. Got: {[r.getMessage() for r in offending]}"
        )

    @pytest.mark.asyncio
    async def test_mismatch_is_still_recorded_at_info(self, adapter, caplog):
        caplog.set_level(logging.DEBUG, logger="plugins.platforms.signal.adapter")
        await _resolve(adapter)

        info_msgs = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO and "getUserStatus" in r.getMessage()
        ]
        assert info_msgs, (
            "the cross-check must survive at INFO — gateway.log's handler is "
            "INFO+, so DEBUG would remove it from the component log entirely "
            "while INFO already clears home_log_router's WARNING floor"
        )
        assert any(PNI[:12] in m and ACI[:12] in m for m in info_msgs), (
            "the INFO line should still carry both identifiers"
        )

    @pytest.mark.asyncio
    async def test_aci_from_list_identities_wins(self, adapter):
        """Behavior is unchanged by the level change: never cache the PNI."""
        await _resolve(adapter)
        assert adapter._own_uuid == ACI
        assert adapter._own_uuid != PNI

    @pytest.mark.asyncio
    async def test_no_log_when_values_agree(self, adapter, caplog):
        """If getUserStatus ever returns the ACI, there is nothing to report."""
        async def _rpc(method, params=None):
            return [{"uuid": ACI}]

        adapter._rpc = _rpc
        caplog.set_level(logging.DEBUG, logger="plugins.platforms.signal.adapter")
        await _resolve(adapter)

        assert not [
            r for r in caplog.records if "getUserStatus self-lookup" in r.getMessage()
        ]
        assert adapter._own_uuid == ACI
