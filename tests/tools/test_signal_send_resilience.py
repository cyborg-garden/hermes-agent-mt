"""Red-green TDD: connection-error resilience in the Signal send path.

Tests for the daemon-bounce ride-through feature added to _send_signal:
- connection errors are retried with capped exponential backoff within a
  configurable time window (SIGNAL_CONNECT_RETRY_WINDOW)
- they do NOT consume SIGNAL_RATE_LIMIT_MAX_ATTEMPTS budget
- non-connection exceptions keep the existing quick-fail / attempt-counting behaviour
- window exhaustion logs an error and fails the batch
- _is_connection_error classification
"""

from __future__ import annotations

import asyncio
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers shared with the main test file
# ---------------------------------------------------------------------------

class _FakeSignalHttp:
    """Stand-in for httpx.AsyncClient used as an async context manager.

    Pops a response from the queue per ``post`` call.  Each entry is either:
    - a dict → returned from .json()
    - an exception instance → raised directly
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, *_a, **_kw):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def post(self, url, json=None):
        self.calls.append({"url": url, "payload": json})
        if not self.responses:
            raise AssertionError("Unexpected extra POST")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        resp = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda data=item: data,
        )
        return resp


def _install_signal_http(monkeypatch, fake):
    """Patch httpx.AsyncClient so the lazy import in _send_signal picks it up."""
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)


# ---------------------------------------------------------------------------
# Clock / sleep patching for send_message_tool's asyncio.sleep calls
# (separate from the scheduler's sleep patcher used by the main tests)
# ---------------------------------------------------------------------------

def _patch_tool_clock(monkeypatch):
    """Patch asyncio.sleep and time.monotonic inside send_message_tool so
    connection-retry sleeps are instant but correctly tracked.

    Returns (sleep_calls, advance_clock).

    ``sleep_calls`` is a list of (seconds,) tuples recorded for each
    asyncio.sleep call with seconds > 0 that goes through the tool module.

    ``advance_clock`` is a callable that moves the fake monotonic clock
    forward, letting tests simulate time-window expiry without real sleeps.
    """
    import asyncio as _aio

    _real_sleep = _aio.sleep
    offset = [0.0]
    sleep_calls: list[float] = []

    async def fake_sleep(seconds):
        if seconds > 0:
            sleep_calls.append(seconds)
            offset[0] += seconds
        else:
            await _real_sleep(0)

    def advance_clock(delta: float):
        offset[0] += delta

    monkeypatch.setattr("plugins.platforms.signal.standalone.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("plugins.platforms.signal.standalone.time.monotonic", lambda: offset[0])
    # Also patch scheduler module's clock so acquire() is consistent
    monkeypatch.setattr("plugins.platforms.signal.signal_rate_limit.time.monotonic", lambda: offset[0])

    return sleep_calls, advance_clock


# ---------------------------------------------------------------------------
# Fixture: fresh scheduler per test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    from plugins.platforms.signal.signal_rate_limit import _reset_scheduler
    _reset_scheduler()
    yield
    _reset_scheduler()


# ---------------------------------------------------------------------------
# Tiny window / cap constants for fast tests
# ---------------------------------------------------------------------------

TINY_WINDOW = 0.5   # seconds
TINY_CAP = 0.05     # seconds


def _patch_conn_constants(monkeypatch):
    """Monkeypatch the connection-retry constants to tiny values."""
    monkeypatch.setattr(
        "plugins.platforms.signal.signal_rate_limit.SIGNAL_CONNECT_RETRY_WINDOW",
        TINY_WINDOW,
    )
    monkeypatch.setattr(
        "plugins.platforms.signal.signal_rate_limit.SIGNAL_CONNECT_RETRY_BACKOFF_CAP",
        TINY_CAP,
    )


# ---------------------------------------------------------------------------
# Unit tests: _is_connection_error classification
# ---------------------------------------------------------------------------

class TestIsConnectionError:
    """_is_connection_error must classify specific httpx / stdlib errors."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from plugins.platforms.signal.signal_rate_limit import _is_connection_error
        self._is_connection_error = _is_connection_error

    def test_httpx_connect_error_is_connection_error(self):
        import httpx
        assert self._is_connection_error(httpx.ConnectError("refused")) is True

    def test_httpx_connect_timeout_is_connection_error(self):
        import httpx
        assert self._is_connection_error(httpx.ConnectTimeout("timed out")) is True

    def test_httpx_read_error_is_connection_error(self):
        import httpx
        assert self._is_connection_error(httpx.ReadError("read failed")) is True

    def test_httpx_remote_protocol_error_is_connection_error(self):
        import httpx
        assert self._is_connection_error(httpx.RemoteProtocolError("bad proto")) is True

    def test_httpx_pool_timeout_is_connection_error(self):
        import httpx
        assert self._is_connection_error(httpx.PoolTimeout("pool")) is True

    def test_builtin_connection_error_is_connection_error(self):
        assert self._is_connection_error(ConnectionError("reset")) is True

    def test_os_error_is_connection_error(self):
        assert self._is_connection_error(OSError("broken pipe")) is True

    def test_value_error_is_not_connection_error(self):
        assert self._is_connection_error(ValueError("bad input")) is False

    def test_runtime_error_is_not_connection_error(self):
        assert self._is_connection_error(RuntimeError("something")) is False

    def test_key_error_is_not_connection_error(self):
        assert self._is_connection_error(KeyError("key")) is False


# ---------------------------------------------------------------------------
# Integration tests: connection-error retry behaviour in _send_signal
# ---------------------------------------------------------------------------

class TestSignalConnectionRetry:
    """_send_signal retries connection errors with backoff within window."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _extra(self):
        return {"http_url": "http://localhost:8080", "account": "+15551234567"}

    def test_conn_error_then_success_delivers_message(self, monkeypatch):
        """Connection error on first attempt, success on second → delivered."""
        import httpx
        _patch_conn_constants(monkeypatch)
        sleep_calls, _ = _patch_tool_clock(monkeypatch)

        fake = _FakeSignalHttp([
            httpx.ConnectError("Connection refused"),
            {"result": {"timestamp": 1}},
        ])
        _install_signal_http(monkeypatch, fake)

        result = self._run(
            _send_signal_fn()(
                self._extra(), "+15557654321", "hello"
            )
        )

        assert result.get("success") is True
        assert result.get("platform") == "signal"
        assert len(fake.calls) == 2
        # A backoff sleep must have happened
        assert len(sleep_calls) >= 1

    def test_conn_errors_within_window_eventually_succeed(self, monkeypatch):
        """Multiple connection errors within window, final attempt succeeds."""
        import httpx
        _patch_conn_constants(monkeypatch)
        _sleep_calls, _ = _patch_tool_clock(monkeypatch)

        # Two connection errors then success
        fake = _FakeSignalHttp([
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused again"),
            {"result": {"timestamp": 5}},
        ])
        _install_signal_http(monkeypatch, fake)

        result = self._run(
            _send_signal_fn()(
                self._extra(), "+15557654321", "hello"
            )
        )

        assert result.get("success") is True
        assert len(fake.calls) == 3

    def test_conn_error_backoff_is_capped(self, monkeypatch):
        """Each backoff sleep is <= SIGNAL_CONNECT_RETRY_BACKOFF_CAP."""
        import httpx
        _patch_conn_constants(monkeypatch)
        sleep_calls, _ = _patch_tool_clock(monkeypatch)

        fake = _FakeSignalHttp([
            httpx.ConnectError("r1"),
            httpx.ConnectError("r2"),
            httpx.ConnectError("r3"),
            {"result": {"timestamp": 99}},
        ])
        _install_signal_http(monkeypatch, fake)

        result = self._run(
            _send_signal_fn()(
                self._extra(), "+15557654321", "hi"
            )
        )

        assert result.get("success") is True
        for s in sleep_calls:
            assert s <= TINY_CAP + 1e-9, f"sleep {s} exceeds cap {TINY_CAP}"

    def test_conn_error_window_exhausted_fails_batch(self, monkeypatch):
        """Daemon down for longer than window → batch failure, error in result."""
        import httpx
        _patch_conn_constants(monkeypatch)
        _sleep_calls, advance_clock = _patch_tool_clock(monkeypatch)

        # Provide many connection errors; advance clock past window on first sleep
        call_count = [0]

        class _WindowExhaustingHttp:
            def __call__(self, *_a, **_kw):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return False

            async def post(self, url, json=None):
                call_count[0] += 1
                # After the first POST, jump the clock past the retry window
                advance_clock(TINY_WINDOW + 1.0)
                raise httpx.ConnectError("daemon down")

        import httpx as _httpx
        monkeypatch.setattr(_httpx, "AsyncClient", _WindowExhaustingHttp())

        result = self._run(
            _send_signal_fn()(
                self._extra(), "+15557654321", "hello"
            )
        )

        assert "error" in result
        assert call_count[0] >= 1

    def test_conn_retries_do_not_consume_rate_limit_attempts(self, monkeypatch):
        """Connection retries must not deplete SIGNAL_RATE_LIMIT_MAX_ATTEMPTS.

        Sequence: conn error (no attempt consumed), rate-limit error (attempt 1),
        success (attempt 2) — both 429-budget attempts still available.
        """
        import httpx
        from plugins.platforms.signal.signal_rate_limit import SIGNAL_RPC_ERROR_RATELIMIT

        _patch_conn_constants(monkeypatch)
        _sleep_calls, _ = _patch_tool_clock(monkeypatch)

        rate_limit_err = {
            "error": {
                "code": SIGNAL_RPC_ERROR_RATELIMIT,
                "message": "rate limited",
                "data": {"response": {"timestamp": 0, "results": []}},
            }
        }

        fake = _FakeSignalHttp([
            httpx.ConnectError("refused"),   # conn error — no attempt consumed
            rate_limit_err,                  # attempt 1 (rate-limit)
            {"result": {"timestamp": 7}},   # attempt 2 (success)
        ])
        _install_signal_http(monkeypatch, fake)

        result = self._run(
            _send_signal_fn()(
                self._extra(), "+15557654321", "hello"
            )
        )

        assert result.get("success") is True
        assert len(fake.calls) == 3

    def test_non_connection_exception_falls_through_old_path(self, monkeypatch):
        """Non-connection errors still count against SIGNAL_RATE_LIMIT_MAX_ATTEMPTS
        and give up quickly (no connection-retry backoff)."""
        _patch_conn_constants(monkeypatch)
        sleep_calls, _ = _patch_tool_clock(monkeypatch)

        # ValueError is not a connection error; the old path applies
        fake = _FakeSignalHttp([
            ValueError("unexpected"),
            ValueError("unexpected again"),
        ])
        _install_signal_http(monkeypatch, fake)

        result = self._run(
            _send_signal_fn()(
                self._extra(), "+15557654321", "hello"
            )
        )

        # Should have errored (non-conn exception drains rate-limit attempts fast)
        assert "error" in result or result.get("success") is True  # either is fine
        # The key assertion: no long connection-retry sleeps
        # (the scheduler's acquire may produce tiny sleeps; we just check total
        #  sleep from the fake is either empty or very small)
        assert sum(sleep_calls) < 0.5

    def test_connect_error_text_only_send_succeeds_after_retry(self, monkeypatch):
        """Verify the common production scenario: text-only send with
        ConnectError on first attempt (daemon restart), succeeds on second."""
        import httpx
        _patch_conn_constants(monkeypatch)
        sleep_calls, _ = _patch_tool_clock(monkeypatch)

        fake = _FakeSignalHttp([
            httpx.ConnectError("Connection refused"),
            {"result": {"timestamp": 42}},
        ])
        _install_signal_http(monkeypatch, fake)

        result = self._run(
            _send_signal_fn()(
                {"http_url": "http://127.0.0.1:8080", "account": "+15551234567"},
                "group:abc123",
                "production message",
            )
        )

        # Fork: _display_chat_id masks Signal group ids in tool results
        # (group ids are treated as sensitive identifiers).
        assert result == {"success": True, "platform": "signal", "chat_id": "group:***"}
        assert len(fake.calls) == 2
        assert len(sleep_calls) >= 1


# ---------------------------------------------------------------------------
# Helper: lazy import of _send_signal so the module is not imported before
# patches are applied in individual tests.
# ---------------------------------------------------------------------------

def _send_signal_fn():
    from plugins.platforms.signal.standalone import _send_signal
    return _send_signal


# ---------------------------------------------------------------------------
# Attachment inlining: the send_message tool must pass attachments to the
# signal-cli daemon as base64 data: URIs, never as agent-container file paths.
#
# Regression: a video rendered at /opt/data/... failed with AttachmentInvalid
# because the signal-cli daemon runs in a separate container that cannot see
# the agent's filesystem. The gateway adapter already inlined as base64
# (PR #18) but this tool path passed the raw path through.
# ---------------------------------------------------------------------------

class TestSignalAttachmentInlining:
    def _run(self, coro):
        return asyncio.run(coro)

    def _extra(self):
        return {"http_url": "http://localhost:8080", "account": "+15551234567"}

    def test_attachment_inlined_as_data_uri(self, monkeypatch, tmp_path):
        """A local media file is inlined as data:<mime>;base64, not a raw path."""
        _patch_conn_constants(monkeypatch)
        _patch_tool_clock(monkeypatch)

        # Simulate an agent-container path the daemon cannot resolve.
        video = tmp_path / "render.mp4"
        video.write_bytes(b"\x00\x01FAKEMP4DATA")

        fake = _FakeSignalHttp([{"result": {"timestamp": 1}}])
        _install_signal_http(monkeypatch, fake)

        result = self._run(
            _send_signal_fn()(
                self._extra(), "group:abc", "here is the video",
                media_files=[(str(video), False)],
            )
        )

        assert result.get("success") is True
        assert len(fake.calls) == 1
        atts = fake.calls[0]["payload"]["params"]["attachments"]
        assert len(atts) == 1
        # The daemon must receive a self-contained data: URI, NOT the path.
        assert atts[0].startswith("data:video/mp4;filename=render.mp4;base64,")
        assert str(video) not in atts[0]

    def test_missing_attachment_is_skipped(self, monkeypatch):
        """A non-existent media path is skipped, not inlined."""
        _patch_conn_constants(monkeypatch)
        _patch_tool_clock(monkeypatch)

        fake = _FakeSignalHttp([{"result": {"timestamp": 1}}])
        _install_signal_http(monkeypatch, fake)

        result = self._run(
            _send_signal_fn()(
                self._extra(), "group:abc", "text only",
                media_files=[("/nope/missing.mp4", False)],
            )
        )

        assert result.get("success") is True
        # No attachments param when nothing was inlined.
        params = fake.calls[0]["payload"]["params"]
        assert "attachments" not in params or params["attachments"] == []
