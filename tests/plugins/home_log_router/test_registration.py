"""registration.register(ctx) — wiring: enable gate, root handler, sender, lifecycle.

Teardown is a process-lifetime concern (atexit), NOT on_session_end: that hook
fires at the end of every conversation turn, which would tear the plugin down
after the first message.
"""
import logging
import threading

from plugins.observability.home_log_router import registration
from plugins.observability.home_log_router.handler import HomeLogHandler


class FakeCtx:
    def __init__(self):
        self.dispatched = []
        self.hooks = {}
        self.tool_event = threading.Event()

    def dispatch_tool(self, name, args, **kw):
        self.dispatched.append((name, args))
        self.tool_event.set()
        return "{}"

    def register_hook(self, name, cb):
        self.hooks.setdefault(name, []).append(cb)


def _our_handlers():
    return [h for h in logging.getLogger().handlers if isinstance(h, HomeLogHandler)]


def test_active_when_plugin_enabled_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_HOME_LOG_ENABLED", raising=False)
    ctx = FakeCtx()
    try:
        registration.register(ctx)
        assert len(_our_handlers()) == 1
        # Must NOT use on_session_end (fires per turn -> self-destruct).
        assert "on_session_end" not in ctx.hooks
    finally:
        registration._teardown()
    assert _our_handlers() == []


def test_kill_switch_disables(monkeypatch):
    for value in ("0", "false", "off"):
        monkeypatch.setenv("HERMES_HOME_LOG_ENABLED", value)
        before = len(_our_handlers())
        ctx = FakeCtx()
        registration.register(ctx)
        assert len(_our_handlers()) == before, value
        registration._teardown()


def test_register_is_idempotent(monkeypatch):
    monkeypatch.delenv("HERMES_HOME_LOG_ENABLED", raising=False)
    try:
        registration.register(FakeCtx())
        registration.register(FakeCtx())  # second call must not stack a handler
        assert len(_our_handlers()) == 1
    finally:
        registration._teardown()


def _capture_sends(monkeypatch):
    """Patch _send_to_home (the real sink) and capture (platform, message) calls.

    The sink calls the send_message tool FUNCTION directly — NOT ctx.dispatch_tool,
    which returns "Unknown tool: send_message" in the gateway worker context.
    """
    sent = []
    evt = threading.Event()

    def fake(platform, message):
        sent.append((platform, message))
        evt.set()

    monkeypatch.setattr(registration, "_send_to_home", fake)
    return sent, evt


def test_forwards_record_to_send_message_home(monkeypatch):
    monkeypatch.delenv("HERMES_HOME_LOG_ENABLED", raising=False)
    sent, evt = _capture_sends(monkeypatch)
    try:
        registration.register(FakeCtx())
        log = logging.getLogger("plugins.platforms.signal.adapter")
        log.setLevel(logging.DEBUG)
        log.warning("disk full on %s", "agent-7")
        assert evt.wait(timeout=2.0), "log was never forwarded to home"
        platform, message = sent[0]
        assert platform == "signal"
        assert "disk full on agent-7" in message
    finally:
        registration._teardown()


def test_respects_platform_env(monkeypatch):
    monkeypatch.delenv("HERMES_HOME_LOG_ENABLED", raising=False)
    monkeypatch.setenv("HERMES_HOME_LOG_PLATFORM", "telegram")
    sent, evt = _capture_sends(monkeypatch)
    try:
        registration.register(FakeCtx())
        log = logging.getLogger("model_tools")
        log.setLevel(logging.DEBUG)
        log.error("provider down")
        assert evt.wait(timeout=2.0)
        assert sent[0][0] == "telegram"
    finally:
        registration._teardown()
