"""routing.mode: llm-router — called only when the heuristic cannot decide, least
text, never logged, strict timeout, fail open to premium, skipped on protected content."""
import logging
import time
from unittest.mock import patch

import pytest

from plugins.intelligent_routing import llm_router, registration

CRON_UNDECIDED = "Nightly: go through the feeds folder."  # no premium signal, no cheap evidence


@pytest.fixture
def router_env():
    calls = []
    settings = {
        "model": "router-model", "provider": "openrouter", "timeout_s": 0.3, "min_confidence": 0.85,
        "max_chars": 200, "on_interactive": False, "exclude_platforms": frozenset(), "extra_body": {},
    }
    reply = {"content": '{"type": "lookup", "confidence": 0.95}'}

    def fake(text, s):
        calls.append(text)
        if isinstance(reply["content"], Exception):
            raise reply["content"]
        if reply["content"] == "SLEEP":
            time.sleep(2)
        return reply["content"]

    with patch.object(registration, "is_intelligent_routing_enabled", return_value=True), \
            patch.object(registration, "routing_mode", return_value="llm-router"), \
            patch.object(registration, "router_settings", return_value=settings), \
            patch.object(llm_router, "_completion", fake):
        yield calls, settings, reply


def _hook(message, **kw):
    kw.setdefault("platform", "cron")
    return registration.on_pre_llm_call(user_message=message, model="kimi", provider="openrouter", **kw)


def test_undecided_background_turn_asks_the_router_and_can_go_cheap(router_env):
    calls, _, _ = router_env
    out = _hook(CRON_UNDECIDED)
    assert len(calls) == 1
    assert out["route"]["escalate"] is True


def test_confident_heuristic_never_calls_the_router(router_env):
    calls, _, _ = router_env
    assert "route" in _hook("list the files in docs/")       # confident cheap
    assert "route" not in _hook("should we ship this?")      # confident premium
    assert calls == []


def test_interactive_turns_skip_the_router_by_default(router_env):
    calls, settings, _ = router_env
    assert "route" not in _hook(CRON_UNDECIDED, platform="discord")
    assert calls == []
    settings["on_interactive"] = True
    _hook(CRON_UNDECIDED, platform="discord")
    assert len(calls) == 1


def test_excluded_platform_skips_the_router(router_env):
    calls, settings, _ = router_env
    settings["exclude_platforms"] = frozenset({"cron"})
    assert "route" not in _hook(CRON_UNDECIDED)
    assert calls == []


def test_protected_content_in_history_never_reaches_the_router_or_cheap_model(router_env):
    calls, _, _ = router_env
    history = [{"role": "user", "content": "...\nThis text transcribed by Papa Reo API. Please be sure "
                                           "to comply with the terms of the Kaitiakitanga license..."}]
    out = _hook(CRON_UNDECIDED, conversation_history=history)
    assert calls == []
    assert "route" not in out
    out = _hook("list the files in docs/", conversation_history=history)
    assert "route" not in out  # not even a confident cheap turn leaves the primary


def test_configured_protected_marker(router_env):
    calls, _, _ = router_env
    with patch.object(registration, "protected_markers", return_value=("do-not-route",)):
        assert "route" not in _hook("list the files DO-NOT-ROUTE")
    assert calls == []


@pytest.mark.parametrize("content", [
    '{"type": "lookup", "confidence": 0.5}',      # low confidence
    '{"type": "judgment", "confidence": 0.99}',   # premium type
    "no json here",                               # unparsed
    '{"type": "made_up", "confidence": 0.99}',    # unknown type
    RuntimeError("boom"),                         # error
    "SLEEP",                                      # timeout
])
def test_router_fails_open_to_premium(router_env, content):
    _, _, reply = router_env
    reply["content"] = content
    started = time.monotonic()
    out = _hook(CRON_UNDECIDED)
    assert "route" not in out
    assert time.monotonic() - started < 1.5  # the wall-clock timeout holds


def test_router_gets_least_text(router_env):
    calls, _, _ = router_env
    _hook("[IMPORTANT: You are running as a scheduled cron job.]\n\n" + CRON_UNDECIDED + " " + "x" * 500,
          conversation_history=[{"role": "user", "content": "SECRET EARLIER TURN"}])
    sent = calls[0]
    assert "IMPORTANT" not in sent and "SECRET" not in sent
    assert len(sent) <= 200 + len("\n[...]\n")


def test_router_text_and_reply_are_never_logged(router_env, caplog):
    _, _, reply = router_env
    reply["content"] = '{"type": "lookup", "confidence": 0.95, "note": "REPLYMARK"}'
    with caplog.at_level(logging.DEBUG):
        _hook("Nightly: go through the TEXTMARK folder.")
    assert "TEXTMARK" not in caplog.text and "REPLYMARK" not in caplog.text
    assert "routing_decision" in caplog.text and '"router": "llm"' in caplog.text


def test_parse_reply():
    assert llm_router.parse_reply('sure: {"type": "summarise", "confidence": 0.9}') == ("summarise", 0.9)
    assert llm_router.parse_reply('{"type": "lookup", "confidence": 3}') is None
    assert llm_router.parse_reply(None) is None


def test_http_client_debug_logs_from_the_router_thread_are_dropped(router_env, caplog):
    import plugins.intelligent_routing.llm_router as lr

    def fake(text, s):
        logging.getLogger("openai._base_client").debug("Request options: %s", text)
        logging.getLogger("httpcore.http11").debug("send_request_body %s", text)
        logging.getLogger("anthropic._base_client").debug("Request options: %s", text)
        logging.getLogger("botocore.endpoint").debug("Sending %s", text)
        logging.getLogger("some.client.created.later").warning("body %s", text)
        return '{"type": "lookup", "confidence": 0.95}'

    with patch.object(lr, "_completion", fake), caplog.at_level(logging.DEBUG):
        _hook("Nightly: go through the LEAKMARK folder.")
        logging.getLogger("openai._base_client").debug("main-thread record stays")
    assert "LEAKMARK" not in caplog.text
    assert "main-thread record stays" in caplog.text
