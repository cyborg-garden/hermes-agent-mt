"""Protected text stays on the primary: the real fallback-chain walk, a fallback kept
across turns by a rate-limit cooldown, auxiliary calls, and children of a protected
parent."""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.agent.test_routed_turn_escalation_loop import (
    CHEAP, PAPA_REO, PRIMARY, ROUTE, _Http500, _agent, _reply, _run, _sent_text,
)

FB = "fallback/other"
FB_ENTRY = {"provider": "openrouter", "model": FB, "base_url": "http://127.0.0.1:8/v1",
            "api_key": "fb-key"}


def _fake_resolve(provider, model=None, **_kw):
    return SimpleNamespace(base_url="http://127.0.0.1:8/v1", api_key="fb-key"), model


def _leaks(calls):
    return [c for c in calls if c["model"] != PRIMARY and "papa reo" in _sent_text(c).lower()]


# ── (d) the real chain walk, not the _chain_walk stand-in ─────────────────────

def test_real_chain_walk_protected_primary_error_never_falls_back():
    a = _agent()
    a._fallback_chain = [dict(FB_ENTRY)]
    a._fallback_index = 0
    with patch("agent.auxiliary_client.resolve_provider_client", _fake_resolve), \
         patch.object(a, "_create_openai_client", return_value=MagicMock()):
        _result, calls = _run(a, lambda m, i: _Http500() if m == PRIMARY else _reply("ok"),
                             hook_results=[{"context": PAPA_REO}])
    assert calls and not _leaks(calls), [c["model"] for c in calls]


def test_real_chain_walk_unprotected_still_falls_back():
    a = _agent()
    a._fallback_chain = [dict(FB_ENTRY)]
    a._fallback_index = 0
    with patch("agent.auxiliary_client.resolve_provider_client", _fake_resolve), \
         patch.object(a, "_create_openai_client", return_value=MagicMock()):
        result, calls = _run(a, lambda m, i: _Http500() if m == PRIMARY else _reply("ok"),
                             hook_results=[])
    assert calls[-1]["model"] == FB and result["final_response"] == "ok"


# ── Leak 1: fallback that stuck across turns (rate-limit cooldown) ───────────

def _stick_on_fallback(a):
    """Turn 1 (no protected text): primary 429s, real chain walk lands on FB and arms
    the 60 s cooldown, so restore_primary_runtime keeps FB for turn 2."""
    class _RL(Exception):
        status_code = 429
        def __str__(self):
            return "Error code: 429 - rate limit exceeded"
    a._fallback_chain = [dict(FB_ENTRY)]
    a._fallback_index = 0
    with patch("agent.auxiliary_client.resolve_provider_client", _fake_resolve), \
         patch.object(a, "_create_openai_client", return_value=MagicMock()):
        _r1, c1 = _run(a, lambda m, i: _RL() if m == PRIMARY else _reply("ok"), hook_results=[],
                      message="hi")
    return c1


def test_protected_turn_after_sticky_fallback_goes_to_fallback_model_routing_premium():
    """Routing on, protected turn -> classifier says premium -> plugin emits no route.
    The agent is still on FB from the previous turn's cooldown, so the transcript goes to FB."""
    a = _agent()
    c1 = _stick_on_fallback(a)
    assert c1[-1]["model"] == FB, [c["model"] for c in c1]
    assert a._rate_limited_until > time.monotonic()
    with patch.object(a, "_create_openai_client", return_value=MagicMock()):
        _r2, c2 = _run(a, lambda m, i: _reply("summary"), hook_results=[],
                      message="summarise this:\n" + PAPA_REO)
    assert not _leaks(c2), "protected text sent to " + str([c["model"] for c in c2])


def test_cheap_route_after_sticky_fallback_escalates_to_fallback_not_primary():
    """Routing on, cheap route armed while sticky on FB: escalation target = pre-turn route = FB,
    so protected plugin context 'escalates' to FB."""
    a = _agent()
    _stick_on_fallback(a)
    with patch.object(a, "_create_openai_client", return_value=MagicMock()):
        _r2, c2 = _run(a, lambda m, i: _reply("summary"),
                      hook_results=[{"route": dict(ROUTE)}, {"context": PAPA_REO}])
    assert not _leaks(c2), "protected text sent to " + str([c["model"] for c in c2])


# ── Leak 2: auxiliary compression/title fallback chain ───────────────────────

def test_aux_compression_rate_limit_falls_back_with_protected_text():
    from agent import auxiliary_client as ac

    class _RL(Exception):
        status_code = 429
        def __str__(self):
            return "Error code: 429 - rate limit exceeded"

    primary_client = MagicMock()
    primary_client.base_url = "http://127.0.0.1:9/v1"
    primary_client.chat.completions.create.side_effect = _RL()
    sent = {}

    def _fb_call(client, model, label, *, messages, **_kw):
        sent["model"], sent["messages"] = model, messages
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="s"))])

    with patch.object(ac, "_resolve_task_provider_model",
                      return_value=("auto", None, None, None, None)), \
         patch.object(ac, "_get_cached_client", return_value=(primary_client, PRIMARY)), \
         patch.object(ac, "_try_configured_fallback_chain", return_value=(None, None, "")), \
         patch.object(ac, "_try_main_fallback_chain", return_value=(MagicMock(), FB, "fb")), \
         patch.object(ac, "_call_fallback_candidate_sync", _fb_call), \
         patch("time.sleep"):
        try:
            ac.call_llm(task="compression",
                        main_runtime={"model": PRIMARY, "provider": "custom",
                                      "base_url": "http://127.0.0.1:9/v1", "api_key": "k"},
                        messages=[{"role": "user", "content": "Summarise:\n" + PAPA_REO}])
        except Exception:
            pass
    assert not (sent.get("model") == FB and "papa reo" in str(sent.get("messages")).lower()), \
        "compression summary request carrying protected text sent to " + str(sent.get("model"))


# ── A child marked protected by its parent, holding the body without the marker ──

def test_protected_child_without_the_marker_never_falls_back():
    a = _agent()
    a._protected_session = True
    a._fallback_chain = [dict(FB_ENTRY)]
    a._fallback_index = 0
    with patch("agent.auxiliary_client.resolve_provider_client", _fake_resolve), \
         patch.object(a, "_create_openai_client", return_value=MagicMock()):
        _result, calls = _run(a, lambda m, i: _Http500() if m == PRIMARY else _reply("ok"),
                             hook_results=[], message="Summarise: kia ora, notes from the meeting.")
    assert calls and all(c["model"] == PRIMARY for c in calls), [c["model"] for c in calls]


def test_aux_call_in_a_protected_session_without_the_marker_does_not_fall_back():
    from agent import auxiliary_client as ac

    class _RL(Exception):
        status_code = 429
        def __str__(self):
            return "Error code: 429 - rate limit exceeded"

    primary_client = MagicMock()
    primary_client.base_url = "http://127.0.0.1:9/v1"
    primary_client.chat.completions.create.side_effect = _RL()
    sent = {}

    def _fb_call(client, model, label, *, messages, **_kw):
        sent["model"] = model
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="s"))])

    token = ac.set_protected_session(True)
    try:
        with patch.object(ac, "_resolve_task_provider_model",
                          return_value=("auto", None, None, None, None)), \
             patch.object(ac, "_get_cached_client", return_value=(primary_client, PRIMARY)), \
             patch.object(ac, "_try_configured_fallback_chain", return_value=(None, None, "")), \
             patch.object(ac, "_try_main_fallback_chain", return_value=(MagicMock(), FB, "fb")), \
             patch.object(ac, "_call_fallback_candidate_sync", _fb_call), \
             patch("time.sleep"):
            try:
                ac.call_llm(task="compression",
                            main_runtime={"model": PRIMARY, "provider": "custom",
                                          "base_url": "http://127.0.0.1:9/v1", "api_key": "k"},
                            messages=[{"role": "user", "content": "Summarise: kia ora."}])
            except Exception:
                pass
    finally:
        ac.reset_protected_session(token)
    assert sent.get("model") != FB


def test_turn_start_log_records_message_length_not_text(caplog):
    import logging

    a = _agent()
    secret = "kia ora, this sentence must never reach a log record"
    with caplog.at_level(logging.DEBUG):
        _run(a, lambda m, i: _reply("ok"), hook_results=[], message=secret)
    turn = [r for r in caplog.records if r.getMessage().startswith("conversation turn:")]
    assert turn, "turn-start log record missing"
    assert all("kia ora" not in r.getMessage() for r in turn)
    assert f"msg_len={len(secret)}" in turn[0].getMessage()
