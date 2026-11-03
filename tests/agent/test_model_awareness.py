"""Tests for model self-awareness — the live (post-fallback) model line.

Re-ports the pre-migration ``hermes-swarm`` behaviour: a
``[Current model: X via Y]`` line is appended to the ephemeral per-turn
context so the LLM can answer "what model are you?" truthfully instead of
hallucinating. The line must reflect the LIVE runtime — i.e. after
``try_activate_fallback`` mutates ``agent.model`` / ``agent.provider`` in
place it must show the FALLBACK model, not the configured primary.
"""

import pytest

from agent.model_awareness import (
    format_current_model_line,
    with_current_model_line,
)


class _FakeAgent:
    """Minimal stand-in exposing the two live runtime fields the loop reads."""

    def __init__(self, model: str, provider: str) -> None:
        self.model = model
        self.provider = provider


def test_format_with_provider():
    assert (
        format_current_model_line("claude-opus-4-8", "anthropic")
        == "[Current model: claude-opus-4-8 via anthropic]"
    )


def test_format_without_provider():
    assert format_current_model_line("glm-4-9b", "") == "[Current model: glm-4-9b]"


def test_format_unknown_model_is_empty():
    # No model configured yet → inject nothing rather than a half-line.
    assert format_current_model_line("", "anthropic") == ""
    assert format_current_model_line("   ", "") == ""


# ---------------------------------------------------------------------------
# with_current_model_line: the stamped (sidecar) form, idempotent for replay
# ---------------------------------------------------------------------------

@pytest.fixture
def awareness_on(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL_AWARENESS", "1")


def test_with_line_appends_once(awareness_on):
    agent = _FakeAgent("glm-5.2", "zai")
    once = with_current_model_line("hello", agent)
    assert once == "hello\n\n[Current model: glm-5.2 via zai]"
    # Re-deriving with the same model is byte-identical (cache-stable replay).
    assert with_current_model_line(once, agent) == once


def test_with_line_replaces_after_fallback(awareness_on):
    agent = _FakeAgent("primary", "openrouter")
    stamped = with_current_model_line("hi\n\nPLUGIN-CTX", agent)
    agent.model, agent.provider = "fallback", "zai"
    swapped = with_current_model_line(stamped, agent)
    assert swapped == "hi\n\nPLUGIN-CTX\n\n[Current model: fallback via zai]"
    assert swapped.count("[Current model:") == 1


def test_with_line_disabled_or_non_string(monkeypatch):
    agent = _FakeAgent("m", "p")
    monkeypatch.setenv("HERMES_MODEL_AWARENESS", "0")
    assert with_current_model_line("hello", agent) == "hello"
    monkeypatch.setenv("HERMES_MODEL_AWARENESS", "1")
    parts = [{"type": "text", "text": "x"}]
    assert with_current_model_line(parts, agent) is parts
    assert with_current_model_line("hello", _FakeAgent("", "p")) == "hello"


# ---------------------------------------------------------------------------
# Wire path: build_api_messages is the only caller that reaches the provider.
# The suite runs with the feature off, so this is its sole on-path coverage.
# ---------------------------------------------------------------------------


class _WireAgent:
    api_mode = "chat_completions"
    ephemeral_system_prompt = None
    _compression_warning = None
    _current_turn_timestamp = 10_000.0

    def __init__(self, model: str, provider: str) -> None:
        self.model, self.provider = model, provider

    @staticmethod
    def _copy_reasoning_content_for_api(_source, _target):
        return None

    @staticmethod
    def _should_sanitize_tool_calls():
        return False


def _wire(agent, history):
    from agent.turn_context import build_api_messages

    request, _ = build_api_messages(
        agent, history, current_turn_user_idx=len(history) - 1,
        ext_prefetch_cache="", plugin_user_context="", moa_config=None, active_system_prompt="")
    return [m for m in request if m["role"] == "user"]


def test_wire_current_turn_carries_line_history_replays_verbatim(awareness_on):
    stamped_earlier = "earlier\n\n[Current model: old via p]"
    history = [
        {"role": "user", "content": "earlier", "api_content": stamped_earlier},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "now"},
    ]
    users = _wire(_WireAgent("live", "prov"), history)
    # A historical row replays the bytes it was sent with, never restamped.
    assert users[0]["content"] == stamped_earlier
    assert users[-1]["content"] == "now\n\n[Current model: live via prov]"
    # The persisted list is untouched beyond the sidecar.
    assert history[-1]["content"] == "now"


def test_wire_mid_turn_fallback_rewrites_line_and_sidecar(awareness_on):
    agent = _WireAgent("primary", "openrouter")
    turn = {"role": "user", "content": "q", "api_content": "q\n\n[Current model: primary via openrouter]"}
    first = _wire(agent, [turn])
    assert first[-1]["content"].endswith("[Current model: primary via openrouter]")
    agent.model, agent.provider = "fallback", "zai"  # try_activate_fallback mutates in place
    second = _wire(agent, [turn])
    assert second[-1]["content"] == "q\n\n[Current model: fallback via zai]"
    assert turn["api_content"] == second[-1]["content"]


def test_wire_feature_off_sends_plain_content(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL_AWARENESS", "0")
    users = _wire(_WireAgent("live", "prov"), [{"role": "user", "content": "now"}])
    assert users[-1]["content"] == "now"
