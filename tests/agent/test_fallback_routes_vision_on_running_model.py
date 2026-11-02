"""During a fallback, image routing must follow the model actually running."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.auxiliary_client import (
    _read_main_model,
    _read_main_provider,
    clear_runtime_main,
    set_runtime_main,
)

VISION_MODEL = "claude-opus-5-5"
BLIND_MODEL = "z-ai/glm-5.3"


def _capability(_provider, model, _cfg=None):
    return {VISION_MODEL: True, BLIND_MODEL: False}.get(model)


@pytest.fixture(autouse=True)
def _known_capabilities(monkeypatch):
    monkeypatch.setattr("agent.image_routing._lookup_supports_vision", _capability)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    yield
    # These tests set the runtime, and it would outlive them in one pytest process.
    clear_runtime_main()


def _fallback_agent():
    """A real activation of the first fallback entry, offline."""
    from run_agent import AIAgent

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.anthropic.com",
            provider="anthropic",
            model=VISION_MODEL,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{"provider": "openrouter", "model": BLIND_MODEL}],
        )
    return agent


def _activate(agent):
    fb_client = MagicMock()
    fb_client.api_key = "test-key"
    fb_client.base_url = "https://openrouter.ai/api/v1"
    fb_client._custom_headers = None
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(fb_client, BLIND_MODEL),
    ):
        assert agent._try_activate_fallback() is True


def test_activating_a_fallback_updates_the_runtime_tools_read():
    agent = _fallback_agent()
    set_runtime_main("anthropic", VISION_MODEL)
    _activate(agent)
    assert agent.model == BLIND_MODEL
    assert _read_main_model() == BLIND_MODEL
    assert _read_main_provider() == "openrouter"


def test_vision_tool_takes_the_auxiliary_path_after_fallback(tmp_path):
    from tools.vision_tools import _should_use_native_vision_fast_path

    agent = _fallback_agent()
    set_runtime_main("anthropic", VISION_MODEL)
    assert _should_use_native_vision_fast_path() is True
    _activate(agent)
    assert _should_use_native_vision_fast_path() is False


def test_failed_auxiliary_vision_does_not_cascade_to_a_blind_main_model():
    from tools.vision_tools import _handle_vision_analyze

    set_runtime_main("openrouter", BLIND_MODEL)
    failed = json.dumps({"success": False, "analysis": "aux route refused"})

    async def _aux(*_a, **_k):
        return failed

    native_calls = []

    async def _native(*_a, **_k):
        native_calls.append(1)
        return "NATIVE"

    with patch("tools.vision_tools.vision_analyze_tool", _aux), patch(
        "tools.vision_tools._vision_analyze_native", _native
    ):
        result = asyncio.run(_handle_vision_analyze({"image_url": "/tmp/a.png", "question": "q"}))
    assert native_calls == []
    assert result == failed


def test_failed_auxiliary_vision_still_cascades_to_a_vision_main_model():
    from tools.vision_tools import _handle_vision_analyze

    set_runtime_main("anthropic", VISION_MODEL)
    failed = json.dumps({"success": False, "analysis": "aux route refused"})

    async def _aux(*_a, **_k):
        return failed

    async def _native(*_a, **_k):
        return "NATIVE"

    with patch("tools.vision_tools._should_use_native_vision_fast_path", lambda: False), patch(
        "tools.vision_tools.vision_analyze_tool", _aux
    ), patch("tools.vision_tools._vision_analyze_native", _native):
        result = asyncio.run(_handle_vision_analyze({"image_url": "/tmp/a.png", "question": "q"}))
    assert result == "NATIVE"


def _runner(cached_agent):
    import threading
    from collections import OrderedDict

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    if cached_agent is not None:
        runner._agent_cache["sess"] = (cached_agent, "sig")
    runner._resolve_session_agent_runtime = lambda **_kw: (
        VISION_MODEL,
        {"provider": "anthropic"},
    )
    return runner


def test_gateway_routes_on_the_fallback_a_session_is_stuck_on():
    agent = SimpleNamespace(
        _fallback_activated=True, provider="openrouter", model=BLIND_MODEL
    )
    assert _runner(agent)._decide_image_input_mode(session_key="sess", user_config={}) == "text"


def test_gateway_routes_on_the_primary_when_no_fallback_is_active():
    agent = SimpleNamespace(
        _fallback_activated=False, provider="anthropic", model=VISION_MODEL
    )
    assert _runner(agent)._decide_image_input_mode(session_key="sess", user_config={}) == "native"
    assert _runner(None)._decide_image_input_mode(session_key="sess", user_config={}) == "native"
