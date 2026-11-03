"""A failed vision call must not end up with a model that cannot see.

Such a model may answer anyway, and its words would come back as the image's
description. Each test runs the real call path and real fallback functions.
"""

from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.auxiliary_client as ac
from agent.image_routing import _lookup_supports_vision as real_lookup

SEES = "moonshotai/kimi-k3.1-pro"
BLIND = "z-ai/glm-5.3"
UNKNOWN = "qwen3:30b"
INVENTED = "A red castle on a hill."
MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }
]
CREDIT = (
    'Error code: 400 - {"error":{"type":"invalid_request_error","message":'
    '"Your credit balance is too low to access the Anthropic API."}}'
)


class ProviderError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.status_code = code


def _reply(text):
    message = types.SimpleNamespace(content=text, tool_calls=None, reasoning=None)
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], usage=None, model="x")


def _client(error=None, reply=None):
    """Sync client, plus the async client ``_to_async_client`` would turn it into."""
    sync = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="k")
    sync.chat.completions.create = MagicMock(side_effect=error, return_value=reply)
    sync.as_async = MagicMock(base_url=sync.base_url, api_key="k")
    sync.as_async.chat.completions.create = AsyncMock(side_effect=error, return_value=reply)
    return sync


def _calls(client):
    return client.chat.completions.create.call_count + client.as_async.chat.completions.create.await_count


def _call(mode, dead):
    if mode == "sync":
        with patch("agent.auxiliary_client.resolve_vision_provider_client", return_value=("anthropic", dead, "dead-model")), \
             patch("agent.auxiliary_client._get_cached_client", return_value=(dead, "dead-model")):
            return ac.call_llm(task="vision", messages=MESSAGES)
    with patch("agent.auxiliary_client.resolve_vision_provider_client", return_value=("anthropic", dead.as_async, "dead-model")), \
         patch("agent.auxiliary_client._get_cached_client", return_value=(dead.as_async, "dead-model")):
        return asyncio.run(ac.async_call_llm(task="vision", messages=MESSAGES))


@pytest.fixture(autouse=True)
def _world(monkeypatch):
    monkeypatch.setattr(
        "agent.image_routing._lookup_supports_vision",
        lambda _p, model, _cfg=None: {SEES: True, BLIND: False}.get(model),
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(ac, "_to_async_client", lambda c, m, is_vision=False: (getattr(c, "as_async", c), m))
    monkeypatch.setattr(ac, "_is_provider_unhealthy", lambda _p: False)
    monkeypatch.setattr(ac, "_mark_provider_unhealthy", lambda _p: None)
    monkeypatch.setattr(ac, "_read_main_provider", lambda: "openrouter")
    monkeypatch.setattr(ac, "_read_main_model", lambda: BLIND)


def _explicit_route(monkeypatch):
    monkeypatch.setattr(
        ac, "_resolve_task_provider_model", lambda *_a, **_k: ("anthropic", "dead-model", None, None, None)
    )


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("main_model", [BLIND, UNKNOWN])
def test_main_model_that_cannot_be_shown_to_see_is_never_asked(monkeypatch, mode, main_model):
    _explicit_route(monkeypatch)
    monkeypatch.setattr(ac, "_read_main_model", lambda: main_model)
    fallback = _client(reply=_reply(INVENTED))
    monkeypatch.setattr(ac, "resolve_provider_client", lambda *_a, **_k: (fallback, main_model))

    with pytest.raises(ProviderError):
        _call(mode, _client(error=ProviderError(400, CREDIT)))

    assert _calls(fallback) == 0


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_main_model_known_to_see_still_rescues_the_call(monkeypatch, mode):
    _explicit_route(monkeypatch)
    monkeypatch.setattr(ac, "_read_main_model", lambda: SEES)
    fallback = _client(reply=_reply("A real description."))
    monkeypatch.setattr(ac, "resolve_provider_client", lambda *_a, **_k: (fallback, SEES))

    result = _call(mode, _client(error=ProviderError(400, CREDIT)))

    assert result.choices[0].message.content == "A real description."
    assert _calls(fallback) == 1


def test_main_fallback_chain_walks_past_blind_entries_to_one_that_sees(monkeypatch):
    """The first entries are what the standard chain's GLM and qwen look like."""
    blind = _client(reply=_reply(INVENTED))
    sees = _client(reply=_reply("A real description."))
    entries = [
        {"provider": "zai", "model": BLIND},
        {"provider": "ollama", "model": UNKNOWN},
        {"provider": "moonshot", "model": SEES},
    ]
    monkeypatch.setattr("hermes_cli.fallback_config.get_fallback_chain", lambda _cfg: entries)
    monkeypatch.setattr(
        ac, "_resolve_fallback_entry", lambda e: (sees if e["model"] == SEES else blind, e["model"])
    )

    client, model, _label = ac._try_main_fallback_chain("vision", "anthropic")

    assert client is sees and model == SEES


def test_discovery_chain_skips_a_default_model_that_cannot_see(monkeypatch):
    blind = _client(reply=_reply(INVENTED))
    sees = _client(reply=_reply("A real description."))
    monkeypatch.setattr(
        ac,
        "_get_provider_chain",
        lambda: [("nous", lambda: (blind, BLIND)), ("local/custom", lambda: (blind, UNKNOWN)), ("openrouter", lambda: (sees, SEES))],
    )

    client, model, label = ac._try_payment_fallback("anthropic", task="vision")

    assert client is sees and label == "openrouter"


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_a_person_configured_vision_chain_is_held_to_the_same_rule(monkeypatch, mode):
    _explicit_route(monkeypatch)
    blind = _client(reply=_reply(INVENTED))
    monkeypatch.setattr(
        ac, "_get_auxiliary_task_config", lambda _t: {"fallback_chain": [{"provider": "zai", "model": BLIND}]}
    )
    monkeypatch.setattr(ac, "_resolve_fallback_entry", lambda e: (blind, e["model"]))
    monkeypatch.setattr(ac, "resolve_provider_client", lambda *_a, **_k: (None, None))

    with pytest.raises(ProviderError):
        _call(mode, _client(error=ProviderError(400, CREDIT)))

    assert _calls(blind) == 0


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_unbuildable_vision_provider_does_not_land_on_a_model_with_no_known_sight(monkeypatch, mode):
    """Picked before any request is sent: the configured provider has no client."""
    _explicit_route(monkeypatch)
    unknown = _client(reply=_reply(INVENTED))
    picked = unknown if mode == "sync" else unknown.as_async

    def _resolve(provider=None, **_kw):
        return ("ollama", picked, UNKNOWN) if provider == "auto" else ("anthropic", None, None)

    monkeypatch.setattr(ac, "resolve_vision_provider_client", _resolve)

    with pytest.raises(RuntimeError):
        if mode == "sync":
            ac.call_llm(task="vision", messages=MESSAGES)
        else:
            asyncio.run(ac.async_call_llm(task="vision", messages=MESSAGES))

    assert _calls(unknown) == 0


def test_main_models_vision_flag_does_not_speak_for_another_model(monkeypatch):
    """``model.supports_vision`` wins over everything in the real lookup, for any model."""
    monkeypatch.undo()
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {"supports_vision": True}})
    monkeypatch.setattr(ac, "_read_main_model", lambda: SEES)
    monkeypatch.setattr("agent.models_dev.get_model_capabilities", lambda _p, _m: None)
    assert real_lookup("custom", UNKNOWN, {"model": {"supports_vision": True}}) is True

    assert ac._fallback_candidate_can_see("custom", SEES) is True
    assert ac._fallback_candidate_can_see("custom", UNKNOWN) is False


def test_a_person_can_declare_that_a_model_sees(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"providers": {"custom": {"models": {UNKNOWN: {"supports_vision": True}}}}},
    )
    monkeypatch.setattr("agent.models_dev.get_model_capabilities", lambda _p, _m: None)

    assert ac._fallback_candidate_can_see("custom", UNKNOWN) is True


def test_other_tasks_keep_their_fallbacks(monkeypatch):
    fallback = MagicMock()
    monkeypatch.setattr(ac, "resolve_provider_client", lambda *_a, **_k: (fallback, BLIND))

    client, _model, _label = ac._try_main_agent_model_fallback("anthropic", task="compression")

    assert client is fallback
