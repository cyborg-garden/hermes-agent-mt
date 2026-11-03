"""What ``routing.cheap_extra_body`` actually does on the wire.

PR #163 shipped the passthrough justified by ``{"think": false}`` for a local
ollama cheap tier. That justification is wrong: ollama's OpenAI-compatible
``/v1/chat/completions`` ignores ``think``, ``chat_template_kwargs`` and an
in-prompt ``/no_think`` (ollama#14820). The only switch that stops a thinking
model is the first-class ``reasoning_effort: "none"``.

The mechanism still works, for a reason nobody wrote down: the OpenAI SDK
merges ``extra_body`` into the JSON body LAST, so a key there OVERRIDES a
same-named TYPED parameter the transport already emitted — one key on the wire,
the extra_body value winning. Every doc site now says so; these tests are what
stops that from silently regressing.

Measured against real ollama (qwen3.5:9b, max_tokens=200) on the Mini
2026-08-26: thinking on → 200 completion tokens, EMPTY content,
finish_reason=length. reasoning_effort="none" → 2 tokens, "OK", stop.
The SDK-merge assertions below reproduce the same request hermetically.
"""
import json

import httpx
import pytest
from openai import OpenAI

from agent.routing_override import apply_routing_override
from agent.transports import get_transport


ROUTED_NONE = {"reasoning_effort": "none"}


@pytest.fixture
def transport():
    import agent.transports.chat_completions
    return get_transport("chat_completions")


class _FakeAgent:
    """Minimal stand-in for the runtime surface apply_routing_override touches."""

    def __init__(self):
        self.model = "z-ai/glm-5.3"
        self.provider = "openrouter"
        self.request_overrides = {}
        self._primary_runtime = {"model": self.model, "provider": self.provider}

    def switch_model(self, model, provider, **kw):
        self.model, self.provider = model, provider


def _capture_wire_body(**create_kwargs) -> dict:
    """Fire one chat.completions.create through a mock transport, return the body."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["raw"] = request.content.decode()
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": "OK"},
                         "finish_reason": "stop"}],
        })

    client = OpenAI(
        api_key="test",
        base_url="http://local.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.chat.completions.create(**create_kwargs)
    seen["parsed"] = json.loads(seen["raw"])
    return seen


def test_extra_body_overrides_a_same_named_typed_param():
    """The load-bearing fact: extra_body is merged LAST and wins."""
    seen = _capture_wire_body(
        model="qwen3.5:9b",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=200,
        reasoning_effort="medium",          # typed SDK param
        extra_body=dict(ROUTED_NONE),       # routed cheap-tier body params
    )
    assert seen["parsed"]["reasoning_effort"] == "none"
    # Exactly ONE key on the wire — not a duplicate, not a dropped param.
    assert seen["raw"].count('"reasoning_effort"') == 1


def test_extra_body_key_without_a_typed_twin_is_simply_added():
    seen = _capture_wire_body(
        model="qwen3.5:9b",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"options": {"num_ctx": 8192}},
    )
    assert seen["parsed"]["options"] == {"num_ctx": 8192}


def test_routed_turn_carries_reasoning_effort_none_end_to_end(transport):
    """cheap_extra_body → request_overrides → transport kwargs → wire body."""
    from providers import get_provider_profile

    agent = _FakeAgent()
    assert apply_routing_override(agent, {
        "model": "qwen3.5:9b", "provider": "ollama", "extra_body": dict(ROUTED_NONE),
    }) is True
    assert agent.request_overrides["extra_body"] == ROUTED_NONE

    profile = get_provider_profile("ollama")
    kwargs = transport.build_kwargs(
        model="qwen3.5:9b",
        messages=[{"role": "user", "content": "hi"}],
        provider_profile=profile,
        provider_name="ollama",
        base_url="http://local.test/v1",
        supports_reasoning=True,
        reasoning_config={"enabled": True, "effort": "medium"},
        request_overrides=agent.request_overrides,
        max_tokens=200,
    )
    # The transport still emits the profile's typed param; the routed value
    # rides in extra_body and only wins at SDK body-assembly time.
    assert kwargs["reasoning_effort"] == "medium"
    assert kwargs["extra_body"]["reasoning_effort"] == "none"

    kwargs.pop("timeout", None)
    seen = _capture_wire_body(**kwargs)
    assert seen["parsed"]["reasoning_effort"] == "none"
    assert seen["raw"].count('"reasoning_effort"') == 1


def test_reasoning_overrides_is_the_supported_route_to_the_same_wire_shape(transport):
    """The first-class mechanism: per-model reasoning config, no extra_body hack.

    ``switch_model`` re-resolves ``reasoning_config`` against the ROUTED model
    (agent_runtime_helpers.resolve_reasoning_config), so
    ``agent.reasoning_overrides: {"qwen3.5:9b": none}`` reaches the transport as
    a disabled reasoning_config — and the custom/ollama profile emits both the
    working switch and ollama's native flag.
    """
    from providers import get_provider_profile

    kwargs = transport.build_kwargs(
        model="qwen3.5:9b",
        messages=[{"role": "user", "content": "hi"}],
        provider_profile=get_provider_profile("ollama"),
        provider_name="ollama",
        base_url="http://local.test/v1",
        supports_reasoning=True,
        reasoning_config={"enabled": False},
        max_tokens=200,
    )
    assert kwargs["reasoning_effort"] == "none"
    # Upstream v0.22.4 stopped emitting ollama's native ``think`` flag on the
    # OpenAI-compatible route (it was inert on /v1); if present it must be off.
    assert (kwargs.get("extra_body") or {}).get("think", False) is False

    kwargs.pop("timeout", None)
    seen = _capture_wire_body(**kwargs)
    assert seen["parsed"]["reasoning_effort"] == "none"


# Verbatim from plugins/intelligent_routing/README.md, "Pointing the cheap
# tier at a LOCAL model". Parsed here by the loader the runtime actually uses,
# so the recipe is proved end to end rather than asserted in prose.
README_RECIPE = """
model:
  default: "z-ai/glm-5.3"

routing:
  intelligent: true
  cheap_model: "qwen3.5:9b"
  cheap_provider: custom

agent:
  reasoning_overrides:
    "qwen3.5:9b": none
"""


def test_documented_reasoning_overrides_recipe_resolves_for_the_routed_model():
    """The README recipe must actually parse — `"qwen3.5:9b": none` → disabled.

    The YAML hop is load-bearing, not incidental. A bare ``none`` parses as the
    STRING "none" (YAML null is ``null``/``~``/empty), and parse_reasoning_effort
    maps none/false/disabled → {"enabled": False}. Had it parsed as a Python
    ``None``, resolve_reasoning_config would return None — NO override at all —
    and the documented recipe would silently no-op, leaving the cheap model
    burning its whole max_tokens budget on reasoning. So parse it for real.
    """
    from hermes_constants import resolve_reasoning_config
    from utils import fast_safe_load

    cfg = fast_safe_load(README_RECIPE)
    assert cfg["agent"]["reasoning_overrides"]["qwen3.5:9b"] == "none", (
        "bare `none` must parse as the STRING 'none', not YAML null"
    )
    # Resolved against the ROUTED model, which is what switch_model passes
    # (agent/agent_runtime_helpers.py: agent.reasoning_config =
    #  resolve_reasoning_config(_reasoning_cfg, agent.model)).
    assert resolve_reasoning_config(cfg, "qwen3.5:9b") == {"enabled": False}
    # The premium primary is untouched by the cheap tier's override.
    assert resolve_reasoning_config(cfg, "z-ai/glm-5.3") != {"enabled": False}
