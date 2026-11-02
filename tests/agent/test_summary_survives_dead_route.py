"""A dead summary route must move on to a route that works."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.auxiliary_client import _is_payment_error, call_llm
from agent.context_compressor import ContextCompressor

CREDIT_400 = (
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
    "'message': 'Your credit balance is too low to access the Anthropic API. "
    "Please go to Plans & Billing to upgrade or purchase credits.'}}"
)
DATA_POLICY_404 = (
    "Error code: 404 - {'error': {'message': '0 endpoints out of 1 requested are available "
    "matching your guardrail restrictions and data policy. We removed them for the following "
    "reasons: ZDR violation (account settings): 1 endpoint excluded'}}"
)


class _ApiError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class _Message:
    content = "## Goal\nreal summary from the working route"


class _Choice:
    message = _Message()
    finish_reason = "stop"


class _Response:
    choices = [_Choice()]
    usage = None
    model = "m"


def _client(base_url, *, fails_with=None):
    client = MagicMock()
    client.base_url = base_url
    if fails_with is not None:
        client.chat.completions.create.side_effect = fails_with
    else:
        client.chat.completions.create.return_value = _Response()
    return client


def _compressor(provider, model):
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        return ContextCompressor(model=model, provider=provider, quiet_mode=True)


TURNS = [
    {"role": "user", "content": "please do the thing"},
    {"role": "assistant", "content": "doing it"},
]


def test_anthropic_credit_400_counts_as_a_payment_error():
    assert _is_payment_error(_ApiError(400, CREDIT_400)) is True


@pytest.mark.parametrize(
    "message",
    ["invalid value for billing_address", "unknown parameter: credits", "max_tokens too large"],
)
def test_ordinary_400_is_not_a_payment_error(message):
    assert _is_payment_error(_ApiError(400, message)) is False


def test_data_policy_404_is_recognised():
    from agent.auxiliary_client import _is_data_policy_refusal

    assert _is_data_policy_refusal(_ApiError(404, DATA_POLICY_404)) is True
    assert _is_data_policy_refusal(_ApiError(404, "page not found")) is False


def test_blocked_vision_route_is_not_handed_to_a_model_that_may_be_blind(monkeypatch):
    blocked = _client("https://openrouter.ai/api/v1", fails_with=_ApiError(404, DATA_POLICY_404))
    other = _client("https://openrouter.ai/api/v1")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("openrouter", "google/gemini-2.5-flash", None, None, None),
    ), patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        return_value=("openrouter", blocked, "google/gemini-2.5-flash"),
    ), patch(
        "agent.auxiliary_client.resolve_provider_client", return_value=(other, "glm")
    ), patch(
        "agent.auxiliary_client._read_main_provider", return_value="openrouter"
    ), patch(
        "agent.auxiliary_client._read_main_model", return_value="glm"
    ):
        with pytest.raises(_ApiError):
            call_llm(task="vision", messages=[{"role": "user", "content": "see"}])
    other.chat.completions.create.assert_not_called()


def test_summary_moves_to_the_fallback_chain_when_main_has_no_credit(monkeypatch):
    """The logged failure: main out of credit, a working fallback configured."""
    dead = _client("https://api.anthropic.com", fails_with=_ApiError(400, CREDIT_400))
    working = _client("https://openrouter.ai/api/v1")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"fallback_providers": [{"provider": "openrouter", "model": "deepseek/flash"}]},
    )
    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("auto", None, None, None, None),
    ), patch(
        "agent.auxiliary_client._get_cached_client", return_value=(dead, "claude-opus-5-5")
    ), patch(
        "agent.auxiliary_client.resolve_provider_client", return_value=(working, "deepseek/flash")
    ), patch(
        "agent.auxiliary_client._read_main_provider", return_value="anthropic"
    ):
        summary = _compressor("anthropic", "claude-opus-5-5")._generate_summary(TURNS)

    assert summary is not None
    assert "real summary from the working route" in summary
    working.chat.completions.create.assert_called_once()


def test_blocked_configured_route_uses_the_main_model_on_the_same_provider(monkeypatch):
    """auxiliary.compression names a model the account's data policy refuses."""
    blocked = _client("https://openrouter.ai/api/v1", fails_with=_ApiError(404, DATA_POLICY_404))
    working = _client("https://openrouter.ai/api/v1")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("openrouter", "google/gemini-2.5-flash", None, None, None),
    ), patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(blocked, "google/gemini-2.5-flash"),
    ), patch(
        "agent.auxiliary_client.resolve_provider_client", return_value=(working, "kimi")
    ), patch(
        "agent.auxiliary_client._read_main_provider", return_value="openrouter"
    ), patch(
        "agent.auxiliary_client._read_main_model", return_value="kimi"
    ):
        summary = _compressor("openrouter", "kimi")._generate_summary(TURNS)

    assert summary is not None
    assert "real summary from the working route" in summary


def test_blocked_configured_route_walks_the_fallback_chain_without_a_main_client(monkeypatch):
    blocked = _client("https://openrouter.ai/api/v1", fails_with=_ApiError(404, DATA_POLICY_404))
    working = _client("https://openrouter.ai/api/v1")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"fallback_providers": [{"provider": "openrouter", "model": "deepseek/flash"}]},
    )

    def _resolve(provider, model=None, **_kw):
        return (working, model) if model == "deepseek/flash" else (None, None)

    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("openrouter", "google/gemini-2.5-flash", None, None, None),
    ), patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(blocked, "google/gemini-2.5-flash"),
    ), patch(
        "agent.auxiliary_client.resolve_provider_client", side_effect=_resolve
    ), patch(
        "agent.auxiliary_client._read_main_provider", return_value="openrouter"
    ), patch(
        "agent.auxiliary_client._read_main_model", return_value="kimi"
    ):
        summary = _compressor("openrouter", "kimi")._generate_summary(TURNS)

    assert summary is not None
    assert "real summary from the working route" in summary


def test_refused_main_model_on_auto_tries_other_models_on_the_same_provider(monkeypatch):
    """The whole chain can sit on one aggregator, as it does on this fleet."""
    blocked = _client("https://openrouter.ai/api/v1", fails_with=_ApiError(404, DATA_POLICY_404))
    working = _client("https://openrouter.ai/api/v1")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"fallback_providers": [{"provider": "openrouter", "model": "deepseek/flash"}]},
    )
    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("auto", None, None, None, None),
    ), patch(
        "agent.auxiliary_client._get_cached_client", return_value=(blocked, "kimi")
    ), patch(
        "agent.auxiliary_client.resolve_provider_client", return_value=(working, "deepseek/flash")
    ), patch(
        "agent.auxiliary_client._read_main_provider", return_value="openrouter"
    ):
        summary = _compressor("openrouter", "kimi")._generate_summary(TURNS)

    assert summary is not None
    assert "real summary from the working route" in summary
