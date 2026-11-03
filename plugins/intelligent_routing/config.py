"""Per-agent config for intelligent routing (read from config.yaml).

Follows the ``image_gen`` plugin's pattern (``from hermes_cli.config import
load_config``; read a named section; best-effort with ``{}`` on failure). The
toggle lives in its own ``routing:`` namespace — a plugin owns its config
namespace, so this sidesteps needing a core ``routing:`` section (per the spec's
2026-07-09 update: teknium1 steered this to a plugin precisely so it doesn't
fight for space in core).

Defaults, on purpose:
  - ``routing.intelligent`` defaults to **False** — opt-in, because it changes
    which model answers a user's message (visible behavior). A failed/absent
    config reads as OFF, never silently ON.
  - ``routing.mode`` defaults to ``"heuristic"`` (Option A). ``"llm-router"``
    (Option B) is Phase 2 and not implemented here.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_DEFAULT_MODE = "heuristic"

# The cheap-metered tier target a mechanical/orchestration turn routes TO.
# Default: OpenRouter deepseek/deepseek-v3.2 — deliberately v3.2, NOT v4-Pro
# (grounded in PR #43534's DeepSWE data: V4-Pro's coding throughput is
# directionally lower, so we route DeepSeek only to mechanical/orchestration
# work where its Terminal-Bench data shows it competitive AND cheap, never to
# code-gen). Overridable per-agent via routing.cheap_model / routing.cheap_provider.
_DEFAULT_CHEAP_MODEL = "deepseek/deepseek-v3.2"
_DEFAULT_CHEAP_PROVIDER = "openrouter"

# Provider-specific request-BODY params to send WITH a cheap-routed turn — e.g.
# ollama ``{"options": {"num_ctx": 8192}}``, or an aggregator's ``provider``
# preferences. Empty by default: changes nothing unless set.
#
# NOT the place to disable thinking. Measured against ollama's OpenAI-compat
# /v1 endpoint (2026-08-26): ``think: false``, ``chat_template_kwargs`` and an
# in-prompt ``/no_think`` all do NOTHING (ollama#14820). The only switch that
# works is the first-class ``reasoning_effort: "none"``, and the supported way
# to set it for a routed model is per-model config:
#
#     agent:
#       reasoning_overrides:
#         "qwen3.5:9b": none
#
# ``switch_model`` re-resolves reasoning_config against the ROUTED model, and
# the custom/ollama provider profile then emits ``reasoning_effort: "none"``.
# Without it a thinking model burns its whole budget and returns an EMPTY
# completion — measured on qwen3.5:9b at max_tokens=200: 200 completion tokens,
# empty content, finish_reason=length; with it, 2 tokens and the right answer.
#
# NOTE: anything set here is merged into the JSON body LAST by the OpenAI SDK,
# so it OVERRIDES a same-named typed parameter the transport emitted.
_DEFAULT_CHEAP_EXTRA_BODY: dict[str, Any] = {}


def _load_config() -> dict[str, Any]:
    """Load the full config dict (``{}`` on any failure). Patched in tests."""
    from hermes_cli.config import load_config

    cfg = load_config()
    return cfg if isinstance(cfg, dict) else {}


def _routing_section() -> dict[str, Any]:
    try:
        cfg = _load_config()
    except Exception as exc:
        logger.debug("intelligent_routing: could not load config: %s", exc)
        return {}
    section = cfg.get("routing") if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return bool(value) if isinstance(value, (int, float)) else False


def is_intelligent_routing_enabled() -> bool:
    """True only when ``routing.intelligent`` is explicitly truthy. Default OFF."""
    return _coerce_bool(_routing_section().get("intelligent", False))


def routing_mode() -> str:
    """``heuristic`` (Option A, default) or ``llm-router`` (Option B, Phase 2)."""
    mode = _routing_section().get("mode", _DEFAULT_MODE)
    mode = str(mode).strip().lower() if mode else _DEFAULT_MODE
    return mode or _DEFAULT_MODE


def cheap_tier_target() -> tuple[str, str]:
    """Return ``(model, provider)`` for the cheap tier. Defaults to v3.2/openrouter."""
    section = _routing_section()
    model = str(section.get("cheap_model") or _DEFAULT_CHEAP_MODEL).strip() or _DEFAULT_CHEAP_MODEL
    provider = str(section.get("cheap_provider") or _DEFAULT_CHEAP_PROVIDER).strip() or _DEFAULT_CHEAP_PROVIDER
    return model, provider


def cheap_extra_body() -> dict[str, Any]:
    """Return ``routing.cheap_extra_body`` as a dict (``{}`` when unset/malformed).

    Returns a shallow copy so callers cannot mutate the loaded config.
    """
    value = _routing_section().get("cheap_extra_body", _DEFAULT_CHEAP_EXTRA_BODY)
    if not isinstance(value, dict):
        if value not in (None, "", {}):
            logger.debug(
                "intelligent_routing: routing.cheap_extra_body is %s, expected a "
                "mapping — ignoring", type(value).__name__,
            )
        return {}
    return dict(value)
