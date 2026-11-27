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
  - ``routing.mode`` defaults to ``"heuristic"``: the task-type classifier in
    ``task_router.py``, no extra call. ``"llm-router"`` adds a small model call
    for turns the heuristic cannot decide (see ``llm_router.py``).
  - ``legacy`` keeps the old length-gated binary classifier, for comparison.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_DEFAULT_MODE = "heuristic"

# The cheap tier a cheap-safe task routes TO. Pinned to a dated snapshot: the bare
# ``deepseek/deepseek-v4-flash`` slug resolves to an older preview on OpenRouter.
# Only cheap-safe task types ever reach it (lookups, formatting, summaries,
# gathering); code, design and judgment stay on the primary. Overridable per agent
# via routing.cheap_model / routing.cheap_provider.
_DEFAULT_CHEAP_MODEL = "deepseek/deepseek-v4-flash-0731"
_DEFAULT_CHEAP_PROVIDER = "openrouter"

_MODES = frozenset({"heuristic", "llm-router", "legacy"})

# LLM router defaults. The router is asked only about turns the heuristic cannot
# decide, and only on non-interactive turns unless router_on_interactive is set:
# a human waiting should not pay the extra round trip.
_DEFAULT_ROUTER_TIMEOUT_S = 2.5
_DEFAULT_ROUTER_MIN_CONFIDENCE = 0.85
_DEFAULT_ROUTER_MAX_CHARS = 800

# Text that marks content as protected (the Papa Reo transcript footer by default):
# a turn carrying it never reaches the LLM router and never leaves the primary.
# The mid-turn check in core (agent/routing_override.py) keeps the same list; a test
# holds the two equal. Extend per agent with routing.protected_markers.
DEFAULT_PROTECTED_MARKERS: tuple[str, ...] = (
    "transcribed by papa reo api",
)

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


def _full_config() -> dict[str, Any]:
    """The whole config, or ``{}`` when it cannot be read (callers then see nothing configured)."""
    try:
        return _load_config()
    except Exception:  # boundary: unreadable config must never break a turn or a spawn
        logger.debug("intelligent_routing: could not load config", exc_info=True)
        return {}


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
    """``heuristic`` (default), ``llm-router`` or ``legacy``. Unknown values read as heuristic."""
    mode = _routing_section().get("mode", _DEFAULT_MODE)
    mode = str(mode).strip().lower().replace("_", "-") if mode else _DEFAULT_MODE
    return mode if mode in _MODES else _DEFAULT_MODE


def cheap_tier_target() -> tuple[str, str]:
    """Return ``(model, provider)`` for the cheap tier."""
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


def _float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if lo <= f <= hi else default


def escalation_enabled() -> bool:
    """``routing.escalate`` (default ON): retry a failed, empty or refused cheap turn on the primary."""
    value = _routing_section().get("escalate", True)
    return True if value is None else _coerce_bool(value)


def extra_protected_markers() -> list[str]:
    """``routing.protected_markers`` only (lower-cased), for passing along on a route."""
    return list(protected_markers()[len(DEFAULT_PROTECTED_MARKERS):])


def protected_markers() -> tuple[str, ...]:
    """Lower-cased marker strings; the defaults plus ``routing.protected_markers``."""
    extra = _routing_section().get("protected_markers") or []
    if isinstance(extra, str):
        extra = [extra]
    if not isinstance(extra, (list, tuple)):
        extra = []
    return DEFAULT_PROTECTED_MARKERS + tuple(str(m).strip().lower() for m in extra if str(m).strip())


def router_settings() -> dict[str, Any]:
    """Everything the LLM router reads, coerced. Model/provider default to the cheap tier."""
    section = _routing_section()
    cheap_model, cheap_provider = cheap_tier_target()
    exclude = section.get("router_exclude_platforms") or []
    if isinstance(exclude, str):
        exclude = [exclude]
    return {
        "model": str(section.get("router_model") or cheap_model).strip() or cheap_model,
        "provider": str(section.get("router_provider") or cheap_provider).strip() or cheap_provider,
        "timeout_s": _float(section.get("router_timeout_s"), _DEFAULT_ROUTER_TIMEOUT_S, 0.2, 30.0),
        "min_confidence": _float(section.get("router_min_confidence"), _DEFAULT_ROUTER_MIN_CONFIDENCE, 0.5, 1.0),
        "max_chars": int(_float(section.get("router_max_chars"), _DEFAULT_ROUTER_MAX_CHARS, 100, 4000)),
        "on_interactive": _coerce_bool(section.get("router_on_interactive", False)),
        "exclude_platforms": frozenset(str(p).strip().lower() for p in exclude if str(p).strip()),
    }


def cheap_reasoning_effort() -> Optional[str]:
    """``routing.cheap_reasoning_effort`` for cheap subagents, or None to inherit.

    Leave unset for z-ai/glm-5.3-flash: that endpoint rejects a request with
    reasoning turned off (HTTP 400, "Reasoning is mandatory"), so ``none`` there
    would fail every cheap child (escalation would then retry it on the primary).
    """
    value = _routing_section().get("cheap_reasoning_effort")
    value = str(value).strip().lower() if value not in (None, "") else ""
    return value or None


def primary_route() -> dict[str, Any]:
    """The profile's configured primary (``model.default`` / ``model.provider`` / ``model.base_url``).

    The escalation target for a cheap subagent: its fallback chain starts here.
    Empty dict when the config has no usable primary.
    """
    cfg = _full_config()
    model_cfg = cfg.get("model") if isinstance(cfg, dict) else None
    if isinstance(model_cfg, str):
        return {"model": model_cfg} if model_cfg.strip() else {}
    if not isinstance(model_cfg, dict):
        return {}
    model = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
    if not model:
        return {}
    route = {"model": model}
    provider = str(model_cfg.get("provider") or "").strip()
    if provider:
        route["provider"] = provider
    base_url = str(model_cfg.get("base_url") or "").strip()
    if base_url:
        route["base_url"] = base_url
    return route


def top_level_fallback_chain() -> list[dict[str, Any]]:
    """``fallback_providers`` from the profile config (entries that are mappings)."""
    cfg = _full_config()
    chain = cfg.get("fallback_providers") if isinstance(cfg, dict) else None
    return [dict(e) for e in chain if isinstance(e, dict)] if isinstance(chain, list) else []
