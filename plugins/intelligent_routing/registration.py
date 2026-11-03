"""register(ctx) — wire intelligent routing onto the ``pre_llm_call`` hook.

Enabling the plugin (``hermes plugins enable intelligent_routing``) attaches the
hook; the PER-AGENT ``routing.intelligent`` toggle (config.yaml, default OFF)
gates whether it does anything. When the toggle is OFF the hook returns ``None``
— zero behavior change, the opt-in guarantee.

When ON, the hook runs the Option-A heuristic classifier over cheap local
signals derived from the hook kwargs and returns:
  1. a context injection that EXTENDS the model-awareness line
     (``agent/model_awareness.py``, spec B1) with the routing reason, e.g.::

         [Current model: claude-sonnet-5 via anthropic — routed: mechanical → cheap]

     so a user is never confused by a silent tier switch mid-thread; and
  2. for a cheap-tier turn, a ``route`` override
     (``{"model": ..., "provider": ...}``) that the runtime ACTUATES via the
     routing-override interface extension (``agent/routing_override.py``), which
     swaps the model for THIS turn (turn-scoped, fail-safe). Premium / uncertain
     / public-facing turns emit NO override and stay on the configured primary.

This is the clean plugin form of upstream PR #43534, and the interface extension
is teknium1's offered "add to the plugin interface" made concrete: a
``pre_llm_call`` result may now carry a per-turn model/provider override.

The reactive fallback chain (``try_activate_fallback`` / ``FailoverReason``) is
deliberately untouched — intelligent routing is a proactive pre-turn layer on a
different axis; reactive escalation still applies underneath the chosen tier.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from .classifier import (
    TIER_CHEAP,
    TurnSignals,
    classify_turn,
    decide_tier,
)
from .config import (
    cheap_extra_body,
    cheap_tier_target,
    is_intelligent_routing_enabled,
)

logger = logging.getLogger(__name__)

# Platforms that are non-interactive by nature (cron/scheduled/background jobs).
# A turn arriving on one of these is mechanical unless a stronger judgment
# signal fires. Kept as a small, explicit set rather than guessing.
_NON_INTERACTIVE_PLATFORMS = frozenset({"cron", "background", "scheduler", "batch"})

# --- User model directive ("use fable for this") ---
# A user can explicitly request a model by name. The directive short-circuits
# the classifier entirely — fires BEFORE the mechanical/judgment split.
# Critical: "use" IS in _IMPERATIVE_VERBS, so without this early intercept
# a bare "use fable" would be classified mechanical → cheap → deepseek.
#
# Map: lowercase keyword → (model_id, provider). None means "stay on primary".
_USER_MODEL_MAP = {
    "fable": ("claude-fable-5", "anthropic"),
    "opus": ("claude-opus-4-8", "anthropic"),
    "sonnet": ("claude-sonnet-4-6", "anthropic"),
    "haiku": ("claude-haiku-4-5-20251001", "anthropic"),
    "deepseek": ("deepseek/deepseek-v3.2", "openrouter"),
    "kimi": ("moonshotai/kimi-k3", "openrouter"),
    "kimi3": ("moonshotai/kimi-k3", "openrouter"),
    "k3": ("moonshotai/kimi-k3", "openrouter"),
    "glm": ("z-ai/glm-5.2", "openrouter"),
    "claude": None,  # "use claude" = stay on configured primary
}
_DIRECTIVE_RE = re.compile(
    r"\buse\s+({models})\b".format(models="|".join(re.escape(k) for k in _USER_MODEL_MAP)),
    re.IGNORECASE,
)
# Sentinel: "no directive found" — separate from None ("use claude" / no override).
_NO_DIRECTIVE = object()


def _parse_model_directive(message: str) -> object:
    """Return (model, provider), None (stay on primary), or ``_NO_DIRECTIVE``."""
    m = _DIRECTIVE_RE.search(message)
    if not m:
        return _NO_DIRECTIVE
    return _USER_MODEL_MAP.get(m.group(1).lower(), _NO_DIRECTIVE)

# Shared multi-user (group) sessions prepend a sender attribution to the message
# before it reaches pre_llm_call — gateway/run.py emits ``f"[{user_name}] {text}"``.
# Left in place, that "[mare] " prefix makes the classifier's leading-imperative
# check read "[mare]" instead of the real first word ("list"), so EVERY group
# turn fails the mechanical test and routes premium — the router goes inert in
# exactly the place agents live. Peel a SINGLE leading "[token] " attribution
# (one line, bounded length, no nested "]") so the classifier sees the real ask.
# Deliberately conservative: only one leading token, and the peeled remainder
# still runs the full conservative classifier (fails open to premium on anything
# non-mechanical), so an over-peel can never wrongly cheap-route real work.
_SENDER_PREFIX_RE = re.compile(r"^\[[^\]\n]{1,64}\]\s+")

# Metrics-only tier label. NOT a classifier tier (the classifier never sees a
# directive turn — the intercept runs before it), so it does not belong beside
# TIER_CHEAP/TIER_PREMIUM in classifier.py. It keeps user-directed turns
# countable and separable from classified ones in the same log stream.
TIER_DIRECTIVE = "directive"


def _strip_sender_prefix(message: str) -> str:
    """Peel one leading ``[sender] `` group-attribution prefix, if present."""
    return _SENDER_PREFIX_RE.sub("", message, count=1)


def _signals_from_kwargs(**kwargs: Any) -> TurnSignals:
    """Derive cheap ``TurnSignals`` from the ``pre_llm_call`` kwargs.

    Only uses signals that are genuinely free to read here: the message text,
    and the platform (to tell a background job from a live human chat). Richer
    signals (tool-call history, kanban shape, public-facing) are wired as they
    become reliably available from the hook context; absent, they default to the
    conservative value that keeps a turn OUT of the cheap tier.
    """
    message = kwargs.get("user_message") or ""
    if not isinstance(message, str):
        message = str(message)
    # Peel the group sender-attribution prefix before classifying (see above).
    message = _strip_sender_prefix(message)
    platform = (kwargs.get("platform") or "")
    platform = platform.strip().lower() if isinstance(platform, str) else ""
    is_background = platform in _NON_INTERACTIVE_PLATFORMS
    return TurnSignals(
        user_message=message,
        is_interactive=not is_background,
        is_background=is_background,
        is_public_facing=bool(kwargs.get("is_public_facing", False)),
    )


def probe_route(message: str, **signals: Any) -> dict:
    """Probe the routing decision for a message in ISOLATION — no config, no hook.

    Returns ``{"tier": "cheap"|"premium", "classification": ...}`` so a live-probe
    or re-audit can feed a batch of representative messages and see exactly what
    each would route to, without mocking config or the pre_llm_call plumbing.
    ``signals`` accepts the same kwargs as the hook (e.g. ``platform="cron"``,
    ``is_public_facing=True``).
    """
    sig = _signals_from_kwargs(user_message=message, **signals)
    return {"tier": decide_tier(sig), "classification": classify_turn(sig)}


def _reason_line(model: Any, provider: Any, reason: str) -> str:
    """Build the extended model-awareness line: ``[... — routed: <reason>]``.

    Reuses ``format_current_model_line`` so the base format stays byte-identical
    to the existing model-awareness injection, then appends the routing reason.
    """
    from agent.model_awareness import format_current_model_line

    base = format_current_model_line(
        str(model or ""), str(provider or "")
    )
    if not base:
        # No known model -> emit a minimal, still-informative line.
        base = "[Current model: unknown]"
    # Splice the reason inside the trailing bracket: "[...]" -> "[... — routed: x]".
    if base.endswith("]"):
        return f"{base[:-1]} — routed: {reason}]"
    return f"{base} — routed: {reason}"


def on_pre_llm_call(**kwargs: Any) -> Optional[dict]:
    """pre_llm_call hook. Inert when disabled; otherwise inject the routing line.

    Never raises into the turn loop — any error is swallowed and treated as
    "inject nothing" (the safe, behavior-preserving default).
    """
    try:
        if not is_intelligent_routing_enabled():
            return None

        # Strip sender prefix before any further parsing (group-chat attribution).
        raw_message = kwargs.get("user_message") or ""
        if not isinstance(raw_message, str):
            raw_message = str(raw_message)
        stripped = _strip_sender_prefix(raw_message)
        # Whether a "[sender] " attribution was peeled IS the group signal — a
        # Discord DM and a Discord group turn share platform="discord", so
        # platform cannot separate them. The group/DM split is the dimension the
        # harvest needs most: a past bug made group chat inert (what
        # _SENDER_PREFIX_RE fixes), and without this flag a recurrence is
        # invisible in the log.
        metric_dims = {
            "session_id": kwargs.get("session_id"),
            "turn_id": kwargs.get("turn_id"),
            "platform": kwargs.get("platform") or "",
            "is_group": stripped != raw_message,
        }

        # ── User model directive ("use fable for this") ───────────────────────
        # Check BEFORE the classifier. "use" is in _IMPERATIVE_VERBS so without
        # this intercept "use fable" would be classified mechanical → deepseek.
        directive = _parse_model_directive(stripped)
        if directive is not _NO_DIRECTIVE:
            if directive is None:
                # "use claude" — stay on configured primary; inject a context note.
                reason = "user-requested → primary"
                line = _reason_line(kwargs.get("model"), kwargs.get("provider"), reason)
                # A directive is the pushback signal — "the user rejected the
                # cheap answer" is exactly what the upstream numbers need, and
                # it is also a real turn. Returning here without a row both lost
                # that signal and left the log with no total-turn denominator,
                # so any cheap-share read off it described classified turns only.
                log_routing_decision(
                    tier=TIER_DIRECTIVE, model=kwargs.get("model"),
                    provider=kwargs.get("provider"), reason=reason,
                    primary_model=kwargs.get("model"), **metric_dims,
                )
                return {"context": line}
            model, provider = directive
            reason = f"user-requested → {model.split('/')[0] if '/' in model else model}"
            line = _reason_line(kwargs.get("model"), kwargs.get("provider"), reason)
            log_routing_decision(
                tier=TIER_DIRECTIVE, model=model, provider=provider, reason=reason,
                primary_model=kwargs.get("model"), **metric_dims,
            )
            return {"route": {"model": model, "provider": provider}, "context": line}

        # ── Heuristic classifier (no explicit directive) ───────────────────────
        sig = _signals_from_kwargs(**kwargs)

        # THE routing decision — deliberately SIMPLE (the classifier is not the
        # contribution; the interface extension is). decide_tier() gates cheap on
        # the CONSERVATIVE binary classifier: cheap ONLY on a confident MECHANICAL,
        # else premium. No catch-all-to-cheap.
        tier = decide_tier(sig)

        # Reason line reflects the ACTUAL decider (the binary classification), so
        # the label never disagrees with the tier.
        classification = classify_turn(sig)  # mechanical | judgment | uncertain
        reason = f"{classification} → {tier}"

        line = _reason_line(kwargs.get("model"), kwargs.get("provider"), reason)
        result: dict = {"context": line}

        # Emit a `route` override ONLY for the cheap tier — this is what the
        # runtime's routing-override extension (agent/routing_override.py)
        # actuates. Premium / fail-open turns emit no override and stay on the
        # configured primary.
        routed_model, routed_provider = kwargs.get("model"), kwargs.get("provider")
        if tier == TIER_CHEAP:
            cheap_model, cheap_provider = cheap_tier_target()
            if cheap_model:
                route = {"model": cheap_model, "provider": cheap_provider}
                # Provider-specific body params for the cheap tier (e.g.
                # {"options": {"num_ctx": N}} on a local ollama model). Omitted
                # entirely when unset, so nothing changes for existing cloud
                # tiers. Thinking control is NOT set here — see
                # config.cheap_extra_body for why and what to use instead.
                extra_body = cheap_extra_body()
                if extra_body:
                    route["extra_body"] = extra_body
                result["route"] = route
                routed_model, routed_provider = cheap_model, cheap_provider

        log_routing_decision(
            tier=tier,
            model=routed_model,
            provider=routed_provider,
            reason=reason,
            primary_model=kwargs.get("model"),
            **metric_dims,
        )
        return result
    except Exception as exc:
        logger.warning("intelligent_routing: pre_llm_call failed (inert): %s", exc)
        return None


def log_routing_decision(
    *, tier, model, provider, reason, primary_model,
    session_id=None, turn_id=None, platform=None, is_group=None,
) -> None:
    """Emit ONE machine-parseable decision record per classified turn.

    Format: ``routing_decision {json}`` at INFO on this module's logger, so a
    window of fleet data can be harvested with a grep + ``json.loads`` on the
    suffix. This is the instrumentation the upstream-PR spec
    (2026-07-23, "numbers needed for the Nous upstream PR") names as its first
    prerequisite: *"Routing decisions must be countable: parseable per-turn log
    line of tier, model, reason."* Absent it, coverage/cost/fail-open rates
    cannot be computed, which is why Stage 2 never had numbers to attach.

    Never raises — metrics must not be able to break a turn.
    """
    try:
        logger.info(
            "routing_decision %s",
            json.dumps(
                {
                    "tier": tier,
                    "model": model,
                    "provider": provider,
                    "reason": reason,
                    "primary_model": primary_model,
                    # Join keys. Cost / latency / 429-rate live on the POST-call
                    # side (usage, api_duration, finish_reason), which this
                    # pre-call hook structurally cannot see; carrying turn_id +
                    # session_id lets those rows be joined against an already
                    # deployed observability plugin instead of blocking on a
                    # post_llm_call hook here.
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "platform": platform,
                    "is_group": is_group,
                },
                default=str,
                ensure_ascii=False,
            ),
        )
    except Exception:
        logger.debug("intelligent_routing: metric emit failed", exc_info=True)


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    logger.info(
        "intelligent_routing active: pre_llm_call classifier registered "
        "(per-agent toggle routing.intelligent gates behavior; default OFF)"
    )
