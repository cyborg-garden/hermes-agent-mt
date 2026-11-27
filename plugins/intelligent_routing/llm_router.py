"""Optional LLM router (``routing.mode: llm-router``).

Asked ONLY about turns the heuristic in ``task_router.py`` cannot decide (no premium
signal, no cheap evidence). A confident heuristic decision never makes this call.

Rules:

* Least text: the current instruction only, sender prefix and runtime scaffolding
  stripped, cut to ``router_max_chars``. No history, no system prompt, no names.
* Never logged: neither the text sent nor the reply. Logs carry the outcome only.
* Never called on protected content (see ``config.DEFAULT_PROTECTED_MARKERS``) or on
  a platform in ``routing.router_exclude_platforms``.
* Strict wall-clock timeout, enforced by joining a worker thread (an HTTP client
  timeout bounds each socket read, not the whole call).
* Fails open: timeout, error, unparseable reply, low confidence, or a premium type
  all mean premium.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .task_router import ALL_TYPES, CHEAP_TYPES

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = (
    "You label one task for a model router. Reply with JSON only, no prose: "
    '{"type": "<type>", "confidence": <number 0-1>}.\n'
    "Types a small model does as well as a large one: lookup (read-only fetch, list, "
    "search, status), formatting (convert, extract or sort text that was supplied), "
    "summarise (a summary for the person asking), mechanical (mark done, tag, remind, "
    "run tests), orchestration (gather raw material and report it back).\n"
    "Types that need the large model: code_gen, architecture, research, public_writing, "
    "creative, judgment, sensitive, side_effect, contextual, uncertain.\n"
    "Anything that writes for other people, changes or deletes things, needs advice or "
    "reasoning, involves money, health, law, security, personal matters or te reo Maori, "
    "or depends on earlier conversation needs the large model. If unsure, answer "
    '"uncertain" with low confidence.'
)

OUTCOME_OK = "ok"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_ERROR = "error"
OUTCOME_UNPARSED = "unparsed"

_JSON_RX = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass(frozen=True)
class RouterResult:
    outcome: str
    task_type: str = ""
    confidence: float = 0.0
    latency_ms: int = 0

    def allows_cheap(self, min_confidence: float) -> bool:
        return (
            self.outcome == OUTCOME_OK
            and self.task_type in CHEAP_TYPES
            and self.confidence >= min_confidence
        )


def trim_for_router(text: str, max_chars: int) -> str:
    """Keep the head (where the ask usually is) and a little of the tail."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.75)
    tail = max_chars - head
    return text[:head].rstrip() + "\n[...]\n" + text[-tail:].lstrip()


def parse_reply(content: Any) -> Optional[tuple[str, float]]:
    if not isinstance(content, str):
        return None
    match = _JSON_RX.search(content)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    task_type = str(data.get("type") or "").strip().lower()
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        return None
    if task_type not in ALL_TYPES or not 0.0 <= confidence <= 1.0:
        return None
    return task_type, confidence


def _default_completion(text: str, settings: dict) -> Any:
    """One chat completion against the configured router model. Returns reply content."""
    from agent.auxiliary_client import resolve_provider_client

    client, resolved_model = resolve_provider_client(settings["provider"], model=settings["model"])
    if client is None:
        raise RuntimeError("router provider has no client")
    kwargs: dict[str, Any] = {
        "model": resolved_model or settings["model"],
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        # Room for a reasoning-mandatory model to think briefly before the JSON.
        "max_tokens": 400,
        "timeout": settings["timeout_s"],
    }
    if settings.get("extra_body"):
        kwargs["extra_body"] = dict(settings["extra_body"])
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# Patched in tests; never set from config.
_completion: Callable[[str, dict], Any] = _default_completion

_WORKER_NAME = "intelligent-routing-router"
class _DropRouterThread(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.threadName != _WORKER_NAME


_DROP_FILTER = _DropRouterThread()


def _install_log_filter() -> None:
    """Drop every record the router thread emits, whatever logger or client it comes
    from (openai, anthropic, httpx, botocore, ...): the filter sits on every handler,
    which is where a propagated record is written. Idempotent; called per router call."""
    loggers = [logging.getLogger()] + [
        lg for lg in list(logging.root.manager.loggerDict.values()) if isinstance(lg, logging.Logger)
    ]
    for lg in loggers:
        if _DROP_FILTER not in lg.filters:
            lg.addFilter(_DROP_FILTER)
        for handler in lg.handlers:
            if _DROP_FILTER not in handler.filters:
                handler.addFilter(_DROP_FILTER)


def ask_router(text: str, settings: dict) -> RouterResult:
    """Classify ``text`` with the router model. Never raises; never logs the text or reply."""
    payload = trim_for_router(text, settings["max_chars"])
    box: dict[str, Any] = {}

    def _work() -> None:
        try:
            box["content"] = _completion(payload, settings)
        # health: allow BLE001 -- fail-open boundary of the router call; the traceback is not logged because exception text can echo the routed message
        except Exception as exc:
            box["error"] = type(exc).__name__

    _install_log_filter()
    started = time.monotonic()
    # Keeps the caller's profile and secret scope (ContextVars do not cross a bare Thread),
    # so the router resolves this profile's credentials.
    from agent.memory_provider import spawn_context_thread
    worker = spawn_context_thread(_work, name=_WORKER_NAME, daemon=True)
    worker.start()
    worker.join(settings["timeout_s"])
    latency_ms = int((time.monotonic() - started) * 1000)
    if worker.is_alive():
        return RouterResult(OUTCOME_TIMEOUT, latency_ms=latency_ms)
    if "error" in box:
        logger.debug("intelligent_routing: router call failed (%s)", box["error"])
        return RouterResult(OUTCOME_ERROR, latency_ms=latency_ms)
    parsed = parse_reply(box.get("content"))
    if parsed is None:
        return RouterResult(OUTCOME_UNPARSED, latency_ms=latency_ms)
    return RouterResult(OUTCOME_OK, parsed[0], parsed[1], latency_ms)
