"""Legacy length-gated classifier, used only by ``routing.mode: legacy``.

The live router is ``task_router.py``; the cheap model default lives in
``config.py``. The model names below describe the original tier choice.

Classifies an incoming turn using ONLY cheap, local signals available before the
LLM call — zero extra API cost, zero added latency. When the signals don't add
up to a confident cheap-tier classification, it FAILS OPEN to premium: a judgment
turn must never be silently downgraded, because a visibly worse answer to a human
is the expensive failure mode (spec open-question #4). This mirrors the fail-open
discipline HSM's ``validateCascadeEntries`` already uses for credential checks.

Everything here is a PURE function of ``TurnSignals`` — deterministic, no I/O,
no globals — so it is trivially testable and safe to call on the hot path.

── Two layers ──
1. **Binary** (``classify_turn`` / ``route_for``) — mechanical vs judgment. The
   original, minimal split. Retained for back-compat.
2. **Task-TYPE** (``classify_task_type`` / ``tier_for_task_type`` / ``route_task``)
   — adopted from upstream **PR #43534** (model-task-router, closed on
   NousResearch/hermes-agent), which routes by task TYPE rather than a binary
   split and grounds the mapping in DeepSWE reliability data.

── What we ADOPTED vs ADAPTED from #43534 ──
ADOPTED: the five task categories (code-gen / hard-architecture / orchestration /
research / mechanical), its keyword-priority decision tree (code > architecture >
mechanical > research > orchestration), and its core principle — *route by task
TYPE, not brand loyalty; treat benchmarks as directional* (see
``references/prior-art.md``, credited to Sugumaran Balasubramaniyan +
the deep-swe#21 correction thread).
ADAPTED: the tier mapping is to OUR fleet's actual tiers, NOT #43534's non-fleet
GPT-5.4/5.5. Specifically we route the cheap tier to OpenRouter
``deepseek/deepseek-v3.2`` — deliberately v3.2, NOT v4-Pro. That choice is
grounded directly in #43534's data: V4-Pro's DeepSWE *coding* throughput is
directionally lower (~8%, heavily caveated), so we do NOT route hard code-gen to
DeepSeek at all (code-gen -> premium/Claude); we route DeepSeek only where its
own data shows it competitive AND cheap — CLI/tool orchestration and mechanical
work (Terminal-Bench 67.9% @ $0.87/1M). See ``tier_for_task_type`` for the
per-category grounding.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- Binary classification labels (layer 1) ---
MECHANICAL = "mechanical"
JUDGMENT = "judgment"
UNCERTAIN = "uncertain"

# --- Task-TYPE categories (layer 2, adopted from PR #43534) ---
TASK_CODE_GEN = "code_gen"
TASK_ARCHITECTURE = "architecture"
TASK_ORCHESTRATION = "orchestration"
TASK_RESEARCH = "research"
TASK_MECHANICAL = "mechanical_task"
TASK_UNCERTAIN = "uncertain_task"

# Routing tiers (our fleet):
#   premium — Claude (per-agent primary: sonnet-5 / fable-5 / opus)
#   cheap   — OpenRouter deepseek/deepseek-v3.2 (see module docstring for why v3.2)
#   local   — ollama qwen/glm floor (not auto-selected by this classifier)
TIER_CHEAP = "cheap"
TIER_PREMIUM = "premium"
TIER_LOCAL = "local"

# A short mechanical ask is bounded in characters. Above this, a message carries
# enough substance that we no longer treat it as a one-shot mechanical request
# and defer to the other signals (fail open if nothing else fires). Kept
# deliberately conservative — false-mechanical on a real question is the costly
# error, so the length gate errs toward "not mechanical."
_SHORT_MESSAGE_CHARS = 120

# Short length is NECESSARY but NOT SUFFICIENT for mechanical: plenty of short
# human messages are judgment calls (e.g. "Can you look at the thing we
# discussed and get back to me?", or "what's the best architecture for this?").
# A short interactive ask is only mechanical when it ALSO leads with an imperative
# command verb (a positive signal). Question-shaped short asks fail open to
# premium — we do NOT special-case "what's X" prefixes (they front real judgment
# questions far more often than trivial lookups).

# Leading imperative verbs that mark a direct single-action command. Includes the
# unambiguous read-only shell verbs (grep/find/cat/ls/search/count) that are
# plainly mechanical work.
_IMPERATIVE_VERBS = frozenset({
    "mark", "add", "remove", "delete", "close", "open", "set", "move", "assign",
    "list", "show", "get", "fetch", "run", "start", "stop", "restart", "rename",
    "tag", "label", "archive", "unarchive", "create", "update", "check", "sort",
    "clear", "reset", "enable", "disable", "send", "post", "pin", "unpin",
    "grep", "find", "cat", "ls", "search", "count", "diff", "print",
})

# Markers of a hedged / deferred / context-dependent request — a defensive
# early-out that keeps such short messages OUT of mechanical. Kept as a
# belt-and-braces guard; the POSITIVE mechanical signal below is a leading
# imperative verb, so this blocklist is not load-bearing on its own.
_DEFERRAL_MARKERS = (
    "get back to me", "look at the", "look into", "we discussed", "we talked",
    "the thing", "figure out", "think about", "your thoughts", "what do you think",
    "not sure", "somehow", "whenever you", "when you get a chance",
)


def _is_short_mechanical_ask(message: str) -> bool:
    """True when a short message is a direct single-action command.

    Mechanical requires a POSITIVE signal — a leading imperative verb
    (grep/list/mark/close/…). We deliberately DO NOT treat trivial-query prefixes
    ("what's X" / "how many" / "when is") as mechanical: those routinely front
    real JUDGMENT questions ("what's the best architecture for this?",
    "when is it worth adding a cache?"), and a fail-*closed*-to-cheap prefix rule
    can't be patched with a blocklist. A genuinely trivial "what's 2+2" going
    premium is harmless — fail open to premium is the whole point.
    """
    text = message.strip().lower()
    if not text:
        return False
    if any(marker in text for marker in _DEFERRAL_MARKERS):
        return False
    first_word = text.split(maxsplit=1)[0].strip(".,!?:")
    return first_word in _IMPERATIVE_VERBS


@dataclass(frozen=True)
class TurnSignals:
    """Cheap local signals for one turn, all available before the LLM call.

    These map directly onto what the ``pre_llm_call`` hook already receives
    (``user_message``, ``platform``, conversation/tool context) plus a few
    booleans the registration layer derives from the runtime/session context.
    """

    user_message: str = ""
    # Live human chat vs. non-interactive invocation.
    is_interactive: bool = True
    # Cron / scheduled / background job (non-interactive by nature).
    is_background: bool = False
    # The request is kanban-triage-shaped (assign/prioritize/sort a backlog).
    is_kanban_triage: bool = False
    # The turn is expected to require multi-step reasoning.
    expects_multi_step: bool = False
    # The output is public-facing (announcement, published content, DM to a
    # non-operator). Public-facing ALWAYS goes premium.
    is_public_facing: bool = False


def classify_turn(sig: TurnSignals) -> str:
    """Return ``MECHANICAL`` / ``JUDGMENT`` / ``UNCERTAIN`` for one turn.

    Order matters: the strongest *judgment* signals are checked first so they
    can never be overridden by a coincidental mechanical signal (e.g. a short
    but public-facing message must not be classified mechanical).
    """
    message = (sig.user_message or "").strip()

    # No signal at all -> we cannot claim it's mechanical. Fail open.
    if not message and not sig.is_background:
        return UNCERTAIN

    # --- Judgment dominators (checked first, never overridden) ---
    if sig.is_public_facing:
        return JUDGMENT
    if sig.expects_multi_step:
        return JUDGMENT
    if len(message) > _SHORT_MESSAGE_CHARS:
        return JUDGMENT

    # --- Mechanical signals ---
    if sig.is_background or not sig.is_interactive:
        return MECHANICAL
    if sig.is_kanban_triage:
        return MECHANICAL
    if (
        message
        and len(message) <= _SHORT_MESSAGE_CHARS
        and _is_short_mechanical_ask(message)
    ):
        # Short, interactive, not multi-step, not public-facing, AND leading with
        # an imperative command verb: a one-shot mechanical ask.
        return MECHANICAL

    # Nothing decisive -> fail open.
    return UNCERTAIN


def route_for(sig: TurnSignals) -> str:
    """Collapse a binary classification to a routing tier, failing open to premium.

    ``mechanical`` -> ``TIER_CHEAP``. Everything else (``judgment`` AND
    ``uncertain``) -> ``TIER_PREMIUM``. Only a confident mechanical result ever
    reaches the cheap tier.
    """
    return TIER_CHEAP if classify_turn(sig) == MECHANICAL else TIER_PREMIUM


def decide_tier(sig: TurnSignals) -> str:
    """THE routing decision — the single source of truth, easy to probe in isolation.

    A turn goes to ``TIER_CHEAP`` ONLY when the CONSERVATIVE binary classifier is
    confidently ``MECHANICAL``. Everything the binary reads as ``JUDGMENT`` or
    ``UNCERTAIN`` — hard architecture, code-gen, open-ended reasoning,
    hedged/ambiguous asks, long messages, public-facing — FAILS OPEN to
    ``TIER_PREMIUM``. There is NO catch-all-to-cheap: the default is premium and
    only a proven-mechanical turn escapes it.

    This is intentionally identical to ``route_for``; it exists as a named,
    stable entry point the live-probe / re-audit calls directly. (The 5-way
    ``classify_task_type`` is NOT consulted here — it is a descriptive utility
    only, kept out of the routing path after the live-deploy catch-all bug.)
    """
    return TIER_CHEAP if classify_turn(sig) == MECHANICAL else TIER_PREMIUM


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: task-TYPE classification (adopted from PR #43534's model-task-router)
# ─────────────────────────────────────────────────────────────────────────────
#
# Keyword sets track #43534's decision tree (SKILL.md lines 101-121). Matched on
# a lowercased message. Priority order (first match wins) is #43534's:
#   code-gen > architecture > mechanical > research > orchestration.
# Architecture is checked before code-gen here ONLY when an explicit architecture
# phrase is present, because "design a system" would otherwise be miscaught by a
# bare code verb; the tests pin both orderings.

# Hard-architecture / high-stakes reasoning (checked first when explicit).
_ARCH_MARKERS = (
    "architecture", "design a system", "design the", "design an",
    "how would you structure", "security review", "complex debugging",
    "distributed race", "threat model", "system design",
)

# Code generation: implement / fix / refactor / write code / PR.
_CODE_MARKERS = (
    "implement", "refactor", "write code", "fix the bug", "fix bug",
    "add a feature", "add feature", "open a pr", "patch", "write a function",
    "write the migration", "migration script",
)
# Bare code verbs (need "first word" or standalone match to avoid false hits).
_CODE_VERBS = frozenset({"implement", "refactor", "code"})

# Research / analysis / lookup. Note: bare "what is"/"what's" are deliberately
# NOT here — they collide with the idiomatic "what's up" and, more importantly,
# front real judgment questions; research needs a stronger verb. (This is the
# descriptive task-type layer, not the routing path.)
_RESEARCH_MARKERS = (
    "research", "summarize", "summarise", "explain", "look up", "look up the",
    "search the web", "analyze", "analyse", "compare the", "what are the tradeoffs",
)

# Mechanical: read-only or single-command shell/file ops.
_MECHANICAL_MARKERS = (
    "grep", "find all", "list files", "list all files", "run the test",
    "run tests", "check if", "what's the content of", "list all", "show me the file",
)


def _has_any(text: str, markers) -> bool:
    return any(m in text for m in markers)


def classify_task_type(sig: TurnSignals) -> str:
    """Classify a turn into one of #43534's five task types (or uncertain).

    Priority (per #43534, SKILL.md line 92): code-gen > architecture > mechanical
    > research > orchestration; orchestration is the catch-all. Structural
    signals (background / kanban-triage) resolve to mechanical directly — those
    are unambiguous non-interactive/triage work.
    """
    message = (sig.user_message or "").strip()
    if not message and not (sig.is_background or sig.is_kanban_triage):
        return TASK_UNCERTAIN

    # Structural mechanical signals (no keyword needed).
    if sig.is_background or not sig.is_interactive or sig.is_kanban_triage:
        return TASK_MECHANICAL

    text = message.lower()

    # code-gen has top priority (a coding task that also says "run" is still
    # code-gen — #43534 priority order).
    if _has_any(text, _CODE_MARKERS) or text.split(maxsplit=1)[0].strip(".,!?:") in _CODE_VERBS:
        # ...unless it's explicitly an architecture ask, which outranks routine
        # coding (#43534: "do NOT use the architecture model for routine coding"
        # — but DO for explicit design/security/hard-debug).
        if _has_any(text, _ARCH_MARKERS):
            return TASK_ARCHITECTURE
        return TASK_CODE_GEN

    if _has_any(text, _ARCH_MARKERS):
        return TASK_ARCHITECTURE

    # Mechanical: an explicit mechanical marker, OR a short SINGLE-CLAUSE direct
    # command. Multi-clause asks ("check the deploy status AND tell me...") are
    # orchestration/diagnostics, not a single mechanical op — so a conjunction
    # disqualifies the short-command shortcut.
    _is_multi_clause = any(c in text for c in (" and ", " then ", ", ", " tell me"))
    if _has_any(text, _MECHANICAL_MARKERS) or (
        not _is_multi_clause and _is_short_mechanical_ask(message)
    ):
        return TASK_MECHANICAL

    if _has_any(text, _RESEARCH_MARKERS):
        return TASK_RESEARCH

    # Everything else — tool calls, shell, navigation, diagnostics, config,
    # planning — is orchestration (the catch-all, per #43534).
    return TASK_ORCHESTRATION


def tier_for_task_type(task_type: str) -> str:
    """Map a task type to one of our fleet tiers. Fail open to premium.

    Grounding (see ``references/prior-art.md``):
      - ARCHITECTURE -> premium: highest reasoning; Claude Opus 4.7 competitive on
        DeepSWE (54%). Never cheap-route hard design/security/debugging.
      - CODE_GEN     -> premium: DeepSWE indicates DeepSeek coding throughput is
        directionally lower (V4-Pro ~8%, caveated); keep multi-file code-gen on
        Claude for our fleet rather than risk a visibly worse implementation.
      - RESEARCH     -> premium: analysis/judgment, frequently public-facing.
      - ORCHESTRATION-> cheap:   DeepSeek is competitive at CLI/tool orchestration
        (V4-Pro Terminal-Bench 67.9% @ $0.87/1M) — this is the empirical basis
        for putting v3.2 here, NOT on code-gen.
      - MECHANICAL   -> cheap:   fast/cheap delegated workhorse tier.
      - anything else / UNCERTAIN -> premium (fail open).
    """
    if task_type in (TASK_ORCHESTRATION, TASK_MECHANICAL):
        return TIER_CHEAP
    # architecture, code_gen, research, uncertain, and any unknown -> premium.
    return TIER_PREMIUM


def route_task(sig: TurnSignals) -> str:
    """Task-type-aware routing: classify -> map to tier, with fail-open guards.

    Public-facing ALWAYS goes premium (dominates task type) — a public-facing
    turn must never be cheap-routed regardless of its shape.
    """
    if sig.is_public_facing:
        return TIER_PREMIUM
    return tier_for_task_type(classify_task_type(sig))
