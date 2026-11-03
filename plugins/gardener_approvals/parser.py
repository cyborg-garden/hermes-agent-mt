"""Pure intent parser: a Signal reply + the open decisions -> a structured
action. No I/O, no subprocess — fully unit-testable. The gate re-validates
everything; this parser only decides what (if anything) to ask the gate to do.

Intent mapping:
  - hold-intent              → action="resolve", value="hold"
  - approve-intent + PR dec  → action="merge",   value=str(pr_number)
  - approve-intent + no PR   → action="resolve",  value="approve"
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .decisions import Decision

_DEC_ID_RE = re.compile(r"D-\d{4}-\d{2}-\d{2}-\d+")

# Single-word keywords — matched by word boundary (token membership).
# Short ambiguous words ("go", "ok") are excluded on purpose — a false trigger
# resolves a decision the user didn't mean.
_APPROVE_WORDS = frozenset(("approve", "approved", "approves", "yes", "yep", "yeah", "lgtm",
                             "merge"))
_APPROVE_PHRASES = ("do it", "go ahead", "ship it", "merge it")   # substring match
_APPROVE_EMOJI = ("\U0001F44D", "✅")                               # raw text substring match

# Hold keywords — single-word boundary-matched; "not"/"never"/"dont" also serve
# as negation so that "not approved" / "I disapprove" → conflict → no-op.
_HOLD_WORDS = frozenset(("hold", "wait", "nope", "later", "no", "not", "never", "dont"))
_HOLD_PHRASES = ("not yet", "do not", "don't")                     # substring match


@dataclass
class ParseResult:
    matched: bool
    action: Optional[str] = None        # "resolve" | "merge"
    value: Optional[str] = None         # "approve" | "hold" | str(pr_number)
    decision_id: Optional[str] = None
    ask: Optional[str] = None           # set -> inject this, DO NOT call the gate


def _has_phrase(text: str, phrases) -> bool:
    return any(p in text for p in phrases)


def _has_word(text: str, words) -> bool:
    toks = set(re.findall(r"[a-z]+", text))
    return bool(toks & words)


def has_approval_intent(text: str) -> bool:
    """Return True if *text* contains any approve-, hold-, or merge-intent keyword
    using the same word-boundary / phrase / emoji logic as parse_reply.  Conflicting
    signals (approve + hold together) are still treated conservatively as False,
    matching the no-op behaviour of parse_reply.
    """
    raw = text or ""
    low = raw.lower()
    approve = (_has_word(low, _APPROVE_WORDS)
               or _has_phrase(low, _APPROVE_PHRASES)
               or any(e in raw for e in _APPROVE_EMOJI))
    hold = (_has_word(low, _HOLD_WORDS)
            or _has_phrase(low, _HOLD_PHRASES))
    if approve and hold:
        return False   # conflicting → conservative no-op
    return approve or hold


def parse_reply(text: str, decisions: list[Decision]) -> ParseResult:
    if not decisions:
        return ParseResult(matched=False)
    open_ids = [d.id for d in decisions]
    raw = text or ""
    low = raw.lower()

    approve = (_has_word(low, _APPROVE_WORDS)
               or _has_phrase(low, _APPROVE_PHRASES)
               or any(e in raw for e in _APPROVE_EMOJI))
    hold = (_has_word(low, _HOLD_WORDS)
            or _has_phrase(low, _HOLD_PHRASES))
    if approve and hold:
        return ParseResult(matched=False)            # conflicting -> no-op
    if not approve and not hold:
        return ParseResult(matched=False)            # no intent -> no-op

    # Which decision: explicit id wins; else exactly-one-open; else ambiguous.
    m = _DEC_ID_RE.search(raw)
    if m:
        dec_id = m.group(0)
    elif len(open_ids) == 1:
        dec_id = open_ids[0]
    else:
        return ParseResult(
            matched=False,
            ask=f"Which decision? Reply with the D-… id (e.g. {open_ids[0]}).",
        )

    if hold:
        return ParseResult(matched=True, action="resolve", value="hold", decision_id=dec_id)

    # approve intent — check whether the target decision has a PR number
    target = next((d for d in decisions if d.id == dec_id), None)
    if target is not None and target.pr_number is not None:
        return ParseResult(matched=True, action="merge",
                           value=str(target.pr_number), decision_id=dec_id)
    return ParseResult(matched=True, action="resolve", value="approve", decision_id=dec_id)
