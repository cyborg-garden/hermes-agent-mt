"""Keep ``tool_call_id`` unique across a whole conversation.

Every OpenAI-compatible pairing rule in Hermes — the pre-call sanitizer's
orphan/duplicate passes, the compressor's ``_sanitize_tool_pairs`` and
``_prune_old_tool_results``, the Codex Responses adapter — keys a tool result
to the assistant call that produced it by ``tool_call_id``.  All of them treat
that id as unique *within the conversation*, because that is what the OpenAI
schema implies.

Some providers do not honour that.  They mint ids scoped to the assistant
message that issued them and restart the numbering on every round.  Moonshot
``kimi-k3`` served through OpenRouter returns::

    round 1:  read_file:0   read_file:1   terminal:0
    round 2:  read_file:0   read_file:1
    round 3:  read_file:0

Locally each result still follows its own call, so the transcript looks fine.
Conversation-wide it is a catastrophe: every id-keyed pass sees round 2 and 3
as duplicates of round 1.  In production session ``20260809_040444_3aca1c0f``
that collapsed 191 tool results onto 10 distinct ids.  The pre-call sanitizer
deleted the later 181, so the model received the FIRST ``read_file`` output
and nothing afterwards — its prompt grew a constant ~23 tokens per API call
whether the tool returned 286 characters or 10,533.  The model did the correct
thing (re-read the file it could not see), the anti-repeat guard blocked it,
and the turn ended at ``api_calls=60/60`` with ``response_len=0``.

This module fixes the collision **at the source**: it rewrites the stored
conversation so ids are conversation-unique before anything keys on them.
That is deliberately upstream of every consumer, so the compressor and the
persistence layer see clean ids too, and a resumed session whose transcript
already carries colliding ids is repaired on its next request.

Pairing is positional, which is the adjacency the rest of the pipeline
already enforces: the tool results of a round immediately follow the
assistant message that opened it, and no non-tool message intervenes.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

# Several OpenAI-compatible providers validate the length of tool_call_id.
# Keep rewritten ids comfortably inside the smallest limit we have seen (64).
MAX_TOOL_CALL_ID_LEN = 64

# Suffix marker for a rewritten id.  ``[A-Za-z0-9_]`` only, so a provider that
# accepted the original id accepts the rewrite.
_SUFFIX_MARKER = "_r"


def _tool_call_id(tc: Any) -> str:
    """Read the id off a tool_call in dict or SDK-object form."""
    if isinstance(tc, dict):
        raw = tc.get("call_id") or tc.get("id") or ""
    else:
        raw = getattr(tc, "call_id", None) or getattr(tc, "id", None) or ""
    return raw.strip() if isinstance(raw, str) else ""


def _rewrite_tool_call_id(tc: Any, old_id: str, new_id: str) -> Any:
    """Return a copy of ``tc`` whose id fields read ``new_id``.

    Copies rather than mutates: an SDK tool_call object can be shared with the
    live response object and the trajectory writer.  Only fields that actually
    hold ``old_id`` are touched, so the Codex Responses split of
    ``id``/``call_id`` (where the two differ on purpose) survives intact.
    """
    if isinstance(tc, dict):
        clone = dict(tc)
        for key in ("id", "call_id"):
            value = clone.get(key)
            if isinstance(value, str) and value.strip() == old_id:
                clone[key] = new_id
        return clone
    try:
        clone = copy.copy(tc)
    except Exception:  # pragma: no cover — exotic provider object
        return tc
    for key in ("id", "call_id"):
        value = getattr(clone, key, None)
        if isinstance(value, str) and value.strip() == old_id:
            try:
                setattr(clone, key, new_id)
            except Exception:  # pragma: no cover — frozen model
                return tc
    return clone


def _mint_unique_id(base_id: str, used: set, counters: dict[str, int]) -> str:
    """Derive an unused id from ``base_id``.

    Deterministic by construction: the same conversation prefix always yields
    the same ids, so the provider's prompt cache still hits after a repair.
    ``counters`` carries the next ordinal per stem, keeping a session with
    thousands of collisions linear instead of rescanning from 2 each time.
    """
    stem = base_id or "tool_call"
    budget = MAX_TOOL_CALL_ID_LEN - len(_SUFFIX_MARKER) - 6
    if len(stem) > budget:
        stem = stem[:budget]
    n = counters.get(stem, 2)
    while True:
        candidate = f"{stem}{_SUFFIX_MARKER}{n}"
        n += 1
        if candidate not in used:
            counters[stem] = n
            return candidate


def audit_tool_call_ids(messages: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Return ``(tool_results, distinct_ids, results_that_would_be_dropped)``.

    A pure read used for logging and for tests.  ``results_that_would_be_dropped``
    counts tool results whose id was already claimed by an earlier tool result
    — exactly the set the pre-call de-duplication pass deletes.
    """
    seen: set = set()
    total = 0
    dropped = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        total += 1
        cid = (msg.get("tool_call_id") or "").strip()
        if not cid:
            continue
        if cid in seen:
            dropped += 1
        else:
            seen.add(cid)
    return total, len(seen), dropped


def repair_conversation_tool_call_ids(
    messages: list[dict[str, Any]],
    *,
    logger: Optional[Any] = None,
    model: str = "",
    provider: str = "",
    session_id: str = "",
) -> int:
    """Make every ``tool_call_id`` in ``messages`` conversation-unique.

    Walks the conversation in order.  The first use of an id is kept verbatim
    (so an already-correct transcript is byte-identical afterwards and the
    prompt cache is untouched).  A later assistant round that reuses an id
    gets a freshly minted one, and the tool results that follow that round are
    repointed to it.

    Mutates ``messages`` in place — the list is the live conversation, so the
    repair also lands in the session DB, in the compressor's view, and in the
    trajectory.  That is the point: the collision never reaches a consumer.

    Duplicate ids *within a single assistant message* are left alone.  Their
    results are genuinely indistinguishable, so inventing a pairing would be a
    guess; the existing de-duplication pass still handles them.

    Returns the number of tool_calls renamed.
    """
    # Snapshot the damage before repairing it — after the walk every id is
    # unique by construction, so the "what would have been deleted" numbers
    # are only observable from here.
    pre_total, pre_distinct, pre_dropped = audit_tool_call_ids(messages)

    used: set = set()
    counters: dict[str, int] = {}
    renamed = 0
    # Original id -> queue of ids minted for it in the round being walked.
    # Cleared by every assistant/user/system boundary, which is what makes the
    # pairing positional rather than global.
    round_map: dict[str, list[str]] = {}

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "tool":
            cid = (msg.get("tool_call_id") or "").strip()
            queue = round_map.get(cid)
            if queue:
                msg["tool_call_id"] = queue.pop(0)
            continue

        round_map = {}
        if role != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if not isinstance(tcs, list) or not tcs:
            continue

        new_tcs: list[Any] = []
        within: set = set()
        changed = False
        for tc in tcs:
            cid = _tool_call_id(tc)
            if not cid or cid in within:
                # Nothing to key on, or a within-round duplicate.
                new_tcs.append(tc)
                continue
            within.add(cid)
            if cid in used:
                new_id = _mint_unique_id(cid, used, counters)
                new_tcs.append(_rewrite_tool_call_id(tc, cid, new_id))
                round_map.setdefault(cid, []).append(new_id)
                used.add(new_id)
                renamed += 1
                changed = True
            else:
                new_tcs.append(tc)
                used.add(cid)
        if changed:
            msg["tool_calls"] = new_tcs

    if renamed and logger is not None:
        logger.error(
            "Provider reuses tool_call ids across rounds (model=%s provider=%s "
            "session=%s): renamed %d tool_call id(s). Before the repair this "
            "conversation held %d tool result(s) on only %d distinct id(s), and "
            "%d result(s) would have been deleted before the request — the model "
            "would have re-requested output it could not see.",
            model or "?",
            provider or "?",
            session_id or "-",
            renamed,
            pre_total,
            pre_distinct,
            pre_dropped,
        )
    return renamed


def count_visible_tool_results(messages: list[dict[str, Any]]) -> tuple[int, int]:
    """Return ``(tool_result_count, total_content_chars)`` for ``messages``.

    Used as a request-time invariant: whatever the assembly pipeline does to
    the payload, the number of tool results and the volume of tool output the
    model can see must not silently collapse.
    """
    count = 0
    chars = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        count += 1
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif content is not None:
            try:
                chars += len(str(content))
            except Exception:  # pragma: no cover — defensive
                pass
    return count, chars


__all__ = [
    "MAX_TOOL_CALL_ID_LEN",
    "audit_tool_call_ids",
    "count_visible_tool_results",
    "repair_conversation_tool_call_ids",
]
