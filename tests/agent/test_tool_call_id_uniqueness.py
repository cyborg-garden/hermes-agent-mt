"""Regression: round-scoped tool_call ids must not delete the conversation.

Reproduces production session ``20260809_040444_3aca1c0f`` (hermes-matilde,
moonshotai/kimi-k3 via OpenRouter, 2026-08-09).  The provider mints tool_call
ids scoped to the assistant message that issued them and restarts the
numbering every round, so the session's 191 tool results landed on 10 distinct
ids::

    read_file:0   x102      execute_code:0  x7      terminal:1      x1
    read_file:1   x 38      skill_view:0    x2      terminal:2      x1
    terminal:0    x 36      search_files:0  x2      read_file:2     x1
                            discord:0       x1

Every id-keyed pass in the request pipeline assumes conversation-unique ids, so
the pre-call sanitizer deleted the 181 later results — 386k of the session's
395,594 characters of tool output — before the request.  The model's prompt
grew ~23 tokens per API call whether the tool returned 286 characters or
10,533, it re-read files it could not see, the anti-repeat guard blocked it,
and the turn ended at api_calls=60/60 with response_len=0.
"""

import logging

import pytest

from agent.tool_call_ids import (
    MAX_TOOL_CALL_ID_LEN,
    audit_tool_call_ids,
    count_visible_tool_results,
    repair_conversation_tool_call_ids,
)

# --- The real incident, to the digit -------------------------------------

INCIDENT_ROUNDS = 150
INCIDENT_TOOL_RESULTS = 191
INCIDENT_DISTINCT_IDS = 10
INCIDENT_DELETED_RESULTS = INCIDENT_TOOL_RESULTS - INCIDENT_DISTINCT_IDS  # 181
INCIDENT_TOOL_CHARS = 395_594


def _incident_round_shapes():
    """The per-round tool_call shapes that reproduce the incident histogram."""
    shapes = []
    shapes.append([("discord", 0)])
    shapes.extend([[("read_file", 0), ("read_file", 1)] for _ in range(37)])
    shapes.append([("read_file", 0), ("read_file", 1), ("read_file", 2)])
    shapes.extend([[("read_file", 0)] for _ in range(64)])
    shapes.extend([[("terminal", 0)] for _ in range(35)])
    shapes.append([("terminal", 0), ("terminal", 1), ("terminal", 2)])
    shapes.extend([[("execute_code", 0)] for _ in range(7)])
    shapes.extend([[("skill_view", 0)] for _ in range(2)])
    shapes.extend([[("search_files", 0)] for _ in range(2)])
    return shapes


def build_incident_transcript(separator=":"):
    """A conversation shaped exactly like the wedged session.

    ``separator`` selects the id flavour kimi-k3 emits: ``read_file:0`` (the
    round-scoped form that wedged) or ``read_file_0``.
    """
    shapes = _incident_round_shapes()
    total_results = sum(len(s) for s in shapes)
    per_result = INCIDENT_TOOL_CHARS // total_results
    remainder = INCIDENT_TOOL_CHARS - per_result * total_results

    messages = [{"role": "user", "content": "keep going with the corollaries!"}]
    emitted = 0
    for round_idx, shape in enumerate(shapes):
        tool_calls = []
        for tool, index in shape:
            tool_calls.append({
                "id": f"{tool}{separator}{index}",
                "type": "function",
                "function": {"name": tool, "arguments": '{"path":"a.py"}'},
            })
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls,
        })
        for tool, index in shape:
            emitted += 1
            size = per_result + (remainder if emitted == total_results else 0)
            messages.append({
                "role": "tool",
                "name": tool,
                "tool_call_id": f"{tool}{separator}{index}",
                "content": f"r{round_idx}:{tool}:{index}:".ljust(size, "x")[:size],
            })
    return messages


def _sanitize(messages):
    from agent.agent_runtime_helpers import sanitize_api_messages
    return sanitize_api_messages([dict(m) for m in messages])


def _assistant_round_ids(messages):
    """[(assistant_index, [ids]), ...] in order."""
    out = []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            out.append((i, [tc["id"] for tc in m["tool_calls"]]))
    return out


# --- The transcript really is the incident -------------------------------


def test_fixture_reproduces_the_incident_numbers():
    messages = build_incident_transcript()
    total, distinct, would_drop = audit_tool_call_ids(messages)
    assert total == INCIDENT_TOOL_RESULTS
    assert distinct == INCIDENT_DISTINCT_IDS
    assert would_drop == INCIDENT_DELETED_RESULTS
    assert len(_assistant_round_ids(messages)) == INCIDENT_ROUNDS
    count, chars = count_visible_tool_results(messages)
    assert (count, chars) == (INCIDENT_TOOL_RESULTS, INCIDENT_TOOL_CHARS)


# --- The bug: unrepaired, the pipeline deletes the session ---------------


def test_unrepaired_transcript_keeps_results_but_still_collides():
    """Documents what the repair still prevents after upstream v0.22.4.

    Pre-merge, the unrepaired incident transcript lost 181 of 191 tool results
    in the pre-call sanitizer (one result per distinct id). Upstream v0.22.4
    independently rewrote ``_dedupe_tool_call_ids`` to track OUTSTANDING calls
    (llama.cpp constant-id fix), so the sanitizer no longer deletes them. The
    payload still carries only 10 distinct ids for 191 results, though: every
    other id-keyed pass (compressor tool-pair sanitizer, Codex Responses
    adapter, strict providers that 400 on duplicate ``tool_call_id``) still
    sees round 2+ as duplicates of round 1. That is what the repair fixes.
    """
    messages = build_incident_transcript()
    sanitized = _sanitize(messages)
    count, chars = count_visible_tool_results(sanitized)
    assert (count, chars) == (INCIDENT_TOOL_RESULTS, INCIDENT_TOOL_CHARS)
    _total, distinct, would_drop = audit_tool_call_ids(sanitized)
    assert distinct == INCIDENT_DISTINCT_IDS, (
        "without the repair the payload still reuses round-scoped ids"
    )
    assert would_drop == INCIDENT_DELETED_RESULTS


# --- The fix -------------------------------------------------------------


def test_repair_preserves_every_tool_result_through_assembly():
    messages = build_incident_transcript()
    renamed = repair_conversation_tool_call_ids(messages)
    assert renamed == INCIDENT_TOOL_RESULTS - INCIDENT_DISTINCT_IDS

    total, distinct, would_drop = audit_tool_call_ids(messages)
    assert (total, distinct, would_drop) == (INCIDENT_TOOL_RESULTS, INCIDENT_TOOL_RESULTS, 0)

    sanitized = _sanitize(messages)
    count, chars = count_visible_tool_results(sanitized)
    assert count == INCIDENT_TOOL_RESULTS
    assert chars == INCIDENT_TOOL_CHARS


def test_repair_keeps_each_result_paired_with_its_own_call():
    messages = build_incident_transcript()
    repair_conversation_tool_call_ids(messages)

    open_ids = None
    round_idx = -1
    seen_ids = set()
    pairs = 0
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            round_idx += 1
            open_ids = [tc["id"] for tc in msg["tool_calls"]]
            for cid in open_ids:
                assert cid not in seen_ids, f"duplicate id survived: {cid}"
                seen_ids.add(cid)
        elif msg.get("role") == "tool":
            assert open_ids is not None
            assert msg["tool_call_id"] in open_ids, (
                "a result was repointed to a call from another round"
            )
            # The synthetic content encodes the round that produced it, so a
            # mis-pairing across rounds is visible rather than plausible.
            assert msg["content"].startswith(f"r{round_idx}:")
            pairs += 1
    assert pairs == INCIDENT_TOOL_RESULTS


def test_repair_preserves_result_content_order():
    """Result N after the repair is still the same bytes as result N before."""
    before = [m["content"] for m in build_incident_transcript() if m["role"] == "tool"]
    messages = build_incident_transcript()
    repair_conversation_tool_call_ids(messages)
    after = [m["content"] for m in messages if m["role"] == "tool"]
    assert before == after


def test_repair_is_idempotent():
    messages = build_incident_transcript()
    first = repair_conversation_tool_call_ids(messages)
    ids_after_first = [ids for _, ids in _assistant_round_ids(messages)]
    second = repair_conversation_tool_call_ids(messages)
    assert first > 0
    assert second == 0
    assert [ids for _, ids in _assistant_round_ids(messages)] == ids_after_first


def test_clean_transcript_is_left_byte_identical():
    """A provider with conversation-unique ids must not be touched at all.

    Rewriting ids on a healthy session would break the upstream prompt cache
    for no reason — the same model on 2026-08-06 emitted ``session_search:0``,
    ``session_search:1``, ``terminal:2`` (a counter that keeps climbing) and
    ran fine.
    """
    messages = []
    for r in range(20):
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"read_file_{r}",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        })
        messages.append({
            "role": "tool",
            "name": "read_file",
            "tool_call_id": f"read_file_{r}",
            "content": "x" * 100,
        })
    snapshot = [dict(m) for m in messages]
    assert repair_conversation_tool_call_ids(messages) == 0
    assert messages == snapshot


def test_within_round_duplicates_are_left_to_the_dedupe_pass():
    """Two identical ids inside ONE assistant message are not repairable.

    Their results are indistinguishable, so inventing a pairing would be a
    guess.  The repair must leave them alone rather than mis-pair them.
    """
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "read_file:0", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "read_file:0", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "name": "read_file", "tool_call_id": "read_file:0", "content": "a"},
        {"role": "tool", "name": "read_file", "tool_call_id": "read_file:0", "content": "b"},
    ]
    assert repair_conversation_tool_call_ids(messages) == 0
    assert [tc["id"] for tc in messages[0]["tool_calls"]] == ["read_file:0", "read_file:0"]


def test_a_user_turn_closes_a_round():
    """A result must never be repointed across a user turn."""
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "terminal:0", "type": "function",
             "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "name": "terminal", "tool_call_id": "terminal:0", "content": "first"},
        {"role": "user", "content": "sigue"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "terminal:0", "type": "function",
             "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "name": "terminal", "tool_call_id": "terminal:0", "content": "second"},
    ]
    assert repair_conversation_tool_call_ids(messages) == 1
    assert messages[1]["tool_call_id"] == "terminal:0"
    assert messages[1]["content"] == "first"
    new_id = messages[3]["tool_calls"][0]["id"]
    assert new_id != "terminal:0"
    assert messages[4]["tool_call_id"] == new_id
    assert messages[4]["content"] == "second"


def test_minted_ids_stay_inside_the_provider_length_limit():
    """A rewritten id must fit the strictest provider limit we have seen.

    The provider's own first id is left verbatim however long it is — that
    one already round-tripped successfully, and shortening it would break
    pairing for no reason.
    """
    long_stem = "a" * 120
    messages = []
    for _ in range(3):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": long_stem, "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": long_stem, "content": "x"})
    assert repair_conversation_tool_call_ids(messages) == 2
    minted = [
        cid
        for _, ids in _assistant_round_ids(messages)
        for cid in ids
        if cid != long_stem
    ]
    assert len(minted) == 2
    for cid in minted:
        assert len(cid) <= MAX_TOOL_CALL_ID_LEN
    # The results follow their own calls.
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == [
        long_stem, minted[0], minted[1],
    ]


# --- The loud guard ------------------------------------------------------


def test_repair_logs_an_error_naming_the_real_numbers(caplog):
    messages = build_incident_transcript()
    logger = logging.getLogger("test.tool_call_ids")
    with caplog.at_level(logging.ERROR, logger="test.tool_call_ids"):
        repair_conversation_tool_call_ids(
            messages,
            logger=logger,
            model="moonshotai/kimi-k3",
            provider="openrouter",
            session_id="20260809_040444_3aca1c0f",
        )
    assert caplog.records, "a silent repair is how this went unnoticed for a whole turn"
    text = caplog.records[0].getMessage()
    assert "moonshotai/kimi-k3" in text
    assert "20260809_040444_3aca1c0f" in text
    assert str(INCIDENT_TOOL_RESULTS) in text
    assert str(INCIDENT_DISTINCT_IDS) in text
    assert str(INCIDENT_DELETED_RESULTS) in text


# --- The wiring ----------------------------------------------------------


def test_repair_runs_before_the_request_payload_is_assembled():
    """The repair is worthless if it lands after api_messages is built.

    Upstream v0.22.4 split run_conversation into per-iteration phases: the
    repair lives in ``turn_iteration_prep.prepare_iteration`` and the request
    copy is built in ``turn_request_assembly.assemble_api_request``. The loop
    must run the prep phase before the assembly phase.
    """
    import inspect

    from agent import conversation_loop, turn_iteration_prep

    prep_src = inspect.getsource(turn_iteration_prep.prepare_iteration)
    assert "repair_conversation_tool_call_ids(" in prep_src, (
        "prepare_iteration no longer repairs tool_call ids"
    )
    loop_src = inspect.getsource(conversation_loop)
    prep_at = loop_src.find("_run_phase(prepare_iteration,")
    assemble_at = loop_src.find("_run_phase(assemble_api_request,")
    assert prep_at != -1 and assemble_at != -1
    assert prep_at < assemble_at, (
        "tool_call ids must be made unique before the request copy is built"
    )


def test_dropped_tool_results_are_reported_loudly():
    """The request-assembly invariant fires when results go missing."""
    import inspect

    from agent import turn_request_assembly

    src = inspect.getsource(turn_request_assembly.assemble_api_request)
    assert "count_visible_tool_results(messages)" in src
    assert "count_visible_tool_results(api_messages)" in src
    assert "Request assembly dropped" in src


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
