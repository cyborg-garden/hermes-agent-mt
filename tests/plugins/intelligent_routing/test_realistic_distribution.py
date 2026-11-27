"""Realistic-distribution test — the gap the unit tests + audit missed.

The live deploy found that `classify_task_type`'s catch-all default
(`TASK_ORCHESTRATION → cheap`) silently cheap-routed hard architecture questions,
multi-file code-gen requests, and hedged/ambiguous asks — violating the
"fail open to premium" guarantee. The prior tests only checked the tier MAPPING
(`tier_for_task_type(ARCHITECTURE) == premium`) in isolation, never that real
messages get ASSIGNED premium.

This test feeds a batch of REPRESENTATIVE messages through the FULL
`on_pre_llm_call` path and asserts what actually gets routed (the emitted `route`
override, or its absence) — not an explicit-type mapping.

Invariant: only a CONFIDENTLY mechanical turn emits a cheap `route`. Everything
judgment / architecture / code-gen / uncertain / hedged / public-facing emits NO
route (stays on the configured primary).
"""
from unittest.mock import patch

import pytest

from plugins.intelligent_routing import registration


def _route_of(message, **kwargs):
    """Run the full hook (routing enabled) and return the emitted route dict or None."""
    with patch(
        "plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
        return_value=True,
    ):
        result = registration.on_pre_llm_call(
            user_message=message, model="claude-sonnet-5", provider="anthropic",
            platform=kwargs.pop("platform", "cli"), **kwargs,
        )
    if not isinstance(result, dict):
        return None
    return result.get("route")


# Messages that MUST stay on premium (no cheap route). This is the batch the live
# run exposed — hard architecture, real code-gen, and hedged/ambiguous asks that
# previously fell through the orchestration catch-all onto cheap.
PREMIUM_MESSAGES = [
    # hard architecture
    "How should we structure the multi-tenant routing layer so tenants stay isolated?",
    "Design a system for multi-tenant routing across the fleet with failure isolation.",
    # multi-file / real code-gen
    "Write a Python function that merges two sorted linked lists.",
    "Implement the retry logic in agent/chat_completion_helpers.py and open a PR.",
    # ambiguous / hedged
    "Can you look at the thing we discussed and get back to me?",
    "Not sure how to approach this — what do you think we should do about the parser?",
    # open-ended reasoning
    "Think through the tradeoffs between OpenRouter and z.ai and recommend an approach.",
    # a plain question with no mechanical shape
    "Why is the deploy flaky lately?",
    # SHORT judgment questions fronted by a trivial-query prefix — these are the
    # round-3 leak: "what's/what is/how many/when is" prefixes routinely front
    # real judgment questions and must NOT be cheap-routed just because they're
    # short. Dropping _TRIVIAL_QUERY_PREFIXES fixes this (fail open to premium).
    "what's the best architecture for this?",
    "what's the right way to handle auth here?",
    "whats your opinion on the design?",
    "how many ways could this deadlock?",
    "when is it worth adding a cache layer?",
]

# Messages that SHOULD route cheap — confidently mechanical only.
CHEAP_MESSAGES = [
    "grep for TODO in the repo",
    "list the files in plugins/",
    "mark card 42 done",
    "run the tests",
]


@pytest.mark.parametrize("message", PREMIUM_MESSAGES)
def test_premium_messages_never_cheap_routed(message):
    assert _route_of(message) is None, f"LEAK: {message!r} was cheap-routed"


@pytest.mark.parametrize("message", CHEAP_MESSAGES)
def test_confident_mechanical_messages_cheap_routed(message):
    route = _route_of(message)
    assert route is not None, f"expected cheap route for {message!r}"
    assert route["model"] == "deepseek/deepseek-v4-flash-0731"


def test_background_cron_job_cheap_routed():
    """A non-interactive/background job is confidently mechanical -> cheap."""
    assert _route_of("run the nightly digest", platform="cron") is not None


def test_public_facing_never_cheap_routed():
    """Public-facing dominates even a mechanical-shaped message."""
    assert _route_of("grep the announcement", is_public_facing=True) is None


def test_empty_message_fails_open_to_premium():
    assert _route_of("") is None


def test_distribution_bias_is_conservative():
    """Sanity on the whole batch: the premium set emits zero cheap routes, and the
    cheap set emits only cheap routes — the safe asymmetry."""
    assert all(_route_of(m) is None for m in PREMIUM_MESSAGES)
    assert all(_route_of(m) is not None for m in CHEAP_MESSAGES)


# ------------------------------------------------- isolation probe (re-audit) --

def test_probe_route_matches_hook_decision():
    """probe_route() is a config-free, plumbing-free view of the SAME decision the
    hook makes — for the live-probe / re-audit."""
    from plugins.intelligent_routing.registration import probe_route

    for m in PREMIUM_MESSAGES:
        assert probe_route(m)["tier"] == "premium", m
    for m in CHEAP_MESSAGES:
        assert probe_route(m)["tier"] == "cheap", m
    # signal passthrough
    assert probe_route("run the nightly digest", platform="cron")["tier"] == "cheap"
    assert probe_route("grep the announcement", is_public_facing=True)["tier"] == "premium"


def test_decide_tier_is_the_single_source_of_truth():
    """decide_tier == route_for (both gate cheap on a confident MECHANICAL)."""
    from plugins.intelligent_routing.classifier import (
        decide_tier, route_for, TurnSignals,
    )
    for m in [m for m in PREMIUM_MESSAGES] + [m for m in CHEAP_MESSAGES] + [""]:
        sig = TurnSignals(user_message=m)
        assert decide_tier(sig) == route_for(sig), m
