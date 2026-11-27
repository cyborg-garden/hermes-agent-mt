"""Task router quality gates on the offline eval sets, plus targeted behaviour.

The gate that matters: zero premium-worthy tasks routed cheap, on every set,
including the blind set written without sight of the router's rules. The
sentinels prove the harness can fail: a router that sends everything cheap, and
the length-gated classifier this replaces, both leak on the same cases.
"""
import pytest

from plugins.intelligent_routing.eval.cases import ALL_CASES, SUBAGENT_CASES, TURN_CASES
from plugins.intelligent_routing.eval.cases_audit import AUDIT_CASES
from plugins.intelligent_routing.eval.cases_blind import BLIND_CASES
from plugins.intelligent_routing.eval.cases_holdout import HOLDOUT_CASES
from plugins.intelligent_routing.eval.run_eval import legacy_router, score, task_router
from plugins.intelligent_routing.task_router import (
    SCOPE_SUBAGENT, TIER_CHEAP, TIER_PREMIUM, TaskSignals, classify_task, clean_instruction,
)

SETS = {"dev": ALL_CASES, "holdout": HOLDOUT_CASES, "blind": BLIND_CASES}
# Premium-only: every case here was routed cheap by an earlier version (independent audit).
GATED = {**SETS, "audit": AUDIT_CASES}


@pytest.mark.parametrize("name", sorted(GATED))
def test_no_premium_task_is_routed_cheap(name):
    result = score(task_router, GATED[name])
    assert result["premium->cheap"] == 0, result["leaks"]


@pytest.mark.parametrize("name", sorted(SETS))
def test_router_still_routes_a_real_share_cheap(name):
    """All-premium would pass the quality gate trivially; demand real cheap recall."""
    assert score(task_router, SETS[name])["cheap_recall"] >= 0.55


def test_sets_are_balanced_enough_to_mean_something():
    for name, cases in SETS.items():
        labels = [label for _, label, _, _ in cases]
        assert labels.count(TIER_PREMIUM) >= 30 and labels.count(TIER_CHEAP) >= 25, name


def test_sentinel_always_cheap_router_fails_the_gate():
    result = score(lambda *_: TIER_CHEAP, ALL_CASES)
    assert result["premium->cheap"] == sum(1 for _, l, _, _ in ALL_CASES if l == TIER_PREMIUM)


def test_sentinel_legacy_classifier_fails_the_gate():
    """The length-gated classifier leaks on this set; the task router must not."""
    assert score(legacy_router, TURN_CASES)["premium->cheap"] > 0
    assert score(task_router, TURN_CASES)["premium->cheap"] == 0


def test_legacy_never_routed_a_subagent_cheap():
    assert score(legacy_router, SUBAGENT_CASES)["cheap->cheap"] == 0


# ── targeted behaviour ──

def _tier(text, **kw):
    return classify_task(TaskSignals(instruction=text, **kw)).tier


def test_length_is_not_a_signal():
    long_check = ("Check each of these hosts and list the ones that are down: "
                  + ", ".join(f"host{i}.example.org" for i in range(60)))
    assert len(long_check) > 120
    assert _tier(long_check) == TIER_CHEAP
    assert _tier("should we ship?") == TIER_PREMIUM


def test_protected_and_attachments_are_premium():
    assert _tier("list the files", is_protected=True) == TIER_PREMIUM
    assert _tier("list the files", has_attachments=True) == TIER_PREMIUM
    assert _tier("list the files", is_public_facing=True) == TIER_PREMIUM


def test_cron_scaffolding_is_not_classified():
    hint = ("[IMPORTANT: You are running as a scheduled cron job. DELIVERY: do NOT use send_message. "
            "FAILURE: If a delegated child fails ... NEVER create or update a cron job.]\n\n"
            "[CURRENT RUN TIME: Monday 09:00]\n\n")
    assert clean_instruction(hint + "list new issues") == "list new issues"
    assert _tier(hint + "list new issues", platform="cron", is_background=True) == TIER_CHEAP


def test_cron_data_block_and_notepad_are_not_classified():
    prompt = ("## Job notepad (persistent across runs)\nThis durable scratchpad survives. Update it via "
              "the CLI, e.g.:\n`hermes cron notepad j set <key> <value>` (also: get/delete/list).\n\n"
              "- last: 3\n\n## Script Output\nThe following data was collected.\n\n```\nfix deploy delete\n```"
              "\n\nCount the rows and report the number.")
    assert clean_instruction(prompt) == "Count the rows and report the number."
    assert _tier(prompt) == TIER_CHEAP


def test_a_hard_later_step_makes_the_task_premium():
    assert _tier("check the code in utils.py") == TIER_PREMIUM  # code is a signal
    assert _tier("list the open cards and rank them") == TIER_PREMIUM
    assert _tier("list the open cards and their owners") == TIER_CHEAP
    # Caught only by the later-step rule: "handle" is no keyword signal on its own.
    assert _tier("fetch the new tickets and handle them") == TIER_PREMIUM


def test_subagent_context_can_veto_a_cheap_goal():
    assert _tier("Summarise the context.", context="Notes on the weather.", scope=SCOPE_SUBAGENT) == TIER_CHEAP
    assert _tier("Summarise the context.", context="Then publish it on the blog.",
                 scope=SCOPE_SUBAGENT) == TIER_PREMIUM


def test_follow_ups_that_lean_on_history_stay_premium():
    for text in ("yes", "do it", "same for the other two", "thanks!", "the second one"):
        assert _tier(text) == TIER_PREMIUM, text
