"""Score a router against the offline eval set.

    python -m plugins.intelligent_routing.eval.run_eval

Prints a confusion matrix per scope for the task router and, as a baseline, the
length-gated classifier it replaces. ``premium->cheap`` is the quality-loss count:
a task that needed the primary model and was sent to the cheap one.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable

from plugins.intelligent_routing.eval.cases import ALL_CASES, SUBAGENT_CASES, TURN_CASES
from plugins.intelligent_routing.eval.cases_audit import AUDIT_CASES
from plugins.intelligent_routing.eval.cases_blind import BLIND_CASES
from plugins.intelligent_routing.eval.cases_holdout import HOLDOUT_CASES
from plugins.intelligent_routing.task_router import (
    SCOPE_SUBAGENT, SCOPE_TURN, TIER_CHEAP, TIER_PREMIUM, TaskSignals, classify_task,
)

Router = Callable[[str, str, dict], str]  # (scope, text, extra) -> tier


def signals_for(scope: str, text: str, extra: dict) -> TaskSignals:
    platform = extra.get("platform", "")
    return TaskSignals(
        instruction=text, context=extra.get("context", ""),
        scope=SCOPE_SUBAGENT if scope == "subagent" else SCOPE_TURN,
        platform=platform, is_background=platform == "cron",
        is_public_facing=bool(extra.get("public")), has_attachments=bool(extra.get("attachments")),
    )


def task_router(scope: str, text: str, extra: dict) -> str:
    return classify_task(signals_for(scope, text, extra)).tier


def legacy_router(scope: str, text: str, extra: dict) -> str:
    """What shipped before: length-gated binary classifier on turns; children always primary."""
    if scope == "subagent":
        return TIER_PREMIUM
    from plugins.intelligent_routing.registration import probe_route
    kwargs = {"platform": extra.get("platform", "cli")}
    if extra.get("public"):
        kwargs["is_public_facing"] = True
    return probe_route(text, **kwargs)["tier"]


def score(router: Router, cases: Iterable[tuple[str, str, str, dict]]) -> dict:
    matrix: Counter = Counter()
    leaks: list[str] = []
    for scope, label, text, extra in cases:
        got = router(scope, text, extra)
        matrix[(label, got)] += 1
        if label == TIER_PREMIUM and got == TIER_CHEAP:
            leaks.append(text[:80])
    total = sum(matrix.values())
    cheap_ok = matrix[(TIER_CHEAP, TIER_CHEAP)] + matrix[(TIER_CHEAP, TIER_PREMIUM)]
    sent_cheap = matrix[(TIER_CHEAP, TIER_CHEAP)] + matrix[(TIER_PREMIUM, TIER_CHEAP)]
    return {
        "total": total,
        "premium->premium": matrix[(TIER_PREMIUM, TIER_PREMIUM)],
        "premium->cheap": matrix[(TIER_PREMIUM, TIER_CHEAP)],
        "cheap->cheap": matrix[(TIER_CHEAP, TIER_CHEAP)],
        "cheap->premium": matrix[(TIER_CHEAP, TIER_PREMIUM)],
        "cheap_share": sent_cheap / total if total else 0.0,
        "cheap_recall": matrix[(TIER_CHEAP, TIER_CHEAP)] / cheap_ok if cheap_ok else 0.0,
        "leaks": leaks,
    }


def _print(name: str, result: dict) -> None:
    print(f"\n{name}  (n={result['total']})")
    print("                 routed premium   routed cheap")
    print(f"  needs premium  {result['premium->premium']:>14}   {result['premium->cheap']:>12}   <- quality loss")
    print(f"  cheap is fine  {result['cheap->premium']:>14}   {result['cheap->cheap']:>12}")
    print(f"  sent cheap: {result['cheap_share']:.0%} of cases; cheap-safe cases caught: {result['cheap_recall']:.0%}")
    for leak in result["leaks"]:
        print(f"  LEAK: {leak!r}")


def main() -> None:
    for scope_name, cases in (("dev main turn", TURN_CASES), ("dev subagent", SUBAGENT_CASES),
                              ("dev all", ALL_CASES), ("holdout", HOLDOUT_CASES), ("blind", BLIND_CASES),
                              ("audit", AUDIT_CASES)):
        _print(f"task router - {scope_name}", score(task_router, cases))
        _print(f"legacy router - {scope_name}", score(legacy_router, cases))


if __name__ == "__main__":
    main()
