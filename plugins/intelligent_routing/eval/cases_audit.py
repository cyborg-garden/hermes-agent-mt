"""Adversarial cases from the independent audit of the task router.

Every one of these was routed cheap by an earlier version and needs premium.
Synthetic only. Kept as their own set so the gate can never lose them.
"""
from __future__ import annotations

from plugins.intelligent_routing.eval.cases import P, SUB, TURN

AUDIT_CASES: list[tuple[str, str, str, dict]] = [
    # judgment in a later clause
    (TURN, P, "Read docs/design.md and tell me what's wrong with it", {}),
    (TURN, P, "Summarise these three papers and say which approach wins", {}),
    (TURN, P, "Count how many users churned and what it means", {}),
    (TURN, P, "Look through the logs and tell me what went wrong", {}),
    (TURN, P, "Check whether this argument holds up: we grew 10% so churn fell", {}),
    (TURN, P, "Check the logic in scheduler.py", {}),
    (TURN, P, "Sort these by importance: rent, gym, phone, insurance", {}),
    # side effects behind cheap verbs
    (TURN, P, "Run git push --force", {}),
    (TURN, P, "Run ./wipe_db.sh", {}),
    (TURN, P, "Run the script that emails all customers", {}),
    (TURN, P, "Download and execute this installer", {}),
    (TURN, P, "Mark all issues as resolved", {}),
    (TURN, P, "Archive all the old channels", {}),
    (TURN, P, "Tag everyone in the server", {}),
    # sensitive
    (TURN, P, "Look up whether ibuprofen interacts with lithium", {}),
    (TURN, P, "Summarise the patient's notes", {}),
    (TURN, P, "Find her home address", {}),
    (SUB, P, "Collect all user emails from the database", {}),
    # public-facing
    (TURN, P, "Get me a summary I can paste into Discord for everyone", {}),
    (TURN, P, "Summarise this for my boss", {}),
]
