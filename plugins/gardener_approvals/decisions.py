"""Read OPEN decisions from the gardener decisions.md. Mirrors the gate's
`open_ids()` grep so the hook and the gate agree on what 'open' means. Real
dated ids only — the `D-YYYY-MM-DD-NN` format example is never matched.

Each open block is returned as a Decision dataclass carrying the id and,
when the block contains a Link: line that references a PR, the PR number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_OPEN_RE = re.compile(r"^## (D-\d{4}-\d{2}-\d{2}-\d+) .*status: open", re.MULTILINE)
_RESOLVED_RE = re.compile(r"^## (D-\d{4}-\d{2}-\d{2}-\d+) .*status: resolved", re.MULTILINE)
_RESOLVED_DATE_RE = re.compile(r"^\s*-\s+\*\*Resolved:\*\*\s+(\d{4}-\d{2}-\d{2})", re.MULTILINE)

# Matches any of:
#   PR #27   |  #27   |  pull/27   |  /pull/27
_PR_NUM_RE = re.compile(r"(?:PR\s+#|/pull/)(\d+)|#(\d+)|pull/(\d+)", re.IGNORECASE)


@dataclass
class Decision:
    id: str
    pr_number: Optional[int]


@dataclass
class ResolvedDecision:
    id: str
    date: str   # ISO date string "YYYY-MM-DD", or "" if absent


def read_open_decisions(decisions_path: str) -> list[Decision]:
    try:
        with open(decisions_path, "r", encoding="utf-8-sig") as fh:
            text = fh.read()
    except OSError:
        return []

    decisions: list[Decision] = []
    for m in _OPEN_RE.finditer(text):
        dec_id = m.group(1)
        block_start = m.start()
        # Find end of block: next ## header or EOF
        next_header = re.search(r"^## ", text[block_start + 1:], re.MULTILINE)
        block_end = (block_start + 1 + next_header.start()) if next_header else len(text)
        block_text = text[block_start:block_end]

        pr_number: Optional[int] = None
        # Only scan Link: lines for PR numbers
        for line in block_text.splitlines():
            if "link:" in line.lower():
                pm = _PR_NUM_RE.search(line)
                if pm:
                    # groups: (PR #N, #N, pull/N) — first non-None group wins
                    raw = pm.group(1) or pm.group(2) or pm.group(3)
                    pr_number = int(raw)
                    break

        decisions.append(Decision(id=dec_id, pr_number=pr_number))

    return decisions


def read_open_ids(decisions_path: str) -> list[str]:
    return [d.id for d in read_open_decisions(decisions_path)]


def most_recent_resolved(path: str) -> Optional[ResolvedDecision]:
    """Return the most recently resolved decision, or None if there are none.

    "Most recent" is determined by the lexically-greatest ISO date found on a
    ``- **Resolved:** YYYY-MM-DD`` line inside the block.  When dates tie or are
    absent the decision appearing LAST in the file wins (decisions are appended
    chronologically, so last = newest).
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            text = fh.read()
    except OSError:
        return None

    matches = list(_RESOLVED_RE.finditer(text))
    if not matches:
        return None

    best: Optional[ResolvedDecision] = None

    for m in matches:
        dec_id = m.group(1)
        block_start = m.start()
        next_header = re.search(r"^## ", text[block_start + 1:], re.MULTILINE)
        block_end = (block_start + 1 + next_header.start()) if next_header else len(text)
        block_text = text[block_start:block_end]

        dm = _RESOLVED_DATE_RE.search(block_text)
        date = dm.group(1) if dm else ""

        candidate = ResolvedDecision(id=dec_id, date=date)
        if best is None:
            best = candidate
        elif candidate.date > best.date:
            # Strictly later date wins
            best = candidate
        elif candidate.date == best.date:
            # Tie: prefer the one appearing later in the file (i.e. the current candidate,
            # since we iterate forward through the file)
            best = candidate

    return best
