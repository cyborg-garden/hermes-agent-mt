#!/usr/bin/env python3
"""Pick the upstream release tag the weekly Upstream Sync should merge.

Reads newline-separated tag names on stdin and prints the chosen tag. When no
release tag qualifies it prints nothing and still exits 0, so the workflow can
tell "no tag" (empty output) from "the picker crashed" (non-zero exit), and a
crash fails the run instead of silently syncing to the moving upstream/main.

Upstream changed tag schemes in 2026-09:

* **Legacy date tags** — ``v2026.9.24``, ``v2026.8.16.2``. Last one shipped
  2026-09-24.
* **Semver tags** — ``v0.22.4``. Current scheme, plus nightly
  ``v0.22.2+canary.20261021T070232Z`` builds.

``git tag --sort=-v:refname`` ranks ``v2026.9.24`` above ``v0.22.4`` because
2026 > 0, so the workflow kept targeting a month-old tag. Rules here:

1. A plain ``vMAJOR.MINOR.PATCH`` semver tag always beats a legacy date tag.
2. Canary/pre-release/build tags (anything with ``+`` or ``-``) never qualify.
3. Date tags are a fallback only when no semver tag exists at all.
"""

from __future__ import annotations

import re
import sys

_SEMVER = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
_DATE = re.compile(r"^v(20\d\d)\.(\d{1,2})\.(\d{1,2})(?:\.(\d+))?$")
# A semver major this large is a year, i.e. a legacy date tag.
_DATE_MAJOR_FLOOR = 2000


def _key(tag: str) -> tuple[int, tuple[int, ...]] | None:
    """Rank key: (scheme, version). Scheme 1 = semver, 0 = legacy date."""
    m = _SEMVER.match(tag)
    if m and int(m.group(1)) < _DATE_MAJOR_FLOOR:
        return (1, tuple(int(g) for g in m.groups()))
    m = _DATE.match(tag)
    if m:
        return (0, tuple(int(g or 0) for g in m.groups()))
    return None


def pick(tags: list[str]) -> str | None:
    ranked = [(k, t) for t in tags if (k := _key(t.strip())) is not None]
    if not ranked:
        return None
    return max(ranked)[1].strip()


def main() -> int:
    chosen = pick(sys.stdin.read().splitlines())
    if chosen is not None:
        print(chosen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
