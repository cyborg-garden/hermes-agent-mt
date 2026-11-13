"""Fork-only: findings an upstream-sync PR inherits from the upstream release it merges.

The ratchet judges a PR against this repository's main. A sync PR merges an upstream release, so
every function upstream had over its own cap looks new here, although nobody in this fork wrote
it and upstream's ratchet already carries it. ``HERMES_HEALTH_INHERITED_FROM`` names that upstream
revision (CI fetches it from the upstream repository itself, see lint.yml). A finding is dropped
only when judging the same head against that revision does not report it: the file is unchanged
since upstream, or the hit sits in code upstream already had. Anything this fork added or edited
is still reported, so the variable cannot forgive the fork's own code.
"""

from __future__ import annotations

from pathlib import Path

from scripts.code_health import gitio
from scripts.code_health.compare import compare
from scripts.code_health.config import in_scope
from scripts.code_health.model import Finding

ENV = "HERMES_HEALTH_INHERITED_FROM"


def _key(finding: Finding) -> tuple[str, str, str, int]:
    return finding.path, finding.rule, finding.scope, finding.line


def drop_inherited(repo: Path, measurer, upstream: str, head: str | None, head_m: dict,
                   findings: list[Finding]) -> tuple[list[Finding], int]:
    """``(findings still owed, number dropped as inherited)`` for ``head`` judged against ``upstream``."""
    paths = {f.path for f in findings}
    if not paths:
        return findings, 0
    changes = [c for c in gitio.changed_files(repo, upstream, head) if c.new in paths]
    upstream_paths = sorted({c.old for c in changes if c.old and in_scope(c.old)})
    changed = {c.new for c in changes}
    still_new = {_key(f) for f in compare(measurer.measure(upstream, upstream_paths),
                                          {p: m for p, m in head_m.items() if p in changed}, changes)}
    kept = [f for f in findings if _key(f) in still_new]
    return kept, len(findings) - len(kept)
