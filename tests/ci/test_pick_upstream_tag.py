"""Tests for scripts/ci/pick_upstream_tag.py.

The Upstream Sync workflow targeted v2026.9.24 for weeks after upstream moved
to semver (v0.22.x), because a version sort ranks 2026 above 0.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "pick_upstream_tag.py"
_spec = importlib.util.spec_from_file_location("pick_upstream_tag", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load pick_upstream_tag.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pick = _mod.pick

# Shape of the real upstream tag list on 2026-10-22.
REAL = [
    "v2026.7.7",
    "v2026.7.7.2",
    "v2026.7.30",
    "v2026.8.16.2",
    "v2026.9.24",
    "v0.21.4+canary.20260925T065930Z",
    "v0.21.7",
    "v0.22.0",
    "v0.22.2+canary.20261021T070232Z",
    "v0.22.3",
    "v0.22.4",
]


def test_semver_beats_newer_looking_date_tag():
    assert pick(REAL) == "v0.22.4"


def test_canary_tags_never_win():
    assert pick(["v0.22.4", "v0.22.5+canary.20261022T070000Z"]) == "v0.22.4"
    assert pick(["v0.23.0-rc1", "v0.22.4"]) == "v0.22.4"


def test_semver_ordering_is_numeric():
    assert pick(["v0.9.0", "v0.10.0", "v0.2.11"]) == "v0.10.0"


def test_date_tags_are_fallback_and_ordered_numerically():
    assert pick(["v2026.7.7", "v2026.7.30", "v2026.7.7.2"]) == "v2026.7.30"
    assert pick(["v2026.8.16", "v2026.8.16.2"]) == "v2026.8.16.2"


def test_nothing_qualifies():
    assert pick(["v0.22.2+canary.x", "nightly", ""]) is None


def test_cli_matches_function():
    out = subprocess.run(
        [sys.executable, str(_PATH)],
        input="\n".join(REAL),
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "v0.22.4"
    # No qualifying tag: empty output, exit 0. The workflow treats a non-zero
    # exit as a crash and fails the run rather than syncing to upstream/main.
    empty = subprocess.run([sys.executable, str(_PATH)], input="junk\n", capture_output=True, text=True)
    assert empty.returncode == 0 and empty.stdout == ""
    none = subprocess.run([sys.executable, str(_PATH)], input="", capture_output=True, text=True)
    assert none.returncode == 0 and none.stdout == ""
