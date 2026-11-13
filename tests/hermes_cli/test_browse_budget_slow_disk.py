"""Fork: the browse schema catch-up refuses the same way however slow the disk is.

``test_browse_schema_drift`` slows every SQLite step by 10 ms. On a slower host (GitHub's
macOS runner) the time budget ran out before the DDL began, in the durability pragmas or in
``BEGIN IMMEDIATE``. That raised a raw "interrupted", which the dashboard reported as a corrupt
store.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.hermes_cli.test_browse_schema_drift import (
    _SLOW_BROWSE,
    _file_state,
    _lagging_store_with_history,
    _profile_home,
    _session_columns,
)


@pytest.mark.platforms("linux", "macos")
def test_browse_on_a_slower_disk_still_gives_up_inside_its_budget(tmp_path):
    # 30 ms a step is the slowdown that reproduced the macOS runner's failure. Much slower disks
    # (200 ms a step) still outlast the gateway's boot wait on Linux; that is not fixed here.
    step_seconds = 0.03
    import gateway.run_bootstrap as bootstrap
    import hermes_state_reconcile as reconcile
    from gateway.runtime_ownership import ProfileOwnership

    assert "time.sleep(0.01)" in _SLOW_BROWSE
    slower = _SLOW_BROWSE.replace("time.sleep(0.01)", f"time.sleep({step_seconds})")
    home = _profile_home(tmp_path)
    path = _lagging_store_with_history(home, rows=400)
    before = _file_state(path)
    browse = subprocess.Popen([sys.executable, "-c", slower, str(path), "ddl"], env=os.environ,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert browse.stdout.readline().strip() == "started"
    assert browse.stdout.readline().strip() == "slowed", "the slow phase never ran under the lock"
    booting = ProfileOwnership()
    try:
        bootstrap._reserve_waiting_out_browse_reconcile(booting, [home])  # default wait: must not raise
        assert booting.owns(home)
    finally:
        booting.close()
    out = browse.communicate(timeout=60)[0]
    result = next(line for line in out.splitlines() if line.startswith("RESULT"))
    _, status, elapsed, *detail = result.split(" ", 3)
    assert status == "503", out
    assert float(elapsed) < reconcile.BROWSE_RECONCILE_BUDGET_S + 2, out
    detail = " ".join(detail)
    assert "nothing was changed" in detail, detail
    assert "`hermes -p scribe sessions repair`" in detail and "gateway" not in detail.lower()
    assert _file_state(path) == before
    assert "last_read_at" not in _session_columns(path)
