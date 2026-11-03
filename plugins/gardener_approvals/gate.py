"""Thin subprocess wrapper around decision-approve.sh. The gate enforces ALL
security (auth/allowlist/idempotency); this only marshals args and reads back
the `<TOKEN> <message>` protocol line. `runner` is injectable for tests.
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional, Tuple


def call_gate(
    *,
    gate_path: str,
    uuid: str,
    decision_id: Optional[str],
    action: str,
    value: str,
    approver: str,
    decisions_file: Optional[str] = None,
    runner=subprocess.run,
    timeout: int = 60,
) -> tuple[str, str]:
    cmd = [
        gate_path,
        "--requester-uuid", uuid,
        "--decision-id", decision_id or "",
        "--action", action,
        "--value", value,
        "--approver-name", approver,
    ]
    env = dict(os.environ)
    if decisions_file:
        env["DECISIONS_FILE"] = decisions_file
    try:
        proc = runner(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except Exception:
        return "ERROR", ""
    first = (getattr(proc, "stdout", "") or "").splitlines()
    if not first:
        return "ERROR", ""
    token, _, msg = first[0].partition(" ")
    return token, msg
