#!/usr/bin/env python3
"""Delete signal-cli's own copies of attachments from disappearing chats.

signal-cli keeps every attachment it receives in its data folder, named by
Signal's attachment id, and never removes them. It keeps no message history,
so it has nothing to expire. The agents run in containers that can't see that
folder, so the disappearing_messages plugin records each attachment id and its
deadline in its ledger, and this script, run on the host, deletes the file when
it is due.

Read-only on every ledger; it only deletes files in the attachments folder.
Attachments from ordinary chats are never in a ledger, so they are left alone.
Prints counts only.

    sweep_signal_cli.py --signal-data ~/.hermes-swarm/signal-data [--homes '~/.hermes*'] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sqlite3
import sys
import time

# Signal attachment ids are short base64url-ish tokens. Anything else is refused,
# so a bad ledger row can never name a path outside the attachments folder.
ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,128}(\.[A-Za-z0-9]{1,10})?$")


def due_ids(ledger: str, now: float) -> set[str]:
    try:
        conn = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return set()
    try:
        rows = conn.execute(
            "SELECT attachment_id FROM signal_cli_files WHERE expires_at <= ?", (now,)
        ).fetchall()
    except sqlite3.Error:
        return set()  # an agent that predates this table
    finally:
        conn.close()
    return {r[0] for r in rows if isinstance(r[0], str) and ID_RE.match(r[0])}


def sweep(signal_data: str, homes: str, now: float, dry_run: bool = False) -> tuple[int, int]:
    folder = os.path.join(os.path.expanduser(signal_data), "attachments")
    if not os.path.isdir(folder):
        return (0, 0)
    due: set[str] = set()
    for ledger in glob.glob(os.path.join(os.path.expanduser(homes), "disappearing", "ledger.db")):
        due |= due_ids(ledger, now)
    if not due:
        return (0, 0)
    stems = {d.split(".", 1)[0] for d in due}
    deleted = 0
    with os.scandir(folder) as it:
        for entry in it:
            if not entry.is_file(follow_symlinks=False):
                continue
            stem = entry.name.split(".", 1)[0]
            if entry.name in due or stem in stems:
                if not dry_run:
                    try:
                        os.unlink(entry.path)
                    except OSError:
                        continue
                deleted += 1
    return (len(due), deleted)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signal-data", default="~/.hermes-swarm/signal-data")
    ap.add_argument("--homes", default="~/.hermes*")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    due, deleted = sweep(args.signal_data, args.homes, time.time(), args.dry_run)
    if deleted or args.dry_run:
        print(f"signal-cli attachments: {due} due, {deleted} {'would be ' if args.dry_run else ''}deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
