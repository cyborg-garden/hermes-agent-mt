#!/usr/bin/env python3
"""Delete signal-cli's own copies of attachments from disappearing chats.

signal-cli keeps every attachment it receives in its data folder, named by
Signal's attachment id, and never removes them. It keeps no message history,
so it has nothing to expire. The agents run in containers that can't see that
folder, so the disappearing_messages plugin lists the ids that are due in
``<agent home>/disappearing/signal_cli_due.json``, and this script, run on the
host, deletes the matching files and writes the ids it has dealt with to
``signal_cli_done.json`` so the plugin can forget them.

Plain files, not the plugin's SQLite ledger: the ledger is in WAL mode inside
the Docker VM, and WAL needs every reader on the same host.

Only ever deletes inside the attachments folder, and only files whose name is
a listed id. Attachments from ordinary chats are never listed. Prints a line
when it deletes something or something is wrong, nothing otherwise.

    sweep_signal_cli.py --signal-data ~/.hermes-swarm/signal-data [--homes '~/.hermes*'] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

# Signal attachment ids are base64url tokens, sometimes with an extension.
ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,128}(\.[A-Za-z0-9]{1,10})?$")
DUE_FILE = "signal_cli_due.json"
DONE_FILE = "signal_cli_done.json"


def _read_due(path: str) -> tuple[list[str], float]:
    try:
        with open(path, encoding="utf-8") as fh:
            body = json.load(fh)
        ids, written = body.get("ids", []), float(body.get("written_at", 0) or 0)
    except (OSError, ValueError, AttributeError, TypeError):
        return [], 0.0
    return [i for i in ids if isinstance(i, str) and ID_RE.match(i)], written


def _write_done(folder: str, ids: list[str], due_written_at: float) -> None:
    path = os.path.join(folder, DONE_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        # due_written_at ties this answer to the list it answers, so the plugin
        # never applies it to a row added after that list.
        json.dump({"written_at": time.time(), "due_written_at": due_written_at,
                   "ids": sorted(set(ids))}, fh)
    os.replace(tmp, path)


def sweep(signal_data: str, homes: str, now: float | None = None, dry_run: bool = False) -> tuple[int, int]:
    attachments = os.path.join(os.path.expanduser(signal_data), "attachments")
    if not os.path.isdir(attachments):
        print(f"signal-cli attachments folder not found: {attachments}", file=sys.stderr)
        return (0, 0)
    by_home: dict[str, tuple[list[str], float]] = {}
    for due_path in glob.glob(os.path.join(os.path.expanduser(homes), "disappearing", DUE_FILE)):
        by_home[os.path.dirname(due_path)] = _read_due(due_path)
    wanted = {i for ids, _ in by_home.values() for i in ids}
    if not wanted:
        return (0, 0)
    stems = {i.split(".", 1)[0] for i in wanted}
    deleted = 0
    with os.scandir(attachments) as it:
        for entry in it:
            if not entry.is_file(follow_symlinks=False):
                continue
            if entry.name in wanted or entry.name.split(".", 1)[0] in stems:
                if not dry_run:
                    try:
                        os.unlink(entry.path)
                    except OSError as e:
                        print(f"could not delete a signal-cli attachment: {e.strerror}", file=sys.stderr)
                        continue
                deleted += 1
    if not dry_run:
        # Whatever was listed is now gone (deleted here, or already absent).
        present = {e.name.split(".", 1)[0] for e in os.scandir(attachments)}
        absent = len(wanted) - deleted
        if absent:
            # Normal for a file signal-cli never kept; many of these can mean
            # --signal-data points at the wrong folder.
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} signal-cli attachments: "
                  f"{absent} due ids had no file to delete", file=sys.stderr)
        for folder, (ids, written) in by_home.items():
            gone = [i for i in ids if i.split(".", 1)[0] not in present]
            try:
                _write_done(folder, gone, written)
            except OSError as e:
                print(f"could not confirm deletions to {folder}: {e.strerror}", file=sys.stderr)
    return (len(wanted), deleted)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signal-data", default="~/.hermes-swarm/signal-data")
    ap.add_argument("--homes", default="~/.hermes*")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    due, deleted = sweep(args.signal_data, args.homes, dry_run=args.dry_run)
    if deleted or args.dry_run:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{stamp} signal-cli attachments: {due} due, {deleted} {'would be ' if args.dry_run else ''}deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
