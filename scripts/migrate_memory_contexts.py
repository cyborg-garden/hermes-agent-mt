#!/usr/bin/env python3
"""Migrate memory context directories onto the current context-id scheme.

Why this exists
---------------
The memory context id has changed shape twice:

  gen0  ``<chat_id>``                      (raw id; DMs/threads went to global)
  gen1  ``{platform}:{chat_type}:{chat_id}``
  gen2  ``{platform}:{dm|chat}:{chat_id}`` (threads pool into parent channel)

A directory left under an older name is never loaded again, so its scoped
entries silently drop out of the agent's read view. Older directories may also
still contain copies of global entries, from the era when a scoped write
persisted the merged global+scoped view; the store self-heals that only for a
context it actually loads, so an orphaned directory keeps them indefinitely.

How the target is chosen
------------------------
Not by guessing from the directory name. Each agent's ``state.db`` records the
``platform``, ``chat_type`` and ``parent_chat_id`` actually observed for every
``chat_id``, so the target is computed by handing those to the SAME
``derive_context_id`` the runtime uses. A chat_id the database has never seen
is reported and skipped rather than renamed on a hunch.

Because gen2 pools a thread into its parent channel, several source
directories can collapse into one target. Their entries are merged and
deduplicated, most-recently-written last.

Safety
------
Dry-run by default: prints the plan and changes nothing. ``--apply`` takes a
timestamped backup of each memories directory before touching it.

Usage
-----
    python3 scripts/migrate_memory_contexts.py ~/.hermes-cyborg
    python3 scripts/migrate_memory_contexts.py ~/.hermes-* --apply
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.memory_tool import ENTRY_DELIMITER, derive_context_id

TARGETS = (("memory", "MEMORY.md"), ("user", "USER.md"))

#: Platform tokens that may appear as the first segment of a gen1/gen2 id.
#: Used only to strip a known prefix back off to recover the raw chat_id.
_KNOWN_PLATFORMS = (
    "signal", "telegram", "discord", "slack", "api_server", "msgraph_webhook",
    "feishu", "whatsapp", "matrix", "local", "cli", "irc", "unknown",
)
_KNOWN_SCOPES = ("dm", "chat", "group", "channel", "forum", "thread")


def read_entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError:
        return []
    if not raw.strip():
        return []
    return [e for e in (x.strip() for x in raw.split(ENTRY_DELIMITER)) if e]


def write_entries(path: Path, entries: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")


def raw_chat_id(dir_name: str) -> str:
    """Recover the underlying chat_id from a directory name of any generation."""
    parsed = parse_qualified_name(dir_name)
    return parsed[2] if parsed else dir_name


def parse_qualified_name(dir_name: str) -> Optional[tuple[str, str, str]]:
    """Split a gen1/gen2 directory name into (platform, chat_type, chat_id).

    Returns None for a gen0 (raw chat_id) name. A qualified name already
    states its own platform and chat_type, so it can be re-derived without
    consulting state.db — which matters because a chat_id whose sessions have
    been pruned is absent from the database but its memory directory is still
    on disk and still wanted.
    """
    parts = dir_name.split(":", 2)
    if (len(parts) == 3
            and parts[0].lower() in _KNOWN_PLATFORMS
            and parts[1].lower() in _KNOWN_SCOPES):
        return parts[0], parts[1], parts[2]
    return None


def load_chat_facts(agent_dir: Path) -> dict[str, tuple[str, str, Optional[str]]]:
    """chat_id -> (platform, chat_type, parent_chat_id) from the agent's sessions.

    The most recently started session for a chat_id wins.
    """
    db = agent_dir / "state.db"
    facts: dict[str, tuple[str, str, Optional[str]]] = {}
    if not db.exists():
        return facts
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        return facts
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(sessions)")}
        if not {"chat_id", "chat_type"} <= cols:
            return facts
        has_origin = "origin_json" in cols
        order = "started_at" if "started_at" in cols else "id"
        sel = "chat_id, chat_type" + (", origin_json" if has_origin else "")
        q = (f"SELECT {sel} FROM sessions "
             f"WHERE chat_id IS NOT NULL AND chat_id != '' ORDER BY {order} ASC")
        for row in con.execute(q):
            chat_id, chat_type = row[0], row[1]
            platform, parent = None, None
            if has_origin and row[2]:
                try:
                    o = json.loads(row[2])
                    platform = o.get("platform")
                    parent = o.get("parent_chat_id")
                except (ValueError, TypeError):
                    pass
            prev = facts.get(str(chat_id))
            # Keep a previously-known platform/parent if this row lacks them.
            if prev:
                platform = platform or prev[0]
                parent = parent or prev[2]
                chat_type = chat_type or prev[1]
            facts[str(chat_id)] = (platform, chat_type, parent)
    except sqlite3.Error:
        pass
    finally:
        con.close()
    return facts


def plan_for_agent(agent_dir: Path) -> tuple[list[dict], list[dict]]:
    """Return (moves, skipped) for one agent directory."""
    moves: list[dict] = []
    skipped: list[dict] = []
    mem_dir = agent_dir / "memories"
    contexts = mem_dir / "contexts"
    if not contexts.is_dir():
        return moves, skipped

    facts = load_chat_facts(agent_dir)
    globals_by_target = {
        target: set(read_entries(mem_dir / fname)) for target, fname in TARGETS
    }

    # target_dir_name -> list of (source_dir, mtime)
    grouped: dict[str, list[tuple[Path, float]]] = defaultdict(list)

    for entry in sorted(contexts.iterdir()):
        if not entry.is_dir():
            continue
        qualified = parse_qualified_name(entry.name)
        cid = qualified[2] if qualified else entry.name
        fact = facts.get(cid)

        if qualified:
            # The name already states platform and chat_type. Trust it, and
            # take only the parent from state.db (a thread needs one; if its
            # sessions were pruned it falls back to its own id).
            platform, chat_type = qualified[0], qualified[1]
            parent = fact[2] if fact else None
        elif fact:
            platform, chat_type, parent = fact
        else:
            skipped.append({"dir": entry.name,
                            "reason": f"unqualified name and chat_id {cid!r} is not "
                                      f"in state.db — cannot determine platform"})
            continue

        target = derive_context_id(platform, chat_type, cid, parent)
        if not target:
            skipped.append({"dir": entry.name, "reason": "derives to unscoped (None)"})
            continue
        safe = str(target).replace("/", "_").replace("\\", "_").replace("..", "_")
        mtimes = [p.stat().st_mtime for p in entry.glob("*.md")] or [0.0]
        grouped[safe].append((entry, max(mtimes)))

    for target_name, sources in sorted(grouped.items()):
        target_dir = contexts / target_name
        # Oldest first so the most recent writes land last in the merged list.
        sources.sort(key=lambda t: t[1])
        if [s for s, _ in sources] == [target_dir]:
            continue  # already correct, nothing to do

        merged: dict[str, list[str]] = {}
        stripped: dict[str, int] = {}
        for _, fname in TARGETS:
            acc: list[str] = []
            dropped = 0
            for src, _mt in sources:
                for e in read_entries(src / fname):
                    if e in globals_by_target["user" if fname == "USER.md" else "memory"]:
                        dropped += 1
                        continue
                    acc.append(e)
            merged[fname] = list(dict.fromkeys(acc))
            stripped[fname] = dropped

        moves.append({
            "target_name": target_name,
            "target_dir": target_dir,
            "sources": [s for s, _ in sources],
            "merged": merged,
            "stripped": stripped,
        })

    return moves, skipped


def apply_move(move: dict) -> None:
    target_dir: Path = move["target_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)
    for _, fname in TARGETS:
        entries = move["merged"].get(fname, [])
        if entries:
            write_entries(target_dir / fname, entries)
        elif (target_dir / fname).exists():
            (target_dir / fname).unlink()
    for src in move["sources"]:
        if src.resolve() != target_dir.resolve():
            shutil.rmtree(src)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent_dirs", nargs="+", type=Path,
                    help="agent home directories (e.g. ~/.hermes-cyborg)")
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (default is a dry run)")
    args = ap.parse_args(argv)

    stamp = time.strftime("%Y%m%dT%H%M%S")
    total_moves = total_stripped = total_merged = 0

    for agent_dir in args.agent_dirs:
        if not (agent_dir / "memories" / "contexts").is_dir():
            continue
        moves, skipped = plan_for_agent(agent_dir)
        if not moves and not skipped:
            continue

        print(f"\n=== {agent_dir.name}")
        for s in skipped:
            print(f"  -- skip {s['dir']}\n       {s['reason']}")
        for m in moves:
            names = [p.name for p in m["sources"]]
            if len(names) > 1:
                total_merged += 1
                print(f"  -> {m['target_name']}   (merging {len(names)} dirs)")
                for n in names:
                    print(f"       + {n}")
            else:
                print(f"  -> {m['target_name']}\n       from {names[0]}")
            for _, fname in TARGETS:
                kept = len(m["merged"].get(fname, []))
                drop = m["stripped"].get(fname, 0)
                if kept or drop:
                    print(f"       {fname}: keep {kept}, drop {drop} global-copied")
            total_stripped += sum(m["stripped"].values())
        total_moves += len(moves)

        if args.apply and moves:
            dest = agent_dir / "memories"
            bak = dest.parent / f"memories.pre-migration-{stamp}"
            shutil.copytree(dest, bak)
            print(f"  backup: {bak}")
            for m in moves:
                apply_move(m)
            print(f"  applied {len(moves)} move(s)")

    verb = "applied" if args.apply else "planned (dry run — pass --apply to write)"
    print(f"\n{total_moves} move(s) {verb}; {total_merged} merged into a shared "
          f"parent; {total_stripped} global-copied entries removed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
