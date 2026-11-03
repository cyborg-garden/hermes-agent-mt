"""MemoryStore — bounded, file-backed curated memory (MEMORY.md / USER.md).
Entries are joined by ``ENTRY_DELIMITER``; budgets are in chars (model-independent).
Module state that tests monkeypatch (``get_memory_dir``, ``fcntl``/``msvcrt``) stays
in ``tools.memory_tool`` and is read lazily."""

import logging
import os
import re
import time
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import atomic_write_text
from tools.threat_patterns import first_threat_message as _first_threat_message

logger = logging.getLogger("tools.memory_tool")

# Block header prefixes rendered by _render_block; agent/conversation_compression.py
# matches them to detect a leftover block for an emptied target — keep in lockstep.
MEMORY_BLOCK_HEADERS = {
    "memory": "MEMORY (your personal notes)", "user": "USER PROFILE (who the user is)"}

ENTRY_DELIMITER = "\n§\n"


def _scan_memory_content(content: str) -> Optional[str]:
    """Error string if *content* matches injection/exfil patterns. Strict scope:
    memory enters the system prompt, so a poisoned entry persists across sessions."""
    return _first_threat_message(content, scope="strict")


# Why the last refusal/failure in this context happened, for the shared metric only (a closed name from
# shared_metrics_contract.MEMORY_OP_FAILURE_CLASSES); never part of the result the model sees.
FAILURE_CLASS: ContextVar[str] = ContextVar("memory_failure_class", default="other")


def _error(message: str, failure_class: str, **extra) -> dict[str, Any]:
    FAILURE_CLASS.set(failure_class)
    return {"success": False, "error": message, **extra}


def _store_target(filename: str) -> str:
    """The memory tool target that owns a store file, for hints that show the call to make."""
    return "user" if filename == "USER.md" else "memory"


def _drift_error(path: Path, bak_path: str) -> dict[str, Any]:
    """External drift: the file wouldn't round-trip, so flushing would discard content."""
    return _error((
        f"Refusing to write {path.name}: it holds an entry larger than the whole store's char limit, "
        f"i.e. free-form text written without § delimiters (the patch tool, a shell append, a manual "
        f"edit). Rewriting it would lose that text. A snapshot was saved to {bak_path}. Resolve the drift first — "
        f"either rewrite the file as a clean §-delimited list of entries, or move the extra "
        f"content out — then retry. This guard exists to prevent silent data loss (issue #26045)."
    ), "drift", drift_backup=bak_path, remediation=(
        "Open the .bak file, integrate the missing entries into the memory tool one at a time via "
        f"memory(action=add, target='{_store_target(path.name)}', content=...), then remove or rewrite the "
        "original file to a clean state."))


def _read_failed_error(path: Path) -> dict[str, Any]:
    """Existing-but-unreadable file: saving from an assumed-empty view would wipe it."""
    return _error(
        f"Refusing to write {path.name}: the file exists on disk but could not be read right now "
        f"(temporarily locked by another program, a permission change, invalid/corrupt text encoding, "
        f"or a filesystem error). Treating an unreadable file as empty and saving would wipe existing "
        f"memory, so the write is refused. Nothing was changed — retry in a moment.", "read_failed")


# Typography a model re-types differently from the stored entry: every quote/backtick -> "'",
# every dash/minus -> "-". Plus whitespace runs (re-wrapped lines, NBSP) -> one space.
_MATCH_FOLD = str.maketrans({**dict.fromkeys("\"`\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f", "'"),
                             **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-")})


def _normalize_for_match(text: str) -> str:
    """Quote/dash/whitespace-folded form for tolerant old_text matching. Models re-emit an
    entry with straight quotes for curly ones, '-' for an em dash, unwrapped lines, or a
    newline over-escaped to a literal backslash-n, and a byte-exact substring test then
    reports "No entry matched" for an unambiguous target."""
    return " ".join(str(text).replace("\\n", " ").translate(_MATCH_FOLD).split())


def _substring_matches(entries: list[str], old_text: str) -> list[int]:
    """Indices of entries containing *old_text*: byte-exact first, else typography-folded."""
    if exact := [i for i, e in enumerate(entries) if old_text in e]:
        return exact
    needle = _normalize_for_match(old_text)
    if not any(c.isalnum() for c in needle):  # punctuation/space alone would select a whole entry
        return []
    return [i for i, e in enumerate(entries) if needle in _normalize_for_match(e)]


def _find_unique_match(entries: list[str], old_text: str) -> tuple[Optional[int], bool]:
    """``(index, ambiguous)`` for entries matching *old_text*. A whole-entry
    EXACT match (``old_text == entry``) takes absolute priority — substring
    matches are only considered when no entry equals *old_text*, so a short
    entry stays addressable even when its full text is contained inside a
    longer sibling entry (remove('test') vs '...tests pass...'). Exact-duplicate
    matches are safe (first wins); distinct matches → ``(None, True)``."""
    exact = [i for i, e in enumerate(entries) if e == old_text]
    matches = exact or _substring_matches(entries, old_text)
    if len({entries[i] for i in matches}) > 1:
        return None, True
    return (matches[0] if matches else None), False


def _no_match_error(entries: list[str], old_text: str, verb: str, **extra) -> dict[str, Any]:
    """Zero-match error that names the fix: the entries sharing the most words with
    *old_text* (a paraphrased or stale old_text usually still shares most of them), and a
    pointer when old_text spans several entries joined by the delimiter."""
    def _words(text: str) -> set:
        return set(re.findall(r"\w+", text.lower()))
    words = _words(old_text)
    scored = sorted(((len(words & _words(e)) / len(words), i) for i, e in enumerate(entries)) if words else [],
                    key=lambda p: (-p[0], p[1]))
    closest = [entries[i] for score, i in scored[:3] if score >= 0.3]
    hint = (" old_text must come from ONE entry ('§' separates entries); use one op per entry."
            if "§" in old_text else "")
    return _error(f"No entry matched '{old_text}'.{hint} Retry the {verb} with old_text copied verbatim "
                  f"from the intended entry" + (" (closest_entries below)." if closest else "."), "no_match",
                  **({"closest_entries": closest} if closest else {}), **extra)


def _over_limit(new_total: int, limit: int) -> str:
    return f"would put memory at {new_total:,}/{limit:,} chars: free at least {new_total - limit:,} chars"


# Patch content cannot be empty, and 'remove' would drop the sibling facts in the entry.
PATCH_DROP_HINT = "To drop text inside an entry, patch a longer span that contains it; 'remove' deletes the whole entry."


def _repeated_span_message(old_text: str, count: int) -> str:
    return (f"'{old_text}' appears {count} times in the matched entry, and patch edits one span. "
            f"Include more surrounding text so old_text occurs once.")


def _span_occurrences(text: str, span: str) -> int:
    """Occurrences of *span* in *text*, overlapping ones included: '00' occurs twice in '1000'."""
    return sum(text.startswith(span, i) for i in range(len(text) - len(span) + 1))


def _patch_entry(entry: str, old_text: str, new_text: str) -> tuple[Optional[str], str, str]:
    """``(patched entry, "", "none")``, or ``(None, error, failure class)`` when the span occurs more
    than once in the entry, the edit changes nothing, or the patched entry fails the threat scan.
    Single and batch patches both go through here, so they share one rule."""
    if (count := _span_occurrences(entry, old_text)) > 1:
        return None, _repeated_span_message(old_text, count), "ambiguous"
    patched = entry.replace(old_text, new_text, 1)
    if patched == entry:
        return None, f"Patch is a no-op: '{old_text}' -> '{new_text}' leaves the entry unchanged.", "invalid_args"
    if scan_error := _scan_memory_content(patched):
        return None, scan_error, "scan_blocked"
    return patched, "", "none"


def _pinned_index(entries: list[str], matched_entry: str) -> Optional[int]:
    """Index of the exact entry a staged write was reviewed against; None once it is gone (stale)."""
    return entries.index(matched_entry) if matched_entry in entries else None


def _stale_entry_message(entry: str) -> str:
    return (f"Entry changed since it was staged, so this write was not applied: '{entry}' is no longer "
            f"in memory as reviewed. Recreate the change against the current entry or reject it; the "
            f"pending record has been preserved.")


# Optional third value of an _apply closure: a dict merged into the success payload —
# e.g. the full text an entry-level replace overwrote, so a whole-entry write is never
# silent about the loss (#117952). Same convention as _error(**extra).


class MemoryStore:
    """Bounded curated memory with file persistence; one instance per AIAgent.
    ``_system_prompt_snapshot`` is frozen at load time (prefix-cache stable);
    ``memory_entries`` / ``user_entries`` are live state persisted to disk."""

    # Failed consolidation attempts (overflow / zero-match) allowed per turn before
    # a TERMINAL "save skipped" result, so a fragile replace/add can't loop the turn
    # to budget exhaustion and suppress the user's reply.
    # See #42405.
    _MAX_CONSOLIDATION_FAILURES_PER_TURN = 3

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375,
                 context_id: Optional[str] = None, include_global: bool = False, *,
                 memory_enabled: bool = True, user_profile_enabled: bool = True):
        # ``memory_entries`` / ``user_entries`` are the DERIVED read view (what
        # the system prompt, ``read``, and success responses show). They are
        # never persisted as one blob. The two partitions below are the source
        # of truth and are written to their own files independently.
        self.memory_entries: list[str] = []
        self.user_entries: list[str] = []
        self.memory_char_limit, self.user_char_limit = memory_char_limit, user_char_limit
        self.memory_enabled, self.user_profile_enabled = memory_enabled, user_profile_enabled
        # Partition state (per target). Global is the shared, backward-compatible
        # layer; scoped is per-context and isolated. Keeping them separate is
        # what guarantees a scoped file only ever holds scoped entries.
        self._global_entries: dict[str, list[str]] = {"memory": [], "user": []}
        self._scoped_entries: dict[str, list[str]] = {"memory": [], "user": []}
        # Sanitize context_id: treat empty/falsy as None; replace path
        # separators and parent-ref sequences so chat IDs from any platform
        # can't escape the contexts/ directory via path traversal. ``:`` is kept
        # verbatim (a within-name separator, not a directory boundary).
        if context_id:
            safe_id = str(context_id).replace("/", "_").replace("\\", "_").replace("..", "_")
            self.context_id: Optional[str] = safe_id or None
        else:
            self.context_id = None
        # Opt-in, read-only global layer for scoped stores. Default OFF: a
        # scoped store's read view is scoped-only unless the operator turns
        # this on. Ignored for unscoped stores, which are always global-only.
        self.include_global = bool(include_global)
        self._system_prompt_snapshot: dict[str, str] = {"memory": "", "user": ""}
        self._consolidation_failures = 0  # per turn; reset by reset_consolidation_failures()

    # Per-turn counter of failed at-capacity consolidation attempts; reset at each turn boundary by
    # reset_consolidation_failures() (#42405).
    def target_enabled(self, target: str) -> bool:
        return self.user_profile_enabled if target == "user" else self.memory_enabled

    def reset_consolidation_failures(self) -> None:
        """Call at turn start."""
        self._consolidation_failures = 0

    def _consolidation_failure(self, response: dict[str, Any]) -> dict[str, Any]:
        """Count a consolidation failure: under the per-turn cap return ``response``
        (it says how to retry); past it a TERMINAL result so the model stops looping.

        Once the cap is exceeded, drop the retry instruction and return a TERMINAL result so the model stops
        looping memory calls and proceeds to answer the user — a failed memory side effect must never block
        the turn's reply (#42405).
        """
        self._consolidation_failures += 1
        if self._consolidation_failures <= self._MAX_CONSOLIDATION_FAILURES_PER_TURN:
            return response
        FAILURE_CLASS.set("retry_cap")
        return {"success": False, "done": True, "error": (
            f"Memory consolidation failed {self._consolidation_failures} times this turn. Stop retrying "
            "memory calls — leave memory unchanged for now and continue with your reply to the user. "
            "The fact can be saved in a later turn.")}

    # Plain-string form of the scoped-context guard, for the batch path — a
    # batch returns one aggregate error, so it can't carry the richer guard.
    _GLOBAL_READONLY_MSG = (
        "Cannot modify a global entry from a scoped context — global memory is "
        "read-only here. Edit it from an unscoped (global) session instead."
    )

    def _global_entry_guard_response(self, target: str) -> dict[str, Any]:
        """Response for an attempt to modify a global entry from a scoped context.

        Global entries are read-only from a scoped context: writes land in the
        scoped file only, so the edit would silently vanish on the next reload.
        At capacity the model was told to consolidate — it would pick a global
        entry, hit this guard, retry, and loop the turn to budget exhaustion
        (fork issue #39). Give scope-honest guidance AND route it through the
        per-turn consolidation-failure cap so repeated attempts degrade to a
        terminal result (shares the #42405 loop-breaker).
        """
        return self._consolidation_failure(_error(
            "That entry belongs to global memory and can't be edited from "
            "this scoped context. To free space, consolidate one of your "
            "context-scoped entries instead (patch or remove an entry you "
            "added in this context), or continue without saving — the fact "
            "can be recorded in a later turn.", "global_readonly"))

    def load_from_disk(self):
        """Load MEMORY.md / USER.md and capture the frozen system-prompt snapshot.
        Threat hits are replaced by a ``[BLOCKED: …]`` placeholder in the SNAPSHOT only;
        live lists keep the raw text so the user can see and remove poisoned entries
        (dropping them silently would hide the attack)."""
        from tools.threat_patterns import scan_for_threats

        def _sanitize(entry, filename):
            # Strict scope, same as writes; empty / already-blocked entries pass through.
            findings = scan_for_threats(entry, scope="strict") if entry and not entry.startswith("[BLOCKED:") else None
            if not findings:
                return entry
            logger.warning("Memory entry from %s blocked at load time: %s", filename, ", ".join(findings))
            return (f"[BLOCKED: {filename} entry contained threat pattern(s): {', '.join(findings)}. "
                    f"Removed from system prompt; use memory(action=remove, target='{_store_target(filename)}') "
                    f"to delete the original.]")

        for target in ("memory", "user"):
            path = self._path_for(target)
            from hermes_constants import mkdir_under_hermes_home

            mkdir_under_hermes_home(self._global_path_for(target).parent)
            # Load global + scoped partitions (deduplicated; a scoped file that an
            # older build polluted with global entries self-heals) and derive the
            # read view. Unscoped stores: view == global file, as upstream.
            self._load_partitions(target)
            entries = self._derive_view(target)
            self._set_entries(target, entries)
            # External writers (MCP bridges, hand edits) can exceed the cap; the limit only fires on
            # add/replace, so the oversized block would silently ride in the prompt while every later
            # add is refused with no visible cause (#10877). Warn; never truncate a user's memories.
            if (count := self._char_count(target)) > (limit := self._char_limit(target)):
                logger.warning("%s exceeds its char limit on load: %d/%d chars. Entries stay loaded; "
                               "further additions are blocked until it is back under the limit.",
                               path.name, count, limit)
            self._system_prompt_snapshot[target] = self._render_block(target, [_sanitize(e, path.name) for e in entries])

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Exclusive lock on a separate .lock file so the memory file itself can
        still be atomically replaced."""
        from tools import memory_tool as _mt  # fcntl/msvcrt live (and are patched) there
        fcntl, msvcrt = _mt.fcntl, _mt.msvcrt
        lock_path = path.with_suffix(path.suffix + ".lock")
        from hermes_constants import mkdir_under_hermes_home

        mkdir_under_hermes_home(lock_path.parent)
        if fcntl is None and msvcrt is None:
            yield
            return
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        raw_fd = os.open(lock_path, flags, 0o600)
        try:
            # The creation mode is filtered through the process umask and does
            # not repair a lock left loose by an older Hermes process. Tighten
            # the opened inode before acquiring the lock so both cases are
            # owner-only. Operating on the fd avoids a path-swap window.
            if hasattr(os, "fchmod"):
                os.fchmod(raw_fd, 0o600)
            fd = os.fdopen(raw_fd, "r+", encoding="utf-8")
        except Exception:
            os.close(raw_fd)
            raise
        with fd:
            def _flock(unlock: bool):
                if fcntl:
                    fcntl.flock(fd, fcntl.LOCK_UN if unlock else fcntl.LOCK_EX)
                else:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK if unlock else msvcrt.LK_LOCK, 1)
            _flock(False)
            try:
                yield
            finally:
                with suppress(OSError):
                    _flock(True)

    @staticmethod
    def _global_path_for(target: str) -> Path:
        """The global (unscoped) file for a target — memories-dir root, backward compatible."""
        from tools import memory_tool  # get_memory_dir is monkeypatched there
        return memory_tool.get_memory_dir() / ("USER.md" if target == "user" else "MEMORY.md")

    def _scoped_path_for(self, target: str) -> Path:
        """The scoped file for a target under contexts/{context_id}/ (only meaningful when scoped)."""
        from tools import memory_tool
        base = memory_tool.get_memory_dir() / "contexts" / str(self.context_id)
        return base / ("USER.md" if target == "user" else "MEMORY.md")

    def _path_for(self, target: str) -> Path:
        """The file this store MUTATES: global when unscoped, the scoped file otherwise.
        Global files are never written from a scoped store."""
        if self.context_id is None:
            return self._global_path_for(target)
        return self._scoped_path_for(target)

    # -- Context-scoping partitions (fork: per-channel/tenant memory) --

    def _heal_scoped(self, target: str, scoped_raw: list[str]) -> list[str]:
        """Drop entries that also live in the global partition from a scoped
        partition — an older build copied the merged view into scoped files."""
        global_set = set(self._global_entries.get(target, []))
        return [e for e in dict.fromkeys(scoped_raw) if e not in global_set]

    def _load_partitions(self, target: str) -> None:
        """Read global and (when scoped) scoped partitions from disk, deduplicated."""
        self._global_entries[target] = list(dict.fromkeys(self._read_file(self._global_path_for(target))))
        if self.context_id is None:
            self._scoped_entries[target] = []
            return
        self._scoped_entries[target] = self._heal_scoped(target, self._read_file(self._scoped_path_for(target)))

    def _project_view(self, target: str, writable: list[str]) -> list[str]:
        """Read view that results from a candidate writable partition.

        Unscoped: the partition itself. Scoped, include_global off (default):
        scoped only. Scoped, include_global on: global first, then scoped, deduped."""
        if self.context_id is None or not self.include_global:
            return list(writable)
        return list(dict.fromkeys(list(self._global_entries.get(target, [])) + list(writable)))

    def _derive_view(self, target: str) -> list[str]:
        writable = self._global_entries if self.context_id is None else self._scoped_entries
        return self._project_view(target, writable.get(target, []))

    def _projected_len(self, target: str, writable: list[str]) -> int:
        """Budget is measured against the resulting READ view, not the bare partition."""
        return len(ENTRY_DELIMITER.join(self._project_view(target, writable)))

    def _is_global_entry(self, target: str, entry: str) -> bool:
        return entry in set(self._global_entries.get(target, []))

    def _global_match_guard(self, target: str, old_text: str) -> Optional[dict[str, Any]]:
        """Refusal when *old_text* selects a read-only global entry from a scoped
        store (only possible when include_global exposes globals in the view)."""
        if self.context_id is None or not self.include_global:
            return None
        view = self._project_view(target, self._scoped_entries.get(target, []))
        idx, ambiguous = _find_unique_match(view, old_text)
        if not ambiguous and idx is not None and self._is_global_entry(target, view[idx]):
            return self._global_entry_guard_response(target)
        return None

    def _entries_for(self, target: str) -> list[str]:
        return self.user_entries if target == "user" else self.memory_entries

    def _set_entries(self, target: str, entries: list[str]):
        setattr(self, "user_entries" if target == "user" else "memory_entries", entries)

    def _char_count(self, target: str) -> int:
        return len(ENTRY_DELIMITER.join(self._entries_for(target)))

    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    def _usage(self, target: str) -> str:
        return f"{self._char_count(target):,}/{self._char_limit(target):,}"

    def _usage_pct(self, target: str, current: int) -> str:
        limit = self._char_limit(target)
        return f"{min(100, int((current / limit) * 100)) if limit > 0 else 0}% — {current:,}/{limit:,} chars"

    def _failure_with_entries(self, target: str, message: str) -> dict[str, Any]:
        """Over-budget consolidation failure carrying the live entries so the model can consolidate."""
        return self._consolidation_failure(
            _error(message, "over_budget", current_entries=self._entries_for(target), usage=self._usage(target)))

    def _batch_failure(self, target: str, message: str, failure_class: str,
                       closest_entries: Optional[list[str]] = None) -> dict[str, Any]:
        """Batch-abort failure WITHOUT ``current_entries``: the store did not change and the
        caller already holds the inventory, so echoing it made each consolidation retry
        grow the context it was invoked to shrink (#97316). A zero-match op carries only
        its few ``closest_entries``."""
        return self._consolidation_failure(
            _error(message + " No operations were applied (batch is all-or-nothing).", failure_class,
                   usage=self._usage(target),
                   **({"closest_entries": closest_entries} if closest_entries else {})))

    def _mutate(self, target: str, mutate, *, skip_drift: bool = False) -> dict[str, Any]:
        """Lock, re-read from disk, run ``mutate(entries, limit)`` -> ``(new_entries, message)``
        or an error dict, then persist and return the success response. The reload aborts
        on an existing-but-unreadable file (even append-only ``add`` rewrites the whole
        file) and, unless *skip_drift*, on external drift (flushing would discard
        un-roundtrippable content). Drift check and parse use the SAME raw snapshot —
        a failed second read used to count as "no drift". The closure may return a
        third value, a dict merged into the success payload (``_error``'s ``**extra``
        convention) — e.g. the full text a replace overwrote (#117952). ANY dict the closure
        returns is passed through verbatim and nothing is persisted: error dicts, or the
        success payload of a read-only closure (``resolve_entry``, ``resolve_batch_entries``)."""
        path = self._path_for(target)
        with self._file_lock(path):
            raw, read_ok = self._read_raw_checked(path)
            if not read_ok:
                return _read_failed_error(path)
            bak = None if skip_drift else self._detect_external_drift(target, raw)
            parsed = list(dict.fromkeys(self._parse_entries(raw)))
            if self.context_id is None:
                self._global_entries[target] = parsed
            else:
                # Global is read-only here; refresh it so guards and the view are current.
                self._global_entries[target] = list(dict.fromkeys(self._read_file(self._global_path_for(target))))
                parsed = self._heal_scoped(target, parsed)
                self._scoped_entries[target] = parsed
            self._set_entries(target, self._derive_view(target))
            if bak:
                return _drift_error(path, bak)
            # Closures operate on the WRITABLE partition (scoped for a scoped store),
            # so the merged read view is never written back into a partition file.
            result = mutate(list(parsed), self._char_limit(target))
            if isinstance(result, dict):
                return result
            if self.context_id is None:
                self._global_entries[target] = result[0]
            else:
                self._scoped_entries[target] = result[0]
            self._set_entries(target, self._derive_view(target))
            from hermes_constants import mkdir_under_hermes_home

            mkdir_under_hermes_home(path.parent)
            self._write_file(path, result[0])
            extra_fields = result[2] if len(result) > 2 else {}
            return self._success_response(target, result[1], **extra_fields)

    def add(self, target: str, content: str) -> dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        content = content.strip()
        if not content:
            return _error("Content cannot be empty.", "missing_content")
        if scan_error := _scan_memory_content(content):
            return _error(scan_error, "scan_blocked")

        def _add(entries, limit):
            # Duplicate check also consults the global partition: a scoped copy of a
            # global fact would be stripped by load-time healing and not survive.
            if content in entries or content in self._entries_for(target) or self._is_global_entry(target, content):
                return self._success_response(target, "Entry already exists (no duplicate added).")
            new_total = self._projected_len(target, entries + [content])
            if new_total > limit:
                # In a scoped context global entries can't be edited; say so, so the
                # model consolidates scoped entries instead of looping on the guard (#39).
                scope_hint = ""
                if self.context_id is not None and self._global_entries.get(target) and self.include_global:
                    scope_hint = (" Note: entries from global memory can't be edited from this context — "
                                  "consolidate only your context-scoped entries (or continue without saving).")
                return self._failure_with_entries(target, (
                    f"Memory at {self._char_count(target):,}/{limit:,} chars; adding this entry ({len(content)} chars) "
                    f"would exceed the limit by {new_total - limit:,} chars. Retry as ONE 'operations' batch that "
                    f"removes or shortens (patch) stale entries from current_entries below to free at least "
                    f"{new_total - limit:,} chars AND adds this entry — the limit is checked only on the batch result."
                    f"{scope_hint}"))
            return entries + [content], "Entry added."
        # Append-only: skip the drift guard (appending never clobbers foreign
        # content) but still refuse a failed read — add rewrites the WHOLE file.
        return self._mutate(target, _add, skip_drift=True)

    def replace(self, target: str, old_text: str, new_content: str,
                matched_entry: Optional[str] = None) -> dict[str, Any]:
        """Find the entry containing old_text (whole-entry exact match first) and
        replace the WHOLE entry with new_content — old_text only locates the entry;
        the matched span is not spliced into it."""
        new_content = new_content.strip()
        if not old_text.strip():
            return _error("old_text cannot be empty.", "missing_old_text")
        if not new_content:
            return _error("new_content cannot be empty. Use 'remove' to delete entries.", "missing_content")
        if scan_error := _scan_memory_content(new_content):
            return _error(scan_error, "scan_blocked")
        return self._edit(target, old_text.strip(), new_content, matched_entry)

    def patch(self, target: str, old_text: str, new_text: str,
              matched_entry: Optional[str] = None) -> dict[str, Any]:
        """Find the entry containing *old_text* and replace that one span with *new_text*;
        the rest of the entry is preserved verbatim. This is the partial-edit shape
        ``replace`` deliberately is not (replace overwrites the WHOLE entry). A span that
        occurs more than once in the entry is rejected rather than edited everywhere."""
        if not (old_text or "").strip():
            return _error("old_text cannot be empty.", "missing_old_text")
        if new_text is None or not str(new_text).strip():
            return _error(f"new_text cannot be empty. {PATCH_DROP_HINT}", "missing_content")
        # The span keeps its whitespace: it is part of the text being replaced ("tea " -> "coffee ").
        return self._patch(target, old_text, str(new_text), matched_entry)

    def _patch(self, target: str, old_text: str, new_text: str,
               matched_entry: Optional[str] = None) -> dict[str, Any]:
        """Locked find-and-splice inside the matched entry (#124582)."""
        def _apply(entries, limit):
            idx = self._locate(entries, old_text, "patch", matched_entry, target=target)
            if isinstance(idx, dict):
                return idx
            before = entries[idx]
            after, message, failure_class = _patch_entry(before, old_text, new_text)
            if after is None:
                return _error(message, failure_class)
            patched = entries[:idx] + [after] + entries[idx + 1:]
            new_total = self._projected_len(target, patched)
            if new_total > limit:
                return self._failure_with_entries(target, (
                    f"Patch would put memory at {new_total:,}/{limit:,} chars. Shorten the new "
                    f"text, or 'remove' other stale or less important entries to make room "
                    f"(see current_entries below), then retry — all in this turn."))
            return patched, "Entry patched.", {
                "patched_entry_before": before, "patched_entry_after": after}
        return self._mutate(target, _apply)

    def remove(self, target: str, old_text: str, matched_entry: Optional[str] = None) -> dict[str, Any]:
        """Remove the entry containing old_text substring."""
        if not old_text.strip():
            return _error("old_text cannot be empty.", "missing_old_text")
        return self._edit(target, old_text.strip(), None, matched_entry)

    def _locate(self, entries: list[str], old_text: str, verb: str, matched_entry: Optional[str] = None,
                target: Optional[str] = None):
        """Index of the entry *old_text* selects, or the error dict the edit returns. A write
        staged for approval carries the FULL entry it was reviewed against (*matched_entry*):
        only that exact entry qualifies, so replay never hits a newer entry that still
        contains old_text. A scoped store refuses a selection that lands on a read-only
        global entry (fork #39) instead of reporting a confusing no-match."""
        if target is not None:
            guard = self._global_match_guard(target, matched_entry if matched_entry is not None else old_text)
            if guard is not None:
                return guard
        if matched_entry is not None:
            idx = _pinned_index(entries, matched_entry)
            return idx if idx is not None else _error(_stale_entry_message(matched_entry), "stale_entry")
        idx, ambiguous = _find_unique_match(entries, old_text)
        if ambiguous:
            return _error(f"Multiple entries matched '{old_text}'. Be more specific.", "ambiguous",
                          matches=[entries[i][:80] + ("..." if len(entries[i]) > 80 else "")
                                   for i in _substring_matches(entries, old_text)])
        if idx is None:
            return self._consolidation_failure(_no_match_error(entries, old_text, verb, current_entries=entries))
        return idx

    def resolve_entry(self, target: str, old_text: str, verb: str) -> dict[str, Any]:
        """``{"success": True, "matched_entry": <full entry>}`` for the entry *old_text* selects
        now, read under the lock, or the error the direct edit would return."""
        def _resolve(entries, limit):
            idx = self._locate(entries, old_text.strip(), verb, target=target)
            return idx if isinstance(idx, dict) else {"success": True, "matched_entry": entries[idx]}
        return self._mutate(target, _resolve, skip_drift=True)

    def _edit(self, target: str, old_text: str, new_content: Optional[str],
              matched_entry: Optional[str] = None) -> dict[str, Any]:
        """Locked replace (``new_content`` set) or remove (None) of the entry matching *old_text*."""
        def _apply(entries, limit):
            idx = self._locate(entries, old_text, "replace" if new_content else "remove", matched_entry,
                               target=target)
            if isinstance(idx, dict):
                return idx
            replaced = entries[:idx] + ([] if new_content is None else [new_content]) + entries[idx + 1:]
            if new_content is None:
                return replaced, "Entry removed.", {"removed_entry": entries[idx]}
            new_total = self._projected_len(target, replaced)
            if new_total > limit:
                return self._failure_with_entries(target, (
                    f"Replacement {_over_limit(new_total, limit)}. Shorten the new content, or retry as ONE "
                    f"'operations' batch with this replace plus removes/shortenings of other stale entries "
                    f"(see current_entries below) — the limit is checked only on the batch result."))
            return replaced, "Entry replaced.", {"replaced_entry": entries[idx]}
        return self._mutate(target, _apply)

    @staticmethod
    def _apply_batch_op(working: list[str], act: str, content: str, old_text: str, pos: str,
                        matched_entry: Optional[str] = None) -> tuple[Optional[str], Any, str, Optional[str]]:
        """Apply one batch op to *working*; return ``(error message, previous content, failure class,
        patched entry)`` — on a zero-match error the second value is that op's ``closest_entries``
        instead, and the fourth is the entry a patch produced (None for every other op).
        Previous content is captured before each replace/remove, under the store lock.
        It is published only after the entire batch has been validated and persisted.
        A staged op's *matched_entry* selects exactly that entry, as in ``_locate``.
        """
        if act == "add":
            if not content:
                return f"{pos}: content is required.", None, "missing_content", None
            if content not in working:  # idempotent -- skip duplicate, don't fail the batch
                working.append(content)
            return None, None, "none", None
        if act not in ("replace", "remove", "patch"):
            return f"{pos}: unknown action. Use add, patch, or remove.", None, "invalid_args", None
        if not old_text.strip():
            return f"{pos}: old_text is required.", None, "missing_old_text", None
        if act == "replace" and not content:
            return (f"{pos}: content is required — set this operation object's own 'content' to the COMPLETE "
                    f"new entry (or use action='remove' to delete)."), None, "missing_content", None
        if act == "patch" and not content.strip():
            return (f"{pos}: content is required — set this operation object's own 'content' to the new "
                    f"text for the old_text span. {PATCH_DROP_HINT}"), None, "missing_content", None
        if matched_entry is not None:
            idx = _pinned_index(working, matched_entry)
            if idx is None:
                return f"{pos}: {_stale_entry_message(matched_entry)}", None, "stale_entry", None
        else:
            idx, ambiguous = _find_unique_match(working, old_text)
            if ambiguous:
                return (f"{pos}: '{old_text}' matched multiple distinct entries -- be more specific.", None,
                        "ambiguous", None)
            if idx is None:
                miss = _no_match_error(working, old_text, act)
                return f"{pos}: {miss['error']}", miss.get("closest_entries"), "no_match", None
        previous_content = working[idx]
        if act != "patch":
            working[idx:idx + 1] = [content] if act == "replace" else []
            return None, previous_content, "none", None
        patched, message, failure_class = _patch_entry(previous_content, old_text, content)
        if patched is None:
            return f"{pos}: {message}", None, failure_class, None
        working[idx:idx + 1] = [patched]
        return None, previous_content, "none", patched

    def apply_batch(self, target: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply add/replace/remove ops atomically against the FINAL budget, so one call
        can free space and add entries. All-or-nothing: any malformed / unmatched op or
        an over-limit result writes NOTHING and returns the first failure. Aborts do not
        echo ``current_entries`` — the store is unchanged and the model already has it."""
        return self._batch(target, operations, commit=True)

    def resolve_batch_entries(self, target: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
        """Dry-run ``apply_batch`` under the lock without persisting: the same content scan,
        op walk, empty-store and budget checks, so it fails exactly where the direct batch
        would; on success ``{"success": True, "matched_entries": [...]}`` — per op, the FULL
        entry its replace/remove selects now (None for add), in batch order."""
        return self._batch(target, operations, commit=False)

    def _batch(self, target: str, operations: list[dict[str, Any]], *, commit: bool) -> dict[str, Any]:
        from tools.memory_tool import _memory_target_error
        if target_error := _memory_target_error(self, target, operations):
            return target_error
        if not operations:
            return _error("operations list is empty.", "invalid_args")
        ops = [op or {} for op in operations]
        # Scan every add/replace content BEFORE touching disk -- one poisoned op rejects the batch.
        for i, op in enumerate(ops):
            scan_error = op.get("action") in {"add", "replace"} and op.get("content") and _scan_memory_content(op["content"])
            if scan_error:
                return _error(f"Operation {i + 1}: {scan_error}", "scan_blocked")

        def _apply(entries, limit):
            working = list(entries)  # only committed if the whole batch validates
            matched = []  # per op, the entry a replace/patch/remove selected (None for add)
            produced = {}  # 1-based op index -> the entry a patch produced
            view = self._entries_for(target)
            for i, op in enumerate(ops):
                act = op.get("action")
                content, old_text = op.get("content") or op.get("new_text") or "", op.get("old_text") or ""
                if act != "patch":  # whole-entry actions; a patch span keeps its whitespace
                    content, old_text = content.strip(), old_text.strip()
                pos = f"Operation {i + 1} ({act or 'unknown'})"
                if self.context_id is not None:
                    # Scoped store: never shadow a global fact with a scoped copy
                    # (load-time healing would strip it), and refuse edits that
                    # target a read-only global entry visible in the view.
                    if act == "add" and content and (content in view or self._is_global_entry(target, content)):
                        matched.append(None)
                        continue
                    if act in ("replace", "remove", "patch") and old_text.strip() and any(
                        self._is_global_entry(target, e) for e in view if old_text in e
                    ):
                        return self._batch_failure(target, f"{pos}: {self._GLOBAL_READONLY_MSG}", "global_readonly")
                msg, previous_content, failure_class, after = self._apply_batch_op(
                    working, act, content, old_text, pos,
                    op.get("matched_entry"))
                if msg:  # a failed op's second value is its closest_entries (or None)
                    return self._batch_failure(target, msg, failure_class, closest_entries=previous_content)
                matched.append(previous_content)
                if after is not None:
                    produced[i + 1] = after
            if entries and not working:
                # #103419: a consolidation batch that removes the last entry would
                # commit an empty file as a normal successful write. Refuse; single
                # remove() is the deliberate-wipe path.
                label = self._path_for(target).name
                return self._batch_failure(target, (
                    f"Refusing to empty {label}: this batch would remove every entry from a "
                    f"previously non-empty store. Keep at least one entry — merge overlapping "
                    f"entries into a shorter one instead of removing the last one. To delete the "
                    f"final entry deliberately, use single remove() calls."), "would_empty")
            new_total = self._projected_len(target, working)  # budget: FINAL read view only
            if new_total > limit:
                return self._batch_failure(target, (
                    f"Applying all {len(operations)} operations {_over_limit(new_total, limit)} more. Add "
                    f"removes or shortening patches of other stale entries to the same batch, then retry."),
                    "over_budget")
            if not commit:
                return {"success": True, "matched_entries": matched}
            # op index -> full entry text its replace/remove selected (#117952), 1-based to
            # match the "Operation N" error numbering the model sees for failed ops.
            replaced, removed, patched = {}, {}, {}
            for i, (op, previous_content) in enumerate(zip(ops, matched), 1):
                if previous_content is not None:
                    result_map = {"replace": replaced, "patch": patched, "remove": removed}.get(op.get("action"))
                    if result_map is not None:
                        result_map[i] = previous_content
            replaced_fields = {"replaced_entries": replaced} if replaced else {}
            if patched:
                replaced_fields["patched_entries"] = patched
                replaced_fields["patched_entries_after"] = produced
            if removed:
                replaced_fields["removed_entries"] = removed
            return working, f"Applied {len(operations)} operation(s).", replaced_fields
        return self._mutate(target, _apply, skip_drift=not commit)

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """Frozen load-time snapshot (NOT live state — mid-session writes don't touch
        it, preserving the prefix cache); None if empty."""
        return self._system_prompt_snapshot.get(target, "") or None

    def _success_response(self, target: str, message: str | None = None, **extra) -> dict[str, Any]:
        """TERMINAL and WITHOUT the entries list: echoing entries invites the model to
        "find more to fix" and re-issue the same ops. A successful write resets the
        per-turn failure budget. ``**extra`` mirrors ``_error``'s convention — e.g. the
        full text a replace overwrote (#117952), deliberately visible despite the
        no-entries rule: silent data loss is the failure this field exists to prevent."""
        # A successful write means the consolidation loop made progress, so the per-turn failure budget
        # resets (the cap counts consecutive failures, not lifetime ones within a turn) (#42405).
        self._consolidation_failures = 0
        return {"success": True, "done": True, "target": target,
                "usage": self._usage_pct(target, self._char_count(target)),
                "entry_count": len(self._entries_for(target)), **({"message": message} if message else {}),
                **extra,
                "note": "Write saved. This update is complete — do not repeat it."}

    def _render_block(self, target: str, entries: list[str]) -> str:
        """System prompt block: header + usage indicator + entries ("" when empty)."""
        if not entries:
            return ""
        content, sep = ENTRY_DELIMITER.join(entries), "═" * 46
        title = MEMORY_BLOCK_HEADERS["user" if target == "user" else "memory"]
        return f"{sep}\n{title} [{self._usage_pct(target, len(content))}]\n{sep}\n{content}"

    @staticmethod
    def _read_raw_checked(path: Path) -> tuple[str, bool]:
        """``(raw, read_ok)``; ``read_ok`` is False ONLY when the file EXISTS but can't be
        read. Decoding stays STRICT (``errors="replace"`` would hand callers a lossy view
        a save then persists); ``utf-8-sig`` strips a Notepad BOM off the first entry."""
        if not path.exists():
            return "", True
        try:
            # utf-8-sig strips a leading UTF-8 BOM (Notepad-edited memory files on Windows) and is
            # byte-identical to utf-8 otherwise. Plain utf-8 kept U+FEFF glued to the first entry,
            # corrupting matching/dedup for that entry forever (#10878 / PR #10888). Decode errors stay
            # STRICT on purpose: errors="replace" would hand read-modify-write callers a lossy view that a
            # subsequent save persists over the real bytes — the wipe class documented above. Undecodable
            # bytes must surface as read_ok=False.
            return path.read_text(encoding="utf-8-sig"), True
        except (OSError, UnicodeDecodeError):
            return "", False

    @staticmethod
    def _parse_entries(raw: str) -> list[str]:
        """Stripped, non-empty entries; splits on the FULL delimiter so a bare "§" survives."""
        return [e for e in (x.strip() for x in raw.split(ENTRY_DELIMITER)) if e]

    @staticmethod
    def _read_file(path: Path) -> list[str]:
        """Entries of a memory file ([] on any error). Read-only callers only; mutation
        paths use ``_read_raw_checked`` so they can refuse to overwrite an unreadable file."""
        return MemoryStore._parse_entries(MemoryStore._read_raw_checked(path)[0])

    @staticmethod
    def _write_file(path: Path, entries: list[str]):
        """Atomic temp-file + rename: readers never see a truncated file. Callers
        hold ``_file_lock`` (via ``_mutate``): a bare write from an earlier snapshot
        drops concurrent entries (#119668)."""
        try:
            atomic_write_text(path, ENTRY_DELIMITER.join(entries), tmp_prefix=".mem_")
        except OSError as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")

    def _detect_external_drift(self, target: str, raw: str) -> Optional[str]:
        """``.bak.<ts>`` snapshot path if *raw* shows external drift, else None. The one
        signal is an entry over the whole-file limit: no tool-written entry can be, so it is
        free-form text appended without delimiters, and replacing that "entry" would discard
        it. Layout differences (blank lines around ``§``, empty entries, padding) are NOT
        drift: parsing only strips whitespace, so the flush loses nothing but whitespace, and
        refusing there blocked every replace/remove on a hand-formatted file for good (#107270)."""
        if max(map(len, self._parse_entries(raw)), default=0) <= self._char_limit(target):
            return None
        path = self._path_for(target)
        bak_path = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
        try:
            bak_path.write_text(raw, encoding="utf-8")
        except OSError:
            return str(bak_path) + " (BACKUP FAILED — file unchanged on disk)"
        return str(bak_path)
