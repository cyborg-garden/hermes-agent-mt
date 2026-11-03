"""Structural guard: every subprocess-env builder must inject a fresh GH_TOKEN.

The defect this file exists to prevent has already happened once. The GH_TOKEN
injection was added to the terminal paths (``_sanitize_subprocess_env``,
``_make_run_env``) and ``hermes_subprocess_env`` — a sibling env builder for
the browser worker, ACP/codex/copilot executors, the TUI Node host, dep-ensure
and the detached gateway — was simply not updated. Nothing caught it, because
the omission does not fail: ``gh`` falls back to the boot-written
``~/.config/gh/hosts.yml``, whose GitHub App token is dead within the hour. The
symptom is a 401 deep inside a tool, weeks later, with a credential file
sitting right there looking perfectly plausible.

A behavioural test per known builder cannot catch the NEXT builder somebody
adds. This one reads the source instead: any function that finalizes HOME via
``apply_subprocess_home_env`` is, by definition, building an environment for a
spawned process, and must therefore also call ``_inject_github_app_gh_token``.

Non-vacuity is asserted explicitly — a test that silently finds zero builders
would pass forever after a rename.
"""
from __future__ import annotations

import ast
from pathlib import Path

import tools.environments.local as local_mod

HOME_FINALIZER = "apply_subprocess_home_env"
INJECTOR = "_inject_github_app_gh_token"

# Upstream v0.22.4 centralized child-env construction: every scrubbing spawn
# surface funnels through ``_finalize_child_env`` (the guard every child-env
# builder shares), which finalizes HOME via ``_apply_profile_home`` ->
# ``apply_subprocess_home_env``. The injection therefore lives once, in that
# shared guard, and every builder must reach it.
SHARED_GUARD = "_finalize_child_env"
HOME_WRAPPER = "_apply_profile_home"

# Builders known to exist today. Kept as a floor, not as the definition: the
# scan finds builders, this set only proves the scan is actually finding them.
KNOWN_BUILDERS = {
    "_sanitize_subprocess_env",
    "hermes_subprocess_env",
    "_make_run_env",
}

# Functions allowed to finalize HOME without going through the shared guard:
# the guard itself, the wrapper, and build_subprocess_env's scrub_secrets=False
# branch, which keeps the base env byte-for-byte (git credential flows,
# bws/op) — its ambient GitHub credentials are never stripped, so there is no
# dead hosts.yml fallback to defend against.
HOME_FINALIZER_ALLOWLIST = {SHARED_GUARD, HOME_WRAPPER, "build_subprocess_env"}


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _functions() -> dict[str, ast.AST]:
    source = Path(local_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _reaches(funcs: dict[str, ast.AST], start: str, target: str) -> bool:
    """Does ``start`` call ``target`` directly or through module-local helpers?"""
    seen: set[str] = set()
    stack = [start]
    while stack:
        name = stack.pop()
        if name in seen or name not in funcs:
            continue
        seen.add(name)
        calls = _called_names(funcs[name])
        if target in calls:
            return True
        stack.extend(calls & funcs.keys())
    return False


def _env_builders() -> dict[str, ast.AST]:
    """Functions that finalize a child's HOME (and so build a spawn env)."""
    funcs = _functions()
    return {
        name: node for name, node in funcs.items()
        if name not in HOME_FINALIZER_ALLOWLIST
        and _reaches(funcs, name, HOME_FINALIZER)
    }


def test_scan_finds_the_known_builders():
    """Non-vacuity floor. Without this, a rename of the finalizer would make
    every assertion below pass over an empty set."""
    found = set(_env_builders())
    assert KNOWN_BUILDERS <= found, f"scan lost builders: {KNOWN_BUILDERS - found}"


def test_every_env_builder_injects_a_fresh_gh_token():
    funcs = _functions()
    offenders = sorted(
        name for name in _env_builders() if not _reaches(funcs, name, INJECTOR)
    )
    assert not offenders, (
        f"{offenders} build a subprocess environment but never reach {INJECTOR}(). "
        f"Without it the spawn authenticates to GitHub out of the boot-written "
        f"~/.config/gh/hosts.yml, which holds a GitHub App token that expires "
        f"within the hour and is refreshed on no schedule the spawn controls. "
        f"That failure is silent — the credential is present, merely dead."
    )


def test_injection_runs_after_home_is_final():
    """Ordering is load-bearing: the mint layer derives the agent's .env dir as
    HOME's parent, so injecting before HOME is finalized reads the wrong home
    (or none) and silently declines to inject. The injection lives in the
    shared guard; it must come after the guard finalizes HOME."""
    guard = _functions()[SHARED_GUARD]
    home_lines = [
        c.lineno for c in ast.walk(guard)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        and c.func.id in (HOME_WRAPPER, HOME_FINALIZER)
    ]
    inject_lines = [
        c.lineno for c in ast.walk(guard)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        and c.func.id == INJECTOR
    ]
    assert home_lines, f"{SHARED_GUARD} no longer finalizes HOME"
    assert inject_lines, f"{SHARED_GUARD} no longer calls {INJECTOR}()"
    assert min(inject_lines) > max(home_lines), (
        f"{SHARED_GUARD}: {INJECTOR}() must run after HOME is finalized"
    )
