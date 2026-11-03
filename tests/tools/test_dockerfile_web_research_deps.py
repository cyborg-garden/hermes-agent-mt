"""Contract: the Docker image bakes the web-research backend SDKs.

The image sets ``HERMES_DISABLE_LAZY_INSTALLS=1``, so an opt-in backend whose
SDK is not installed at build time can never work in a container. Fleet
agents use the self-hosted Firecrawl (``web_extract``) and the keyless ddgs
search fallback, so both SDKs must come in through the locked
``pm.build_env`` venv build (transitives pinned by uv.lock), not a runtime
install. The venv is sealed, so a later ``uv pip install`` is not an option.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# extra name -> (distribution, import name used by plugins/web/<backend>)
WEB_RESEARCH_EXTRAS = {
    "firecrawl": ("firecrawl-py", "firecrawl"),
    "ddgs": ("ddgs", "ddgs"),
}


def _uv_sync_line() -> str:
    """The pm.build_env RUN instruction, continuation lines joined."""
    text = DOCKERFILE.read_text().replace("\\\n", " ")
    lines = [
        line for line in text.splitlines()
        if line.startswith("RUN python3 -m pm.build_env ")
    ]
    assert len(lines) == 1, f"expected one pm.build_env venv build in Dockerfile, got {lines}"
    return lines[0]


def _optional_deps() -> dict[str, list[str]]:
    data = tomllib.loads(PYPROJECT.read_text())
    return data["project"]["optional-dependencies"]


def test_dockerfile_bakes_web_research_extras() -> None:
    line = _uv_sync_line()
    for extra in WEB_RESEARCH_EXTRAS:
        assert re.search(rf"--extra {re.escape(extra)}(\s|$)", line), (
            f"Dockerfile uv sync is missing --extra {extra}; with lazy installs "
            f"disabled the {extra} backend can never import in the image"
        )


def test_web_research_extras_are_exact_pinned() -> None:
    extras = _optional_deps()
    for extra, (dist, _module) in WEB_RESEARCH_EXTRAS.items():
        assert extra in extras, f"pyproject has no [{extra}] extra"
        specs = extras[extra]
        assert any(re.fullmatch(rf"{re.escape(dist)}==[0-9][0-9A-Za-z.]*", s) for s in specs), (
            f"[{extra}] must exact-pin {dist} (no ranges), got {specs}"
        )


def test_web_research_extras_have_pm_import_anchors() -> None:
    """pm decides an extra is installed by importing its anchor module."""
    from pm.extras import ANCHORS

    for extra, (_dist, module) in WEB_RESEARCH_EXTRAS.items():
        assert ANCHORS.get(extra) == module, (extra, ANCHORS.get(extra))


def test_web_research_extras_stay_out_of_all() -> None:
    all_specs = _optional_deps()["all"]
    for extra in WEB_RESEARCH_EXTRAS:
        assert f"hermes-agent[{extra}]" not in all_specs
