"""Contract: the optional per-agent apt layer (HERMES_EXTRA_APT_PACKAGES).

Swarm Map renders an agent's ``extraAptPackages`` as this build arg so one
agent (e.g. a file-converting public bot that needs headless LibreOffice) can
carry system packages the rest of the fleet does not. The layer must:

* default to empty, so every other agent's image is unchanged;
* sit AFTER the heavy npm/uv dependency layers, so an agent that sets it
  still shares their build cache;
* validate names before they reach apt (the value is shell-split).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"


def _lines() -> list[str]:
    return DOCKERFILE.read_text().splitlines()


def _index(pattern: str) -> int:
    hits = [i for i, line in enumerate(_lines()) if re.search(pattern, line)]
    assert len(hits) == 1, f"expected exactly one line matching {pattern!r}, got {hits}"
    return hits[0]


def test_arg_defaults_to_empty() -> None:
    assert "ARG HERMES_EXTRA_APT_PACKAGES=" in _lines()


def test_layer_sits_after_dependency_layers_and_before_source() -> None:
    arg = _index(r"^ARG HERMES_EXTRA_APT_PACKAGES=")
    assert arg > _index(r"^RUN uv sync --frozen")
    assert arg > _index(r"^RUN npm install ")
    assert arg < _index(r"^COPY --link --chmod=a\+rX,go-w \. \.")


def _layer() -> str:
    text = DOCKERFILE.read_text()
    start = text.index("ARG HERMES_EXTRA_APT_PACKAGES=")
    end = text.index("\n\n", start)
    return text[start:end]


def _accepts(name: str) -> bool:
    """Run the layer's own validation loop (everything before apt) in sh."""
    body = _layer().split("RUN ", 1)[1]
    check = body.split("apt-get", 1)[0].replace("\\\n", "\n").rstrip().removesuffix("&&")
    script = check + "\nfi\n"
    r = subprocess.run(["sh", "-c", script], env={"HERMES_EXTRA_APT_PACKAGES": name, "LC_ALL": "C"},
                       capture_output=True, text=True)
    return r.returncode == 0


def test_validator_accepts_real_package_names() -> None:
    for name in ("libreoffice-writer-nogui", "pandoc", "fonts-liberation2", "g++", "libc6.1"):
        assert _accepts(name), name


def test_validator_rejects_shell_and_apt_tricks() -> None:
    for name in ("-o", "--allow-unauthenticated", "foo;rm", "$(id)", "Foo", "a/b", "pkg=1.0", "../x"):
        assert not _accepts(name), name
