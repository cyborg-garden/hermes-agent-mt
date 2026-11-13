"""Fork: the broken-trampoline self-heal finds the real git on Git for Windows 2.56+.

2.56 renamed ``mingw64`` to ``ucrt64`` (git-for-windows v2.56.0 release notes), so the locator
found nothing on a current install and ``hermes update`` fell back to the ZIP path.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath

import pytest

from hermes_cli import update_cmd_git


@pytest.mark.parametrize("arch", ["ucrt64", "mingw64", "clangarm64"])
def test_locator_finds_git_core_under_each_git_for_windows_layout(monkeypatch, arch):
    real = PureWindowsPath(rf"C:\Program Files\Git\{arch}\libexec\git-core\git.exe")
    monkeypatch.setattr(update_cmd_git, "_portable_git_candidates", list)
    monkeypatch.setattr(Path, "exists", lambda self: PureWindowsPath(str(self)) == real)
    monkeypatch.setattr(update_cmd_git, "_probe_fork_bomb", lambda cmd: False)

    found = update_cmd_git._locate_real_git()

    assert found is not None and PureWindowsPath(str(found)) == real

