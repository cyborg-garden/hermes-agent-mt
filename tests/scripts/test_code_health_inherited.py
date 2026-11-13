"""Fork: an upstream-sync PR is not charged for code it merges unchanged from upstream.

``HERMES_HEALTH_INHERITED_FROM`` names the upstream release a sync PR merges. A finding is dropped
only when the same head, judged against that release, would not report it. The fork's own code
stays charged, in a new file and inside a file that came from upstream.
"""

from __future__ import annotations

from scripts.code_health import cli, inherited
from tests.scripts.test_code_health_cli import _SWALLOW, _commit, _git, _ratchet_repo

_SECOND = "\n\ndef swallow_again():\n    try:\n        pass\n    except Exception:\n        pass\n"


def _sync_repo(tmp_path):
    """main -> an "upstream" branch that adds a swallowing function -> merged back as the sync head."""
    repo, base = _ratchet_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "upstream")
    upstream = _commit(repo, {"pkg/up.py": _SWALLOW})
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", "sync", "upstream")
    return repo, base, upstream


def _run(repo, base, monkeypatch, capsys, upstream=None):
    monkeypatch.chdir(repo)
    if upstream:
        monkeypatch.setenv(inherited.ENV, upstream)
    else:
        monkeypatch.delenv(inherited.ENV, raising=False)
    code = cli.main(["--base", base])
    return code, capsys.readouterr().out


def test_code_merged_unchanged_from_upstream_is_charged_only_without_the_upstream_revision(
        tmp_path, monkeypatch, capsys):
    repo, base, upstream = _sync_repo(tmp_path)

    code, out = _run(repo, base, monkeypatch, capsys)
    assert code == 1 and "pkg/up.py" in out and "BLE001" in out

    code, out = _run(repo, base, monkeypatch, capsys, upstream)
    assert code == 0, out
    assert "pkg/up.py" not in out and "inherited from upstream" in out


def test_the_forks_own_code_is_still_charged_on_a_sync(tmp_path, monkeypatch, capsys):
    repo, base, upstream = _sync_repo(tmp_path)
    # The fork adds a violation of its own to a file that came from upstream, and a new file.
    _commit(repo, {"pkg/up.py": _SWALLOW + _SECOND, "pkg/fork.py": _SWALLOW})

    code, out = _run(repo, base, monkeypatch, capsys, upstream)

    assert code == 1, out
    assert "swallow_again" in out and "pkg/fork.py" in out
    # upstream's own function in the same file (its except is on line 4) is still not charged
    assert "pkg/up.py:4 " not in out


def test_an_unknown_upstream_revision_fails_the_run_instead_of_forgiving_everything(
        tmp_path, monkeypatch, capsys):
    repo, base, _upstream = _sync_repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setenv(inherited.ENV, "0" * 40)
    assert cli.main(["--base", base]) == 2  # a broken run, never a pass
