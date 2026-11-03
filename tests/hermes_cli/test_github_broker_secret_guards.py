"""github-broker.json sits beside .env and is just as live, so every list that names .env names it.

Upstream merges have rewritten these lists and moved them between files. This holds the entry
through that: a list that loses it fails here, by name, and not only in a test of that feature.
"""
from __future__ import annotations

import importlib
import re
import tarfile
from pathlib import Path

import pytest

BROKER_SECRET = "github-broker.json"
REPO = Path(__file__).resolve().parents[2]


GUARDS = [
    ("acp_adapter.edit_approval", "SENSITIVE_AUTO_APPROVE_NAMES"),      # ACP auto-approve
    ("hermes_cli.backup", "_SECRET_FILE_NAMES"),                        # restore file modes
    ("hermes_cli.profile_distribution", "USER_OWNED_EXCLUDE"),          # distribution install
    ("hermes_cli.profiles", "_DEFAULT_EXPORT_EXCLUDE_ROOT"),            # default profile export
    ("hermes_cli.web_server", "_SENSITIVE_MANAGED_FILE_BASENAMES"),     # dashboard files API
    ("tools.checkpoint_manager", "DEFAULT_EXCLUDES"),                   # checkpoint snapshots
]


@pytest.mark.parametrize("module,name", GUARDS)
def test_every_list_that_names_env_names_the_broker_secret(module, name):
    assert BROKER_SECRET in getattr(importlib.import_module(module), name)


def test_read_and_write_guards_name_it(tmp_path, monkeypatch):
    # These lists sit inside functions on this base, so ask the guards themselves.
    from agent.file_safety import get_read_block_error, is_write_denied

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    secret = tmp_path / BROKER_SECRET
    assert get_read_block_error(str(secret))
    assert is_write_denied(str(secret))


def test_stage2_chown_list_names_it():
    # A shell list, so it can't be imported. Root-owned, the agent couldn't read its own secret.
    hook = (REPO / "docker" / "stage2-hook.sh").read_text(encoding="utf-8")
    assert re.search(r"for f in \\\n\s+auth\.json auth\.lock \.env github-broker\.json", hook)


def test_export_drops_it_at_any_depth(tmp_path, monkeypatch):
    from hermes_cli.profiles import export_profile

    profile = tmp_path / "profiles" / "coder"
    nested = profile / "skills" / "demo"
    nested.mkdir(parents=True)
    (profile / "config.yaml").write_text("ok")
    (profile / BROKER_SECRET).write_text('{"secret": "x"}')
    (nested / BROKER_SECRET).write_text('{"secret": "x"}')
    (nested / "SKILL.md").write_text("kept")
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "profiles")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda n: profile)
    monkeypatch.setattr("hermes_cli.profiles.validate_profile_name", lambda n: None)

    output = tmp_path / "out.tar.gz"
    export_profile("coder", str(output))

    with tarfile.open(str(output), "r:gz") as tf:
        names = tf.getnames()
    assert "coder/skills/demo/SKILL.md" in names
    assert not [n for n in names if n.endswith(BROKER_SECRET)]
