"""github-broker.json sits beside .env and is just as live, so every list that names .env names it.

Upstream merges have rewritten these lists and moved them between files. This holds the entry
through that: a list that loses it fails here, by name, and not only in a test of that feature.
"""
from __future__ import annotations

import re
import tarfile
from pathlib import Path

import pytest

BROKER_SECRET = "github-broker.json"
REPO = Path(__file__).resolve().parents[2]


def _guards():
    from acp_adapter.edit_approval import SENSITIVE_AUTO_APPROVE_NAMES
    from agent.file_safety import _CREDENTIAL_FILE_NAMES
    from gateway.platforms.base import _ROOT_CREDENTIAL_PATHS
    from hermes_cli.backup import _SECRET_FILE_NAMES
    from hermes_cli.profile_distribution import USER_OWNED_EXCLUDE
    from hermes_cli.profiles import (
        _DEFAULT_EXPORT_EXCLUDE_ROOT,
        _EXPORT_CREDENTIAL_FILES,
        PROFILE_CREDENTIAL_PATHS,
    )
    from hermes_cli.web_routers.files import _SENSITIVE_MANAGED_FILE_BASENAMES
    from tools.checkpoint_manager import DEFAULT_EXCLUDES

    return {
        "acp auto-approve": SENSITIVE_AUTO_APPROVE_NAMES,
        "file_safety read deny": _CREDENTIAL_FILE_NAMES,
        "media delivery deny": _ROOT_CREDENTIAL_PATHS,
        "backup restore modes": _SECRET_FILE_NAMES,
        "distribution install": USER_OWNED_EXCLUDE,
        "default profile export": _DEFAULT_EXPORT_EXCLUDE_ROOT,
        "export at any depth": _EXPORT_CREDENTIAL_FILES,
        "profile credential stores": PROFILE_CREDENTIAL_PATHS,
        "dashboard files api": _SENSITIVE_MANAGED_FILE_BASENAMES,
        "checkpoint snapshots": DEFAULT_EXCLUDES,
    }


@pytest.mark.parametrize("name", sorted(_guards()))
def test_every_list_that_names_env_names_the_broker_secret(name):
    assert BROKER_SECRET in _guards()[name], f"{name} lost {BROKER_SECRET}"


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
