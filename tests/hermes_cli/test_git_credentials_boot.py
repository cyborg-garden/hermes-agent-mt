"""Tests for hermes_cli.git_credentials_boot — boot-time provisioning of
per-agent git credentials from the agent's own configured GitHub token.

This is the runtime-side port of HSM's lib/services/git-credentials.ts.
It runs in-process at container boot (a cont-init.d step), where the agent's
.env is readable on the mounted volume and HERMES_HOME / get_subprocess_home()
are known directly — unlike the tool subprocess, which cannot see the token
(GITHUB_TOKEN is blocklisted from the subprocess env by design).

Tests run against a fake $HERMES_HOME under tmp_path. No container, no git.
"""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

from hermes_cli.git_credentials_boot import (
    build_gh_hosts_content,
    build_git_config_content,
    build_git_credentials_content,
    provision_all,
    provision_git_credentials,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_env(home: Path, body: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(body)


def _cred_path(home: Path) -> Path:
    return home / "home" / ".git-credentials"


def _cfg_path(home: Path) -> Path:
    return home / "home" / ".gitconfig"


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


def _gh_hosts_path(home: Path) -> Path:
    return home / "home" / ".config" / "gh" / "hosts.yml"


# ---------------------------------------------------------------------------
# Pure content builders
# ---------------------------------------------------------------------------


def test_credentials_content_uses_x_access_token_https_line():
    assert build_git_credentials_content("ghp_abc") == (
        "https://x-access-token:ghp_abc@github.com\n"
    )


def test_config_content_has_store_helper_identity_and_insteadof():
    cfg = build_git_config_content(name="cyborg", email="cyborg@users.noreply.github.com")
    assert "helper = store" in cfg
    assert "name = cyborg" in cfg
    assert "email = cyborg@users.noreply.github.com" in cfg
    # ssh-style remotes get rewritten through HTTPS so the agent doesn't die on
    # "Host key verification failed" when it reaches for an SSH URL.
    assert 'insteadOf = git@github.com:' in cfg
    assert 'insteadOf = ssh://git@github.com/' in cfg


# ---------------------------------------------------------------------------
# provision_git_credentials — single home
# ---------------------------------------------------------------------------


def test_writes_credentials_and_config_from_env_token(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_secret\n")

    result = provision_git_credentials(tmp_path)

    assert result.provisioned is True
    assert result.source == "GITHUB_PAT"
    assert _cred_path(tmp_path).read_text() == "https://x-access-token:ghp_secret@github.com\n"
    assert "helper = store" in _cfg_path(tmp_path).read_text()


def test_credentials_file_is_chmod_600(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_secret\n")
    provision_git_credentials(tmp_path)
    assert _mode(_cred_path(tmp_path)) == 0o600


def test_config_file_is_chmod_644(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_secret\n")
    provision_git_credentials(tmp_path)
    assert _mode(_cfg_path(tmp_path)) == 0o644


def test_token_precedence_pat_beats_github_token_beats_gh_token(tmp_path):
    _write_env(tmp_path, "GH_TOKEN=gh_v\nGITHUB_TOKEN=ght_v\nGITHUB_PAT=pat_v\n")
    result = provision_git_credentials(tmp_path)
    assert result.source == "GITHUB_PAT"
    assert "pat_v" in _cred_path(tmp_path).read_text()


def test_falls_back_to_gh_token_when_only_one_present(tmp_path):
    _write_env(tmp_path, "GH_TOKEN=gh_only\n")
    result = provision_git_credentials(tmp_path)
    assert result.source == "GH_TOKEN"
    assert "gh_only" in _cred_path(tmp_path).read_text()


def test_strips_quotes_and_whitespace_from_token(tmp_path):
    _write_env(tmp_path, 'GITHUB_PAT="ghp_quoted"  \n')
    provision_git_credentials(tmp_path)
    assert _cred_path(tmp_path).read_text() == "https://x-access-token:ghp_quoted@github.com\n"


def test_no_token_is_inert(tmp_path):
    _write_env(tmp_path, "OPENAI_API_KEY=sk-whatever\n")
    result = provision_git_credentials(tmp_path)
    assert result.provisioned is False
    assert result.reason == "no GitHub token configured"
    assert not _cred_path(tmp_path).exists()
    assert not (tmp_path / "home").exists()


def test_missing_env_is_inert(tmp_path):
    result = provision_git_credentials(tmp_path)
    assert result.provisioned is False
    assert not _cred_path(tmp_path).exists()


def test_empty_token_value_is_inert(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=\n")
    result = provision_git_credentials(tmp_path)
    assert result.provisioned is False


# ---------------------------------------------------------------------------
# Manage-if-ours (the data-loss guard from HSM PR #53, re-based on ownership)
#
# The guard's intent — never clobber a config we did not write — is unchanged.
# What changed is the test for it: "the file exists" also protected our OWN
# stale output, which is how a 1h token survived 16 days in hosts.yml and a
# git identity kept authoring as "data". Ownership separates the two cases.
# ---------------------------------------------------------------------------


def test_does_not_clobber_foreign_credentials_file(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_new\n")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".git-credentials").write_text("https://human:theirpat@github.com\n")

    result = provision_git_credentials(tmp_path)

    assert result.provisioned is True  # gitconfig + gh still provisioned
    assert (
        home / ".git-credentials"
    ).read_text() == "https://human:theirpat@github.com\n"  # untouched


def test_does_not_clobber_foreign_gitconfig(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_new\n")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text("[user]\n\tname = human\n")

    result = provision_git_credentials(tmp_path)

    assert result.provisioned is True
    assert (home / ".gitconfig").read_text() == "[user]\n\tname = human\n"  # untouched


def test_no_op_when_everything_already_current(tmp_path):
    """The true no-op branch: a second boot with nothing changed writes nothing.

    Locks idempotence — manage-if-ours must not rewrite identical content, or
    every boot would churn mtimes and make "when did this last rotate?"
    unanswerable.
    """
    _write_env(tmp_path, "GITHUB_PAT=ghp_new\n")
    provision_git_credentials(tmp_path)
    before = {
        p: p.read_text()
        for p in (_cred_path(tmp_path), _cfg_path(tmp_path), _gh_hosts_path(tmp_path))
    }

    result = provision_git_credentials(tmp_path)

    assert result.provisioned is False
    assert "already current" in (result.reason or "")
    assert all(p.read_text() == text for p, text in before.items())


def test_does_not_clobber_foreign_gh_hosts(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_new\n")
    home = tmp_path / "home"
    (home / ".config" / "gh").mkdir(parents=True)
    (home / ".config" / "gh" / "hosts.yml").write_text(
        "github.com:\n    oauth_token: human_token\n    user: a-human\n"
    )

    provision_git_credentials(tmp_path)

    assert "human_token" in _gh_hosts_path(tmp_path).read_text()


def test_rotated_pat_is_picked_up_on_next_boot(tmp_path):
    """The PAT half of the same fossil: apply-if-absent kept serving the OLD
    token after .env was rotated, until someone deleted the files by hand."""
    _write_env(tmp_path, "GITHUB_PAT=ghp_old\n")
    provision_git_credentials(tmp_path)
    assert "ghp_old" in _cred_path(tmp_path).read_text()

    _write_env(tmp_path, "GITHUB_PAT=ghp_rotated\n")
    result = provision_git_credentials(tmp_path)

    assert result.provisioned is True
    assert "ghp_rotated" in _cred_path(tmp_path).read_text()
    assert "ghp_old" not in _cred_path(tmp_path).read_text()
    assert "oauth_token: ghp_rotated" in _gh_hosts_path(tmp_path).read_text()


def test_stale_identity_in_our_own_gitconfig_is_corrected(tmp_path, monkeypatch):
    """The `name = data` fossil, exactly as found on the fleet 2026-08-09.

    HERMES_AGENT_NAME was wired up in a later build, but the .gitconfig written
    by the earlier one already existed, so the corrected resolution could never
    reach disk.
    """
    _write_env(tmp_path, "GITHUB_PAT=ghp_secret\n")
    home = tmp_path / "home"
    home.mkdir()
    # Byte-shape of a real pre-marker .gitconfig from hermes-matilde.
    (home / ".gitconfig").write_text(
        "[credential]\n\thelper = store\n"
        "[user]\n\tname = data\n\temail = data@users.noreply.github.com\n"
        '[url "https://github.com/"]\n'
        "\tinsteadOf = git@github.com:\n"
        "\tinsteadOf = ssh://git@github.com/\n"
    )
    monkeypatch.setenv("HERMES_AGENT_NAME", "matilde")

    provision_all(tmp_path)

    cfg = _cfg_path(tmp_path).read_text()
    assert "name = matilde" in cfg
    assert "name = data" not in cfg


def test_legacy_unmarked_files_are_recognized_as_ours(tmp_path):
    """Every already-deployed agent has files written before MANAGED_MARKER
    existed. If those read as foreign, the fix would freeze the fossil instead
    of thawing it."""
    from hermes_cli.git_credentials_boot import (
        is_managed_gh_hosts,
        is_managed_git_credentials,
        is_managed_gitconfig,
    )

    legacy_cfg = (
        "[credential]\n\thelper = store\n[user]\n\tname = data\n"
        '[url "https://github.com/"]\n\tinsteadOf = git@github.com:\n'
        "\tinsteadOf = ssh://git@github.com/\n"
    )
    legacy_app_cfg = (
        "[credential]\n\thelper =\n"
        '\thelper = "!/opt/hermes/.venv/bin/python -m hermes_cli.github_app_token get-credential"\n'
        '[user]\n\tname = data\n[url "https://github.com/"]\n'
        "\tinsteadOf = git@github.com:\n\tinsteadOf = ssh://git@github.com/\n"
    )
    assert is_managed_gitconfig(legacy_cfg)
    assert is_managed_gitconfig(legacy_app_cfg)
    assert is_managed_gh_hosts(
        "github.com:\n    oauth_token: ghs_x\n    git_protocol: https\n    user: x-access-token\n"
    )
    assert is_managed_git_credentials("https://x-access-token:ghs_x@github.com\n")

    # …and a human's own files still read as foreign.
    assert not is_managed_gitconfig("[user]\n\tname = human\n\temail = h@example.com\n")
    assert not is_managed_gh_hosts(
        "github.com:\n    oauth_token: gho_x\n    user: a-human\n"
    )
    assert not is_managed_git_credentials("https://human:theirpat@github.com\n")


def test_force_overwrites_existing(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_new\n")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".git-credentials").write_text("PRE-EXISTING\n")

    result = provision_git_credentials(tmp_path, force=True)

    assert result.provisioned is True
    assert "ghp_new" in (home / ".git-credentials").read_text()


# ---------------------------------------------------------------------------
# provision_all — HERMES_HOME root (default profile) + named profiles
# ---------------------------------------------------------------------------


def test_provision_all_does_root_and_each_named_profile(tmp_path):
    # Default profile = HERMES_HOME root.
    _write_env(tmp_path, "GITHUB_PAT=root_tok\n")
    # Named profiles live under profiles/<name>/, each with its own .env.
    _write_env(tmp_path / "profiles" / "cyborg", "GITHUB_PAT=cyborg_tok\n")
    _write_env(tmp_path / "profiles" / "osint", "GH_TOKEN=osint_tok\n")

    results = provision_all(tmp_path)

    provisioned = {r.home: r for r in results if r.provisioned}
    assert tmp_path in provisioned
    assert (tmp_path / "profiles" / "cyborg") in provisioned
    assert (tmp_path / "profiles" / "osint") in provisioned
    assert "root_tok" in _cred_path(tmp_path).read_text()
    assert "cyborg_tok" in _cred_path(tmp_path / "profiles" / "cyborg").read_text()
    assert "osint_tok" in _cred_path(tmp_path / "profiles" / "osint").read_text()


def test_provision_all_skips_profile_without_token(tmp_path):
    _write_env(tmp_path / "profiles" / "notoken", "OPENAI_API_KEY=sk-x\n")
    results = provision_all(tmp_path)
    notoken = tmp_path / "profiles" / "notoken"
    assert all(not r.provisioned for r in results if r.home == notoken)
    assert not _cred_path(notoken).exists()


def test_provision_all_handles_no_profiles_dir(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=root_tok\n")
    results = provision_all(tmp_path)
    assert any(r.provisioned and r.home == tmp_path for r in results)


def test_provision_all_uses_hermes_agent_name_for_root_identity(tmp_path, monkeypatch):
    # The default profile (HERMES_HOME root) basename is "data" in production
    # (/opt/data); HERMES_AGENT_NAME carries the real agent name, so commits
    # are authored as the agent, not "data". (HSM parity.)
    monkeypatch.setenv("HERMES_AGENT_NAME", "cryptids")
    _write_env(tmp_path, "GITHUB_PAT=root_tok\n")

    provision_all(tmp_path)

    cfg = _cfg_path(tmp_path).read_text()
    assert "name = cryptids" in cfg
    assert "email = cryptids@users.noreply.github.com" in cfg


def test_provision_all_profile_identity_uses_profile_name_not_env(tmp_path, monkeypatch):
    # A named profile is its own agent — its identity is the profile name,
    # never the container-level HERMES_AGENT_NAME.
    monkeypatch.setenv("HERMES_AGENT_NAME", "rootname")
    _write_env(tmp_path / "profiles" / "osint", "GITHUB_PAT=tok\n")

    provision_all(tmp_path)

    cfg = _cfg_path(tmp_path / "profiles" / "osint").read_text()
    assert "name = osint" in cfg
    assert "name = rootname" not in cfg


# ---------------------------------------------------------------------------
# gh CLI auth (so `gh` works in the tool subprocess too, not just `git`)
# ---------------------------------------------------------------------------


def test_gh_hosts_content_has_oauth_token_and_https():
    out = build_gh_hosts_content("ghp_abc")
    assert "github.com:" in out
    assert "oauth_token: ghp_abc" in out
    assert "git_protocol: https" in out


def test_writes_gh_hosts_from_env_token(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_secret\n")
    provision_git_credentials(tmp_path)
    assert "oauth_token: ghp_secret" in _gh_hosts_path(tmp_path).read_text()


def test_gh_hosts_is_chmod_600(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_secret\n")
    provision_git_credentials(tmp_path)
    assert _mode(_gh_hosts_path(tmp_path)) == 0o600


def test_gh_provisioned_even_when_git_already_present(tmp_path):
    # The fleet-heal case: an agent provisioned by an older build has git set up
    # but no gh hosts.yml. gh must still be written (git and gh are independent).
    _write_env(tmp_path, "GITHUB_PAT=ghp_secret\n")
    sub = tmp_path / "home"
    sub.mkdir(parents=True)
    (sub / ".gitconfig").write_text("# pre-existing git setup\n")

    result = provision_git_credentials(tmp_path)

    assert result.provisioned is True
    assert "oauth_token: ghp_secret" in _gh_hosts_path(tmp_path).read_text()
    # git config left untouched — not ours (no marker, no generated shape)
    assert _cfg_path(tmp_path).read_text() == "# pre-existing git setup\n"


# ---------------------------------------------------------------------------
# GitHub App mode — mint-on-demand credential helper instead of a static token
#
# When the three GITHUB_APP_* vars are present the agent auths with short-lived
# installation tokens: .gitconfig points credential.helper at the minting CLI
# and NO .git-credentials is written (there is no long-lived token to store).
# Agents still on a PAT must be entirely unaffected.
# ---------------------------------------------------------------------------

import base64 as _b64
import datetime as _dt

from hermes_cli.git_credentials_boot import build_git_config_content_app


def _write_app_env(home: Path, *, extra: str = "") -> None:
    key_b64 = _b64.b64encode(b"-----BEGIN PRIVATE KEY-----\nfake\n").decode("ascii")
    _write_env(
        home,
        "GITHUB_APP_ID=123456\n"
        "GITHUB_APP_INSTALLATION_ID=987654\n"
        f"GITHUB_APP_PRIVATE_KEY_B64={key_b64}\n" + extra,
    )


def _iso_in(seconds: float) -> str:
    return (
        _dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=seconds)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def _stub_mint(monkeypatch):
    """Stub the network mint so boot tests stay offline."""
    import hermes_cli.git_credentials_boot as boot
    from hermes_cli.github_app_token import InstallationToken

    monkeypatch.setattr(
        boot,
        "get_installation_token_detail",
        lambda home, force=False: InstallationToken("ghs_minted", _iso_in(3600)),
    )


def test_app_mode_writes_helper_gitconfig_not_credentials_file(tmp_path, _stub_mint):
    _write_app_env(tmp_path)

    result = provision_git_credentials(tmp_path)

    assert result.provisioned is True
    assert result.source == "GITHUB_APP"
    cfg = _cfg_path(tmp_path).read_text()
    assert "hermes_cli.github_app_token get-credential" in cfg
    assert "helper = store" not in cfg
    # No long-lived token exists in App mode, so nothing to store.
    assert not _cred_path(tmp_path).exists()


def test_app_mode_writes_gh_hosts_with_minted_token(tmp_path, _stub_mint):
    _write_app_env(tmp_path)
    provision_git_credentials(tmp_path)
    assert "oauth_token: ghs_minted" in _gh_hosts_path(tmp_path).read_text()


def test_app_mode_preferred_when_both_app_and_pat_present(tmp_path, _stub_mint):
    _write_app_env(tmp_path, extra="GITHUB_PAT=ghp_legacy\n")

    result = provision_git_credentials(tmp_path)

    assert result.source == "GITHUB_APP"
    assert not _cred_path(tmp_path).exists()
    assert "ghp_legacy" not in _cfg_path(tmp_path).read_text()


def test_pat_path_untouched_when_no_app_vars(tmp_path):
    # Regression: agents still on a PAT keep byte-identical behaviour.
    _write_env(tmp_path, "GITHUB_PAT=ghp_secret\n")

    result = provision_git_credentials(tmp_path)

    assert result.provisioned is True
    assert result.source == "GITHUB_PAT"
    assert "helper = store" in _cfg_path(tmp_path).read_text()
    assert "ghp_secret" in _cred_path(tmp_path).read_text()


def test_build_git_config_content_app_shape():
    out = build_git_config_content_app(
        name="cryptids", email="c@example.com", python_exe="/opt/hermes/.venv/bin/python"
    )
    # Leading empty helper clears any inherited store helper.
    assert "\thelper =\n" in out
    assert '\thelper = "!/opt/hermes/.venv/bin/python -m hermes_cli.github_app_token get-credential"' in out
    assert "\tname = cryptids" in out


# ---------------------------------------------------------------------------
# gh hosts.yml refresh — driven by the token's lifetime, not the file's existence
#
# THE fossil. `_provision_app_mode` gated the gh mint on
# `not gh_hosts_path.exists()`, so the very first boot's ~1h installation token
# was the only one an agent ever received. Observed live 2026-08-09 on
# hermes-matilde / -nimbleco / -cryptids: hosts.yml frozen at 2026-07-23,
# holding a ghs_ token that had been dead for ~16 days, while `git` kept
# working perfectly through the mint-on-demand credential helper — so nothing
# in the container's own view ever said the credential had died.
# ---------------------------------------------------------------------------

from hermes_cli.git_credentials_boot import (
    GH_EXPIRY_STAMP_PREFIX,
    MANAGED_MARKER,
    gh_hosts_needs_refresh,
    read_gh_hosts_expiry,
    refresh_gh_hosts,
    resolve_root_identity,
)


def _managed_hosts(token: str, expires_at: str) -> str:
    return (
        f"{MANAGED_MARKER}\n{GH_EXPIRY_STAMP_PREFIX}{expires_at}\n"
        f"github.com:\n    oauth_token: {token}\n"
        "    git_protocol: https\n    user: x-access-token\n"
    )


class TestGhHostsStaleness:
    def test_absent_needs_refresh(self):
        assert gh_hosts_needs_refresh(None) is True

    def test_foreign_file_never_refreshed(self):
        assert (
            gh_hosts_needs_refresh("github.com:\n    oauth_token: gho_x\n    user: me\n")
            is False
        )

    def test_ours_but_unstamped_needs_refresh(self):
        """Every hosts.yml on the fleet today is in this bucket — written before
        the expiry stamp existed, holding a token that died within the hour.
        Unknown lifetime must mean refresh, not skip."""
        legacy = (
            "github.com:\n    oauth_token: ghs_fossil\n"
            "    git_protocol: https\n    user: x-access-token\n"
        )
        assert gh_hosts_needs_refresh(legacy) is True

    def test_fresh_stamp_does_not_need_refresh(self):
        assert gh_hosts_needs_refresh(_managed_hosts("ghs_a", _iso_in(3600))) is False

    def test_expired_stamp_needs_refresh(self):
        assert gh_hosts_needs_refresh(_managed_hosts("ghs_a", _iso_in(-60))) is True

    def test_refresh_starts_inside_the_margin_not_at_death(self):
        """A `gh` call may run minutes after the env was built; rolling over only
        at expiry hands out a token that dies mid-operation."""
        assert gh_hosts_needs_refresh(_managed_hosts("ghs_a", _iso_in(60))) is True

    def test_unparseable_stamp_needs_refresh(self):
        assert gh_hosts_needs_refresh(_managed_hosts("ghs_a", "not-a-date")) is True

    def test_expiry_round_trips_through_the_written_file(self, tmp_path, _stub_mint):
        _write_app_env(tmp_path)
        provision_git_credentials(tmp_path)
        text = _gh_hosts_path(tmp_path).read_text()
        assert read_gh_hosts_expiry(text) is not None
        assert gh_hosts_needs_refresh(text) is False


class TestAppModeBootRefresh:
    def test_expired_hosts_is_rewritten_not_preserved(self, tmp_path, _stub_mint):
        """The exact production state: a hosts.yml that exists and is dead."""
        _write_app_env(tmp_path)
        gh = _gh_hosts_path(tmp_path)
        gh.parent.mkdir(parents=True)
        gh.write_text(_managed_hosts("ghs_fossil", _iso_in(-86400 * 16)))

        result = provision_git_credentials(tmp_path)

        assert result.provisioned is True
        assert "oauth_token: ghs_minted" in gh.read_text()
        assert "ghs_fossil" not in gh.read_text()

    def test_legacy_unstamped_hosts_is_healed(self, tmp_path, _stub_mint):
        _write_app_env(tmp_path)
        gh = _gh_hosts_path(tmp_path)
        gh.parent.mkdir(parents=True)
        gh.write_text(
            "github.com:\n    oauth_token: ghs_2026_07_23\n"
            "    git_protocol: https\n    user: x-access-token\n"
        )

        provision_git_credentials(tmp_path)

        assert "oauth_token: ghs_minted" in gh.read_text()

    def test_still_valid_hosts_is_left_alone_without_minting(self, tmp_path, monkeypatch):
        """Boot must not mint when the file is fine — a live HTTP call in
        cont-init.d is a boot-hang risk on a flaky network."""
        import hermes_cli.git_credentials_boot as boot

        _write_app_env(tmp_path)
        gh = _gh_hosts_path(tmp_path)
        gh.parent.mkdir(parents=True)
        gh.write_text(_managed_hosts("ghs_still_good", _iso_in(3600)))
        calls = []
        monkeypatch.setattr(
            boot,
            "get_installation_token_detail",
            lambda home, force=False: calls.append(home),
        )

        provision_git_credentials(tmp_path)

        assert calls == []
        assert "ghs_still_good" in gh.read_text()

    def test_mint_failure_leaves_file_and_warns(self, tmp_path, monkeypatch, capsys):
        """Fail-soft, but audibly: a dead credential left in place must say so."""
        import hermes_cli.git_credentials_boot as boot

        _write_app_env(tmp_path)
        gh = _gh_hosts_path(tmp_path)
        gh.parent.mkdir(parents=True)
        gh.write_text(_managed_hosts("ghs_fossil", _iso_in(-3600)))

        def _boom(home, force=False):
            raise RuntimeError("github unreachable")

        monkeypatch.setattr(boot, "get_installation_token_detail", _boom)

        provision_git_credentials(tmp_path)  # must not raise

        assert "ghs_fossil" in gh.read_text()
        assert "WARNING stale gh hosts.yml" in capsys.readouterr().out

    def test_app_mode_stale_identity_is_corrected(self, tmp_path, _stub_mint, monkeypatch):
        """The `name = data` fossil on the App-mode path (matilde/nimbleco/cryptids)."""
        _write_app_env(tmp_path)
        home = tmp_path / "home"
        home.mkdir()
        home.joinpath(".gitconfig").write_text(
            "[credential]\n\thelper =\n"
            '\thelper = "!/x/python -m hermes_cli.github_app_token get-credential"\n'
            "[user]\n\tname = data\n\temail = data@users.noreply.github.com\n"
            '[url "https://github.com/"]\n\tinsteadOf = git@github.com:\n'
            "\tinsteadOf = ssh://git@github.com/\n"
        )
        monkeypatch.setenv("HERMES_AGENT_NAME", "matilde")

        provision_all(tmp_path)

        cfg = _cfg_path(tmp_path).read_text()
        assert "name = matilde" in cfg
        assert "email = matilde@users.noreply.github.com" in cfg
        assert "name = data" not in cfg


class TestRefreshGhHosts:
    """The out-of-boot half: refresh tied to the token's lifetime, invoked from
    the subprocess-env boundary so a weeks-long container still rotates."""

    def test_writes_when_absent(self, tmp_path):
        assert refresh_gh_hosts(tmp_path, "ghs_new", _iso_in(3600)) is True
        assert (
            "oauth_token: ghs_new"
            in (tmp_path / ".config" / "gh" / "hosts.yml").read_text()
        )

    def test_rewrites_when_expired(self, tmp_path):
        path = tmp_path / ".config" / "gh" / "hosts.yml"
        path.parent.mkdir(parents=True)
        path.write_text(_managed_hosts("ghs_old", _iso_in(-60)))

        assert refresh_gh_hosts(tmp_path, "ghs_new", _iso_in(3600)) is True
        assert "oauth_token: ghs_new" in path.read_text()

    def test_no_write_while_still_valid(self, tmp_path):
        path = tmp_path / ".config" / "gh" / "hosts.yml"
        path.parent.mkdir(parents=True)
        path.write_text(_managed_hosts("ghs_old", _iso_in(3600)))
        before = path.read_text()

        assert refresh_gh_hosts(tmp_path, "ghs_new", _iso_in(3600)) is False
        assert path.read_text() == before

    def test_never_touches_a_foreign_gh_login(self, tmp_path):
        path = tmp_path / ".config" / "gh" / "hosts.yml"
        path.parent.mkdir(parents=True)
        path.write_text("github.com:\n    oauth_token: gho_human\n    user: a-human\n")

        assert refresh_gh_hosts(tmp_path, "ghs_new", _iso_in(3600)) is False
        assert "gho_human" in path.read_text()

    def test_written_file_is_chmod_600(self, tmp_path):
        refresh_gh_hosts(tmp_path, "ghs_new", _iso_in(3600))
        assert _mode(tmp_path / ".config" / "gh" / "hosts.yml") == 0o600


# ---------------------------------------------------------------------------
# Identity resolution — never let a mount basename become an author name
# ---------------------------------------------------------------------------


class TestRootIdentity:
    def test_process_env_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_AGENT_NAME", "matilde")
        _write_env(tmp_path, "HERMES_AGENT_NAME=stale-in-file\n")
        assert resolve_root_identity(tmp_path)[0] == "matilde"

    def test_falls_back_to_the_agents_own_env_file(self, tmp_path, monkeypatch):
        """HERMES_AGENT_NAME lives in the agent's .env on the mounted volume
        (verified on hermes-matilde). That file is the durable source when the
        module runs outside the container's with-contenv environment."""
        monkeypatch.delenv("HERMES_AGENT_NAME", raising=False)
        _write_env(tmp_path, "HERMES_AGENT_NAME=matilde\nGITHUB_PAT=ghp_x\n")
        assert resolve_root_identity(tmp_path)[0] == "matilde"

    def test_generic_mount_basename_is_warned_about(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_AGENT_NAME", raising=False)
        opt_data = tmp_path / "opt" / "data"
        opt_data.mkdir(parents=True)
        name, warning = resolve_root_identity(opt_data)
        assert name is None
        assert warning is not None and "data" in warning

    def test_meaningful_basename_is_not_warned_about(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_AGENT_NAME", raising=False)
        agent_home = tmp_path / "cryptids"
        agent_home.mkdir()
        assert resolve_root_identity(agent_home) == (None, None)

    def test_env_file_name_reaches_the_written_gitconfig(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_AGENT_NAME", raising=False)
        _write_env(tmp_path, "HERMES_AGENT_NAME=matilde\nGITHUB_PAT=ghp_x\n")

        results = provision_all(tmp_path)

        assert "name = matilde" in _cfg_path(tmp_path).read_text()
        assert results[0].identity == "matilde"
