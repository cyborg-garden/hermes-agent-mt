"""Tests for short-lived GitHub App token injection as GH_TOKEN.

Tier-1 stripping removes every ambient GitHub credential from tool
subprocesses — deliberately: they are long-lived. But that leaves ``gh``
with no auth at all. git is covered by the ``github_app_token`` credential
helper, which mints a fresh installation token per operation; ``gh`` has no
credential-helper concept and falls back to a ``hosts.yml`` written once at
container boot, whose App token dies within the hour (observed live:
hermes-nimbleco's ``gh`` dead from 2026-07-21 onward while git kept working).

The fix mirrors the git path at the env boundary: mint (cache-first) the
same short-lived installation token and inject it as ``GH_TOKEN`` into the
sanitized subprocess env. Exposure is not widened — the helper already
caches this token under the subprocess HOME, readable by any tool.

That fix originally landed on the terminal paths only. ``hermes_subprocess_env``
— browser worker, ACP/codex/copilot executors, TUI Node host, dep-ensure,
detached gateway — stripped ``GH_TOKEN`` in Tier 1 and never re-injected, so
those spawns kept authenticating out of the frozen ``hosts.yml``. Silent by
construction: the credential was present and merely dead, so the failure
surfaced as an unexplained 401 inside a tool rather than as missing auth.
``TestEverySpawnSurface`` is the guard against that gap reopening.

See hermes_cli/github_app_token.py for the mint/cache layer and
hermes_cli/git_credentials_boot.py for the managed-file refresh.
"""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.environments.local as local_mod
from hermes_cli.github_app_token import InstallationToken
from tools.environments.local import (
    _make_run_env,
    _sanitize_subprocess_env,
    hermes_subprocess_env,
)


MINT_TARGET = "hermes_cli.github_app_token.get_installation_token_detail"

# Captured before the autouse neutering fixture can replace it.
from hermes_cli.git_credentials_boot import refresh_gh_hosts as _REAL_REFRESH_GH_HOSTS


def _detail(token="ghs_minted", ttl_seconds=3600):
    import datetime as _dt

    expires = (
        _dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=ttl_seconds)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return InstallationToken(token, expires)


@pytest.fixture(autouse=True)
def _reset_cooldown():
    """Each test starts with a clean mint-failure cooldown."""
    local_mod._GH_TOKEN_MINT_FAILURE_TS = 0.0
    yield
    local_mod._GH_TOKEN_MINT_FAILURE_TS = 0.0


@pytest.fixture(autouse=True)
def _no_real_hosts_writes(monkeypatch):
    """Injection also refreshes the managed hosts.yml under HOME. Most tests
    here use a synthetic HOME, so neuter the disk write by default;
    TestHostsRefreshSideEffect opts back in with a real tmp_path HOME."""
    import hermes_cli.git_credentials_boot as boot

    monkeypatch.setattr(boot, "refresh_gh_hosts", lambda *a, **k: False)


def _base_env(home="/data/agent/home"):
    return {"HOME": home, "PATH": "/usr/bin"}


class TestInjection:
    def test_minted_token_injected_as_gh_token(self):
        with patch(MINT_TARGET, return_value=_detail()) as mint:
            env = _sanitize_subprocess_env(_base_env())
        assert env.get("GH_TOKEN") == "ghs_minted"
        mint.assert_called_once()

    def test_home_parent_is_the_hermes_home_dir(self):
        """The mint layer's contract: HOME is ``{HERMES_HOME}/home``, so the
        dir holding ``.env`` is HOME's parent (github_app_token._default_home)."""
        with patch(MINT_TARGET, return_value=_detail()) as mint:
            env = _sanitize_subprocess_env(_base_env())
        home = Path(env["HOME"])
        assert mint.call_args.args[0] == home.parent

    def test_run_env_path_also_injects(self):
        """_make_run_env (foreground terminal) mirrors the sanitize path."""
        with patch(MINT_TARGET, return_value=_detail()):
            env = _make_run_env(_base_env())
        assert env.get("GH_TOKEN") == "ghs_minted"

    def test_ambient_github_tokens_still_stripped(self):
        """Injection must not resurrect the long-lived ambient credentials."""
        base = _base_env() | {
            "GITHUB_TOKEN": "ghp_longlived",
            "GITHUB_APP_PRIVATE_KEY_B64": "pem",
        }
        with patch(MINT_TARGET, return_value=_detail()):
            env = _sanitize_subprocess_env(base)
        assert env.get("GH_TOKEN") == "ghs_minted"
        assert "GITHUB_TOKEN" not in env
        assert "GITHUB_APP_PRIVATE_KEY_B64" not in env


class TestFailSoft:
    def test_no_app_creds_no_injection(self):
        """None from the mint layer = this agent has no App creds. Not an
        error: PAT-less / App-less installs see exactly the old behavior."""
        with patch(MINT_TARGET, return_value=None):
            env = _sanitize_subprocess_env(_base_env())
        assert "GH_TOKEN" not in env

    def test_mint_error_swallowed(self):
        with patch(MINT_TARGET, side_effect=RuntimeError("api down")):
            env = _sanitize_subprocess_env(_base_env())  # must not raise
        assert "GH_TOKEN" not in env

    def test_mint_error_starts_cooldown(self):
        """A failed mint must not retry on every spawn while GitHub is down —
        each retry costs a live HTTP timeout on the tool hot path."""
        with patch(MINT_TARGET, side_effect=RuntimeError("api down")) as mint:
            _sanitize_subprocess_env(_base_env())
            _sanitize_subprocess_env(_base_env())
        assert mint.call_count == 1

    def test_cooldown_expires(self):
        with patch(MINT_TARGET, side_effect=RuntimeError("api down")):
            _sanitize_subprocess_env(_base_env())
        local_mod._GH_TOKEN_MINT_FAILURE_TS = (
            time.time() - local_mod._GH_TOKEN_MINT_COOLDOWN_SECONDS - 1
        )
        with patch(MINT_TARGET, return_value=_detail("ghs_recovered")) as mint2:
            env = _sanitize_subprocess_env(_base_env())
        assert env.get("GH_TOKEN") == "ghs_recovered"
        assert mint2.call_count == 1

    def test_none_result_does_not_start_cooldown(self):
        """No-creds is a cheap file read, not a failure — never cool down on
        it, or an agent that gains App creds mid-session waits a minute."""
        with patch(MINT_TARGET, return_value=None) as mint:
            _sanitize_subprocess_env(_base_env())
            _sanitize_subprocess_env(_base_env())
        assert mint.call_count == 2

    def test_missing_home_skips_quietly(self):
        with patch(MINT_TARGET, return_value=_detail()) as mint:
            env = _sanitize_subprocess_env({"PATH": "/usr/bin"})
        # apply_subprocess_home_env may synthesize a HOME; only assert we
        # didn't crash and the mint layer got a real dir if it was called.
        if mint.called:
            assert mint.call_args.args[0] != Path(".")
        assert isinstance(env, dict)


class TestPrecedence:
    def test_forced_gh_token_wins_over_mint(self):
        """An operator force-passing GH_TOKEN (force prefix) is explicit
        intent — the mint must not clobber it."""
        from tools.environments.local import _HERMES_PROVIDER_ENV_FORCE_PREFIX

        extra = {_HERMES_PROVIDER_ENV_FORCE_PREFIX + "GH_TOKEN": "ghp_forced"}
        with patch(MINT_TARGET, return_value=_detail()) as mint:
            env = _sanitize_subprocess_env(_base_env(), extra_env=extra)
        assert env.get("GH_TOKEN") == "ghp_forced"
        mint.assert_not_called()

    def test_opt_out_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_GH_TOKEN_INJECT", "off")
        with patch(MINT_TARGET, return_value=_detail()) as mint:
            env = _sanitize_subprocess_env(_base_env())
        assert "GH_TOKEN" not in env
        mint.assert_not_called()


class TestEverySpawnSurface:
    """``hermes_subprocess_env`` is the OTHER half of the spawn surface —
    browser worker, ACP/codex/copilot executors, TUI Node host, dep-ensure,
    detached gateway. It strips GH_TOKEN in Tier 1 and never re-injected, so
    every one of those paths silently fell back to the boot-written hosts.yml
    and authenticated with a token that had been dead for weeks.

    The invariant this class defends: no spawn surface may reach GitHub through
    a file whose refresh is not tied to the token's lifetime.
    """

    def test_hermes_subprocess_env_injects(self, monkeypatch):
        monkeypatch.setenv("HOME", "/data/agent/home")
        with patch(MINT_TARGET, return_value=_detail()):
            env = hermes_subprocess_env()
        assert env.get("GH_TOKEN") == "ghs_minted"

    def test_injected_even_with_inherit_credentials(self, monkeypatch):
        """Tier 1 strips GH_TOKEN regardless of inherit_credentials, so the
        re-injection has to be unconditional too — codex/copilot otherwise get
        the same silent hosts.yml fallback."""
        monkeypatch.setenv("HOME", "/data/agent/home")
        monkeypatch.setenv("GH_TOKEN", "ghp_ambient_longlived")
        with patch(MINT_TARGET, return_value=_detail()):
            env = hermes_subprocess_env(inherit_credentials=True)
        assert env.get("GH_TOKEN") == "ghs_minted"

    def test_ambient_long_lived_token_is_still_replaced_not_inherited(
        self, monkeypatch
    ):
        monkeypatch.setenv("HOME", "/data/agent/home")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_longlived")
        monkeypatch.setenv("GH_TOKEN", "ghp_longlived")
        monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_B64", "pem")
        with patch(MINT_TARGET, return_value=_detail()):
            env = hermes_subprocess_env()
        assert env["GH_TOKEN"] == "ghs_minted"
        assert "GITHUB_TOKEN" not in env
        assert "GITHUB_APP_PRIVATE_KEY_B64" not in env

    def test_no_app_creds_leaves_env_clean(self, monkeypatch):
        """App-less installs keep exactly the old behaviour: stripped, not
        re-injected with somebody's long-lived PAT."""
        monkeypatch.setenv("HOME", "/data/agent/home")
        monkeypatch.setenv("GH_TOKEN", "ghp_longlived")
        with patch(MINT_TARGET, return_value=None):
            env = hermes_subprocess_env()
        assert "GH_TOKEN" not in env

    def test_mint_failure_does_not_break_the_spawn(self, monkeypatch):
        monkeypatch.setenv("HOME", "/data/agent/home")
        with patch(MINT_TARGET, side_effect=RuntimeError("api down")):
            env = hermes_subprocess_env()  # must not raise
        assert "GH_TOKEN" not in env
        assert env.get("PATH")


class TestHostsRefreshSideEffect:
    """Injection carries the fresh token back into the managed hosts.yml, which
    is what gives that file a refresh schedule tied to the token's ~1h life
    rather than to container boot.

    Exercised through ``_inject_github_app_gh_token`` directly so HOME is
    exactly the tmp_path this test controls — the outer helpers run HOME
    through the subprocess-home contract, which could point the write at a real
    user home.
    """

    @pytest.fixture(autouse=True)
    def _real_refresh(self, monkeypatch):
        """Undo the module-level neutering — these tests want the disk write."""
        import hermes_cli.git_credentials_boot as boot

        monkeypatch.setattr(boot, "refresh_gh_hosts", _REAL_REFRESH_GH_HOSTS)

    def test_stale_hosts_is_rewritten_on_spawn(self, tmp_path):
        from hermes_cli.git_credentials_boot import (
            GH_EXPIRY_STAMP_PREFIX,
            MANAGED_MARKER,
        )

        home = tmp_path / "home"
        hosts = home / ".config" / "gh" / "hosts.yml"
        hosts.parent.mkdir(parents=True)
        hosts.write_text(
            f"{MANAGED_MARKER}\n{GH_EXPIRY_STAMP_PREFIX}2026-07-23T23:09:09Z\n"
            "github.com:\n    oauth_token: ghs_fossil\n"
            "    git_protocol: https\n    user: x-access-token\n"
        )
        env = {"HOME": str(home)}

        with patch(MINT_TARGET, return_value=_detail("ghs_fresh")):
            local_mod._inject_github_app_gh_token(env)

        assert env["GH_TOKEN"] == "ghs_fresh"
        assert "oauth_token: ghs_fresh" in hosts.read_text()
        assert "ghs_fossil" not in hosts.read_text()

    def test_valid_hosts_is_not_rewritten(self, tmp_path):
        """No per-spawn disk churn: the write happens once per token, not once
        per subprocess."""
        from hermes_cli.git_credentials_boot import build_gh_hosts_content

        home = tmp_path / "home"
        hosts = home / ".config" / "gh" / "hosts.yml"
        hosts.parent.mkdir(parents=True)
        detail = _detail("ghs_current")
        hosts.write_text(
            build_gh_hosts_content(detail.token, expires_at=detail.expires_at)
        )
        before = hosts.stat().st_mtime_ns
        env = {"HOME": str(home)}

        with patch(MINT_TARGET, return_value=detail):
            local_mod._inject_github_app_gh_token(env)

        assert hosts.stat().st_mtime_ns == before

    def test_foreign_gh_login_is_never_rotated(self, tmp_path):
        home = tmp_path / "home"
        hosts = home / ".config" / "gh" / "hosts.yml"
        hosts.parent.mkdir(parents=True)
        hosts.write_text("github.com:\n    oauth_token: gho_human\n    user: a-human\n")
        env = {"HOME": str(home)}

        with patch(MINT_TARGET, return_value=_detail("ghs_fresh")):
            local_mod._inject_github_app_gh_token(env)

        assert env["GH_TOKEN"] == "ghs_fresh"  # the env var is still ours to set
        assert "gho_human" in hosts.read_text()  # the file is not

    def test_refresh_failure_never_costs_the_token(self, tmp_path, monkeypatch):
        """The env var is the load-bearing path; a file-write problem (read-only
        mount, wrong owner) must not take GH_TOKEN down with it."""
        import hermes_cli.git_credentials_boot as boot

        def _boom(*a, **k):
            raise OSError("read-only file system")

        monkeypatch.setattr(boot, "refresh_gh_hosts", _boom)
        env = {"HOME": str(tmp_path / "home")}

        with patch(MINT_TARGET, return_value=_detail("ghs_fresh")):
            local_mod._inject_github_app_gh_token(env)

        assert env["GH_TOKEN"] == "ghs_fresh"
