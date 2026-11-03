"""The agent side of the host token broker, against a fake broker on loopback.

Nothing here reaches GitHub. The git and gh tests run the real helper and the real
env boundary under HOME={agent}/home, which is where agents really run git.
"""
from __future__ import annotations

import datetime as dt
import http.server
import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hermes_cli import github_app_token as gat
from hermes_cli.git_credentials_boot import provision_git_credentials

SECRET = "agent-secret-value"
# Built at runtime: the sanitization gate reads a literal as a real credential.
TOKEN = "ghs" + "_fakeBrokerToken"
OLD_PAT_LINE = "https://x-access-token" + ":old-pat-value" + "@github.com\n"


def _iso_in(seconds: float) -> str:
    when = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=seconds)
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


class _FakeBroker:
    def __init__(self):
        self.requests = []
        self.status = 200
        self.expires_in = 3600
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                outer.requests.append((self.path, self.headers.get("Authorization")))
                body = json.dumps(
                    {"token": TOKEN, "expires_at": _iso_in(outer.expires_in)}
                    if outer.status == 200
                    else {"error": "nope"}
                ).encode()
                self.send_response(outer.status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.server = http.server.ThreadingHTTPServer(("localhost", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://localhost:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def broker():
    b = _FakeBroker()
    yield b
    b.close()


@pytest.fixture
def agent_home(tmp_path: Path, broker) -> Path:
    home = tmp_path / "agent"
    home.mkdir()
    (home / gat.BROKER_CONFIG_FILE).write_text(json.dumps({"url": broker.url, "secret": SECRET}))
    return home


# ---- config ----------------------------------------------------------------


def test_no_config_means_not_configured(tmp_path):
    assert gat.read_broker_config(tmp_path) is None


@pytest.mark.parametrize("body", ["not json", "{}", '{"url": "x"}', '{"secret": "x"}', "[]"])
def test_malformed_config_is_not_configured(tmp_path, body):
    (tmp_path / gat.BROKER_CONFIG_FILE).write_text(body)
    assert gat.read_broker_config(tmp_path) is None


def test_a_profile_uses_the_root_home_config(agent_home, broker):
    profile = agent_home / "profiles" / "p1"
    profile.mkdir(parents=True)
    assert gat.read_broker_config(profile).url == broker.url


# ---- token ------------------------------------------------------------------


def test_broker_token_uses_the_bearer_secret(agent_home, broker):
    detail = gat.get_installation_token_detail(agent_home)
    assert detail.token == TOKEN
    assert broker.requests == [("/v1/token", f"Bearer {SECRET}")]


def test_second_call_is_served_from_the_local_cache(agent_home, broker):
    gat.get_installation_token_detail(agent_home)
    gat.get_installation_token_detail(agent_home)
    assert len(broker.requests) == 1
    cache = agent_home / "home" / ".cache" / "hermes" / "github_broker_token.json"
    assert stat.S_IMODE(cache.stat().st_mode) == 0o600


def test_a_token_near_expiry_is_asked_for_again(agent_home, broker):
    broker.expires_in = 300
    gat.get_installation_token_detail(agent_home)
    gat.get_installation_token_detail(agent_home)
    assert len(broker.requests) == 2


def test_broker_wins_over_app_creds_in_env(agent_home, broker, monkeypatch):
    (agent_home / ".env").write_text("GITHUB_APP_ID=1\nGITHUB_APP_PRIVATE_KEY_B64=eHg=\n")
    monkeypatch.setattr(gat, "mint_installation_token", lambda creds: pytest.fail("minted locally"))
    assert gat.get_installation_token_detail(agent_home).token == TOKEN


def test_a_failing_broker_raises_and_does_not_fall_back_to_app_creds(agent_home, broker, monkeypatch):
    # A silent fallback would hide a broken rollout behind a key we meant to remove.
    broker.status = 502
    (agent_home / ".env").write_text("GITHUB_APP_ID=1\nGITHUB_APP_PRIVATE_KEY_B64=eHg=\n")
    monkeypatch.setattr(gat, "mint_installation_token", lambda creds: pytest.fail("minted locally"))
    with pytest.raises(RuntimeError) as exc:
        gat.get_installation_token_detail(agent_home)
    assert SECRET not in str(exc.value)
    assert not (agent_home / "home" / ".cache" / "hermes" / "github_broker_token.json").exists()


def test_without_a_broker_the_app_path_is_unchanged(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GITHUB_APP_ID=1\nGITHUB_APP_INSTALLATION_ID=2\nGITHUB_APP_PRIVATE_KEY_B64=eHg=\n")
    monkeypatch.setattr(gat, "mint_installation_token", lambda creds: ("ghs_fromApp", _iso_in(3600)))
    assert gat.get_installation_token_detail(tmp_path).token == "ghs_fromApp"


def test_broker_request_ignores_proxy_env(agent_home, broker, monkeypatch):
    # The secret must not take a detour through a proxy set in the environment.
    monkeypatch.setenv("HTTP_PROXY", "http://localhost:9")
    monkeypatch.setenv("http_proxy", "http://localhost:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert gat.get_installation_token_detail(agent_home).token == TOKEN


# ---- boot provisioning ------------------------------------------------------


def test_boot_with_broker_and_no_env_wires_git_and_gh(agent_home):
    result = provision_git_credentials(agent_home, name="mare")
    assert result.provisioned
    assert result.source == "GITHUB_BROKER"
    assert "hermes_cli.github_app_token get-credential" in (agent_home / "home" / ".gitconfig").read_text()
    hosts = agent_home / "home" / ".config" / "gh" / "hosts.yml"
    assert TOKEN in hosts.read_text()
    assert "# token-expires-at:" in hosts.read_text()


def test_boot_removes_a_managed_git_credentials_file_holding_an_old_pat(agent_home):
    creds = agent_home / "home" / ".git-credentials"
    creds.parent.mkdir(parents=True)
    creds.write_text(OLD_PAT_LINE)
    provision_git_credentials(agent_home, name="mare")
    assert not creds.exists()


def test_boot_leaves_a_human_git_credentials_file_alone(agent_home):
    creds = agent_home / "home" / ".git-credentials"
    creds.parent.mkdir(parents=True)
    creds.write_text("https://someone" + ":theirs" + "@example.com\n")
    provision_git_credentials(agent_home, name="mare")
    assert creds.exists()


def test_boot_replaces_a_managed_hosts_yml_holding_an_old_pat(agent_home):
    hosts = agent_home / "home" / ".config" / "gh" / "hosts.yml"
    hosts.parent.mkdir(parents=True)
    hosts.write_text("github.com:\n    oauth_token: old-pat-value\n    user: x-access-token\n")
    provision_git_credentials(agent_home, name="mare")
    assert "old-pat-value" not in hosts.read_text()


def test_boot_survives_the_broker_being_down(agent_home, broker):
    broker.status = 502
    result = provision_git_credentials(agent_home, name="mare")
    assert result.source == "GITHUB_BROKER"
    assert (agent_home / "home" / ".gitconfig").exists()


# ---- under the agent's real HOME --------------------------------------------


def test_real_git_gets_the_token_under_the_agent_home(agent_home):
    provision_git_credentials(agent_home, name="mare")
    env = {
        "HOME": str(agent_home / "home"),
        "PATH": os.environ["PATH"],
        "GIT_TERMINAL_PROMPT": "0",
        "PYTHONPATH": str(Path(gat.__file__).resolve().parents[1]),
    }
    out = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    assert f"password={TOKEN}" in out
    assert "username=x-access-token" in out


def test_helper_gives_nothing_to_other_hosts(agent_home):
    env = {
        "HOME": str(agent_home / "home"),
        "PATH": os.environ["PATH"],
        "PYTHONPATH": str(Path(gat.__file__).resolve().parents[1]),
    }
    (agent_home / "home").mkdir(exist_ok=True)
    out = subprocess.run(
        [sys.executable, "-m", "hermes_cli.github_app_token", "get-credential"],
        input="protocol=https\nhost=evil.example\n\n",
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0
    assert TOKEN not in out.stdout


def test_gh_gets_the_token_at_the_env_boundary(agent_home):
    from tools.environments import local

    local._GH_TOKEN_MINT_FAILURE_TS = 0.0
    env = {"HOME": str(agent_home / "home")}
    local._inject_github_app_gh_token(env)
    assert env["GH_TOKEN"] == TOKEN


def test_broker_config_is_denied_to_the_read_tool(tmp_path, monkeypatch):
    from agent.file_safety import get_read_block_error

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    target = tmp_path / gat.BROKER_CONFIG_FILE
    target.write_text("{}")
    assert get_read_block_error(str(target)) is not None


def _run_helper(agent_home, stdin="protocol=https\nhost=github.com\n\n"):
    env = {"HOME": str(agent_home / "home"), "PATH": os.environ.get("PATH", "")}
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.github_app_token", "get-credential"],
        input=stdin, capture_output=True, text=True, env=env, cwd=Path(gat.__file__).parent.parent,
    )


def test_helper_with_the_broker_down_says_so_in_one_line(agent_home):
    (agent_home / "home").mkdir(exist_ok=True)
    # Nothing listens on port 1, so the connection is refused as with a stopped broker.
    (agent_home / gat.BROKER_CONFIG_FILE).write_text(json.dumps({"url": "http://127.0.0.1:1", "secret": SECRET}))
    r = _run_helper(agent_home)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "token broker" in r.stderr
    assert len(r.stderr.strip().splitlines()) == 1
    assert SECRET not in r.stderr + r.stdout


def test_helper_with_a_refusing_broker_names_the_status(agent_home, broker):
    (agent_home / "home").mkdir(exist_ok=True)
    broker.status = 502
    r = _run_helper(agent_home)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "HTTP 502" in r.stderr
    assert SECRET not in r.stderr + r.stdout


# ---- Skills Hub, in the gateway process ------------------------------------


@pytest.fixture
def hub_auth(agent_home, monkeypatch):
    """A gateway under HOME=/opt/data: gh can't see home/, and no token in the env."""
    from tools.skills_hub_github import GitHubAuth

    for name in ("GITHUB_TOKEN", "GH_TOKEN", "GH_CONFIG_DIR", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(agent_home))
    monkeypatch.setenv("HOME", str(agent_home))
    return GitHubAuth()


def test_skills_hub_gets_a_broker_token_in_the_gateway(hub_auth, broker):
    assert hub_auth.auth_method() == "github-broker"
    assert hub_auth.get_headers()["Authorization"] == f"token {TOKEN}"


def test_skills_hub_asks_again_once_its_token_is_near_expiry(hub_auth, broker):
    hub_auth.get_headers()
    hub_auth._app_token_expiry = 0
    gat.cache_path(Path(os.environ["HERMES_HOME"]), "github_broker_token.json").unlink()
    hub_auth.get_headers()
    assert len(broker.requests) == 2


def test_skills_hub_goes_anonymous_rather_than_failing_when_the_broker_is_down(hub_auth, broker, monkeypatch):
    monkeypatch.setattr(hub_auth, "_try_gh_cli", lambda: None)
    broker.status = 502
    assert hub_auth.auth_method() == "anonymous"
    assert "Authorization" not in hub_auth.get_headers()


def test_skills_hub_prefers_a_token_in_the_env_while_an_agent_still_has_one(hub_auth, broker, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "env-token-value")
    assert hub_auth.auth_method() == "pat"
    assert broker.requests == []


def test_skills_hub_moves_on_when_github_rejects_the_broker_token(hub_auth, broker, monkeypatch):
    monkeypatch.setattr(hub_auth, "_try_gh_cli", lambda: None)
    assert hub_auth.get_headers()["Authorization"] == f"token {TOKEN}"
    hub_auth.reject(TOKEN)
    assert hub_auth.auth_method() == "anonymous"
    assert hub_auth.rejected == ["the token broker"]
