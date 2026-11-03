"""Tests for hermes_cli.github_app_token — GitHub App installation-token minting.

Mints a short-lived (1h) installation token from a NimbleCoOrg GitHub App so
fleet agents auth to org repos without a long-lived PAT. Runs in TWO contexts:
at container boot (reads the agent .env directly) and as a git credential helper
inside the tool subprocess (where the App private key is blocklisted from env,
so it MUST read the .env file, not os.environ).

Tests use an in-test RSA keypair (via `cryptography`) and a fake GitHub API —
no network, no real key.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import pytest

from hermes_cli import github_app_token as gat


# ---------------------------------------------------------------------------
# Fixtures: a throwaway RSA keypair, PEM + base64 form
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    return key, pem


@pytest.fixture
def app_env(tmp_path: Path, rsa_keypair) -> Path:
    """A HERMES_HOME-shaped dir whose .env carries the three App vars."""
    _key, pem = rsa_keypair
    b64 = base64.b64encode(pem.encode("utf-8")).decode("ascii")
    home = tmp_path / "agent"
    home.mkdir()
    (home / ".env").write_text(
        f"GITHUB_APP_ID=123456\n"
        f"GITHUB_APP_INSTALLATION_ID=987654\n"
        f"GITHUB_APP_PRIVATE_KEY_B64={b64}\n"
    )
    return home


# ---------------------------------------------------------------------------
# read_app_credentials + decode_private_key
# ---------------------------------------------------------------------------


def test_read_app_credentials_parses_all_three(app_env, rsa_keypair):
    _key, pem = rsa_keypair
    creds = gat.read_app_credentials(app_env / ".env")
    assert creds is not None
    assert creds.app_id == "123456"
    assert creds.installation_id == "987654"
    assert creds.private_key_pem.strip() == pem.strip()


def test_read_app_credentials_none_when_app_id_missing(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("GITHUB_APP_PRIVATE_KEY_B64=eHg=\nGITHUB_APP_INSTALLATION_ID=1\n")
    assert gat.read_app_credentials(env) is None


def test_read_app_credentials_none_when_key_missing(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("GITHUB_APP_ID=1\nGITHUB_APP_INSTALLATION_ID=1\n")
    assert gat.read_app_credentials(env) is None


def test_read_app_credentials_installation_optional(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("GITHUB_APP_ID=1\nGITHUB_APP_PRIVATE_KEY_B64=eHg=\n")
    creds = gat.read_app_credentials(env)
    assert creds is not None and creds.installation_id is None


def test_decode_private_key_roundtrips(rsa_keypair):
    _key, pem = rsa_keypair
    b64 = base64.b64encode(pem.encode("utf-8")).decode("ascii")
    assert gat.decode_private_key(b64).strip() == pem.strip()


# ---------------------------------------------------------------------------
# build_app_jwt — verifiable with the public key
# ---------------------------------------------------------------------------


def test_build_app_jwt_claims_and_alg(rsa_keypair):
    import jwt as pyjwt

    key, pem = rsa_keypair
    now = 1_700_000_000
    token = gat.build_app_jwt("123456", pem, now=now)
    header = pyjwt.get_unverified_header(token)
    assert header["alg"] == "RS256"
    pub = key.public_key()
    decoded = pyjwt.decode(token, pub, algorithms=["RS256"], options={"verify_exp": False})
    assert decoded["iss"] == "123456"
    assert decoded["iat"] == now - 60
    assert decoded["exp"] == now + 600


# ---------------------------------------------------------------------------
# mint_installation_token — fake GitHub API
# ---------------------------------------------------------------------------


def test_mint_uses_bearer_jwt_and_returns_token(app_env, rsa_keypair, monkeypatch):
    calls = {}

    class FakeResp:
        status_code = 201

        def json(self):
            return {"token": "ghs_minted", "expires_at": "2026-01-01T00:00:00Z"}

    def fake_post(url, headers=None, timeout=None):
        calls["url"] = url
        calls["auth"] = headers.get("Authorization", "")
        return FakeResp()

    monkeypatch.setattr(gat.httpx, "post", fake_post)
    creds = gat.read_app_credentials(app_env / ".env")
    token, expires = gat.mint_installation_token(creds)

    assert token == "ghs_minted"
    assert expires == "2026-01-01T00:00:00Z"
    assert "987654/access_tokens" in calls["url"]
    assert calls["auth"].startswith("Bearer ")


def test_resolve_installation_id_discovers_when_absent(tmp_path: Path, rsa_keypair, monkeypatch):
    _key, pem = rsa_keypair
    b64 = base64.b64encode(pem.encode("utf-8")).decode("ascii")
    env = tmp_path / ".env"
    env.write_text(f"GITHUB_APP_ID=1\nGITHUB_APP_PRIVATE_KEY_B64={b64}\n")

    class FakeResp:
        status_code = 200

        def json(self):
            return [{"id": 555}]

    monkeypatch.setattr(gat.httpx, "get", lambda url, headers=None, timeout=None: FakeResp())
    creds = gat.read_app_credentials(env)
    jwt_tok = gat.build_app_jwt(creds.app_id, creds.private_key_pem)
    assert gat.resolve_installation_id(jwt_tok, creds) == "555"


# ---------------------------------------------------------------------------
# get_installation_token — cache-first with expiry
# ---------------------------------------------------------------------------


def _write_cache(home: Path, token: str, expires_epoch: float) -> None:
    import datetime as dt

    cache = gat.cache_path(home)
    cache.parent.mkdir(parents=True, exist_ok=True)
    iso = dt.datetime.fromtimestamp(expires_epoch, tz=dt.UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    cache.write_text(json.dumps({"token": token, "expires_at": iso}))


def test_cache_hit_when_far_from_expiry(app_env, monkeypatch):
    _write_cache(app_env, "ghs_cached", time.time() + 3600)

    def boom(*a, **k):
        raise AssertionError("must not mint on a fresh cache hit")

    monkeypatch.setattr(gat.httpx, "post", boom)
    assert gat.get_installation_token(app_env) == "ghs_cached"


def test_remint_when_within_margin(app_env, monkeypatch):
    _write_cache(app_env, "ghs_stale", time.time() + 120)  # < 600s margin

    class FakeResp:
        status_code = 201

        def json(self):
            return {"token": "ghs_fresh", "expires_at": "2099-01-01T00:00:00Z"}

    monkeypatch.setattr(gat.httpx, "post", lambda *a, **k: FakeResp())
    assert gat.get_installation_token(app_env) == "ghs_fresh"


def test_remint_when_cache_corrupt(app_env, monkeypatch):
    cache = gat.cache_path(app_env)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("{not json")

    class FakeResp:
        status_code = 201

        def json(self):
            return {"token": "ghs_recovered", "expires_at": "2099-01-01T00:00:00Z"}

    monkeypatch.setattr(gat.httpx, "post", lambda *a, **k: FakeResp())
    assert gat.get_installation_token(app_env) == "ghs_recovered"


# ---------------------------------------------------------------------------
# get_installation_token_detail — the expiry travels WITH the token
#
# A token handed around as a bare string is a credential whose lifetime is
# invisible. Every consumer that PERSISTS it (gh's hosts.yml) needs to know
# when its copy dies, or it cannot tell "still good" from "dead for 16 days".
# ---------------------------------------------------------------------------


def test_detail_carries_expiry_from_a_cache_hit(app_env, monkeypatch):
    _write_cache(app_env, "ghs_cached", time.time() + 3600)
    monkeypatch.setattr(
        gat.httpx, "post", lambda *a, **k: pytest.fail("must not mint on cache hit")
    )

    detail = gat.get_installation_token_detail(app_env)

    assert detail.token == "ghs_cached"
    assert detail.expires_epoch - time.time() == pytest.approx(3600, abs=5)


def test_detail_carries_expiry_from_a_fresh_mint(app_env, monkeypatch):
    class FakeResp:
        status_code = 201

        def json(self):
            return {"token": "ghs_fresh", "expires_at": "2099-01-01T00:00:00Z"}

    monkeypatch.setattr(gat.httpx, "post", lambda *a, **k: FakeResp())

    detail = gat.get_installation_token_detail(app_env, force=True)

    assert detail.token == "ghs_fresh"
    assert detail.expires_at == "2099-01-01T00:00:00Z"


def test_detail_none_without_app_credentials(tmp_path: Path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-x\n")
    assert gat.get_installation_token_detail(tmp_path) is None


def test_string_wrapper_still_matches_detail(app_env, monkeypatch):
    """get_installation_token stays the string-only convenience path — callers
    that only authenticate right now must be unaffected."""
    _write_cache(app_env, "ghs_cached", time.time() + 3600)
    monkeypatch.setattr(
        gat.httpx, "post", lambda *a, **k: pytest.fail("must not mint on cache hit")
    )
    assert gat.get_installation_token(app_env) == (
        gat.get_installation_token_detail(app_env).token
    )


def test_cache_written_0600(app_env, monkeypatch):
    import stat

    class FakeResp:
        status_code = 201

        def json(self):
            return {"token": "ghs_x", "expires_at": "2099-01-01T00:00:00Z"}

    monkeypatch.setattr(gat.httpx, "post", lambda *a, **k: FakeResp())
    gat.get_installation_token(app_env, force=True)
    mode = stat.S_IMODE(gat.cache_path(app_env).stat().st_mode)
    assert mode == 0o600


# ---------------------------------------------------------------------------
# get-credential CLI — git credential-helper protocol
# ---------------------------------------------------------------------------


def test_get_credential_stdout_format(app_env, monkeypatch, capsys):
    monkeypatch.setattr(gat, "get_installation_token", lambda home, force=False: "ghs_helper")
    rc = gat.main(["get-credential", "--home", str(app_env)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "username=x-access-token\n" in out
    assert "password=ghs_helper\n" in out
    assert out.endswith("\n\n")


def test_get_credential_accepts_git_get_operation(app_env, monkeypatch, capsys):
    # git invokes the helper as `<helper> get` — the trailing operation must not
    # crash argparse (regression: "unrecognized arguments: get").
    monkeypatch.setattr(gat, "get_installation_token", lambda home, force=False: "ghs_helper")
    rc = gat.main(["get-credential", "--home", str(app_env), "get"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "password=ghs_helper\n" in out


def test_get_alias_matches_get_credential(app_env, monkeypatch, capsys):
    """`get` is an alias for `get-credential` — agents reach for it first
    (observed live 2026-07-21: a fleet agent fumbled the subcommand and gave
    up on the helper entirely)."""
    monkeypatch.setattr(gat, "get_installation_token", lambda home, force=False: "ghs_helper")
    rc = gat.main(["get", "--home", str(app_env)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "password=ghs_helper\n" in out


def test_get_credential_store_and_erase_are_noops(app_env, monkeypatch, capsys):
    # store/erase carry no output and must never mint.
    def boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("mint attempted on store/erase")

    monkeypatch.setattr(gat, "get_installation_token", boom)
    for op in ("store", "erase"):
        assert gat.main(["get-credential", "--home", str(app_env), op]) == 0
        assert capsys.readouterr().out == ""


def test_unknown_subcommand_names_valid_choices(capsys):
    """argparse error must surface the valid choices, not just 'invalid'."""
    with pytest.raises(SystemExit):
        gat.main(["mint"])
    err = capsys.readouterr().err
    assert "get-credential" in err


# ---------------------------------------------------------------------------
# get-credential host allowlist — never answer for a host we don't recognize
# ---------------------------------------------------------------------------


def _pipe_stdin(monkeypatch, text: str) -> None:
    import io

    monkeypatch.setattr(gat.sys, "stdin", io.StringIO(text))


def test_get_credential_refuses_foreign_host(app_env, monkeypatch, capsys):
    def boom(home, force=False):
        raise AssertionError("must not mint for a non-GitHub host")

    monkeypatch.setattr(gat, "get_installation_token", boom)
    _pipe_stdin(monkeypatch, "protocol=https\nhost=evil.example.com\n")
    rc = gat.main(["get-credential", "--home", str(app_env)])
    assert rc == 0  # silent fall-through, git tries the next helper
    assert capsys.readouterr().out == ""


def test_get_credential_refuses_non_https(app_env, monkeypatch, capsys):
    def boom(home, force=False):
        raise AssertionError("must not mint for a non-https request")

    monkeypatch.setattr(gat, "get_installation_token", boom)
    _pipe_stdin(monkeypatch, "protocol=http\nhost=github.com\n")
    rc = gat.main(["get-credential", "--home", str(app_env)])
    assert rc == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("host", ["github.com", "gist.github.com"])
def test_get_credential_answers_for_allowed_hosts(app_env, monkeypatch, capsys, host):
    monkeypatch.setattr(gat, "get_installation_token", lambda home, force=False: "ghs_ok")
    _pipe_stdin(monkeypatch, f"protocol=https\nhost={host}\n")
    rc = gat.main(["get-credential", "--home", str(app_env)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "password=ghs_ok\n" in out


def test_get_credential_answers_on_empty_request(app_env, monkeypatch, capsys):
    """No stdin description (manual smoke test / boot-style call) still answers."""
    monkeypatch.setattr(gat, "get_installation_token", lambda home, force=False: "ghs_ok")
    _pipe_stdin(monkeypatch, "")
    rc = gat.main(["get-credential", "--home", str(app_env)])
    assert rc == 0
    assert "password=ghs_ok\n" in capsys.readouterr().out
