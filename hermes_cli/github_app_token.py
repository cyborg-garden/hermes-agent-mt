"""GitHub App installation-token minting for fleet agents.

A NimbleCoOrg-owned GitHub App lets every agent authenticate to org repos with a
short-lived (~1h), auto-rotating installation token instead of a long-lived,
per-user PAT bound to a single resource owner. This module mints and caches that
token.

It runs in two contexts, and BOTH read the App credentials from the agent's
``.env`` FILE — never from ``os.environ``:

* at container boot (``git_credentials_boot``), where the ``.env`` on the mounted
  volume is readable and the tool subprocess hasn't started;
* as a git **credential helper** inside the tool subprocess, where the App
  private key is deliberately blocklisted from the environment
  (``tools/environments/local.py``), so a file read is the only path.

Credentials (all single-line, so they survive HSM's newline-rejecting ``.env``
writer):

* ``GITHUB_APP_ID``
* ``GITHUB_APP_PRIVATE_KEY_B64`` — the PEM, base64-encoded (the raw PEM is
  multi-line and cannot be stored by HSM)
* ``GITHUB_APP_INSTALLATION_ID`` — optional; auto-discovered if absent

Mint flow mirrors the proven pattern in ``tools/skills_hub.py``: sign a short
RS256 JWT (iss=app id, iat-60, exp+600) → ``POST /app/installations/{id}/
access_tokens`` → ``201 {token, expires_at}``.

**Broker mode.** When ``github-broker.json`` is in the agent home, the token comes
from the host-side broker (swarm-map ``scripts/token-broker``) and the agent holds
no App key at all. The file carries the broker URL and this agent's own secret. It
wins over any App credentials and never falls back to them: a broker failure must
stay visible, not get hidden by a key that was meant to be removed.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import httpx

GITHUB_API = "https://api.github.com"

# Required + optional App vars in the agent .env.
_APP_ID_VAR = "GITHUB_APP_ID"
_APP_KEY_VAR = "GITHUB_APP_PRIVATE_KEY_B64"
_APP_INSTALL_VAR = "GITHUB_APP_INSTALLATION_ID"

# Re-mint when fewer than this many seconds remain, so a token never expires
# mid-operation (a `git push` may run minutes after `git credential fill`).
REMINT_MARGIN_SECONDS = 600

_JWT_LIFETIME_SECONDS = 600  # GitHub caps App JWTs at 10 min.
_HTTP_TIMEOUT = 10.0

BROKER_CONFIG_FILE = "github-broker.json"
# Short: this runs on subprocess spawns, and the host is one hop away.
_BROKER_TIMEOUT = 5.0


@dataclass(frozen=True)
class AppCredentials:
    app_id: str
    private_key_pem: str
    installation_id: Optional[str]


# ---------------------------------------------------------------------------
# .env parsing (flat file, no shell expansion — matches git_credentials_boot)
# ---------------------------------------------------------------------------


def _read_env_var(content: str, key: str) -> Optional[str]:
    m = re.search(rf"^{re.escape(key)}=(.*)$", content, re.MULTILINE)
    if not m:
        return None
    val = re.sub(r'^["\']|["\']$', "", m.group(1).strip())
    return val or None


def read_app_credentials(env_path: Path) -> Optional[AppCredentials]:
    """Return ``AppCredentials`` from ``env_path`` or ``None``.

    ``None`` (any required var absent) is the signal for callers to fall back to
    the static-PAT path — App auth is strictly opt-in per agent.
    """
    try:
        content = env_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None
    app_id = _read_env_var(content, _APP_ID_VAR)
    key_b64 = _read_env_var(content, _APP_KEY_VAR)
    if not app_id or not key_b64:
        return None
    try:
        pem = decode_private_key(key_b64)
    except Exception:
        return None
    return AppCredentials(
        app_id=app_id,
        private_key_pem=pem,
        installation_id=_read_env_var(content, _APP_INSTALL_VAR),
    )


def decode_private_key(b64: str) -> str:
    """Base64-decode the stored PEM back to its multi-line text form."""
    return base64.b64decode(b64).decode("utf-8")


# ---------------------------------------------------------------------------
# Host-side broker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrokerConfig:
    url: str
    secret: str


def read_broker_config(home_dir: Path) -> Optional[BrokerConfig]:
    """The broker settings for this home, or ``None`` when it isn't on the broker.

    A profile shares its container's identity, so it uses the config at the root
    of ``HERMES_HOME``.
    """
    candidates = [home_dir]
    if home_dir.parent.name == "profiles":
        candidates.append(home_dir.parent.parent)
    for base in candidates:
        try:
            data = json.loads((base / BROKER_CONFIG_FILE).read_text(encoding="utf-8"))
            url, secret = data["url"], data["secret"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if isinstance(url, str) and isinstance(secret, str) and url and secret:
            return BrokerConfig(url=url.rstrip("/"), secret=secret)
    return None


def request_broker_token(config: BrokerConfig) -> Tuple[str, str]:
    """Ask the broker for this agent's token. Returns ``(token, expires_at_iso)``.

    ``trust_env=False`` keeps the agent secret from going through any proxy set in
    the environment; the broker is on the host, never behind one.
    """
    resp = httpx.post(
        f"{config.url}/v1/token",
        headers={"Authorization": f"Bearer {config.secret}"},
        timeout=_BROKER_TIMEOUT,
        trust_env=False,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"token broker refused: HTTP {resp.status_code}")
    body = resp.json()
    return body["token"], body["expires_at"]


# ---------------------------------------------------------------------------
# JWT + mint
# ---------------------------------------------------------------------------


def build_app_jwt(app_id: str, private_key_pem: str, *, now: Optional[int] = None) -> str:
    """Sign a short-lived RS256 App JWT. ``now`` is injectable for tests."""
    import jwt  # PyJWT[crypto] — already in pyproject.toml

    ts = int(time.time()) if now is None else now
    payload = {"iat": ts - 60, "exp": ts + _JWT_LIFETIME_SECONDS, "iss": app_id}
    return jwt.encode(payload, private_key_pem, algorithm="RS256")


def _app_headers(jwt_token: str) -> dict:
    return {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github.v3+json",
    }


def resolve_installation_id(jwt_token: str, creds: AppCredentials) -> str:
    """Return the installation id, discovering it via the API if unset.

    An App installed on exactly one org has one installation; if the agent
    didn't pin ``GITHUB_APP_INSTALLATION_ID`` we take the first installation the
    App can see.
    """
    if creds.installation_id:
        return creds.installation_id
    resp = httpx.get(
        f"{GITHUB_API}/app/installations", headers=_app_headers(jwt_token), timeout=_HTTP_TIMEOUT
    )
    if resp.status_code != 200:
        raise RuntimeError(f"installation discovery failed: HTTP {resp.status_code}")
    installs = resp.json()
    if not installs:
        raise RuntimeError("GitHub App has no installations")
    return str(installs[0]["id"])


def mint_installation_token(creds: AppCredentials) -> tuple[str, str]:
    """Mint a fresh installation token. Returns ``(token, expires_at_iso)``."""
    jwt_token = build_app_jwt(creds.app_id, creds.private_key_pem)
    installation_id = resolve_installation_id(jwt_token, creds)
    resp = httpx.post(
        f"{GITHUB_API}/app/installations/{installation_id}/access_tokens",
        headers=_app_headers(jwt_token),
        timeout=_HTTP_TIMEOUT,
    )
    if resp.status_code != 201:
        raise RuntimeError(f"token mint failed: HTTP {resp.status_code}")
    body = resp.json()
    return body["token"], body["expires_at"]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def cache_path(home_dir: Path, name: str = "github_app_token.json") -> Path:
    """Cache file for one agent home. ``home_dir`` is the HERMES_HOME-shaped dir
    (holds ``.env``); the token cache lives under its ``home/`` subprocess HOME."""
    return home_dir / "home" / ".cache" / "hermes" / name


def _expires_epoch(iso: str) -> float:
    # GitHub returns e.g. "2026-01-01T00:00:00Z".
    return _dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=_dt.UTC
    ).timestamp()


def _load_cache_raw(path: Path) -> Optional[tuple[str, str]]:
    """``(token, expires_at_iso)`` from the cache, or ``None`` if absent/corrupt."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        _expires_epoch(data["expires_at"])  # validate parseability
        return data["token"], data["expires_at"]
    except Exception:
        return None  # absent or corrupt → caller re-mints


def _load_cache(path: Path) -> Optional[tuple[str, float]]:
    raw = _load_cache_raw(path)
    return (raw[0], _expires_epoch(raw[1])) if raw else None


def _store_cache(path: Path, token: str, expires_at_iso: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"token": token, "expires_at": expires_at_iso}), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)  # atomic — a concurrent reader never sees a partial file


@dataclass(frozen=True)
class InstallationToken:
    """A minted installation token together with the instant it dies.

    The expiry travels WITH the token because every consumer that persists it
    (``gh``'s ``hosts.yml``) needs to know when its copy goes dead. A token
    handed around as a bare string is a credential whose lifetime is invisible,
    which is exactly how ``hosts.yml`` fossilized for 16 days.
    """

    token: str
    expires_at: str  # ISO-8601 Z, verbatim from GitHub

    @property
    def expires_epoch(self) -> float:
        return _expires_epoch(self.expires_at)


def get_installation_token_detail(
    home_dir: Path, *, force: bool = False
) -> Optional[InstallationToken]:
    """Return a valid :class:`InstallationToken` for ``home_dir``, minting if needed.

    Cache-first: a cached token with more than ``REMINT_MARGIN_SECONDS`` of life
    left is returned as-is. Returns ``None`` only if the home has neither a broker
    nor App creds.

    This is the lifetime-aware primitive; :func:`get_installation_token` is the
    string-only convenience wrapper for callers that only need to authenticate
    right now and never persist the value.
    """
    broker = read_broker_config(home_dir)
    creds = None if broker else read_app_credentials(home_dir / ".env")
    if broker is None and creds is None:
        return None

    # Separate caches, so a wide App token can't outlive the move to the broker.
    path = cache_path(home_dir, "github_broker_token.json") if broker else cache_path(home_dir)
    if not force:
        cached = _load_cache_raw(path)
        if cached and _expires_epoch(cached[1]) - time.time() > REMINT_MARGIN_SECONDS:
            return InstallationToken(cached[0], cached[1])

    token, expires_at = request_broker_token(broker) if broker else mint_installation_token(creds)
    _store_cache(path, token, expires_at)
    return InstallationToken(token, expires_at)


def get_installation_token(home_dir: Path, *, force: bool = False) -> Optional[str]:
    """Return a valid installation token string for ``home_dir``, minting if needed."""
    detail = get_installation_token_detail(home_dir, force=force)
    return detail.token if detail else None


# ---------------------------------------------------------------------------
# CLI — `python -m hermes_cli.github_app_token get-credential` IS the git helper
# ---------------------------------------------------------------------------


def _default_home() -> Path:
    """The HERMES_HOME-shaped dir: parent of the subprocess HOME.

    The credential helper runs with ``HOME={HERMES_HOME}/home``, so the dir
    holding ``.env`` is its parent. Overridable with ``--home`` (boot passes it).
    """
    return Path(os.environ.get("HOME", "/opt/data/home")).parent


# Hosts this helper will answer for. Git invokes credential helpers for EVERY
# https remote, so an unscoped helper that ignored the request's host would
# hand the installation token to any URL a repo (or a malicious submodule/
# redirect) points at. The boot-written gitconfig already scopes the helper to
# https://github.com; this allowlist enforces the same boundary inside the
# helper itself, so a mis-scoped or globally-wired config can't leak the token.
ALLOWED_HOSTS = frozenset({"github.com", "gist.github.com"})


def _read_credential_request() -> dict:
    """Parse git's credential description block (key=value lines) from stdin."""
    request: dict = {}
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return request
        for line in sys.stdin.read().splitlines():
            key, sep, value = line.partition("=")
            if sep:
                request[key.strip()] = value.strip()
    except OSError:
        pass
    return request


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes_cli.github_app_token")
    sub = parser.add_subparsers(dest="cmd", required=True)
    # `get` alias: agents reach for the short form first (observed live
    # 2026-07-21 — a fleet agent fumbled the subcommand, got an argparse
    # error, and abandoned the helper for a broken workaround instead).
    gc = sub.add_parser(
        "get-credential",
        aliases=["get"],
        help="emit git credential-helper output (username/password lines)",
    )
    gc.add_argument("--home", type=Path, default=None)
    # git invokes a credential helper as ``<helper> <operation>`` where operation
    # is get/store/erase. Accept and ignore that trailing arg — without it argparse
    # errors ("unrecognized arguments: get") and git can never read the credential.
    gc.add_argument("op", nargs="?", default="get")
    args = parser.parse_args(argv)

    if args.cmd in ("get-credential", "get"):
        if args.op not in (None, "get"):
            return 0  # store/erase are no-ops for a mint-on-demand helper
        # Only answer for GitHub over https. For anything else, emit nothing
        # and exit 0 so git falls through to other helpers / prompts — never
        # send the token toward a host we don't recognize. An empty request
        # (no stdin, e.g. a manual smoke-test invocation) still answers.
        request = _read_credential_request()
        host = request.get("host", "")
        protocol = request.get("protocol", "")
        if host and host not in ALLOWED_HOSTS:
            return 0
        if protocol and protocol != "https":
            return 0
        home = args.home or _default_home()
        try:
            token = get_installation_token(home)
        except Exception as exc:
            # One line, because git shows this to whoever is stuck. The refusals we
            # raise ourselves carry only an HTTP status; anything else gives its type.
            source = "the token broker" if read_broker_config(home) else "the GitHub App"
            reason = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
            print(f"hermes git credentials: no GitHub token from {source} ({reason})", file=sys.stderr)
            return 1
        if not token:
            return 1  # no App creds → git falls through to any other helper
        # git credential protocol: key=value lines, blank line terminates.
        sys.stdout.write(f"username=x-access-token\npassword={token}\n\n")
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
