"""Boot-time provisioning of per-agent git credentials.

This is the runtime-side port of HSM's ``lib/services/git-credentials.ts``.
It moves "make the agent's own GitHub token usable by git" out of the
swarm-manager and into the agent runtime, where it belongs:

  * The tool subprocess (where the agent runs ``git``) deliberately CANNOT
    see ``GITHUB_TOKEN`` — provider secrets are blocklisted from the
    subprocess env (``tools/environments/local.py`` /
    ``hermes_cli/auth.py``). So a credential file is the only path, and the
    skill that runs *inside* the subprocess can't read the raw token to
    write one.
  * This module runs in-process at container boot (a cont-init.d step),
    where the agent's ``.env`` is readable on the mounted volume and
    ``get_subprocess_home()`` (``{HERMES_HOME}/home``) is known directly.

It writes the agent's OWN token (from its ``.env``) into
``{HERMES_HOME}/home/.git-credentials`` + ``.gitconfig`` so git's ``store``
helper picks it up. Tokens never cross-pollinate — each home reads only its
own ``.env``. Creating ``{HERMES_HOME}/home`` is also what activates the
tool subprocess's HOME override (see ``get_subprocess_home``).

Mirrors ``container_boot.py``: a pure, per-home function plus a profile-aware
``provision_all`` and a ``main()`` entry point wired into the image as a
cont-init.d hook. No HSM dependence on a runtime-internal path — the runtime
owns the path now.

**Manage-if-ours, not apply-if-absent.** Provisioning originally refused to
touch any file that already existed, so that a human's own ``.gitconfig`` was
never clobbered. The rule is right; the test was wrong. "Does a file exist
here?" cannot distinguish someone else's config from our own stale output, so
every file this module wrote became permanent on the first boot that wrote it:
an installation token good for ~1h froze into ``hosts.yml`` for weeks, and a
git identity resolved before ``HERMES_AGENT_NAME`` was consulted kept authoring
commits as "data" long after the resolution was fixed. Files are now stamped
with :data:`MANAGED_MARKER`, ownership is the gate, and a file we own is
reconciled to current content on every boot. Credentials that carry an expiry
additionally record it (``# token-expires-at:``) so refresh is driven by the
token's lifetime rather than by the container's.
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from hermes_cli.github_app_token import (
    get_installation_token_detail,
    read_app_credentials,
    read_broker_config,
)

GIT_HOST = "github.com"

# Checked in order; a dedicated GITHUB_PAT wins over the copilot GITHUB_TOKEN.
TOKEN_VARS = ("GITHUB_PAT", "GITHUB_TOKEN", "GH_TOKEN")

# Stamped as the first line of every file this module owns. It is what lets
# provisioning be *manage-if-ours* rather than *apply-if-absent*: a file we
# wrote is kept current on every boot; a file we did not write is never
# touched. Both `.gitconfig` (INI) and `hosts.yml` (YAML) treat `#` as a
# comment, so the marker is inert to git and gh.
MANAGED_MARKER = "# managed-by: hermes git_credentials_boot"

# Prefix of the expiry stamp written into an App-mode ``hosts.yml``. `gh` has
# no credential-helper hook, so its token is a value at rest with a ~1h life.
# Recording the death time next to it is what makes "this credential is dead"
# an inspectable fact instead of a silent 401.
GH_EXPIRY_STAMP_PREFIX = "# token-expires-at: "

# Rewrite an App-mode ``hosts.yml`` once it is within this many seconds of its
# recorded expiry. Matches ``github_app_token.REMINT_MARGIN_SECONDS`` so the
# file and the mint cache roll over together.
GH_HOSTS_REFRESH_MARGIN_SECONDS = 600


# ---------------------------------------------------------------------------
# Pure content builders
# ---------------------------------------------------------------------------


def build_git_credentials_content(token: str) -> str:
    """The ``~/.git-credentials`` line the ``store`` helper reads for HTTPS."""
    return f"https://x-access-token:{token}@{GIT_HOST}\n"


def build_git_config_content(*, name: str, email: str) -> str:
    """Minimal ``~/.gitconfig``: store helper + identity + ssh→https rewrites."""
    return "\n".join(
        [
            MANAGED_MARKER,
            "[credential]",
            "\thelper = store",
            "[user]",
            f"\tname = {name}",
            f"\temail = {email}",
            f'[url "https://{GIT_HOST}/"]',
            f"\tinsteadOf = git@{GIT_HOST}:",
            f"\tinsteadOf = ssh://git@{GIT_HOST}/",
            "",
        ]
    )


def build_git_config_content_app(
    *, name: str, email: str, python_exe: str, module: str = "hermes_cli.github_app_token"
) -> str:
    """``~/.gitconfig`` for GitHub App mode — mint-on-demand instead of ``store``.

    A GitHub App installation token lives ~1h, so it cannot be written once at
    boot the way a PAT is. Instead git calls this helper on every
    ``credential fill``; the helper returns a cached token or mints a fresh one.
    Always-fresh auth with no daemon and no token at rest in ``.git-credentials``.

    The bare ``helper =`` line first RESETS the helper list, so an inherited
    ``store`` (e.g. from a previous PAT-mode provisioning) can't answer first
    with a stale token.
    """
    return "\n".join(
        [
            MANAGED_MARKER,
            "[credential]",
            "\thelper =",
            f'\thelper = "!{python_exe} -m {module} get-credential"',
            "[user]",
            f"\tname = {name}",
            f"\temail = {email}",
            f'[url "https://{GIT_HOST}/"]',
            f"\tinsteadOf = git@{GIT_HOST}:",
            f"\tinsteadOf = ssh://git@{GIT_HOST}/",
            "",
        ]
    )


def build_gh_hosts_content(token: str, *, expires_at: Optional[str] = None) -> str:
    """The ``~/.config/gh/hosts.yml`` the GitHub CLI reads for HTTPS auth.

    Same rationale as the git credential file: ``gh`` runs in the tool
    subprocess where ``GITHUB_TOKEN`` is blocklisted, so a config file is the
    only way ``gh`` can authenticate. Without this, ``git`` worked from agent
    tools but ``gh`` reported "not logged in" — leaving half the GitHub surface
    unusable to the agent.

    ``expires_at`` (App mode) is stamped as a YAML comment so the file carries
    its own death date. A PAT has no expiry the runtime can know, so it is
    stamped only when supplied.
    """
    lines = [MANAGED_MARKER]
    if expires_at:
        lines.append(f"{GH_EXPIRY_STAMP_PREFIX}{expires_at}")
    lines += [
        f"{GIT_HOST}:",
        f"    oauth_token: {token}",
        "    git_protocol: https",
        "    user: x-access-token",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ownership + staleness — the two predicates that replace "does the file exist"
# ---------------------------------------------------------------------------

# Shape fingerprints for files written by builds that predate MANAGED_MARKER.
# Without these, every already-deployed agent would look "foreign" forever and
# never heal — the fossil would simply be frozen by the fix instead of thawed.
_LEGACY_GITCONFIG_SIGNS = (
    f"insteadOf = git@{GIT_HOST}:",
    f"insteadOf = ssh://git@{GIT_HOST}/",
)
_LEGACY_GH_HOSTS_SIGN = "user: x-access-token"


def is_managed_gitconfig(text: str) -> bool:
    """True when ``text`` is a ``.gitconfig`` this module owns.

    Ownership, not absence, is what licenses a rewrite. A human's own
    ``.gitconfig`` (or an imported agent's) is left strictly alone; one we
    generated is ours to keep current — which is how a fossilized
    ``name = data`` identity gets corrected on the next boot instead of
    surviving forever behind an existence check.
    """
    if MANAGED_MARKER in text:
        return True
    return all(sign in text for sign in _LEGACY_GITCONFIG_SIGNS) and (
        "helper = store" in text or "hermes_cli.github_app_token" in text
    )


def is_managed_gh_hosts(text: str) -> bool:
    """True when ``text`` is a ``hosts.yml`` this module owns."""
    return MANAGED_MARKER in text or _LEGACY_GH_HOSTS_SIGN in text


def is_managed_git_credentials(text: str) -> bool:
    """True when ``text`` is a ``.git-credentials`` this module owns.

    No marker here: git's ``store`` helper parses the file as one credential
    URL per line, so a comment is not safely inert. The generated line is its
    own fingerprint — only this module writes the ``x-access-token`` username,
    a human's own file carries their login.
    """
    return any(
        line.strip().startswith("https://x-access-token:")
        and line.strip().endswith(f"@{GIT_HOST}")
        for line in text.splitlines()
    )


def read_gh_hosts_expiry(text: str) -> Optional[str]:
    """The ISO expiry stamped into a managed ``hosts.yml``, if any."""
    m = re.search(
        rf"^{re.escape(GH_EXPIRY_STAMP_PREFIX)}(\S+)\s*$", text, re.MULTILINE
    )
    return m.group(1) if m else None


def gh_hosts_needs_refresh(
    text: Optional[str], *, now: Optional[float] = None
) -> bool:
    """Should this ``hosts.yml`` be re-minted and rewritten?

    The lifetime question, asked of the file itself:

    * absent → yes (nothing to serve)
    * not ours → no (never touch a file we do not own)
    * ours but unstamped → yes. Every file written before this change is in
      this bucket, and each one holds a token that died within an hour of
      being written. "Unknown lifetime" must mean refresh, not skip.
    * ours and stamped → yes once inside ``GH_HOSTS_REFRESH_MARGIN_SECONDS``
      of the recorded expiry.
    """
    if text is None:
        return True
    if not is_managed_gh_hosts(text):
        return False
    stamped = read_gh_hosts_expiry(text)
    if not stamped:
        return True
    try:
        expiry = _parse_iso_z(stamped)
    except ValueError:
        return True  # unparseable stamp = unknown lifetime = refresh
    reference = time.time() if now is None else now
    return expiry - reference <= GH_HOSTS_REFRESH_MARGIN_SECONDS


def _parse_iso_z(value: str) -> float:
    return (
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )


# ---------------------------------------------------------------------------
# .env token extraction
# ---------------------------------------------------------------------------


def _read_env_token(env_path: Path) -> Optional[tuple[str, str]]:
    """Return ``(token, source_var)`` from ``env_path`` or ``None``.

    Reads a flat ``.env`` file (no shell expansion). The first non-empty
    match across ``TOKEN_VARS`` (in precedence order) wins.
    """
    try:
        content = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for key in TOKEN_VARS:
        m = re.search(rf"^{re.escape(key)}=(.*)$", content, re.MULTILINE)
        if m:
            val = re.sub(r'^["\']|["\']$', "", m.group(1).strip())
            if val:
                return val, key
    return None


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvisionResult:
    home: Path
    provisioned: bool
    reason: Optional[str] = None
    source: Optional[str] = None
    # The git author identity actually written. Reported at boot because
    # "commits are authored as data@users.noreply.github.com" was invisible
    # from inside the container for weeks — nothing ever said which name won.
    identity: Optional[str] = None


def provision_git_credentials(
    home_dir: Path,
    *,
    name: Optional[str] = None,
    email: Optional[str] = None,
    force: bool = False,
) -> ProvisionResult:
    """Provision git auth for one agent home from its own ``.env`` token.

    ``home_dir`` is a HERMES_HOME-shaped directory: it holds ``.env`` at the
    root and the tool subprocess HOME at ``{home_dir}/home``. Idempotent;
    a no-op (``provisioned=False``) when no token is configured.

    Manage-if-ours: a file this module wrote is reconciled to current content
    on every boot (rotated token, corrected identity); a file it did not write
    is never touched unless ``force``. The previous apply-if-absent rule could
    not tell those two cases apart, so it protected the fleet's own fossils as
    carefully as it protected a human's config.
    """
    subprocess_home = home_dir / "home"
    cred_path = subprocess_home / ".git-credentials"
    cfg_path = subprocess_home / ".gitconfig"
    gh_hosts_path = subprocess_home / ".config" / "gh" / "hosts.yml"

    resolved_name = name or home_dir.name
    resolved_email = email or f"{resolved_name}@users.noreply.github.com"

    # The host broker beats everything: an agent on it holds nothing of its own.
    # It runs the App-mode path, since both are mint-on-demand. A PAT from before
    # the move can still be sitting in .git-credentials, so that goes too.
    if read_broker_config(home_dir) is not None:
        existing_creds = _read_text(cred_path)
        if existing_creds is not None and is_managed_git_credentials(existing_creds):
            cred_path.unlink()
        return _provision_app_mode(
            home_dir,
            subprocess_home=subprocess_home,
            cfg_path=cfg_path,
            gh_hosts_path=gh_hosts_path,
            name=resolved_name,
            email=resolved_email,
            force=force,
            source="GITHUB_BROKER",
        )

    # GitHub App mode takes precedence over a static PAT: an org-owned App is
    # the credential we want winning wherever both are configured (an agent
    # mid-migration may still carry its legacy PAT in .env).
    if read_app_credentials(home_dir / ".env") is not None:
        return _provision_app_mode(
            home_dir,
            subprocess_home=subprocess_home,
            cfg_path=cfg_path,
            gh_hosts_path=gh_hosts_path,
            name=resolved_name,
            email=resolved_email,
            force=force,
        )

    found = _read_env_token(home_dir / ".env")
    if found is None:
        return ProvisionResult(home_dir, False, reason="no GitHub token configured")
    token, source = found

    # git and gh are provisioned INDEPENDENTLY. An agent provisioned by an older
    # build may have git set up but no gh hosts.yml — gating both on "git
    # present" left gh permanently unconfigured. Each is reconciled on its own.
    #
    # A rotated PAT is the same fossil class as an expired App token: the .env
    # gets the new value, and under apply-if-absent the credential files kept
    # serving the old one until someone deleted them by hand. Reconciling
    # against desired content fixes rotation for free.
    subprocess_home.mkdir(parents=True, exist_ok=True)
    wrote: list[str] = []

    git_writes = [
        _write_managed(
            cred_path,
            build_git_credentials_content(token),
            mode=0o600,
            owned=is_managed_git_credentials,
            force=force,
        ),
        _write_managed(
            cfg_path,
            build_git_config_content(name=resolved_name, email=resolved_email),
            mode=0o644,
            owned=is_managed_gitconfig,
            force=force,
        ),
    ]
    if any(git_writes):
        wrote.append("git")

    if _write_managed(
        gh_hosts_path,
        build_gh_hosts_content(token),
        mode=0o600,
        owned=is_managed_gh_hosts,
        force=force,
    ):
        wrote.append("gh")

    if not wrote:
        return ProvisionResult(
            home_dir,
            False,
            reason="git+gh already current",
            source=source,
            identity=resolved_name,
        )
    return ProvisionResult(
        home_dir,
        True,
        reason="wrote " + "+".join(wrote),
        source=source,
        identity=resolved_name,
    )


def _provision_app_mode(
    home_dir: Path,
    *,
    subprocess_home: Path,
    cfg_path: Path,
    gh_hosts_path: Path,
    name: str,
    email: str,
    force: bool,
    source: str = "GITHUB_APP",
) -> ProvisionResult:
    """Provision GitHub App auth: helper-based ``.gitconfig`` + a lifetime-refreshed ``gh``.

    No ``.git-credentials`` is written — in App mode there is no long-lived
    token to store; git mints one per ``credential fill``. ``gh`` has no
    credential-helper hook, so its ``hosts.yml`` holds a real ~1h token at rest.
    That file is therefore refreshed against its **recorded expiry**, not
    against its existence: gating on ``not gh_hosts_path.exists()`` meant the
    first boot's token was the only one an agent ever got, and every container
    that had booted once was pinned to a credential that died an hour later.
    """
    subprocess_home.mkdir(parents=True, exist_ok=True)
    wrote: list[str] = []

    if _write_managed(
        cfg_path,
        build_git_config_content_app(
            name=name, email=email, python_exe=sys.executable
        ),
        mode=0o644,
        owned=is_managed_gitconfig,
        force=force,
    ):
        wrote.append("git(app)")

    existing = _read_text(gh_hosts_path)
    if force or gh_hosts_needs_refresh(existing):
        # Best-effort: a mint failure (network, bad key) must not break boot —
        # git still works via the helper, which retries on every invocation,
        # and gh is covered at the env boundary by the GH_TOKEN injection in
        # tools/environments/local.py.
        try:
            detail = get_installation_token_detail(home_dir)
        except Exception as exc:
            detail = None
            print(f"git-credentials: gh token mint failed ({exc})")
        if detail and _write_managed(
            gh_hosts_path,
            build_gh_hosts_content(detail.token, expires_at=detail.expires_at),
            mode=0o600,
            owned=is_managed_gh_hosts,
            force=force,
        ):
            wrote.append(f"gh(expires {detail.expires_at})")
        elif detail is None and existing is not None:
            # Say it out loud rather than leaving a dead credential in place
            # with a silent boot log — the exact failure mode this fixes.
            print(
                f"git-credentials: WARNING stale gh hosts.yml left in place at "
                f"{gh_hosts_path} (mint unavailable); gh falls back to the "
                f"GH_TOKEN injected per subprocess"
            )

    if not wrote:
        return ProvisionResult(
            home_dir,
            False,
            reason="git+gh already current",
            source=source,
            identity=name,
        )
    return ProvisionResult(
        home_dir,
        True,
        reason="wrote " + "+".join(wrote),
        source=source,
        identity=name,
    )


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _write_managed(
    path: Path,
    content: str,
    *,
    mode: int,
    owned,
    force: bool = False,
) -> bool:
    """Reconcile ``path`` to ``content``. Returns True when it wrote.

    The single ownership rule for every credential file this module produces:

    * absent → write
    * present, ``owned(text)`` false → leave alone (a human's or an imported
      agent's config; ``force`` overrides)
    * present, ours, content already equal → no write (keeps boot idempotent
      and mtimes honest)
    * present, ours, content differs → rewrite

    This is the whole of the "apply-if-absent" repair. Existence answered
    "has anyone written here?"; ownership answers "is what is written here
    still what we mean?", which is the question a rotating credential and a
    corrected identity both need asked.
    """
    existing = _read_text(path)
    if existing is not None and not force and not owned(existing):
        return False
    if existing == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_file(path, content, mode=mode)
    return True


def _write_file(path: Path, content: str, *, mode: int) -> None:
    """Write ``content`` then force ``mode`` (umask-independent)."""
    path.write_text(content, encoding="utf-8")
    os.chmod(path, mode)


def refresh_gh_hosts(subprocess_home: Path, token: str, expires_at: str) -> bool:
    """Bring a managed ``hosts.yml`` up to ``token`` if it has gone stale.

    Boot is not a schedule for a credential that lives ~1h and a container that
    lives weeks. This is the other half of the refresh: the tool-subprocess env
    boundary already mints (cache-first) a fresh installation token on every
    spawn, so it can hand the same token to ``hosts.yml`` for free — the file
    is then reconciled on the token's own lifetime, at the exact moment its
    staleness would have mattered.

    Cheap by construction: a stamped, still-valid file short-circuits on a
    ~200-byte read, so the actual write happens at most once per token
    (~50 min), never per spawn. Returns True when it wrote.
    """
    path = subprocess_home / ".config" / "gh" / "hosts.yml"
    existing = _read_text(path)
    if existing is not None and not is_managed_gh_hosts(existing):
        return False  # someone else's gh login — not ours to rotate
    if not gh_hosts_needs_refresh(existing):
        return False
    return _write_managed(
        path,
        build_gh_hosts_content(token, expires_at=expires_at),
        mode=0o600,
        owned=is_managed_gh_hosts,
    )


# Basenames a HERMES_HOME mount point can have that are never an agent's name.
# Falling through to one of these is how commits ended up authored as
# "data <data@users.noreply.github.com>" — /opt/data's basename standing in for
# an identity nobody ever chose.
_NON_IDENTITY_BASENAMES = frozenset(
    {"data", "home", "opt", "root", "srv", "var", "mnt", "app", "workspace", ""}
)


def resolve_root_identity(hermes_home: Path) -> tuple[Optional[str], Optional[str]]:
    """Resolve the default profile's git author name. Returns ``(name, warning)``.

    Precedence, most-to-least specific:

    1. ``HERMES_AGENT_NAME`` in the process env (set by the container runtime);
    2. ``HERMES_AGENT_NAME`` in the agent's own ``.env`` on the mounted volume —
       the durable source that survives being invoked outside ``with-contenv``,
       and the same file the credentials themselves come from;
    3. the mount basename, which in production is ``/opt/data`` → "data" and is
       not an identity at all. Returning it is reported as a warning rather than
       accepted silently, because that fallback is the bug, not the behaviour.
    """
    from_env = (os.environ.get("HERMES_AGENT_NAME") or "").strip()
    if from_env:
        return from_env, None
    content = _read_text(hermes_home / ".env")
    if content:
        from_file = _read_env_var_value(content, "HERMES_AGENT_NAME")
        if from_file:
            return from_file, None
    basename = hermes_home.name
    if basename.lower() in _NON_IDENTITY_BASENAMES:
        return None, (
            f"HERMES_AGENT_NAME unset and {hermes_home} has no agent name; "
            f"git identity would fall back to the mount basename "
            f"{basename!r} — commits will be authored as {basename!r}"
        )
    return None, None


def _read_env_var_value(content: str, key: str) -> Optional[str]:
    m = re.search(rf"^{re.escape(key)}=(.*)$", content, re.MULTILINE)
    if not m:
        return None
    return re.sub(r'^["\']|["\']$', "", m.group(1).strip()) or None


def provision_all(hermes_home: Path) -> list[ProvisionResult]:
    """Provision the default profile (HERMES_HOME root) + each named profile.

    Mirrors ``container_boot``'s model: the HERMES_HOME root is the implicit
    default profile, and named profiles live under ``{HERMES_HOME}/profiles/<name>/``.

    The root's basename is uninformative in production (``/opt/data`` → "data"),
    so the git identity for the default profile comes from
    :func:`resolve_root_identity` — commits are authored as the agent, not
    "data". A named profile is its own agent and is identified by its profile
    directory name.
    """
    root_name, warning = resolve_root_identity(hermes_home)
    if warning:
        print(f"git-credentials: WARNING {warning}")
    results = [provision_git_credentials(hermes_home, name=root_name)]
    profiles_dir = hermes_home / "profiles"
    if profiles_dir.is_dir():
        for profile in sorted(p for p in profiles_dir.iterdir() if p.is_dir()):
            results.append(provision_git_credentials(profile))
    return results


def main() -> int:
    """Entry point invoked from /etc/cont-init.d/03-provision-git-credentials."""
    hermes_home = Path(os.environ.get("HERMES_HOME", "/opt/data"))
    for r in provision_all(hermes_home):
        status = (
            f"provisioned (source={r.source}, {r.reason})"
            if r.provisioned
            else f"skipped ({r.reason})"
        )
        identity = f" identity={r.identity}" if r.identity else ""
        print(f"git-credentials: home={r.home}{identity} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
