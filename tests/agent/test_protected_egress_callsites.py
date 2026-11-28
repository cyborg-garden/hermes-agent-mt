"""Guard: every raw HTTP or model-client call site is listed and classified.

A protected session's text may only reach the configured primary. The runtime
check lives in ``agent/protected_session.py`` (httpx, requests, urllib, aiohttp
and botocore hooks). Libraries outside those hooks, and any new place that sends
text out, need a decision. This test fails when a call site appears in a file
that ``agent/protected_egress_allowlist.py`` does not list, or when a listed
file's count changes, so every new call site gets classified in review.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCAN_DIRS = ("agent", "tools", "plugins", "gateway", "hermes_cli", "tui_gateway",
             "acp_adapter", "cron", "providers")

CALL_SITE = re.compile(
    r"\brequests\.(?:get|post|put|patch|delete|head|options|request|Session)\s*\("
    r"|\burllib\.request\.(?:urlopen|build_opener)\s*\(|(?<![.\w])urlopen\s*\("
    r"|\baiohttp\.(?:ClientSession|request)\s*\("
    r"|\bhttpx\.(?:Client|AsyncClient|get|post|put|patch|delete|head|request|stream)\s*\("
    r"|\b(?:Async)?(?:Azure)?OpenAI\s*\("
    r"|\b(?:Async)?Anthropic(?:Bedrock|Vertex|Foundry)?\s*\("
    r"|\bboto3\.(?:client|Session|resource)\s*\("
    r"|\bhttp\.client\.HTTPS?Connection\s*\("
    r"|\burllib3\.(?:PoolManager|ProxyManager|request)\s*\("
    r"|\bwebsockets?\.(?:connect|create_connection)\s*\("
    r"|\bgenai\.Client\s*\("
)


def _files():
    for d in SCAN_DIRS:
        base = ROOT / d
        if base.is_dir():
            yield from (p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    yield from (p for p in ROOT.glob("*.py"))


def scan_call_sites() -> dict:
    out = {}
    for path in _files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        n = len(CALL_SITE.findall(text))
        if n:
            out[path.relative_to(ROOT).as_posix()] = n
    return out


def test_every_call_site_is_classified():
    from agent.protected_egress_allowlist import CALL_SITES, CATEGORIES

    found = scan_call_sites()
    unlisted = sorted(f for f in found if f not in CALL_SITES)
    changed = sorted(f"{f}: listed {CALL_SITES[f][0]}, found {n}"
                     for f, n in found.items() if f in CALL_SITES and CALL_SITES[f][0] != n)
    stale = sorted(f for f in CALL_SITES if f not in found)
    bad = sorted(f for f, (_n, cat) in CALL_SITES.items() if cat not in CATEGORIES)
    assert not unlisted, (
        "New raw HTTP / model-client call sites. Decide how a protected session is kept "
        "off them and list each in agent/protected_egress_allowlist.py:\n" + "\n".join(unlisted))
    assert not changed, ("Call-site count changed; re-check and update "
                         "agent/protected_egress_allowlist.py:\n" + "\n".join(changed))
    assert not stale, "Listed files with no call sites left:\n" + "\n".join(stale)
    assert not bad, "Unknown categories:\n" + "\n".join(bad)


def test_the_guard_catches_a_new_call_site(tmp_path):
    """Proves the scan can fail: a file using each library is detected."""
    samples = [
        "import requests\nrequests.post(url, json=body)\n",
        "import urllib.request\nurllib.request.urlopen(req)\n",
        "import aiohttp\naiohttp.ClientSession()\n",
        "import httpx\nhttpx.Client(timeout=5)\n",
        "from openai import OpenAI\nOpenAI(api_key=k)\n",
        "from anthropic import Anthropic\nAnthropic()\n",
        "import boto3\nboto3.client('bedrock-runtime')\n",
    ]
    for s in samples:
        assert CALL_SITE.findall(s), s
