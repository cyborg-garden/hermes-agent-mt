"""Protected-session egress guard.

"Protected text" is Papa Reo transcript text, recognised by its markers (the
transcript footer by default, plus ``routing.protected_markers``). It may only
ever reach the agent's configured primary model. This module holds that rule in
one place:

* **One sticky flag per session.** :func:`scan` looks for markers in anything
  entering the session (the user message, history, tool results, plugin context,
  the system prompt) and, on a hit, calls :func:`mark_protected`. The flag lives on
  the agent (``_protected_session``), in the session record (``model_config``
  ``_protected``, so it survives context compression and agent rebuilds) and is
  inherited by children. Nothing clears it. Every check reads the flag; the
  marker scan only ever sets it.

* **One egress check at the lowest layer.** :func:`install` hooks
  ``httpx.Client.send`` / ``httpx.AsyncClient.send`` (every OpenAI- and
  Anthropic-style SDK client), ``requests``, ``urllib``, ``aiohttp`` and
  botocore (Bedrock). While the running session is protected, a model request to
  anything other than the configured primary's endpoint and model is refused
  before a byte is sent, and so is any other outbound request made inside a tool,
  plugin hook or memory-provider scope (:func:`egress_scope`) unless it targets
  loopback or the scope is explicitly allowed. Platform delivery to the user
  (gateway adapters, outside any scope) is unaffected.

The running session travels in a ContextVar. ``ThreadPoolExecutor.submit`` and
``threading.Thread.start`` carry it (and the scope) into worker threads, so a
thread spawned mid-turn is guarded too.

The configured primary is frozen when the agent is built from its config
(``agent._protected_primary``); it is never read from the live runtime, which a
fallback, a cooldown or a cheap route may have moved.

Errors raised here never carry content: no URL, body, message or argument.
"""

from __future__ import annotations

import contextvars
import ipaddress
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Iterator, Optional
from urllib.parse import unquote, urlsplit

logger = logging.getLogger(__name__)

# Shown to the user / model when the guard refuses something. No content, ever.
PROTECTED_EGRESS_MESSAGE = (
    "Blocked: this session holds protected text, which may only be sent to this "
    "agent's configured primary model. Nothing was sent."
)


class ProtectedEgressBlocked(RuntimeError):
    """Raised before sending when a protected session would leave the primary.

    ``kind`` is a category (``"model"``, ``"tool:<name>"``, ...), never content.
    """

    def __init__(self, kind: str = "request") -> None:
        super().__init__(PROTECTED_EGRESS_MESSAGE)
        self.kind = kind


class EgressSession:
    """What the egress check needs to know about the running session."""

    __slots__ = ("agent", "primary", "_flag")

    def __init__(self, agent: Any = None, primary: Optional[dict] = None,
                 protected: bool = False) -> None:
        self.agent = agent
        self.primary = primary
        self._flag = bool(protected)

    @property
    def protected(self) -> bool:
        if self._flag:
            return True
        agent = self.agent
        return agent is not None and getattr(agent, "_protected_session", False) is True

    def mark(self, source: str = "scan") -> None:
        self._flag = True
        if self.agent is not None:
            mark_protected(self.agent, source)


_SESSION: contextvars.ContextVar = contextvars.ContextVar(
    "hermes_protected_egress_session", default=None)
_SCOPE: contextvars.ContextVar = contextvars.ContextVar(
    "hermes_protected_egress_scope", default=None)


# ── markers ──────────────────────────────────────────────────────────────────

def configured_markers(agent: Any = None) -> tuple:
    """Default markers plus ``routing.protected_markers`` (read whether or not
    routing is enabled) plus any a route attached to ``agent``. Lower-cased."""
    from agent.routing_override import DEFAULT_PROTECTED_MARKERS

    extra: list = []
    try:
        from hermes_cli.config import load_config_readonly

        routing = (load_config_readonly() or {}).get("routing") or {}
        cfg = routing.get("protected_markers") if isinstance(routing, dict) else None
        if isinstance(cfg, str):
            cfg = [cfg]
        if isinstance(cfg, (list, tuple)):
            extra.extend(cfg)
    except Exception:
        logger.debug("protected markers: config unreadable; defaults only", exc_info=True)
    if agent is not None:
        route_marks = getattr(agent, "_routing_protected_markers", None)
        if isinstance(route_marks, (list, tuple)):
            extra.extend(route_marks)
    out = list(DEFAULT_PROTECTED_MARKERS)
    for m in extra:
        m = str(m).strip().lower()
        if m and m not in out:
            out.append(m)
    return tuple(out)


def text_has_marker(*texts: Any, agent: Any = None) -> bool:
    """True when any of ``texts`` (strings, messages, content parts) holds a marker."""
    from agent.routing_override import carries_protected_text

    return carries_protected_text([], list(texts), configured_markers(agent))


# ── the flag ─────────────────────────────────────────────────────────────────

def is_protected(agent: Any = None) -> bool:
    """The sticky flag of ``agent``, or of the running session when ``agent`` is None."""
    if agent is not None:
        if getattr(agent, "_protected_session", False) is True:
            return True
    sess = _SESSION.get()
    if sess is None:
        return False
    if agent is not None and sess.agent is not None and sess.agent is not agent:
        return False
    return sess.protected


def mark_protected(agent: Any, source: str = "scan") -> None:
    """Set the sticky flag on ``agent`` and persist it in its session record.

    Never clears anything and never raises. ``source`` is a category for the log
    line, never content."""
    if agent is None:
        return
    already = getattr(agent, "_protected_session", False) is True
    try:
        agent._protected_session = True
    except Exception:
        return
    try:
        cfg = getattr(agent, "_session_init_model_config", None)
        if isinstance(cfg, dict):
            cfg["_protected"] = True
        else:
            agent._session_init_model_config = {"_protected": True}
    except Exception:
        pass
    sid = getattr(agent, "session_id", None)
    if not already or getattr(agent, "_protected_persisted_sid", None) != sid:
        _persist(agent, sid)
    if not already:
        logger.info("protected session: flag set (source=%s, session=%s)", source, sid)


def _persist(agent: Any, sid: Any) -> None:
    db = getattr(agent, "_session_db", None)
    if not sid or db is None or not hasattr(db, "mark_session_protected"):
        return
    try:
        db.mark_session_protected(sid)
        agent._protected_persisted_sid = sid
    except Exception:
        logger.debug("protected session: could not persist the flag", exc_info=True)


def load_persisted(agent: Any) -> bool:
    """Pick the flag back up from the session record (this session or any
    ancestor: compression continuations, branches, delegating parents)."""
    if agent is None or getattr(agent, "_protected_session", False) is True:
        return bool(agent is not None)
    sid = getattr(agent, "session_id", None)
    if not sid or getattr(agent, "_protected_checked_sid", None) == sid:
        return False
    db = getattr(agent, "_session_db", None)
    if db is None or not hasattr(db, "session_is_protected"):
        return False
    try:
        hit = db.session_is_protected(sid) is True
    except Exception:
        logger.debug("protected session: could not read the flag", exc_info=True)
        return False
    agent._protected_checked_sid = sid
    if hit:
        agent._protected_session = True
        agent._protected_persisted_sid = sid
        logger.info("protected session: flag restored from the session record (session=%s)", sid)
    return hit


def scan(agent: Any, messages: Any = None, *extras: Any, source: str = "scan") -> bool:
    """Set the flag if ``messages`` or ``extras`` hold a marker. Returns the flag.

    The only thing that ever sets the flag from content. Fails closed: a scan
    that raises marks the session."""
    if agent is not None and getattr(agent, "_protected_session", False) is True:
        return True
    try:
        from agent.routing_override import agent_extra_text, carries_protected_text

        extra_text = agent_extra_text(agent, tuple(e for e in extras if e)) if agent is not None \
            else [e for e in extras if e]
        hit = carries_protected_text(messages, extra_text, configured_markers(agent))
    except Exception:
        logger.warning("protected session: marker scan failed; treating the session as protected")
        hit = True
    if hit:
        if agent is not None:
            mark_protected(agent, source)
        else:
            sess = _SESSION.get()
            if sess is not None:
                sess.mark(source)
    return hit or is_protected(agent)


def scan_running(*texts: Any, source: str = "scan") -> bool:
    """Scan ``texts`` against the running session (tool results, aux requests)."""
    sess = _SESSION.get()
    if sess is not None and sess.protected:
        return True
    agent = sess.agent if sess is not None else None
    try:
        hit = text_has_marker(*texts, agent=agent)
    except Exception:
        hit = True
    if hit and sess is not None:
        sess.mark(source)
    return hit


def skip_side_task(name: str, agent: Any = None, *texts: Any, session_db: Any = None,
                   session_id: Any = None) -> bool:
    """True when side task ``name`` (title, review, goal judge, MoA, query rewrite)
    must not run: the session is protected, by flag, by the running session, by its
    record, or because ``texts`` hold a marker (which marks it). Fails closed."""
    try:
        if agent is not None:
            hit = scan(agent, list(texts), source=name) if texts else is_protected(agent)
        else:
            hit = is_protected() or (bool(texts) and scan_running(*texts, source=name))
        if not hit and session_db is not None and session_id \
                and hasattr(session_db, "session_is_protected"):
            hit = session_db.session_is_protected(session_id) is True
    except Exception:
        hit = True
    if hit:
        logger.info("protected session: skipped %s", name)
    return hit


# ── the configured primary ───────────────────────────────────────────────────

_PRIMARY_KEYS = ("model", "provider", "base_url", "api_key", "api_mode")


def freeze_primary(agent: Any) -> None:
    """Record the configured primary at construction. Called once, from agent init."""
    try:
        primary = {k: str(getattr(agent, k, "") or "") for k in _PRIMARY_KEYS}
        anthropic_base = getattr(agent, "_anthropic_base_url", None)
        if anthropic_base:
            primary["anthropic_base_url"] = str(anthropic_base)
        agent._protected_primary = primary
    except Exception:
        logger.debug("protected session: could not record the primary", exc_info=True)


def configured_primary(agent: Any) -> Optional[dict]:
    """The configured primary of ``agent`` (frozen at construction, or a
    protected parent's), never its live runtime."""
    primary = getattr(agent, "_protected_primary", None)
    if isinstance(primary, dict) and primary.get("model"):
        return primary
    return None


def inherit(child: Any, parent: Any) -> None:
    """A child inherits its parent's configured primary, and its flag if set."""
    primary = configured_primary(parent)
    if primary is not None:
        child._protected_primary = dict(primary)
    if is_protected(parent):
        mark_protected(child, "parent")


# ── the running session ──────────────────────────────────────────────────────

def activate(agent: Any) -> contextvars.Token:
    """Make ``agent`` the running session in this context. Returns the reset token."""
    return _SESSION.set(EgressSession(agent, configured_primary(agent)))


def reset(token: Optional[contextvars.Token]) -> None:
    if token is None:
        return
    try:
        _SESSION.reset(token)
    except (RuntimeError, ValueError):
        pass


def current_session() -> Optional[EgressSession]:
    return _SESSION.get()


def running_primary() -> Optional[dict]:
    sess = _SESSION.get()
    return sess.primary if sess is not None else None


@contextmanager
def session(agent: Any) -> Iterator[None]:
    """Run a block with ``agent`` as the running session (manual /compress etc.)."""
    token = activate(agent)
    try:
        yield
    finally:
        reset(token)


@contextmanager
def egress_scope(name: str) -> Iterator[None]:
    """Mark a block as a tool, plugin hook or memory provider. In a protected
    session, outbound requests inside it are refused unless they go to loopback
    or the primary, or ``name`` is allowed."""
    token = _SCOPE.set(str(name))
    try:
        yield
    finally:
        try:
            _SCOPE.reset(token)
        except (RuntimeError, ValueError):
            pass


def plugin_scope_name(callback: Any) -> str:
    """``plugin:<name>`` for a hook callback, from the module it was defined in."""
    module = str(getattr(callback, "__module__", "") or "")
    parts = module.split(".")
    for i, part in enumerate(parts[:-1]):
        if part in ("plugins", "hermes_plugins", "hermes_user_plugins"):
            return f"plugin:{parts[i + 1]}"
    return f"plugin:{parts[0] if parts and parts[0] else 'unknown'}"


def refuse_tool_if_protected(tool_name: str) -> Optional[str]:
    """The generic refusal for a tool that cannot run in a protected session, else None."""
    from agent.protected_egress_allowlist import BLOCKED_TOOLS, BLOCKED_TOOL_PREFIXES

    if not is_protected():
        return None
    name = str(tool_name or "")
    if _scope_allowed(f"tool:{name}"):
        return None
    if name in BLOCKED_TOOLS or name.startswith(BLOCKED_TOOL_PREFIXES):
        logger.info("protected session: refused tool %s", name)
        return PROTECTED_EGRESS_MESSAGE
    return None


# ── the egress check ─────────────────────────────────────────────────────────

# Model calls named by URL rather than body (Gemini native, Bedrock runtime). A
# path suffix such as "/messages" alone is not enough: chat platforms use it too.
_MODEL_PATH_MARKERS = (":generatecontent", ":streamgeneratecontent", ":counttokens",
                       ":embedcontent", ":batchembedcontents")
_BEDROCK_SUFFIXES = ("/converse", "/converse-stream", "/invoke", "/invoke-with-response-stream")
_MAX_BODY_PARSE = 64 * 1024 * 1024


def _scope_allowed(scope: Optional[str]) -> bool:
    if not scope:
        return False
    from agent.protected_egress_allowlist import ALLOWED_SCOPES

    if scope in ALLOWED_SCOPES:
        return True
    try:
        from hermes_cli.config import load_config_readonly

        routing = (load_config_readonly() or {}).get("routing") or {}
        extra = routing.get("protected_egress_allow") if isinstance(routing, dict) else None
        if isinstance(extra, str):
            extra = [extra]
        return isinstance(extra, (list, tuple)) and scope in {str(x) for x in extra}
    except Exception:
        return False


def _split(url: Any):
    try:
        parts = urlsplit(str(url))
    except Exception:
        return None
    scheme = (parts.scheme or "").lower()
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is None:
        port = 443 if scheme in ("https", "wss") else 80 if scheme in ("http", "ws") else None
    return scheme, host, port, unquote(parts.path or "")


def _is_loopback(host: str) -> bool:
    if host in ("localhost", "localhost.localdomain") or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _endpoint_matches(url: Any, base_url: Any) -> bool:
    req, base = _split(url), _split(base_url)
    if not req or not base or not base[1]:
        return False
    if req[1] != base[1] or req[2] != base[2]:
        return False
    base_path = base[3].rstrip("/")
    return not base_path or req[3] == base_path or req[3].startswith(base_path + "/")


def _model_core(model: str) -> str:
    """Bedrock-style ids without region and vendor prefixes, ``:N`` version suffix kept."""
    tail = model
    for prefix in ("us.", "eu.", "apac.", "global.", "jp.", "au."):
        if tail.startswith(prefix):
            tail = tail[len(prefix):]
    head = tail.split(".", 1)[0]
    if "." in tail and head in ("anthropic", "meta", "amazon", "mistral", "cohere", "ai21",
                                "deepseek", "openai", "qwen", "moonshotai"):
        tail = tail.split(".", 1)[1]
    return tail


def _same_model(a: Any, b: Any) -> bool:
    """Same model, allowing one side to carry a ``vendor/`` prefix the other lacks
    (an aggregator id vs a native id), Bedrock region/vendor prefixes, and ``.``/``-``
    spelling of versions. Two different ``vendor/name`` ids never match."""
    x, y = str(a or "").strip().lower(), str(b or "").strip().lower()
    if not x or not y:
        return False
    if ("/" in x) != ("/" in y):
        x, y = x.split("/")[-1], y.split("/")[-1]
    x, y = _model_core(x), _model_core(y)
    return x == y or x.replace(".", "-") == y.replace(".", "-")


def _body_json(body: Any) -> Optional[dict]:
    if body is None:
        return None
    if isinstance(body, dict):
        return body
    if isinstance(body, str):
        body = body.encode("utf-8", "replace")
    if not isinstance(body, (bytes, bytearray, memoryview)):
        return None
    raw = bytes(body)
    if not raw or len(raw) > _MAX_BODY_PARSE or raw.lstrip()[:1] != b"{":
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _looks_like_model_request(path: str, body: Optional[dict], headers: Any) -> bool:
    p = (path or "").lower().rstrip("/")
    if any(mk in p for mk in _MODEL_PATH_MARKERS):
        return True
    if "/model/" in p and p.endswith(_BEDROCK_SUFFIXES):
        return True
    if isinstance(body, dict) and isinstance(body.get("model"), str) and body.get("model"):
        return True
    try:
        keys = {str(k).lower() for k in (headers or {}).keys()}
    except Exception:
        keys = set()
    return "anthropic-version" in keys or "x-stainless-lang" in keys


def _primary_bases(primary: dict) -> list:
    return [b for b in (primary.get("base_url"), primary.get("anthropic_base_url")) if b]


def _model_matches(primary: dict, path: str, body: Optional[dict]) -> bool:
    want = primary.get("model")
    sent = body.get("model") if isinstance(body, dict) else None
    if sent:
        return _same_model(sent, want)
    # Model in the URL (Gemini native, Bedrock): any path segment naming the model.
    segs = [s for s in (path or "").split("/") if s]
    names = segs + [s.split(":")[0] for s in segs if ":" in s]
    return any(_same_model(s, want) for s in names) or not any(
        s in ("models", "model", "deployments") for s in segs)


def check_request(url: Any, body: Any = None, headers: Any = None, *, kind: str = "http") -> None:
    """Refuse (raise :class:`ProtectedEgressBlocked`) before sending, or return.

    Acts only while the running session is protected."""
    sess = _SESSION.get()
    if sess is None or not sess.protected:
        return
    parts = _split(url)
    path = parts[3] if parts else ""
    data = _body_json(body)
    primary = sess.primary or {}
    on_primary_endpoint = any(_endpoint_matches(url, b) for b in _primary_bases(primary))
    if _looks_like_model_request(path, data, headers):
        if primary.get("model") and on_primary_endpoint and _model_matches(primary, path, data):
            return
        if primary.get("model") and not _primary_bases(primary) and _model_matches(primary, path, data):
            return  # primary configured without an endpoint: the model is all we can match
        logger.warning("protected session: refused a %s model request off the primary", kind)
        raise ProtectedEgressBlocked("model")
    scope = _SCOPE.get()
    if not scope:
        return  # platform delivery, startup, sync: not tool or plugin egress
    if on_primary_endpoint or (parts and _is_loopback(parts[1])) or _scope_allowed(scope):
        return
    logger.warning("protected session: refused outbound %s request from %s", kind, scope)
    raise ProtectedEgressBlocked(scope)


# ── hooks ────────────────────────────────────────────────────────────────────

_INSTALLED = False
_INSTALL_LOCK = threading.Lock()


def _httpx_body(request: Any) -> Any:
    try:
        return request.content
    except Exception:  # streaming body not read yet: judged on URL and headers
        return None


def _install_httpx() -> None:
    try:
        import httpx
    except Exception:
        return
    orig_send = httpx.Client.send
    orig_async_send = httpx.AsyncClient.send

    def send(self, request, *args, **kwargs):
        if _SESSION.get() is not None:
            check_request(request.url, _httpx_body(request), request.headers, kind="httpx")
        return orig_send(self, request, *args, **kwargs)

    async def async_send(self, request, *args, **kwargs):
        if _SESSION.get() is not None:
            check_request(request.url, _httpx_body(request), request.headers, kind="httpx")
        return await orig_async_send(self, request, *args, **kwargs)

    send.__wrapped__ = orig_send  # type: ignore[attr-defined]
    async_send.__wrapped__ = orig_async_send  # type: ignore[attr-defined]
    httpx.Client.send = send
    httpx.AsyncClient.send = async_send


def _install_requests() -> None:
    try:
        import requests
    except Exception:
        return
    orig = requests.Session.send

    def send(self, request, **kwargs):
        if _SESSION.get() is not None:
            check_request(request.url, request.body, request.headers, kind="requests")
        return orig(self, request, **kwargs)

    send.__wrapped__ = orig  # type: ignore[attr-defined]
    requests.Session.send = send


def _install_urllib() -> None:
    import urllib.request

    orig = urllib.request.OpenerDirector.open

    def open_(self, fullurl, data=None, *args, **kwargs):
        if _SESSION.get() is not None:
            if isinstance(fullurl, urllib.request.Request):
                url, body, headers = fullurl.full_url, data if data is not None else fullurl.data, \
                    dict(fullurl.header_items())
            else:
                url, body, headers = fullurl, data, None
            check_request(url, body, headers, kind="urllib")
        return orig(self, fullurl, data, *args, **kwargs)

    open_.__wrapped__ = orig  # type: ignore[attr-defined]
    urllib.request.OpenerDirector.open = open_


def _install_aiohttp() -> None:
    try:
        import aiohttp
    except Exception:
        return
    orig = aiohttp.ClientSession._request

    async def _request(self, method, str_or_url, *args, **kwargs):
        if _SESSION.get() is not None:
            body = kwargs.get("json")
            if body is None:
                body = kwargs.get("data")
            check_request(str_or_url, body, kwargs.get("headers"), kind="aiohttp")
        return await orig(self, method, str_or_url, *args, **kwargs)

    _request.__wrapped__ = orig  # type: ignore[attr-defined]
    aiohttp.ClientSession._request = _request


def _botocore_before_send(request: Any = None, **_kw: Any) -> None:
    if request is None or _SESSION.get() is None:
        return
    check_request(getattr(request, "url", ""), getattr(request, "body", None),
                  getattr(request, "headers", None), kind="botocore")


def _install_botocore() -> None:
    session_mod = sys.modules.get("botocore.session")
    if session_mod is None:
        try:
            import botocore.session as session_mod  # noqa: PLC0415
        except Exception:
            return
    if getattr(session_mod.Session.create_client, "_hermes_protected_egress", False):
        return
    orig = session_mod.Session.create_client

    def create_client(self, *args, **kwargs):
        client = orig(self, *args, **kwargs)
        try:
            client.meta.events.register("before-send", _botocore_before_send,
                                        unique_id="hermes-protected-egress")
        except Exception:
            logger.debug("protected session: could not hook a botocore client", exc_info=True)
        return client

    create_client.__wrapped__ = orig  # type: ignore[attr-defined]
    create_client._hermes_protected_egress = True  # type: ignore[attr-defined]
    session_mod.Session.create_client = create_client


def _install_thread_propagation() -> None:
    orig_submit = ThreadPoolExecutor.submit
    orig_start = threading.Thread.start

    def submit(self, fn, /, *args, **kwargs):
        sess, scope = _SESSION.get(), _SCOPE.get()

        def run(*a, **k):
            t1, t2 = _SESSION.set(sess), _SCOPE.set(scope)
            try:
                return fn(*a, **k)
            finally:
                try:
                    _SCOPE.reset(t2)
                    _SESSION.reset(t1)
                except (RuntimeError, ValueError):
                    pass

        return orig_submit(self, run, *args, **kwargs)

    def start(self):
        sess = _SESSION.get()
        if sess is not None:
            scope = _SCOPE.get()
            inner = self.run

            def run():
                _SESSION.set(sess)
                _SCOPE.set(scope)
                return inner()

            self.run = run
        return orig_start(self)

    submit.__wrapped__ = orig_submit  # type: ignore[attr-defined]
    start.__wrapped__ = orig_start  # type: ignore[attr-defined]
    ThreadPoolExecutor.submit = submit
    threading.Thread.start = start


# Event loops that deliver to the person (the gateway's loop). Work handed to one
# from a tool thread (a progress line, a clarify question, an approval prompt) is
# delivery, not the tool's own egress, so it leaves the tool's scope behind.
_DELIVERY_LOOPS: "weakref.WeakSet" = None  # type: ignore[assignment]


def register_delivery_loop(loop: Any) -> None:
    """Mark ``loop`` as the person's delivery loop (called by the gateway)."""
    global _DELIVERY_LOOPS
    import weakref

    if _DELIVERY_LOOPS is None:
        _DELIVERY_LOOPS = weakref.WeakSet()
    if loop is not None:
        _DELIVERY_LOOPS.add(loop)


def _install_delivery_hop() -> None:
    import asyncio.base_events as base_events

    orig = base_events.BaseEventLoop.call_soon_threadsafe

    def call_soon_threadsafe(self, callback, *args, context=None):
        if (context is None and _SCOPE.get() is not None and _DELIVERY_LOOPS is not None
                and self in _DELIVERY_LOOPS):
            context = contextvars.copy_context()
            context.run(_SCOPE.set, None)
        return orig(self, callback, *args, context=context)

    call_soon_threadsafe.__wrapped__ = orig  # type: ignore[attr-defined]
    base_events.BaseEventLoop.call_soon_threadsafe = call_soon_threadsafe


class _PatchOnImport:
    """Meta-path finder that applies a hook right after its module first imports, so
    installing the guard never pays for importing httpx, aiohttp or botocore."""

    def __init__(self, hooks: dict) -> None:
        self.hooks = hooks  # module name -> installer

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.hooks:
            return None
        import importlib.util

        installer = self.hooks.pop(fullname)  # popped first: no re-entry while finding
        spec = importlib.util.find_spec(fullname)
        if fullname in sys.modules:
            # Finding the spec imported the parent package, which imported this
            # module already: hook it now and let the import system use it.
            _safe(installer)
            return None
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                installer()
            except Exception:
                logger.warning("protected session: could not install %s", installer.__name__,
                               exc_info=True)

        try:
            loader.exec_module = exec_module  # type: ignore[method-assign]
        except Exception:
            return None  # cannot hook the loader: the import proceeds unhooked
        return spec


def install() -> None:
    """Install every hook once per process. Idempotent; never raises. Libraries not
    imported yet are hooked the moment they are."""
    global _INSTALLED
    if _INSTALLED:
        return
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        lazy = {}
        for module, step in (("httpx", _install_httpx), ("requests", _install_requests),
                             ("aiohttp", _install_aiohttp),
                             ("botocore.session", _install_botocore)):
            if module in sys.modules:
                _safe(step)
            else:
                lazy[module] = step
        for step in (_install_urllib, _install_thread_propagation, _install_delivery_hop):
            _safe(step)
        if lazy:
            sys.meta_path.insert(0, _PatchOnImport(lazy))
        _INSTALLED = True


def _safe(step) -> None:
    try:
        step()
    except Exception:
        logger.warning("protected session: could not install %s", step.__name__, exc_info=True)


install()
