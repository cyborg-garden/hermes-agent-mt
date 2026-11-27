"""Audit round 7 of the protected-session egress guard, as regression tests."""
from __future__ import annotations

import pytest

from agent import protected_session as ps

PRIMARY = {"model": "primary/model", "provider": "custom",
           "base_url": "http://127.0.0.1:9/v1", "api_key": "k", "api_mode": "chat_completions"}
SECRET = "placeholder protected words"


class Agent:
    def __init__(self, **kw):
        self.session_id = "s-a7"
        self._session_db = None
        self._session_init_model_config = {}
        self._protected_session = False
        self._protected_primary = dict(PRIMARY)
        self._cached_system_prompt = ""
        self.ephemeral_system_prompt = None
        self.prefill_messages = []
        self._session_messages = []
        self.__dict__.update(kw)


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import profiles
    from hermes_state import SessionDB
    other_dir = profiles.get_profile_dir("other")
    other_dir.mkdir(parents=True)
    other = SessionDB(db_path=other_dir / "state.db")
    other.create_session("sess-protected-other", "signal")
    other.append_message("sess-protected-other", "user", f"transcript {SECRET}")
    other.mark_session_protected("sess-protected-other")
    current = SessionDB(db_path=home / "state.db")
    current.create_session("sess-current", "cli")
    yield current, other
    other.close()
    current.close()


@pytest.mark.parametrize("sid", ["sess-protected-other", "other/sess-protected-other"])
def test_C1_cross_profile_read_of_a_protected_session_is_refused(two_profiles, sid):
    from tools.session_search_tool import session_search
    current, _ = two_profiles
    out = session_search(session_id=sid, db=current, current_session_id="sess-current")
    assert SECRET not in out


def test_C2_cronjob_is_refused():
    a = Agent()
    tok = ps.activate(a)
    ps.mark_protected(a, "test")
    try:
        assert ps.refuse_tool_if_protected(
            "cronjob", {"action": "create", "schedule": "1d", "prompt": "x"}) \
            == ps.PROTECTED_EGRESS_MESSAGE
    finally:
        ps.reset(tok)


def test_C3_a_registered_endpoint_is_exact_host_and_port():
    from gateway.config import PlatformConfig
    from gateway.platforms.signal import SignalAdapter
    SignalAdapter(PlatformConfig(enabled=True, extra={"http_url": "http://host.docker.internal:8080",
                                                      "account": "+10000000000"}))
    a = Agent()
    tok = ps.activate(a)
    ps.mark_protected(a, "test")
    try:
        with ps.egress_scope(ps.DELIVERY_SCOPE):
            ps.check_request("http://host.docker.internal:8080/api/v1/rpc", b"{}")
            with pytest.raises(ps.ProtectedEgressBlocked):
                ps.check_request("http://host.docker.internal:11434/api/tags", b"{}")
    finally:
        ps.reset(tok)


def test_C3_a_bare_tld_endpoint_does_not_cover_hosts_under_it():
    from gateway.config import PlatformConfig
    ps.register_adapter_endpoints(PlatformConfig(enabled=True, extra={"url": "http://com."}))
    assert not ps.delivery_endpoint_allowed("http://collector.com/x")
