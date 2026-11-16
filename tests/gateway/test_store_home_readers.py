"""Readers of a store's profile directory find the profile, not a linked file's volume.

Follow-up to #224 (#187). ``state.db`` may be a symlink onto a state volume, and the
shared registry opens handles by their resolved path, so ``Path(db.db_path).parent`` is
the volume. Every reader below wants the profile home (its config, ``sessions/``,
``cache/``, roster, mailbox, liveness registry). Each test opens a real linked store
through the registry and checks the directory the reader actually used.
"""
import asyncio
import contextlib
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.gateway.test_state_store_ownership import _linked_home


class _Stop(Exception):
    """Raised by a spy once it has seen the directory, to end the call under test."""


@pytest.fixture
def linked(tmp_path, monkeypatch):
    """A registry handle on a profile whose state.db is a link onto a volume."""
    from hermes_state_registry import acquire, release
    home, volume = _linked_home(tmp_path, monkeypatch)
    db = acquire(home / 'state.db')
    try:
        assert Path(db.db_path).parent == volume.resolve()   # sentinel: the handle is on the volume
        yield SimpleNamespace(home=home.resolve(), volume=volume.resolve(), db=db,
                              authority=SimpleNamespace(db=db, profile_id='default'))
    finally:
        release(db)


def _spy(seen):
    def spy(home, *args, **kwargs):
        seen.append(Path(home))
        raise _Stop
    return spy


# --- the shared helper -------------------------------------------------------------------


def test_store_dir_keeps_a_plain_path_as_given_and_resolves_a_registry_handle(linked, tmp_path):
    from types import SimpleNamespace as NS
    from hermes_state_home import store_dir
    real = tmp_path / 'real'
    real.mkdir()
    (tmp_path / 'alias').symlink_to(real, target_is_directory=True)
    # No entry_path (own-path handle, test double): exactly Path(db_path).parent, unresolved.
    assert store_dir(NS(db_path=str(tmp_path / 'alias' / 'state.db'))) == tmp_path / 'alias'
    assert store_dir(tmp_path / 'alias' / 'state.db') == tmp_path / 'alias'
    assert store_dir(linked.db) == linked.home


def test_the_gateway_helper_is_the_shared_one():
    import hermes_state_home
    from gateway import session_store_home
    assert session_store_home.store_home is hermes_state_home.store_home
    assert session_store_home.entry_path is hermes_state_home.entry_path


# --- agent/ and tools/ -------------------------------------------------------------------


def test_system_prompt_agent_home(linked):
    from agent.system_prompt import _agent_home
    assert _agent_home(SimpleNamespace(_session_db=linked.db)) == linked.home


def test_bot_mode_dm_agent_home(linked):
    from tools.bot_mode_dm import _agent_home
    assert _agent_home(SimpleNamespace(_session_db=linked.db)) == str(linked.home)


def test_delegate_parent_live_home(linked):
    from tools.delegate_tool import _parent_live_home
    assert _parent_live_home(SimpleNamespace(_session_db=linked.db)) == linked.home


# --- hermes_state ------------------------------------------------------------------------


def test_retired_spillover_is_collected_from_the_profile_cache(linked):
    """Spillover archives live under ``<home>/cache/spillover``, never beside the linked file."""
    from hermes_state_media import SPILLOVER_PREFIX, collect_retired_spillover
    spill = linked.home / 'cache' / 'spillover'
    spill.mkdir(parents=True)
    (spill / 'retired.txt').write_text('x')
    linked.db.set_meta(SPILLOVER_PREFIX + 'retired.txt', '"retired.txt"')
    collect_retired_spillover(linked.db)
    assert not (spill / 'retired.txt').exists()
    assert linked.db.get_meta(SPILLOVER_PREFIX + 'retired.txt') is None


def test_session_delete_hook_runs_under_the_profile_home(linked, monkeypatch):
    import hermes_constants
    from hermes_state_common import emit_session_deleted
    seen = []
    monkeypatch.setattr(hermes_constants, 'profile_name_for_home', lambda home: seen.append(Path(home)))
    monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', lambda *a, **k: None)
    emit_session_deleted(linked.db, ['sid'], reason='delete')
    assert seen == [linked.home]


# --- gateway/session_bot -----------------------------------------------------------------


def test_bot_delivery_home_check(linked, monkeypatch):
    from gateway import session_bot
    seen = []
    monkeypatch.setattr('gateway.session_authorities.served_profile_name', _spy(seen))
    with pytest.raises(_Stop):
        session_bot._home(linked.authority, SimpleNamespace(), 'default')
    assert seen == [linked.home]


def test_bot_relay_operation_home(linked, monkeypatch):
    from gateway import session_bot
    seen = []
    monkeypatch.setattr('gateway.session_authorities.served_profile_name', _spy(seen))
    with pytest.raises(_Stop):
        session_bot.relay_operation(SimpleNamespace(authority=linked.authority, actor=SimpleNamespace()), 'roster', {})
    assert seen == [linked.home]


def test_bot_delivery_recovery_scans_the_profile_home(linked, monkeypatch):
    from gateway import session_bot
    seen = []

    async def scan(authority, home):
        seen.append(Path(home))
        raise _Stop

    monkeypatch.setattr(session_bot, '_scan_owner_receipts', scan)
    with pytest.raises(_Stop):
        asyncio.run(session_bot.recover_bot_deliveries(linked.authority))
    assert seen == [linked.home]


def test_api_peer_turn_names_the_profile_home(linked, monkeypatch):
    from gateway.platforms import api_server_bot_chat as chat
    seen = []
    monkeypatch.setattr(chat, '_authority', lambda runner: linked.authority)
    monkeypatch.setattr(chat, 'bot_chat_tip', lambda authority, sid: {'user_id': 'owner'})
    monkeypatch.setattr('gateway.session_bot.displaceable_bot_chat', lambda authority, row: None)
    monkeypatch.setattr('gateway.session_authorities.served_profile_name', _spy(seen))
    with pytest.raises(_Stop):
        asyncio.run(chat.admit_peer_turn(SimpleNamespace(), 'sid', 'hello', None))
    assert seen == [linked.home]


# --- gateway/ other readers --------------------------------------------------------------


def test_onboarding_discard_uses_the_profile_sessions_dir(linked, monkeypatch):
    from gateway.session_onboarding import _discard_unfinished_runs
    seen = []
    monkeypatch.setattr('agent.onboarding_record.delete_chat', lambda db, sid, delete: delete())
    monkeypatch.setattr(linked.db, 'delete_session',
                        lambda sid, sessions_dir=None, **kw: seen.append(Path(sessions_dir)) or True)
    discards = SimpleNamespace(lock=threading.Lock(), ids={'old'})
    assert _discard_unfinished_runs(linked.db, ['old'], discards) == ['old']
    assert seen == [linked.home / 'sessions']


def test_hosted_room_service_root_is_the_profile_home(linked):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    service = SimpleNamespace(authority=linked.authority, db_path=Path(linked.db.db_path))
    assert CanonicalHostedRoomService.root.fget(service) == linked.home


def test_roster_session_fields_are_cached_under_the_profile_home(linked, monkeypatch):
    from gateway import session_group_controls
    seen = []
    monkeypatch.setattr('tui_gateway.profile_roster_cache.cached_reader_session_fields', _spy(seen))
    with pytest.raises(_Stop):
        session_group_controls._bot_sessions(linked.authority, SimpleNamespace(subject='s', capabilities=frozenset()))
    assert seen == [linked.home]


def test_auto_prune_notice_asks_about_the_profile_home(linked, monkeypatch):
    from gateway.config import Platform
    from gateway.run_housekeeping_disk import publish_local_auto_prune_notice
    seen = []
    live = SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL))
    authority = SimpleNamespace(db=linked.db, runner=None, sessions={'sid': live})
    monkeypatch.setattr('gateway.session_policy.policy_for_source', lambda runner, source: SimpleNamespace(source='tui'))
    monkeypatch.setattr('hermes_cli.prune_advisor.notice_due', _spy(seen))
    with pytest.raises(_Stop):
        asyncio.run(publish_local_auto_prune_notice(authority, SimpleNamespace(session_id='sid')))
    assert seen == [linked.home]


def test_api_live_bot_chat_owner_is_looked_up_in_the_profile_home(linked, monkeypatch):
    from gateway.platforms.api_server import APIServerAdapter
    seen = []

    def owner(home):
        seen.append(Path(home))
        raise ValueError('no owner')

    async def db():
        return linked.db

    monkeypatch.setattr('tools.bot_live_delivery.find_canonical_live_owner', owner)
    adapter = SimpleNamespace(gateway_runner=None, _ensure_session_db_async=db)
    assert asyncio.run(APIServerAdapter._admit_to_live_bot_chat(adapter, 'sid', 'hello', None)) is None
    assert seen == [linked.home]


def test_web_mutation_context_accepts_the_launch_authority(linked):
    """The dashboard's launch home is matched against the authority's profile, not its file."""
    from hermes_cli.web_server_sessions import _session_mutation_context
    request = SimpleNamespace(
        state=SimpleNamespace(native_http_principal={'subject': 'owner', 'profile_id': 'default'}, session=None),
        app=SimpleNamespace(state=SimpleNamespace(session_authority=linked.authority, gateway_runner=None)))
    authority, actor = _session_mutation_context(request, None)
    assert authority is linked.authority and actor.subject == 'owner'


def test_session_list_reads_subagent_setting_from_the_profile_home(linked, monkeypatch):
    from tui_gateway import server
    seen = []

    @contextlib.contextmanager
    def profile_db(params):
        yield linked.db

    monkeypatch.setattr(server, '_profile_db', profile_db)
    monkeypatch.setattr('hermes_cli.session_listing.show_subagent_sessions', _spy(seen))
    server._methods['session.list']('r1', {})
    assert seen == [linked.home]


@pytest.mark.asyncio
async def test_legacy_adoption_sees_a_live_owner_registered_in_the_profile_home(tmp_path, monkeypatch):
    """The liveness registry is the profile's. A legacy row whose owner is live there must be
    refused even though the store's file sits on a volume."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    from gateway.session_local_title import resolve_titled_session
    from hermes_cli.active_sessions import release_active_session, try_acquire_active_session
    from hermes_state_runtime import RuntimeStoreError

    home, volume = _linked_home(tmp_path, monkeypatch)
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(home / 'sessions', GatewayConfig())
    db = store._db
    try:
        assert Path(db.db_path).parent == volume.resolve()   # sentinel: the handle is on the volume
        db.create_session('old-live', source='cli', model='old', cwd=str(home))
        runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False)
        authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='first')
        native = SessionActor('native', str(home), frozenset({'session:read', 'session:create'}), 'socket')
        authority._native_legacy_transports = {native.transport_id: native}
        lease, refusal = try_acquire_active_session(session_id='old-live', surface='cli', registry_home=home.resolve(), config={})
        assert lease is not None, refusal
        try:
            with pytest.raises(RuntimeStoreError, match='runtime_coordination_required'):
                resolve_titled_session(authority, native, 'old-live')
        finally:
            release_active_session(lease)
    finally:
        db.close()
