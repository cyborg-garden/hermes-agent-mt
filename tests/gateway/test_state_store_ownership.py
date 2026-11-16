"""A store belongs to the profile whose home holds its ``state.db`` entry.

Swarm Map's state-init links ``/opt/data/state.db`` onto a named volume, so the
file lives outside the profile, and the shared registry opens handles by their
resolved path. Ownership is therefore read from the path the store was opened
by (``entry_path``), never from the resolved file. These tests cover the pure
helper, the real registry handle, and the gateway boot path.
"""
from pathlib import Path

import pytest

from gateway.session_store_home import store_belongs_to_profile, store_home


def _belongs(db_path: Path, home: Path, others=()) -> bool:
    return store_belongs_to_profile(db_path, home, others)


def test_regular_store_in_profile(tmp_path):
    home = (tmp_path / 'profiles' / 'worker').resolve()
    home.mkdir(parents=True)
    assert _belongs(home / 'state.db', home)


def test_symlinked_db_file_still_belongs(tmp_path):
    """swarm-map layout: /opt/data/state.db -> /state/state.db, profile intact."""
    home = (tmp_path / 'profiles' / 'worker').resolve()
    home.mkdir(parents=True)
    volume = tmp_path / 'state-volume'
    volume.mkdir()
    (home / 'state.db').symlink_to(volume / 'state.db')
    assert _belongs(home / 'state.db', home)


def test_dangling_symlinked_db_still_belongs(tmp_path):
    """A fresh swarm-map harness symlinks before the volume has any file (SQLite
    O_CREAT follows the link) — the dangling target must not fail ownership."""
    home = (tmp_path / 'profiles' / 'worker').resolve()
    home.mkdir(parents=True)
    volume = tmp_path / 'state-volume'
    volume.mkdir()
    (home / 'state.db').symlink_to(volume / 'state.db')  # target does not exist yet
    assert _belongs(home / 'state.db', home)


def test_symlinked_profile_home_still_belongs(tmp_path):
    """HERMES_HOME itself is a symlink: canonical home differs, resolved parent matches."""
    home = (tmp_path / 'profile-real').resolve()
    home.mkdir(parents=True)
    link = tmp_path / 'profile-link'
    link.symlink_to(home, target_is_directory=True)
    canonical_home = home  # get_hermes_home().resolve()
    assert _belongs(link / 'state.db', canonical_home)


def test_store_outside_the_profile_is_refused(tmp_path):
    home = (tmp_path / 'profiles' / 'worker').resolve()
    home.mkdir(parents=True)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    assert not _belongs(elsewhere / 'state.db', home)


def test_store_in_a_subdirectory_is_refused(tmp_path):
    """Existing behaviour preserved: a store nested under the profile but not at
    its root is not 'the reserved profile' store."""
    home = (tmp_path / 'profiles' / 'worker').resolve()
    (home / 'nested').mkdir(parents=True)
    assert not _belongs(home / 'nested' / 'state.db', home)


def test_link_into_another_profile_is_refused(tmp_path):
    """A profile whose state.db links onto another reserved profile's store would run
    that store under the wrong authority. A link onto a neutral volume stays allowed."""
    mine = (tmp_path / 'profiles' / 'mine').resolve()
    theirs = (tmp_path / 'profiles' / 'theirs').resolve()
    mine.mkdir(parents=True)
    theirs.mkdir(parents=True)
    (theirs / 'state.db').write_bytes(b'')
    (mine / 'state.db').symlink_to(theirs / 'state.db')
    assert not _belongs(mine / 'state.db', mine, [mine, theirs])
    # sentinel: the same link onto a volume outside every profile is accepted
    volume = tmp_path / 'volume'
    volume.mkdir()
    (mine / 'state.db').unlink()
    (mine / 'state.db').symlink_to(volume / 'state.db')
    assert _belongs(mine / 'state.db', mine, [mine, theirs])


def test_profile_nested_under_the_default_home_is_not_refused(tmp_path):
    """The default profile's home is the root that holds profiles/<name>; a secondary's
    plain store sits inside it and is still its own."""
    root = (tmp_path / 'hermes').resolve()
    alpha = root / 'profiles' / 'alpha'
    alpha.mkdir(parents=True)
    assert _belongs(alpha / 'state.db', alpha, [root, alpha])
    assert _belongs(root / 'state.db', root, [root, alpha])


def test_own_home_in_the_reserved_list_does_not_refuse(tmp_path):
    home = (tmp_path / 'profiles' / 'worker').resolve()
    home.mkdir(parents=True)
    assert _belongs(home / 'state.db', home, [home])


def test_store_home_is_the_canonical_profile_in_every_layout(tmp_path):
    home = (tmp_path / 'profile-real').resolve()
    home.mkdir()
    volume = tmp_path / 'volume'
    volume.mkdir()
    assert store_home(home / 'state.db') == home                 # plain file
    (home / 'state.db').symlink_to(volume / 'state.db')          # dangling link onto a volume
    assert store_home(home / 'state.db') == home
    link = tmp_path / 'profile-link'
    link.symlink_to(home, target_is_directory=True)              # symlinked home directory
    assert store_home(link / 'state.db') == home


def test_cron_owner_is_found_through_a_symlinked_store(tmp_path):
    """The owner table is looked up by canonical home; binding must use the same key
    whether the store is a plain file, a link onto a volume, or under a linked home."""
    import asyncio
    from types import SimpleNamespace
    from gateway import session_cron

    home = (tmp_path / 'profile-real').resolve()
    home.mkdir()
    volume = tmp_path / 'volume'
    volume.mkdir()
    (home / 'state.db').symlink_to(volume / 'state.db')
    link = tmp_path / 'profile-link'
    link.symlink_to(home, target_is_directory=True)

    async def run(db_path):
        authority = SimpleNamespace(db=SimpleNamespace(db_path=str(db_path)))
        session_cron.bind_owner(authority)
        try:
            return session_cron._owners.get(home) is not None
        finally:
            session_cron.unbind_owner(authority)

    assert asyncio.run(run(home / 'state.db'))
    assert asyncio.run(run(link / 'state.db'))
    assert home not in session_cron._owners


def test_session_rows_keep_their_profile_through_a_linked_store(tmp_path, monkeypatch):
    """Session rows name their owning profile from the store path. A store linked onto a
    volume must still name its profile, not fall back to no owner."""
    from types import SimpleNamespace
    import hermes_constants
    from hermes_state_sessions import SessionSessionsMixin

    root = (tmp_path / 'hermes').resolve()
    worker = root / 'profiles' / 'worker'
    worker.mkdir(parents=True)
    volume = tmp_path / 'volume'
    volume.mkdir()
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda: root)

    def owner(db_path):
        return SessionSessionsMixin._own_profile_name(SimpleNamespace(db_path=str(db_path)))

    assert owner(root / 'state.db') == 'default'
    assert owner(worker / 'state.db') == 'worker'
    (worker / 'state.db').symlink_to(volume / 'state.db')
    assert owner(worker / 'state.db') == 'worker'
    (root / 'state.db').symlink_to(volume / 'root.db')
    assert owner(root / 'state.db') == 'default'
    assert owner(volume / 'state.db') is None      # sentinel: outside the tree has no owner


def _linked_home(tmp_path, monkeypatch):
    """A profile home whose state.db is a dangling link onto a volume, as state-init leaves it."""
    import hermes_state
    home = tmp_path / '.hermes'
    home.mkdir(mode=0o700)
    volume = tmp_path / 'state-volume'
    volume.mkdir()
    (home / 'state.db').symlink_to(volume / 'state.db')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    return home, volume


def test_registry_handle_through_a_link_still_names_its_profile(tmp_path, monkeypatch):
    """The real open path: the registry resolves the link, so db_path names the volume.
    Ownership must come from the path the store was asked for."""
    from hermes_state_registry import acquire, release
    home, volume = _linked_home(tmp_path, monkeypatch)
    db = acquire(home / 'state.db')
    try:
        assert Path(db.db_path).parent == volume.resolve()       # sentinel: the handle is on the volume
        assert store_home(db) == home.resolve()
        assert _belongs(db, home.resolve())
        assert not _belongs(db, volume.resolve())
        assert (volume / 'state.db').exists()                    # SQLite created it through the link
    finally:
        release(db)


def test_registry_handle_for_a_plain_store_is_unchanged(tmp_path, monkeypatch):
    from hermes_state_registry import acquire, release
    home = (tmp_path / 'plain').resolve()
    home.mkdir()
    db = acquire(home / 'state.db')
    try:
        assert store_home(db) == home == Path(db.db_path).parent
    finally:
        release(db)


@pytest.mark.asyncio
async def test_gateway_boots_on_a_store_linked_onto_a_volume(tmp_path, monkeypatch):
    """swarm-map#273 / #187: the gateway must serve a profile whose state.db is a link."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    home, volume = _linked_home(tmp_path, monkeypatch)
    (home / 'config.yaml').write_text('{}')
    process_ownership.reserve([home])
    runner = None
    try:
        runner = GatewayRunner(GatewayConfig())
        await initialize_gateway_runtime(runner)
        authority = runner.session_authority
        assert authority is not None
        assert Path(authority.db.db_path).parent == volume.resolve()
        assert store_home(authority.db) == home.resolve()
        from gateway.session_authorities import authority_for_home
        assert authority_for_home(runner, home) is authority
    finally:
        for held in list(getattr(runner, 'session_authorities', []) or []):
            held.db.close()
        process_ownership.release(home)


def test_a_mock_handle_falls_back_to_its_db_path(tmp_path):
    """Test doubles answer any attribute and are path-like; only a real entry_path counts."""
    from unittest.mock import MagicMock
    db = MagicMock()
    db.db_path = str(tmp_path / 'state.db')
    assert store_home(db) == tmp_path.resolve()
