"""The reserved-profile store ownership check tolerates symlinked state stores.

A store belongs to a profile when its directory is that profile's home.
``db.db_path`` is built from the *unresolved* home (hermes_state._default_db_path
→ get_hermes_home() / 'state.db') while the gate's ``home`` is canonical, and
two legitimate layouts make a resolve()-only comparison lie:

- the db FILE is a symlink onto a named volume (hermes-swarm-map's state-init
  service symlinks /opt/data/state.db → /state/state.db so SQLite O_CREAT
  follows it): resolving the file points at the volume, outside the profile,
  even though the store lives inside it;
- the HOME DIRECTORY itself is a symlink (e.g. HERMES_HOME=/tmp/x on macOS,
  where /tmp → /private/tmp): the unresolved parent differs from the canonical
  home even though everything is inside the profile.

The gate accepts either an exact match on the unresolved parent or a canonical
match; a store that genuinely lives elsewhere is still refused.
"""
from pathlib import Path

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
