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

from gateway import run_runtime


def _belongs(db_path: Path, home: Path) -> bool:
    return run_runtime.store_belongs_to_profile(db_path, home)


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
