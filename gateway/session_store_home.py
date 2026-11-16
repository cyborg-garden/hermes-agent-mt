"""Which profile a state store belongs to.

A store belongs to the profile whose home directory holds its ``state.db``
entry. The entry may be a symlink: Swarm Map's state-init service links
``/opt/data/state.db`` onto a named volume, so the FILE lives outside the
profile. A handle opened through such a link carries the link's path as
``db_path`` (see ``hermes_state_registry._handle_path``), so the directory of
``db_path`` is the profile home in every layout: plain file, linked file,
dangling link on a fresh agent, linked home directory.

Resolve the directory, never the file.
"""
from pathlib import Path, PurePath


def _path_of(store) -> Path:
    # Real paths only: a mock handle is path-like too.
    return Path(store) if isinstance(store, (str, PurePath)) else Path(store.db_path)


def store_home(store) -> Path:
    """Canonical home of the profile holding *store* (a SessionDB or a path)."""
    return _path_of(store).parent.resolve()


def store_belongs_to_profile(store, home, other_homes=()) -> bool:
    """True when *store* is *home*'s own store.

    The entry must sit directly in *home*. Its target may live elsewhere (a
    state volume), but never directly in another profile's home: one
    profile's link onto another profile's store would run that store under the
    wrong authority.
    """
    home = Path(home).resolve()
    if store_home(store) != home:
        return False
    target = _path_of(store).resolve()
    for other in other_homes:
        other = Path(other).resolve()
        # Exact directory only: the default profile's home contains the others.
        if other != home and target.parent == other:
            return False
    return True
