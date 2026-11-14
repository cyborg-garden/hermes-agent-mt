"""Which profile a state store belongs to.

A store belongs to the profile whose home directory holds its ``state.db``
entry. The entry may be a symlink: Swarm Map's state-init service links
``/opt/data/state.db`` onto a named volume, so resolving the FILE lands
outside the profile. Resolving the DIRECTORY that holds the entry gives the
canonical profile home in every layout (plain file, symlinked file, dangling
link on a fresh agent, symlinked home directory).
"""
from pathlib import Path


def store_home(db_path) -> Path:
    """Canonical home of the profile that holds *db_path*'s directory entry."""
    return Path(db_path).parent.resolve()


def store_belongs_to_profile(db_path, home, other_homes=()) -> bool:
    """True when *db_path* is *home*'s own store.

    The entry must sit directly in *home*. Its target may live elsewhere (a
    state volume), but never inside another reserved profile: one profile's
    link onto another profile's store would run that store under the wrong
    authority.
    """
    home = Path(home).resolve()
    if store_home(db_path) != home:
        return False
    target = Path(db_path).resolve()
    for other in other_homes:
        other = Path(other).resolve()
        if other != home and (target.parent == other or other in target.parents):
            return False
    return True
