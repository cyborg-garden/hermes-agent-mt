"""Which profile a state store belongs to.

A store belongs to the profile whose home directory holds its ``state.db``
entry. The entry may be a symlink: Swarm Map's state-init service links
``/opt/data/state.db`` onto a named volume, so the FILE lives outside the
profile. The shared registry opens handles by their resolved path, so a
handle's ``db_path`` names the volume; the registry therefore records the
path the store was asked for as ``entry_path``. The directory holding that
entry is the profile home in every layout (plain file, linked file, dangling
link on a fresh agent, linked home directory).
"""
from pathlib import Path

from hermes_state_home import entry_path, store_dir, store_home

__all__ = ['entry_path', 'store_belongs_to_profile', 'store_dir', 'store_home']


def store_belongs_to_profile(store, home, other_homes=()) -> bool:
    """True when *store* is *home*'s own store.

    The entry must sit directly in *home*. Its target may live elsewhere (a
    state volume), but never directly in another reserved profile's home: one
    profile's link onto another profile's store would run that store under the
    wrong authority.
    """
    home = Path(home).resolve()
    if store_home(store) != home:
        return False
    target = entry_path(store).resolve()
    for other in other_homes:
        other = Path(other).resolve()
        # Exact directory only: the default profile's home contains the others.
        if other != home and target.parent == other:
            return False
    return True
