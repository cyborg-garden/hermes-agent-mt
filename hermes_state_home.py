"""The profile directory a state store's entry sits in.

``state.db`` may be a symlink onto a state volume, and the shared registry
opens handles by their resolved path, so ``db_path`` then names the volume.
The registry records the path the store was asked for as ``entry_path``; the
directory holding that entry is the profile home in every layout. Imports
nothing from the project, so any layer can use it.
"""
from pathlib import Path, PurePath


def entry_path(store) -> Path:
    """The path *store* was opened by. *store* is a SessionDB or a path."""
    # Real paths only: a mock handle answers every attribute and is path-like too.
    if isinstance(store, (str, PurePath)):
        return Path(store)
    entry = getattr(store, 'entry_path', None)
    if isinstance(entry, (str, PurePath)):
        return Path(entry)
    return Path(store.db_path)


def store_home(store) -> Path:
    """Canonical home of the profile that holds *store*'s directory entry."""
    return entry_path(store).parent.resolve()


def store_dir(store) -> Path:
    """The profile directory of *store*, spelled as ``Path(db_path).parent`` always was.

    A registry handle's ``db_path`` is resolved, so its directory was the canonical
    home; that stays, read from the entry instead of the file. A handle opened by
    its own path (or a path) keeps the directory exactly as given.
    """
    if not isinstance(store, (str, PurePath)) and isinstance(getattr(store, 'entry_path', None), (str, PurePath)):
        return store_home(store)
    return entry_path(store).parent
