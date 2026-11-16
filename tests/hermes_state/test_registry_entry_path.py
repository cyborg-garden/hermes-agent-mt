"""A shared handle names the link it is reached by, whoever opened it (#187, follow-up to #224).

One handle serves every caller of one resolved file, and callers that hold only a handle
reopen by its ``db_path``, which is already resolved. The entry whose directory names the
owning profile must survive that: a second caller, a replacement generation, a reopen.
"""
import os
import shutil
from pathlib import Path

import pytest

import hermes_state_registry as registry
from hermes_state_home import store_home


@pytest.fixture(autouse=True)
def _clean_registry():
    def clean():
        registry.close_all()
        for table in (registry._generations, registry._retired, registry._opening,
                      registry._tearing_down, getattr(registry, '_entry_paths', {})):
            table.clear()
    clean()
    yield
    clean()


@pytest.fixture
def layout(tmp_path):
    """``home/state.db`` is a link onto ``volume/state.db``, as a state volume leaves it."""
    home, volume = (tmp_path / 'home').resolve(), (tmp_path / 'volume').resolve()
    home.mkdir()
    volume.mkdir()
    (home / 'state.db').symlink_to(volume / 'state.db')
    return home, volume


def test_a_caller_through_the_link_names_the_profile_on_a_handle_opened_by_the_file(layout):
    home, volume = layout
    first = registry.acquire(volume / 'state.db')
    assert store_home(first) == volume               # sentinel: no link known yet
    second = registry.acquire(home / 'state.db')
    try:
        assert second is first
        assert store_home(second) == home
    finally:
        registry.release(second)
        registry.release(first)


def test_a_reopen_by_the_handles_own_path_keeps_the_link(layout):
    home, volume = layout
    first = registry.acquire(home / 'state.db')
    again = registry.acquire(first.db_path)          # how a child or side agent reopens
    try:
        assert again is first and Path(first.db_path).parent == volume
        assert store_home(again) == home
    finally:
        registry.release(again)
        registry.release(first)


def _replace(target):
    """Snapshot restore: a new inode holding the same store."""
    copy = target.with_suffix('.replacement.tmp')
    shutil.copy2(target, copy)
    os.unlink(target)
    os.rename(copy, target)


@pytest.mark.platforms("posix")
def test_a_replacement_generation_opened_by_the_resolved_path_keeps_the_link(layout):
    home, volume = layout
    old = registry.acquire(home / 'state.db')
    _replace(volume / 'state.db')
    new = registry.acquire(old.db_path)
    try:
        assert new is not old                        # sentinel: a new generation was opened
        assert store_home(new) == home
    finally:
        registry.release(new)
        registry.release(old)


@pytest.mark.platforms("posix")
def test_a_link_that_no_longer_leads_to_the_file_is_forgotten(layout):
    """The link is kept across a replacement only while it still leads to the file."""
    home, volume = layout
    old = registry.acquire(home / 'state.db')
    (home / 'state.db').unlink()
    _replace(volume / 'state.db')
    new = registry.acquire(old.db_path)
    try:
        assert new is not old                        # sentinel: a new generation was opened
        assert store_home(new) == volume
    finally:
        registry.release(new)
        registry.release(old)


def _two_homes(tmp_path):
    a, b, volume = ((tmp_path / name).resolve() for name in ('a', 'b', 'volume'))
    for directory in (a, b, volume):
        directory.mkdir()
    (a / 'state.db').symlink_to(volume / 'state.db')
    (b / 'state.db').symlink_to(volume / 'state.db')
    return a, b


def test_a_link_is_not_remembered_after_its_last_handle_closes(tmp_path):
    """Two homes linked onto one file: once the first is released, the second names itself."""
    a, b = _two_homes(tmp_path)
    registry.release(registry.acquire(a / 'state.db'))
    assert not registry._entry_paths                 # nothing outlives the last handle
    second = registry.acquire(b / 'state.db')
    try:
        assert store_home(second) == b
    finally:
        registry.release(second)


@pytest.mark.parametrize('sweep', [lambda root: registry.close_all(), registry.close_all_under],
                         ids=['close_all', 'close_all_under'])
def test_a_forced_close_forgets_the_link(tmp_path, sweep):
    a, b = _two_homes(tmp_path)
    registry.acquire(a / 'state.db')
    assert sweep(tmp_path) == 1
    assert not registry._entry_paths
    second = registry.acquire(b / 'state.db')
    try:
        assert store_home(second) == b
    finally:
        registry.release(second)


def test_the_first_link_names_a_handle_while_it_is_shared(tmp_path):
    """One handle has one entry. While both callers hold it, the first link stays."""
    a, b = _two_homes(tmp_path)
    first = registry.acquire(a / 'state.db')
    second = registry.acquire(b / 'state.db')
    try:
        assert second is first
        assert store_home(second) == a
    finally:
        registry.release(second)
        registry.release(first)


def test_a_directory_alias_of_a_plain_store_is_not_a_link(tmp_path):
    """``home/lnk/../state.db`` is ``other/state.db`` to the OS: its profile is ``other``."""
    from hermes_state_home import store_dir
    home, other = (tmp_path / 'home').resolve(), (tmp_path / 'other').resolve()
    home.mkdir()
    (other / 'deep').mkdir(parents=True)
    (home / 'lnk').symlink_to(other / 'deep', target_is_directory=True)
    db = registry.acquire(home / 'lnk' / '..' / 'state.db')
    try:
        assert Path(db.db_path) == other / 'state.db'    # sentinel: where the OS opened it
        assert store_dir(db) == other == store_home(db)
        assert not registry._entry_paths
    finally:
        registry.release(db)


def test_a_linked_store_under_an_aliased_home_names_the_canonical_home(layout, tmp_path):
    from hermes_state_home import store_dir
    home, _volume = layout
    (tmp_path / 'alias').symlink_to(home, target_is_directory=True)
    db = registry.acquire(tmp_path / 'alias' / 'state.db')
    try:
        assert store_dir(db) == home == store_home(db)
    finally:
        registry.release(db)


# --- the link lives exactly as long as something holds or is opening the file ------------


@pytest.mark.platforms("posix")
def test_the_link_survives_the_retired_generations_release(layout):
    home, volume = layout
    old = registry.acquire(home / 'state.db')
    _replace(volume / 'state.db')
    new = registry.acquire(old.db_path)
    registry.release(old)                            # the live generation still holds the file
    again = registry.acquire(new.db_path)
    try:
        assert again is new and store_home(again) == home
    finally:
        registry.release(again)
        registry.release(new)


@pytest.mark.platforms("posix")
def test_the_link_survives_the_live_generations_release(layout):
    home, volume = layout
    old = registry.acquire(home / 'state.db')
    _replace(volume / 'state.db')
    registry.release(registry.acquire(old.db_path))  # the retired generation still holds the file
    again = registry.acquire(old.db_path)
    try:
        assert again is not old and store_home(again) == home
    finally:
        registry.release(again)
        registry.release(old)


def _release_during_a_replacement_open(layout, monkeypatch, opened):
    """Final-release the retired handle while the replacement open is in flight; *opened*
    is what that open does once the release is admitted. Returns the acquire's outcome."""
    import threading
    home, volume = layout
    old = registry.acquire(home / 'state.db')
    resolved = old.db_path
    _replace(volume / 'state.db')
    real_open, started, gate, outcome = registry._open_session_db, threading.Event(), threading.Event(), []

    def slow_open(path):
        if not started.is_set():
            started.set()
            assert gate.wait(30)
            return opened(real_open, path)
        return real_open(path)

    def acquire():
        try:
            outcome.append(registry.acquire(resolved))
        except OSError as exc:
            outcome.append(exc)

    monkeypatch.setattr(registry, '_open_session_db', slow_open)
    opener = threading.Thread(target=acquire)
    opener.start()
    assert started.wait(30)
    releaser = threading.Thread(target=registry.release, args=(old,))
    releaser.start()
    while registry._retired:                         # the release is admitted; its close waits on the open
        releaser.join(0.01)
    gate.set()
    opener.join(30)
    releaser.join(30)
    return home, outcome[0]


@pytest.mark.platforms("posix")
def test_the_link_survives_a_release_during_the_replacement_open(layout, monkeypatch):
    home, new = _release_during_a_replacement_open(layout, monkeypatch, lambda real_open, path: real_open(path))
    try:
        assert store_home(new) == home
    finally:
        registry.release(new)


@pytest.mark.platforms("posix")
def test_a_failed_replacement_open_leaves_no_link_behind(layout, monkeypatch):
    def fail(real_open, path):
        raise OSError('open failed')

    _home, outcome = _release_during_a_replacement_open(layout, monkeypatch, fail)
    assert isinstance(outcome, OSError)
    assert registry.stats() == {'live_generations': 0, 'retired_generations': 0, 'total_refcounts': 0}
    assert not registry._opening and not registry._entry_paths
