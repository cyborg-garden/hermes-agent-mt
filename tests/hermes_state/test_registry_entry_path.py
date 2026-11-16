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


@pytest.mark.platforms("posix")
def test_a_replacement_generation_opened_by_the_resolved_path_keeps_the_link(layout):
    home, volume = layout
    old = registry.acquire(home / 'state.db')
    target = volume / 'state.db'
    copy = target.with_suffix('.replacement.tmp')
    shutil.copy2(target, copy)                       # snapshot restore: new inode, same store
    os.unlink(target)
    os.rename(copy, target)
    new = registry.acquire(old.db_path)
    try:
        assert new is not old                        # sentinel: a new generation was opened
        assert store_home(new) == home
    finally:
        registry.release(new)
        registry.release(old)


def test_a_link_that_no_longer_leads_to_the_file_is_forgotten(layout):
    home, volume = layout
    registry.release(registry.acquire(home / 'state.db'))
    (home / 'state.db').unlink()
    direct = registry.acquire(volume / 'state.db')
    try:
        assert store_home(direct) == volume
    finally:
        registry.release(direct)
