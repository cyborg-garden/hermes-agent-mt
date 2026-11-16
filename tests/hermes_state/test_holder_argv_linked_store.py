"""A process that names the home linking onto our store is ours, not another instance (#187).

A store linked onto a volume is opened by its resolved path, so the holder scan sees the
volume as the store's directory while processes name the profile home that holds the link.
"""
import pytest

from hermes_state_holders import _argv_scoped_to_other_home


@pytest.fixture
def linked(tmp_path):
    home, volume = (tmp_path / 'home').resolve(), (tmp_path / 'volume').resolve()
    home.mkdir()
    volume.mkdir()
    (volume / 'state.db').write_bytes(b'')
    (home / 'state.db').symlink_to(volume / 'state.db')
    return home, volume / 'state.db'


@pytest.mark.parametrize('argv', [
    lambda home: ['hermes', 'gateway', '--hermes-home', str(home)],
    lambda home: ['env', f'HERMES_HOME={home}', 'hermes', 'gateway'],
    lambda home: ['hermes', 'sessions', f'--db={home}/state.db'],
], ids=['flag', 'env', 'db'])
def test_a_process_naming_the_linking_home_is_not_another_instance(linked, argv):
    home, db_path = linked
    assert _argv_scoped_to_other_home(argv(home), db_path) is False


def test_a_process_naming_an_unrelated_home_is_still_another_instance(linked, tmp_path):
    _home, db_path = linked
    other = tmp_path / 'elsewhere'
    other.mkdir()
    (other / 'state.db').write_bytes(b'')
    assert _argv_scoped_to_other_home(['hermes', 'gateway', '--hermes-home', str(other)], db_path) is True
    assert _argv_scoped_to_other_home(['hermes', 'sessions', f'--db={other}/state.db'], db_path) is True


def test_a_plain_store_is_matched_by_its_paths_as_before(tmp_path):
    home = (tmp_path / 'plain').resolve()
    home.mkdir()
    (home / 'state.db').write_bytes(b'')
    alias = tmp_path / 'alias'
    alias.symlink_to(home, target_is_directory=True)
    db_path = home / 'state.db'
    assert _argv_scoped_to_other_home(['hermes', 'gateway', '--hermes-home', str(home)], db_path) is False
    # A directory alias is not a link to the file: it reads as another home, as it always did.
    assert _argv_scoped_to_other_home(['hermes', 'gateway', '--hermes-home', str(alias)], db_path) is True


def test_a_relative_home_is_not_matched_through_the_link(linked, monkeypatch):
    """A relative home is relative to the other process, not to this one."""
    home, db_path = linked
    monkeypatch.chdir(home)
    assert _argv_scoped_to_other_home(['hermes', 'gateway', '--hermes-home', '.'], db_path) is True


@pytest.mark.parametrize('argv', [
    lambda home: ['hermes', 'gateway', '--hermes-home', str(home)],
    lambda home: ['env', f'HERMES_HOME={home}', 'hermes', 'gateway'],
    lambda home: ['hermes', 'sessions', f'--db={home}/state.db'],
], ids=['flag', 'env', 'db'])
def test_the_link_target_may_have_another_name(tmp_path, argv):
    home, volume = (tmp_path / 'alpha').resolve(), (tmp_path / 'volume').resolve()
    home.mkdir()
    volume.mkdir()
    (volume / 'alpha-state.db').write_bytes(b'')
    (home / 'state.db').symlink_to(volume / 'alpha-state.db')
    assert _argv_scoped_to_other_home(argv(home), volume / 'alpha-state.db') is False
