"""Activating a backup outside a turn must not decide the next test's provider."""

from __future__ import annotations

import agent.auxiliary_client as aux


def test_a_test_records_a_backup_runtime():
    aux.set_runtime_main("openrouter", "backup/model")
    assert aux._runtime_main_value("model") == "backup/model"


def test_the_next_test_starts_without_it():
    assert aux._runtime_main_value("model") == ""
