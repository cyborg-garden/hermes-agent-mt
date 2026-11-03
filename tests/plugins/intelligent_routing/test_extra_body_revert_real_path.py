"""Revert of the turn-scoped ``extra_body`` driven through the REAL
``restore_primary_runtime`` — not a hand-copied mirror of its revert block.

The PR that introduced ``extra_body`` passthrough tested restore with a local
``_simulate_restore()`` mirror. A mirror reproduces whatever assumption the
implementation makes, so it structurally cannot fail on the two bugs the
independent audit found, and it drifted immediately: the mirror had no
``_routing_override_active`` gate while the real block does.

These drive ``restore_primary_runtime`` itself, reusing the fake-agent harness
from ``tests/agent/test_routing_override_revert.py`` (the suite that surfaced
the original three blockers) so the covered code is the shipped code.
"""
import pytest

from agent.agent_runtime_helpers import restore_primary_runtime
from agent.routing_override import apply_routing_override
from tests.agent.test_routing_override_revert import _make_agent


def _agent_with_overrides(overrides=None):
    a = _make_agent()
    a.request_overrides = {} if overrides is None else overrides
    # Upstream v0.22.4 (#75091): request_overrides travel with the
    # _primary_runtime snapshot (primary_runtime_snapshot._build_primary_runtime_snapshot),
    # and restore_primary_runtime reinstates them from it. A real agent's
    # premium snapshot is taken after its init-time overrides are set.
    a._primary_runtime["request_overrides"] = dict(a.request_overrides)
    return a


def test_restore_removes_extra_body_that_did_not_exist_before():
    a = _agent_with_overrides()
    apply_routing_override(a, {"model": "qwen3.5:9b", "provider": "ollama",
                               "extra_body": {"think": False}})
    assert a.request_overrides["extra_body"] == {"think": False}
    restore_primary_runtime(a)
    assert "extra_body" not in a.request_overrides


def test_restore_reinstates_prior_extra_body():
    a = _agent_with_overrides({"extra_body": {"tags": ["x"]}})
    apply_routing_override(a, {"model": "qwen3.5:9b", "provider": "ollama",
                               "extra_body": {"think": False}})
    restore_primary_runtime(a)
    assert a.request_overrides["extra_body"] == {"tags": ["x"]}


def test_routed_turn_without_extra_body_preserves_a_pre_existing_one():
    """BLOCKER (audit finding 3): the DEFAULT path must not destroy extra_body.

    ``routing.cheap_extra_body`` is empty by default, so almost every routed
    turn carries no ``extra_body``. Conflating "this turn wrote nothing" with
    "the key was absent before the turn" made the revert pop a primary
    provider's own body params — set once at agent construction
    (``_custom_provider_request_overrides``, and fast/priority mode), so the
    loss is session-persistent and silent.
    """
    a = _agent_with_overrides({"extra_body": {"think": True, "top_k": 40}})
    apply_routing_override(a, {"model": "deepseek/x", "provider": "openrouter"})
    restore_primary_runtime(a)
    assert a.request_overrides["extra_body"] == {"think": True, "top_k": 40}


def test_revert_cleans_the_dict_it_actually_mutated_not_the_live_one():
    """Audit finding 1: the gateway swaps in a fresh per-turn overrides dict.

    ``gateway/run.py`` assigns a newly built ``request_overrides`` onto the
    CACHED agent before ``run_conversation`` fires the revert, so re-reading
    ``agent.request_overrides`` at revert time reaches a different object than
    the one the routed turn dirtied: the dirty dict keeps the cheap params and
    the next turn's dict gets stomped.
    """
    routed = {"speed": "fast"}
    a = _agent_with_overrides(routed)
    apply_routing_override(a, {"model": "qwen3.5:9b", "provider": "ollama",
                               "extra_body": {"think": False}})
    assert routed["extra_body"] == {"think": False}

    # Gateway builds a brand-new dict for the next turn on the same agent.
    next_turn = {"speed": "fast", "extra_body": {"think": True}}
    a.request_overrides = next_turn
    restore_primary_runtime(a)

    assert "extra_body" not in routed, "the mutated dict was never cleaned"
    assert next_turn["extra_body"] == {"think": True}, "next turn's dict was stomped"


def test_double_apply_keeps_the_pristine_pre_turn_extra_body():
    """Audit finding 4b: a re-apply must not overwrite the pristine snapshot."""
    a = _agent_with_overrides({"extra_body": {"tags": ["x"]}})
    apply_routing_override(a, {"model": "qwen3.5:9b", "provider": "ollama",
                               "extra_body": {"think": False}})
    apply_routing_override(a, {"model": "deepseek/x", "provider": "openrouter",
                               "extra_body": {"reasoning": "none"}})
    restore_primary_runtime(a)
    assert a.request_overrides["extra_body"] == {"tags": ["x"]}


def test_revert_is_gated_on_the_override_flag():
    """No routed turn ⇒ the revert block must not touch request_overrides."""
    a = _agent_with_overrides({"extra_body": {"think": True}})
    restore_primary_runtime(a)
    assert a.request_overrides["extra_body"] == {"think": True}
