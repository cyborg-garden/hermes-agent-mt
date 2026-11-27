"""Cheap-tier `extra_body` passthrough + per-turn routing metrics.

Two gaps this closes:

1. **extra_body** — `routing_override` forwarded only base_url/api_key/api_mode,
   so there was no way to send a provider-specific body param with a routed
   turn (ollama `options.num_ctx`, an aggregator's `provider` preferences).

   ⚠️ CORRECTION: this file originally justified the passthrough as the way to
   send `{"think": false}` to a local ollama tier. That was never measured and
   is wrong — ollama's OpenAI-compat /v1 endpoint ignores `think` entirely
   (ollama#14820). Thinking is disabled per-model via `agent.reasoning_overrides`,
   which `switch_model` re-resolves for the ROUTED model. See
   `test_cheap_extra_body_wire_semantics.py` for the measured behaviour.
   The `{"think": ...}` literals below are just opaque body dicts exercising the
   plumbing; they are NOT a working recipe for disabling thinking.

2. **metrics** — the plugin emitted no per-turn decision record, which is the
   stated blocker on the upstream PR (spec 2026-07-23: "Routing decisions must
   be countable: parseable per-turn log line of tier, model, reason").
"""
import json
import logging
from unittest.mock import patch

from plugins.intelligent_routing import config as cfgmod


# ── 1. config: routing.cheap_extra_body ──────────────────────────────────────

def test_cheap_extra_body_defaults_empty():
    with patch.object(cfgmod, "_load_config", return_value={}):
        assert cfgmod.cheap_extra_body() == {}


def test_cheap_extra_body_read_from_config():
    with patch.object(cfgmod, "_load_config", return_value={
            "routing": {"cheap_extra_body": {"think": False}}}):
        assert cfgmod.cheap_extra_body() == {"think": False}


def test_cheap_extra_body_non_dict_is_ignored():
    """A malformed value must not break the turn — fail to {}."""
    for bad in ("think=false", 42, [1, 2], None):
        with patch.object(cfgmod, "_load_config",
                          return_value={"routing": {"cheap_extra_body": bad}}):
            assert cfgmod.cheap_extra_body() == {}, bad


def test_cheap_extra_body_returns_a_copy():
    """Callers must not be able to mutate the cached config section."""
    section = {"routing": {"cheap_extra_body": {"think": False}}}
    with patch.object(cfgmod, "_load_config", return_value=section):
        got = cfgmod.cheap_extra_body()
        got["think"] = True
        assert section["routing"]["cheap_extra_body"] == {"think": False}


# ── 2. routing_override: apply / scope / restore extra_body ──────────────────

class _FakeAgent:
    def __init__(self):
        self.model = "z-ai/glm-5.3"
        self.provider = "openrouter"
        self.request_overrides = {}
        self._primary_runtime = {"model": "z-ai/glm-5.3", "provider": "openrouter"}
        self.switched = None

    def switch_model(self, model, provider, **kw):
        self.switched = (model, provider, kw)
        self.model, self.provider = model, provider


def test_extra_body_applied_to_request_overrides():
    from agent import routing_override as ro
    a = _FakeAgent()
    ok = ro.apply_routing_override(a, {
        "model": "qwen3.5:9b", "provider": "ollama",
        "extra_body": {"think": False},
    })
    assert ok is True
    assert a.request_overrides.get("extra_body") == {"think": False}


def test_extra_body_not_passed_to_switch_model():
    """switch_model()'s signature has no extra_body — passing it would TypeError."""
    from agent import routing_override as ro
    a = _FakeAgent()
    ro.apply_routing_override(a, {
        "model": "qwen3.5:9b", "provider": "ollama",
        "extra_body": {"think": False},
    })
    assert "extra_body" not in (a.switched[2] or {})


def test_extra_body_merges_over_existing_without_destroying_it():
    from agent import routing_override as ro
    a = _FakeAgent()
    a.request_overrides["extra_body"] = {"tags": ["x"], "think": True}
    ro.apply_routing_override(a, {
        "model": "qwen3.5:9b", "provider": "ollama",
        "extra_body": {"think": False},
    })
    eb = a.request_overrides["extra_body"]
    assert eb["think"] is False      # routed value wins
    assert eb["tags"] == ["x"]       # pre-existing survives


def test_extra_body_stashed_for_restore():
    from agent import routing_override as ro
    a = _FakeAgent()
    a.request_overrides["extra_body"] = {"tags": ["x"]}
    ro.apply_routing_override(a, {
        "model": "qwen3.5:9b", "provider": "ollama",
        "extra_body": {"think": False},
    })
    assert a._routing_override_saved_extra_body == {"tags": ["x"]}
    assert a._routing_override_active is True


def test_no_extra_body_leaves_request_overrides_untouched():
    from agent import routing_override as ro
    a = _FakeAgent()
    ro.apply_routing_override(a, {"model": "deepseek/x", "provider": "openrouter"})
    assert "extra_body" not in a.request_overrides


def test_malformed_extra_body_does_not_break_the_turn():
    from agent import routing_override as ro
    a = _FakeAgent()
    ok = ro.apply_routing_override(a, {
        "model": "qwen3.5:9b", "provider": "ollama", "extra_body": "think=false",
    })
    assert ok is True                       # swap still happened
    assert "extra_body" not in a.request_overrides


# ── 3. registration: per-turn metrics line ───────────────────────────────────

def test_routing_decision_emits_parseable_metric(caplog):
    """One machine-parseable line per turn carrying tier, model, reason."""
    from plugins.intelligent_routing import registration as reg
    with caplog.at_level(logging.INFO, logger=reg.logger.name):
        reg.log_routing_decision(
            tier="cheap", model="qwen3.5:9b", provider="ollama",
            reason="mechanical → cheap", primary_model="z-ai/glm-5.3",
        )
    recs = [r for r in caplog.records if "routing_decision" in r.getMessage()]
    assert len(recs) == 1
    payload = json.loads(recs[0].getMessage().split("routing_decision ", 1)[1])
    assert payload["tier"] == "cheap"
    assert payload["model"] == "qwen3.5:9b"
    assert payload["provider"] == "ollama"
    assert payload["reason"] == "mechanical → cheap"
    assert payload["primary_model"] == "z-ai/glm-5.3"


def test_metric_emission_never_raises():
    from plugins.intelligent_routing import registration as reg
    reg.log_routing_decision(tier=None, model=None, provider=None,
                             reason=None, primary_model=None)


# ── 4. hook path + restore (regression guards) ───────────────────────────────

def test_hook_emits_extra_body_on_cheap_route():
    """Exercises the real hook body — guards the import of cheap_extra_body."""
    from plugins.intelligent_routing import registration as reg
    with patch.object(reg, "is_intelligent_routing_enabled", return_value=True), \
         patch.object(reg, "cheap_tier_target", return_value=("qwen3.5:9b", "ollama")), \
         patch.object(reg, "cheap_extra_body", return_value={"think": False}):
        out = reg.on_pre_llm_call(user_message="list the files",
                                  model="z-ai/glm-5.3", provider="openrouter")
    assert out["route"]["model"] == "qwen3.5:9b"
    assert out["route"]["extra_body"] == {"think": False}


def test_hook_omits_extra_body_when_unset():
    from plugins.intelligent_routing import registration as reg
    with patch.object(reg, "is_intelligent_routing_enabled", return_value=True), \
         patch.object(reg, "cheap_tier_target", return_value=("deepseek/x", "openrouter")), \
         patch.object(reg, "cheap_extra_body", return_value={}):
        out = reg.on_pre_llm_call(user_message="list the files",
                                  model="z-ai/glm-5.3", provider="openrouter")
    assert "extra_body" not in out["route"]


# Restore coverage lives in test_extra_body_revert_real_path.py, which drives the
# REAL restore_primary_runtime. This file used to mirror its revert block in a
# local _simulate_restore() helper; the mirror reproduced the implementation's
# own assumptions, so it could not fail on either revert bug the audit found —
# and it had already drifted (no _routing_override_active gate).


# ── 5. metrics: the dimensions a harvest actually has to slice by ────────────

def _decision_rows(caplog):
    return [json.loads(r.getMessage().split("routing_decision ", 1)[1])
            for r in caplog.records if "routing_decision " in r.getMessage()]


def _routing_on(reg, tier=None):
    """Force the plugin on with a deterministic classification."""
    tier = tier or reg.TIER_CHEAP
    return (patch.object(reg, "is_intelligent_routing_enabled", return_value=True),
            patch.object(reg, "decide_tier", return_value=tier),
            patch.object(reg, "classify_turn", return_value="mechanical"),
            patch.object(reg, "cheap_tier_target", return_value=("deepseek/x", "openrouter")),
            patch.object(reg, "cheap_extra_body", return_value={}))


def test_decision_row_carries_join_keys_and_group_flag(caplog):
    """A row with no session_id/turn_id cannot be joined to post-call usage, and
    without a group flag the group-chat regression _SENDER_PREFIX_RE exists to
    catch stays invisible in the harvest."""
    from plugins.intelligent_routing import registration as reg
    ctxs = _routing_on(reg)
    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4], \
            caplog.at_level(logging.INFO, logger=reg.logger.name):
        reg.on_pre_llm_call(user_message="[alice] list the files",
                            model="z-ai/glm-5.3", platform="discord",
                            session_id="S1", turn_id="T7")
    row = _decision_rows(caplog)[0]
    assert row["session_id"] == "S1"
    assert row["turn_id"] == "T7"
    assert row["platform"] == "discord"
    assert row["is_group"] is True


def test_dm_turn_is_not_flagged_as_group(caplog):
    from plugins.intelligent_routing import registration as reg
    ctxs = _routing_on(reg)
    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4], \
            caplog.at_level(logging.INFO, logger=reg.logger.name):
        reg.on_pre_llm_call(user_message="list the files", model="z-ai/glm-5.3",
                            platform="discord", session_id="S1", turn_id="T7")
    assert _decision_rows(caplog)[0]["is_group"] is False


def test_user_directive_to_primary_emits_a_row(caplog):
    """"use claude" is the pushback signal — it must not be a silent turn."""
    from plugins.intelligent_routing import registration as reg
    ctxs = _routing_on(reg)
    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4], \
            caplog.at_level(logging.INFO, logger=reg.logger.name):
        out = reg.on_pre_llm_call(user_message="use claude please",
                                  model="z-ai/glm-5.3", session_id="S1", turn_id="T8")
    assert "route" not in out
    rows = _decision_rows(caplog)
    assert len(rows) == 1
    assert rows[0]["tier"] == "directive"
    assert rows[0]["turn_id"] == "T8"


def test_user_directive_to_another_model_emits_a_row(caplog):
    from plugins.intelligent_routing import registration as reg
    ctxs = _routing_on(reg)
    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4], \
            caplog.at_level(logging.INFO, logger=reg.logger.name):
        out = reg.on_pre_llm_call(user_message="use fable for this",
                                  model="z-ai/glm-5.3", session_id="S1", turn_id="T9")
    rows = _decision_rows(caplog)
    assert len(rows) == 1
    assert rows[0]["tier"] == "directive"
    assert rows[0]["model"] == out["route"]["model"]
