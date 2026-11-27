"""Tests for intelligent-routing per-agent config (default OFF — opt-in)."""
from unittest.mock import patch

from plugins.intelligent_routing import config as cfgmod


def test_default_is_off():
    """No config -> intelligent routing is OFF (opt-in, changes visible behavior)."""
    with patch.object(cfgmod, "_load_config", return_value={}):
        assert cfgmod.is_intelligent_routing_enabled() is False


def test_missing_routing_section_is_off():
    with patch.object(cfgmod, "_load_config", return_value={"auxiliary": {}}):
        assert cfgmod.is_intelligent_routing_enabled() is False


def test_explicit_true_enables():
    with patch.object(cfgmod, "_load_config",
                      return_value={"routing": {"intelligent": True}}):
        assert cfgmod.is_intelligent_routing_enabled() is True


def test_explicit_false_disables():
    with patch.object(cfgmod, "_load_config",
                      return_value={"routing": {"intelligent": False}}):
        assert cfgmod.is_intelligent_routing_enabled() is False


def test_truthy_string_enables():
    for val in ("true", "1", "yes", "on", "True"):
        with patch.object(cfgmod, "_load_config",
                          return_value={"routing": {"intelligent": val}}):
            assert cfgmod.is_intelligent_routing_enabled() is True, val


def test_falsy_string_disables():
    for val in ("false", "0", "no", "off", ""):
        with patch.object(cfgmod, "_load_config",
                          return_value={"routing": {"intelligent": val}}):
            assert cfgmod.is_intelligent_routing_enabled() is False, val


def test_config_load_failure_fails_off():
    """If config can't be read, default OFF (never silently change behavior)."""
    with patch.object(cfgmod, "_load_config", side_effect=Exception("boom")):
        assert cfgmod.is_intelligent_routing_enabled() is False


def test_mode_defaults_to_heuristic():
    """routing.mode defaults to 'heuristic' (Option A); llm-router is Phase 2."""
    with patch.object(cfgmod, "_load_config", return_value={"routing": {}}):
        assert cfgmod.routing_mode() == "heuristic"


def test_mode_llm_router_is_read():
    with patch.object(cfgmod, "_load_config",
                      return_value={"routing": {"mode": "llm-router"}}):
        assert cfgmod.routing_mode() == "llm-router"


def test_cheap_tier_default_is_pinned_v4_flash_openrouter():
    """Default cheap target is the pinned deepseek-v4-flash-0731 snapshot via openrouter."""
    with patch.object(cfgmod, "_load_config", return_value={"routing": {}}):
        model, provider = cfgmod.cheap_tier_target()
    assert model == "deepseek/deepseek-v4-flash-0731"
    assert provider == "openrouter"


def test_cheap_tier_overridable():
    with patch.object(cfgmod, "_load_config", return_value={
        "routing": {"cheap_model": "qwen/qwen-2.5", "cheap_provider": "ollama"}
    }):
        model, provider = cfgmod.cheap_tier_target()
    assert model == "qwen/qwen-2.5"
    assert provider == "ollama"
