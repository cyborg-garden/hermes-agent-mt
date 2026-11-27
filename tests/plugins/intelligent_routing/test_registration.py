"""Tests for the intelligent-routing pre_llm_call hook wiring.

The hook:
  - registers on ``pre_llm_call``,
  - is INERT when the per-agent toggle is OFF (returns None -> zero behavior
    change, the opt-in guarantee),
  - when ON, classifies the turn and returns a context injection that extends
    the model-awareness line with the routing reason, e.g.
    ``[Current model: <model> via OpenRouter — routed: mechanical]``,
  - fails open (routes premium / does not claim "mechanical") on uncertainty,
  - never raises into the turn loop.
"""
from unittest.mock import MagicMock, patch

from plugins.intelligent_routing import registration


class FakeCtx:
    def __init__(self):
        self.hooks = {}

    def register_hook(self, name, cb):
        self.hooks.setdefault(name, []).append(cb)


def test_registers_pre_llm_call_hook():
    ctx = FakeCtx()
    registration.register(ctx)
    assert "pre_llm_call" in ctx.hooks


def test_inert_when_disabled():
    """Toggle OFF -> hook returns None (no context injected, no behavior change)."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=False):
        result = registration.on_pre_llm_call(
            user_message="mark card 42 done", model="claude-sonnet-5", platform="cli",
        )
    assert result is None


def test_mechanical_turn_injects_cheap_tier_reason():
    """A confidently mechanical turn surfaces the cheap tier it routes to.

    The reason line reflects the ACTUAL decider — the binary classification —
    so the label never disagrees with the tier."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="grep for TODO in the repo", model="claude-sonnet-5",
            platform="cli",
        )
    assert result is not None
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "routed: lookup → cheap" in ctx_text
    assert "Current model:" in ctx_text


def test_architecture_turn_routes_premium():
    """A hard architecture question fails open to premium (binary: judgment)."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="design a system for multi-tenant routing across the fleet",
            model="claude-sonnet-5", platform="cli",
        )
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "→ premium" in ctx_text
    assert "→ cheap" not in ctx_text


def test_code_gen_turn_routes_premium():
    """Code-gen -> premium: the binary classifier reads it as judgment (never
    cheap), so multi-file coding stays on Claude."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="Write a Python function that merges two sorted linked lists.",
            model="claude-sonnet-5", platform="cli",
        )
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "→ premium" in ctx_text
    assert "→ cheap" not in ctx_text


def test_uncertain_turn_fails_open_to_premium_reason():
    """An empty message is uncertain -> must route premium (fail open), never cheap."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="", model="claude-sonnet-5", platform="cli",
        )
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "→ cheap" not in ctx_text
    assert "→ premium" in ctx_text


def test_public_facing_never_cheap_routed():
    """Public-facing dominates: reason must show premium even if mechanical-shaped."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="grep the announcement", model="claude-sonnet-5",
            platform="cli", is_public_facing=True,
        )
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "→ premium" in ctx_text
    assert "→ cheap" not in ctx_text


def test_hook_never_raises_on_bad_input():
    """Defensive: bad/missing kwargs must not raise into the turn loop."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        # No user_message, weird model type.
        result = registration.on_pre_llm_call(model=None)
    # Either None or a well-formed injection, but no exception.
    assert result is None or "Current model:" in (
        result["context"] if isinstance(result, dict) else result
    )


def test_background_context_classified_mechanical_cheap():
    """A non-interactive/background platform routes mechanical → cheap when enabled."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="run nightly digest", model="claude-sonnet-5",
            platform="cron",
        )
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "routed: mechanical → cheap" in ctx_text


# ------------------------------------- route override emission (actuation) ----

def test_cheap_route_emits_route_override():
    """A cheap-tier turn emits a `route` override targeting the configured cheap
    model so the runtime can actuate the swap."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="grep for TODO", model="claude-sonnet-5", platform="cli",
        )
    assert isinstance(result, dict)
    assert result.get("route") == {
        "model": "deepseek/deepseek-v4-flash-0731", "provider": "openrouter",
        "escalate": True,
        "protected_watch": True,
    }


def test_premium_route_emits_no_override():
    """A premium turn must NOT emit a route override (stay on the primary)."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="design a system for multi-tenant routing across the fleet",
            model="claude-sonnet-5", platform="cli",
        )
    assert isinstance(result, dict)
    assert "route" not in result


def test_public_facing_emits_no_override():
    """Public-facing never emits a cheap route override."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="grep the announcement", model="claude-sonnet-5",
            platform="cli", is_public_facing=True,
        )
    assert "route" not in result


def test_route_override_honors_config_target():
    """The emitted override uses the per-agent configured cheap target."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True), \
         patch("plugins.intelligent_routing.registration.cheap_tier_target",
               return_value=("qwen/qwen-2.5", "ollama")):
        result = registration.on_pre_llm_call(
            user_message="grep for TODO", model="claude-sonnet-5", platform="cli",
        )
    assert result["route"] == {"model": "qwen/qwen-2.5", "provider": "ollama", "escalate": True, "protected_watch": True}


# ---- group [sender] attribution prefix (Signal/group classifier bug) ---------
#
# In a shared multi-user (group) session the gateway prepends "[sender] " to the
# message before it reaches pre_llm_call (gateway/run.py: `[{user_name}] {text}`).
# That prefix made the classifier's leading-imperative-verb check see "[mare]"
# instead of "list", so EVERY group turn failed the mechanical test and routed
# premium — the intelligent router was inert in exactly the place agents live.
# _signals_from_kwargs must strip a leading "[sender] " attribution before
# classifying, without over-triggering on genuine judgment content.


def test_group_prefixed_mechanical_turn_routes_cheap():
    """A mechanical ask carrying a "[sender] " group prefix still routes cheap."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="[mare] list the numbers one to five",
            model="claude-sonnet-5", platform="signal",
        )
    assert isinstance(result, dict)
    assert result.get("route") == {
        "model": "deepseek/deepseek-v4-flash-0731", "provider": "openrouter",
        "escalate": True,
        "protected_watch": True,
    }


def test_group_prefix_probe_matches_unprefixed():
    """probe_route: the "[sender] " prefix must not change the tier vs. the bare
    message — the classifier decides on the real content, not the attribution."""
    bare = registration.probe_route("grep for TODO in the repo")
    prefixed = registration.probe_route("[oz] grep for TODO in the repo")
    assert prefixed["tier"] == bare["tier"] == "cheap"


def test_group_prefix_does_not_make_judgment_cheap():
    """Stripping the prefix must NOT over-trigger: a judgment ask that happens to
    carry a group prefix still routes premium (no route override)."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="[mare] what should we prioritize next quarter and why?",
            model="claude-sonnet-5", platform="signal",
        )
    assert "route" not in result


def test_group_prefix_strip_is_single_leading_token_only():
    """Only ONE leading bracketed attribution token is stripped; a second bracket
    in the body is left intact (the strip is an attribution peel, not a cleaner)."""
    # After peeling "[mare] ", the remaining "grep [WIP] logs" still leads with a
    # verb → cheap. This pins that we peel exactly one leading token.
    assert registration.probe_route("[mare] grep [WIP] logs")["tier"] == "cheap"


# ---- user model directive ("use fable for this") ----------------------------
#
# A user can explicitly request a model tier mid-message. The directive wins
# over the classifier: "use fable" routes to Fable even if the turn looks
# mechanical; "use deepseek" routes cheap even if the classifier would go
# premium. The parse fires BEFORE the classifier so "use fable" isn't
# misread as an imperative verb → mechanical → cheap route to deepseek
# (which is what would happen without the directive intercept — "use" IS in
# _IMPERATIVE_VERBS, so a bare "use fable" would currently route deepseek).


def test_user_directive_fable_routes_to_fable():
    """'use fable' overrides the classifier and routes to claude-fable-5."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="hey can you use fable for this task?",
            model="claude-sonnet-5", platform="signal",
        )
    assert isinstance(result, dict)
    assert result["route"] == {"model": "claude-fable-5", "provider": "anthropic", "protected_watch": True}
    assert "user-requested" in result["context"]


def test_user_directive_deepseek_routes_cheap():
    """'use deepseek' routes to the cheap deepseek tier even on a judgment-shaped ask."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="design the multi-tenant auth system, use deepseek",
            model="claude-sonnet-5", platform="cli",
        )
    assert isinstance(result, dict)
    assert result["route"] == {"model": "deepseek/deepseek-v4-flash-0731", "provider": "openrouter", "protected_watch": True}


def test_user_directive_opus_routes_to_opus():
    """'use opus' routes to claude-opus-4-8 / anthropic."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="use opus — this needs max reasoning",
            model="claude-sonnet-5", platform="cli",
        )
    assert isinstance(result, dict)
    assert result["route"]["model"] == "claude-opus-4-8"
    assert result["route"]["provider"] == "anthropic"


def test_user_directive_group_prefix_plus_model():
    """Directive still fires when a group [sender] prefix is present."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="[mare] use fable for this please",
            model="claude-sonnet-5", platform="signal",
        )
    assert isinstance(result, dict)
    assert result["route"] == {"model": "claude-fable-5", "provider": "anthropic", "protected_watch": True}


def test_user_directive_wins_over_mechanical_classifier():
    """'use fable' on a mechanical-shaped message routes to fable, NOT deepseek.

    Without directive intercept, 'use' is in _IMPERATIVE_VERBS so a message
    starting with 'use fable' would be classified mechanical → cheap → deepseek.
    The directive must short-circuit the classifier and route to fable instead.
    """
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="use fable",
            model="claude-sonnet-5", platform="cli",
        )
    assert isinstance(result, dict)
    route = result.get("route", {})
    assert route.get("model") == "claude-fable-5", (
        f"'use fable' must route to fable, not deepseek; got route={route}"
    )


def test_no_directive_falls_through_to_classifier():
    """Without a model directive the classifier runs normally."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="grep for TODO in the repo",
            model="claude-sonnet-5", platform="cli",
        )
    assert isinstance(result, dict)
    assert result.get("route") == {"model": "deepseek/deepseek-v4-flash-0731", "provider": "openrouter", "escalate": True, "protected_watch": True}


# ---- kimi / glm directives (three-tier fleet hierarchy) ----------------------
#
# Fleet cost restructure: PRIMARY = z-ai/glm-5.2 (chat default), CHEAP =
# deepseek/deepseek-v4-flash-0731 (mechanical tier), ON-DEMAND PREMIUM =
# moonshotai/kimi-k3 ("use kimi" runs that turn on Kimi K3). Same directive
# mechanism as fable/opus/deepseek — intercepted before the classifier.


def test_user_directive_kimi_routes_to_kimi_k3():
    """'use kimi' overrides the classifier and routes to moonshotai/kimi-k3."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="use kimi for this — it needs the premium tier",
            model="z-ai/glm-5.2", platform="signal",
        )
    assert isinstance(result, dict)
    assert result["route"] == {"model": "moonshotai/kimi-k3", "provider": "openrouter", "protected_watch": True}
    assert "user-requested" in result["context"]


def test_user_directive_kimi_aliases_kimi3_and_k3():
    """'use kimi3' and 'use k3' are aliases for moonshotai/kimi-k3."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        for phrase in ("please use kimi3 here", "use k3 — max reasoning"):
            result = registration.on_pre_llm_call(
                user_message=phrase, model="z-ai/glm-5.2", platform="cli",
            )
            assert isinstance(result, dict), phrase
            assert result["route"] == {
                "model": "moonshotai/kimi-k3", "provider": "openrouter", "protected_watch": True,
            }, phrase


def test_user_directive_kimi_wins_over_mechanical_classifier():
    """Bare 'use kimi' routes to kimi-k3, NOT deepseek.

    'use' is in _IMPERATIVE_VERBS, so without the directive intercept a bare
    'use kimi' would be classified mechanical → cheap → deepseek. The
    directive must short-circuit the classifier (same trap as 'use fable').
    """
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="use kimi",
            model="z-ai/glm-5.2", platform="cli",
        )
    assert isinstance(result, dict)
    route = result.get("route", {})
    assert route.get("model") == "moonshotai/kimi-k3", (
        f"'use kimi' must route to kimi-k3, not deepseek; got route={route}"
    )


def test_user_directive_glm_routes_to_glm():
    """'use glm' routes to z-ai/glm-5.2 / openrouter (the fleet primary)."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="use glm for this one",
            model="claude-sonnet-5", platform="cli",
        )
    assert isinstance(result, dict)
    assert result["route"] == {"model": "z-ai/glm-5.2", "provider": "openrouter", "protected_watch": True}


def test_user_directive_group_prefix_plus_kimi():
    """Kimi directive still fires when a group [sender] prefix is present."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="[mare] use kimi for this please",
            model="z-ai/glm-5.2", platform="signal",
        )
    assert isinstance(result, dict)
    assert result["route"] == {"model": "moonshotai/kimi-k3", "provider": "openrouter", "protected_watch": True}
