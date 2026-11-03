"""``request_overrides["extra_body"]`` must MERGE on both transport paths.

The intelligent_routing plugin ships ``routing.cheap_extra_body`` into
``agent.request_overrides["extra_body"]`` for a routed turn (see
``agent/routing_override.py``). The provider-profile path merges that dict into
the extra_body the transport already assembled; the legacy (no-profile) path
used ``api_kwargs.update(overrides)``, which REPLACED the assembled extra_body
wholesale — so a cheap tier pointed at an unregistered provider silently lost
its provider preferences and reasoning config.

Both paths agree for DICT values: overriding keys win on collision, everything
else survives, and the merge is SHALLOW — a colliding nested dict is replaced
wholesale, not merged into.

They do NOT agree for non-dict values, and that divergence is pinned below
rather than papered over: the legacy path forwards the raw value verbatim,
while the profile path assigns it top-level and then discards it whenever the
profile assembled an extra_body of its own.
"""

import pytest

from agent.transports import get_transport


@pytest.fixture
def transport():
    import agent.transports.chat_completions
    return get_transport("chat_completions")


MSGS = [{"role": "user", "content": "Hi"}]
PREFS = {"sort": "throughput"}
ROUTED_EXTRA_BODY = {"reasoning_effort": "none"}


def _legacy_kwargs(transport, **extra):
    """Legacy path: no provider_profile (custom / unregistered provider)."""
    return transport.build_kwargs(
        model="qwen3.5:9b",
        messages=MSGS,
        is_openrouter=True,
        provider_preferences=PREFS,
        supports_reasoning=True,
        reasoning_config={"enabled": True, "effort": "medium"},
        **extra,
    )


def _profile_kwargs(transport, **extra):
    from providers import get_provider_profile

    profile = get_provider_profile("openrouter")
    return transport.build_kwargs(
        model="deepseek/deepseek-v3.2",
        messages=MSGS,
        provider_profile=profile,
        provider_name="openrouter",
        base_url=profile.base_url,
        provider_preferences=PREFS,
        supports_reasoning=True,
        reasoning_config={"enabled": True, "effort": "medium"},
        **extra,
    )


class TestLegacyPathMerges:

    def test_baseline_assembles_provider_and_reasoning(self, transport):
        eb = _legacy_kwargs(transport)["extra_body"]
        assert eb["provider"] == PREFS
        assert eb["reasoning"] == {"enabled": True, "effort": "medium"}

    def test_routed_extra_body_does_not_destroy_assembled_keys(self, transport):
        eb = _legacy_kwargs(
            transport, request_overrides={"extra_body": ROUTED_EXTRA_BODY}
        )["extra_body"]

        # The regression: these two used to vanish entirely.
        assert eb["provider"] == PREFS
        assert eb["reasoning"] == {"enabled": True, "effort": "medium"}
        assert eb["reasoning_effort"] == "none"

    def test_routed_value_wins_on_collision(self, transport):
        eb = _legacy_kwargs(
            transport,
            request_overrides={"extra_body": {"provider": {"sort": "price"}}},
        )["extra_body"]
        assert eb["provider"] == {"sort": "price"}
        assert eb["reasoning"] == {"enabled": True, "effort": "medium"}

    def test_non_extra_body_overrides_still_land_top_level(self, transport):
        kw = _legacy_kwargs(
            transport,
            request_overrides={
                "service_tier": "priority",
                "extra_body": ROUTED_EXTRA_BODY,
            },
        )
        assert kw["service_tier"] == "priority"
        assert kw["extra_body"]["provider"] == PREFS

    def test_extra_body_only_from_overrides_when_nothing_assembled(self, transport):
        kw = transport.build_kwargs(
            model="qwen3.5:9b",
            messages=MSGS,
            request_overrides={"extra_body": ROUTED_EXTRA_BODY},
        )
        assert kw["extra_body"] == ROUTED_EXTRA_BODY
        # Defensive copy, not the caller's dict. Callers pass
        # agent.request_overrides BY REFERENCE (chat_completion_helpers.py),
        # and downstream mutators recurse into api_kwargs in place
        # (conversation_loop._sanitize_structure_non_ascii) — aliasing here
        # would permanently corrupt the agent's persistent overrides.
        assert kw["extra_body"] is not ROUTED_EXTRA_BODY

    def test_returned_kwargs_mutation_cannot_reach_the_callers_overrides(
        self, transport
    ):
        overrides = {"extra_body": {"tag": "cafe"}}
        kw = transport.build_kwargs(
            model="qwen3.5:9b", messages=MSGS, request_overrides=overrides
        )
        kw["extra_body"]["tag"] = "MUTATED"
        assert overrides["extra_body"] == {"tag": "cafe"}

    def test_non_dict_extra_body_override_is_passed_through_verbatim(self, transport):
        """Malformed value: keep the old last-write-wins behaviour, don't crash."""
        kw = _legacy_kwargs(transport, request_overrides={"extra_body": "nope"})
        assert kw["extra_body"] == "nope"


class TestNestedCollisionIsShallow:
    """A colliding nested dict is REPLACED, not deep-merged. Pinned, not fixed.

    ``reasoning`` is the live case: the transport assembles the two-key
    ``{"enabled": True, "effort": ...}``, so an override of ``{"effort": "none"}``
    drops ``enabled``. That is deliberate last-write-wins — an override the
    caller wrote out in full should reach the wire in full, and a deep merge
    would make removing an assembled sub-key impossible. Documented in the
    README recipe and in the transport comment; asserted here so a change to
    either semantics is a visible test failure, not a silent one.
    """

    def test_legacy_nested_dict_is_replaced_wholesale(self, transport):
        eb = _legacy_kwargs(
            transport,
            request_overrides={"extra_body": {"reasoning": {"effort": "none"}}},
        )["extra_body"]
        assert eb["reasoning"] == {"effort": "none"}   # "enabled" is gone
        assert eb["provider"] == PREFS                 # siblings survive

    def test_profile_nested_dict_is_replaced_wholesale(self, transport):
        eb = _profile_kwargs(
            transport,
            request_overrides={"extra_body": {"provider": {"order": ["x"]}}},
        )["extra_body"]
        assert eb["provider"] == {"order": ["x"]}      # "sort" is gone


class TestProfilePathParity:

    def test_profile_path_merges_the_same_way(self, transport):
        eb = _profile_kwargs(
            transport, request_overrides={"extra_body": ROUTED_EXTRA_BODY}
        )["extra_body"]
        assert eb["provider"] == PREFS
        assert eb["reasoning_effort"] == "none"
        assert "reasoning" in eb

    def test_both_paths_preserve_assembled_keys(self, transport):
        legacy = _legacy_kwargs(
            transport, request_overrides={"extra_body": ROUTED_EXTRA_BODY}
        )["extra_body"]
        profile = _profile_kwargs(
            transport, request_overrides={"extra_body": ROUTED_EXTRA_BODY}
        )["extra_body"]
        for eb in (legacy, profile):
            assert eb["provider"] == PREFS
            assert eb["reasoning_effort"] == "none"

    def test_non_dict_override_diverges_and_the_profile_path_discards_it(
        self, transport
    ):
        """The one input where the two paths do NOT agree. Pinned, not fixed.

        Neither behaviour is defensible for a malformed value, but the plugin
        cannot emit one (config.py coerces non-dicts to {}; routing_override
        ignores them), so only a hand-written agent.request_overrides reaches
        here. Pin it so a future change to either path is visible.
        """
        legacy = _legacy_kwargs(
            transport, request_overrides={"extra_body": "nope"}
        )["extra_body"]
        assert legacy == "nope"

        profile = _profile_kwargs(
            transport, request_overrides={"extra_body": "nope"}
        )["extra_body"]
        # Assigned top-level, then overwritten by the profile's assembled dict.
        assert isinstance(profile, dict)
        assert profile["provider"] == PREFS

    def test_profile_path_keeps_a_non_dict_when_it_assembled_nothing(
        self, transport
    ):
        """The overwrite is conditional: nothing assembled ⇒ the value survives."""
        from providers import get_provider_profile

        kw = transport.build_kwargs(
            model="deepseek/deepseek-v3.2",
            messages=MSGS,
            provider_profile=get_provider_profile("openrouter"),
            provider_name="openrouter",
            request_overrides={"extra_body": "nope"},
        )
        assert kw["extra_body"] == "nope"
