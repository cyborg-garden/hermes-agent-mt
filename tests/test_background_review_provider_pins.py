"""The background review fork keeps the parent's OpenRouter provider pins.

An agent pinned two hosts out in ``provider_routing.ignore`` because they
dropped tool-call arguments. Its main loop stopped seeing them; its background
review fork did not, and logged hundreds of skill_manage calls a day with
``old_string`` or ``file_path`` missing. The fork now carries the pins.
"""

import types

from agent import background_review as br


def parent(**attrs):
    base = dict(providers_allowed=None, providers_ignored=None, providers_order=None, provider_sort=None,
                provider_require_parameters=False, provider_data_collection=None)
    base.update(attrs)
    return types.SimpleNamespace(**base)


def test_pins_carry_over_to_the_fork():
    agent = parent(providers_ignored=["inference-net", "relace"], provider_require_parameters=True,
                   provider_data_collection="deny")
    assert br.provider_routing_kwargs(agent) == {
        "providers_ignored": ["inference-net", "relace"],
        "provider_require_parameters": True,
        "provider_data_collection": "deny",
    }


def test_control_an_agent_with_no_pins_passes_none():
    assert br.provider_routing_kwargs(parent()) == {}


def test_every_kwarg_is_one_the_agent_takes():
    import inspect
    from run_agent import AIAgent
    params = inspect.signature(AIAgent.__init__).parameters
    agent = parent(providers_allowed=["a"], providers_ignored=["b"], providers_order=["c"], provider_sort="price",
                   provider_require_parameters=True, provider_data_collection="deny")
    for key in br.provider_routing_kwargs(agent):
        assert key in params, key


def test_the_fork_is_built_with_them():
    src = open(br.__file__).read()
    assert "_fork_kwargs: Dict[str, Any] = dict(provider_routing_kwargs(agent))" in src
