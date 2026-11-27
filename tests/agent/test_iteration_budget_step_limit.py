"""Unit tests for the step-limit wrap-up helpers in agent.iteration_budget."""

from types import SimpleNamespace

from agent.conversation_loop import _maybe_append_step_limit_warning
from agent.iteration_budget import (
    IterationBudget,
    should_warn_step_limit,
    step_limit_warning_text,
    step_limit_warning_threshold,
)


def test_threshold_is_eighty_percent_rounded_up():
    assert step_limit_warning_threshold(150, 0.8) == 120
    assert step_limit_warning_threshold(50, 0.8) == 40
    assert step_limit_warning_threshold(7, 0.8) == 6


def test_threshold_leaves_at_least_one_step():
    assert step_limit_warning_threshold(2, 0.99) == 1
    assert step_limit_warning_threshold(3, 0.95) == 2


def test_threshold_disabled_for_bad_fraction_or_tiny_budget():
    for frac in (0, 1, 1.5, -0.2, None, "nope"):
        assert step_limit_warning_threshold(100, frac) is None
    assert step_limit_warning_threshold(1, 0.8) is None


def test_should_warn_flips_at_threshold():
    assert not should_warn_step_limit(119, 150, 0.8)
    assert should_warn_step_limit(120, 150, 0.8)
    assert should_warn_step_limit(149, 150, 0.8)


def test_warning_text_names_counts_and_says_stop_new_work():
    text = step_limit_warning_text(120, 150)
    assert "120 of 150 steps" in text
    assert "Do not start new work" in text


def _agent(used, total, **kw):
    budget = IterationBudget(total)
    for _ in range(used):
        budget.consume()
    base = dict(
        iteration_budget=budget,
        _step_limit_warning=True,
        _step_limit_warning_at=0.8,
        _step_limit_warned=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_hook_appends_to_last_tool_result_once():
    agent = _agent(8, 10)
    messages = [{"role": "tool", "tool_call_id": "a", "content": "result"}]
    assert _maybe_append_step_limit_warning(agent, messages) is True
    assert messages[-1]["content"].startswith("result\n\n[System note: You have used 8 of 10")
    assert len(messages) == 1
    messages.append({"role": "tool", "tool_call_id": "b", "content": "next"})
    assert _maybe_append_step_limit_warning(agent, messages) is False
    assert messages[-1]["content"] == "next"


def test_hook_does_not_refire_after_compression_drops_note():
    agent = _agent(8, 10)
    messages = [{"role": "tool", "tool_call_id": "a", "content": "r"}]
    _maybe_append_step_limit_warning(agent, messages)
    compressed = [{"role": "tool", "tool_call_id": "z", "content": "summary"}]
    agent.iteration_budget.consume()
    assert _maybe_append_step_limit_warning(agent, compressed) is False
    assert compressed[-1]["content"] == "summary"


def test_hook_waits_for_threshold():
    agent = _agent(7, 10)
    messages = [{"role": "tool", "tool_call_id": "a", "content": "r"}]
    assert _maybe_append_step_limit_warning(agent, messages) is False
    assert agent._step_limit_warned is False


def test_hook_respects_disable_flag():
    agent = _agent(9, 10, _step_limit_warning=False)
    messages = [{"role": "tool", "tool_call_id": "a", "content": "r"}]
    assert _maybe_append_step_limit_warning(agent, messages) is False
    assert messages[-1]["content"] == "r"


def test_hook_handles_multimodal_tool_content():
    agent = _agent(9, 10)
    messages = [{"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "r"}]}]
    assert _maybe_append_step_limit_warning(agent, messages) is True
    assert messages[-1]["content"][-1]["type"] == "text"
    assert "9 of 10" in messages[-1]["content"][-1]["text"]


def test_hook_skips_when_last_row_is_not_a_tool_result():
    agent = _agent(9, 10)
    messages = [{"role": "user", "content": "hi"}]
    assert _maybe_append_step_limit_warning(agent, messages) is False
    assert agent._step_limit_warned is False
