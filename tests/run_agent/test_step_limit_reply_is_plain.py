"""At the step limit, the user must learn that the work stopped and why."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _agent(reply="Built the ward. The stairs are not done.", raises=None):
    from run_agent import AIAgent

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=60,
        )
    client = MagicMock()
    if raises is not None:
        client.chat.completions.create.side_effect = raises
    else:
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=reply, tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
            model="test/model",
        )
    agent._ensure_primary_openai_client = lambda **_kw: client
    agent._last_summary_client = client
    return agent, client


def _messages():
    return [{"role": "user", "content": "build it"}]


def test_reply_says_it_stopped_at_its_step_limit():
    agent, _ = _agent()
    reply = agent._handle_max_iterations(_messages(), 60)
    assert "step limit" in reply
    assert "60" in reply
    assert "Built the ward" in reply


def test_reply_says_nothing_is_running_when_work_may_remain():
    agent, _ = _agent()
    reply = agent._handle_max_iterations(_messages(), 60)
    assert "Nothing is running" in reply
    assert "reply" in reply.lower()


def test_model_is_asked_to_say_whether_work_remains():
    agent, client = _agent()
    agent._handle_max_iterations(_messages(), 60)
    sent = client.chat.completions.create.call_args.kwargs["messages"]
    assert any("Work remaining:" in str(m.get("content")) for m in sent)


def test_model_that_says_work_remains_is_not_contradicted():
    agent, _ = _agent(reply="Built the ward.\nWork remaining: the stairs and the music.")
    reply = agent._handle_max_iterations(_messages(), 60)
    assert "the stairs and the music" in reply
    assert "can't say" not in reply
    assert "Nothing is running" in reply


def test_model_that_says_nothing_remains_is_not_told_to_reply():
    agent, _ = _agent(reply="All built.\nWork remaining: none")
    reply = agent._handle_max_iterations(_messages(), 60)
    assert "step limit" in reply
    assert "Nothing is running" not in reply


def test_silent_model_gets_an_honest_unknown():
    agent, _ = _agent(reply="Built the ward.")
    reply = agent._handle_max_iterations(_messages(), 60)
    assert "can't say" in reply


def test_failed_summary_still_says_it_plainly_and_leaks_no_error_text():
    agent, _ = _agent(raises=RuntimeError("Error code: 500 secret-internal-detail"))
    reply = agent._handle_max_iterations(_messages(), 60)
    assert "step limit" in reply
    assert "Nothing is running" in reply
    assert "secret-internal-detail" not in reply


def test_stored_reply_matches_what_the_user_got():
    agent, _ = _agent()
    messages = _messages()
    reply = agent._handle_max_iterations(messages, 60)
    assert messages[-1] == {"role": "assistant", "content": reply}


def test_cron_job_is_not_told_to_reply():
    """A cron job's reply goes to its delivery target. Nobody can answer it."""
    agent, _ = _agent()
    agent.platform = "cron"
    reply = agent._handle_max_iterations(_messages(), 60)
    assert "step limit" in reply
    assert "Nothing is running now." in reply
    assert "Send a reply" not in reply


def test_subagent_is_not_told_to_reply():
    agent, _ = _agent()
    agent.platform = "discord"
    agent._delegate_depth = 1
    reply = agent._handle_max_iterations(_messages(), 60)
    assert "Nothing is running now." in reply
    assert "Send a reply" not in reply


def test_a_person_on_a_platform_is_still_told_to_reply():
    agent, _ = _agent()
    agent.platform = "discord"
    reply = agent._handle_max_iterations(_messages(), 60)
    assert "Send a reply" in reply
