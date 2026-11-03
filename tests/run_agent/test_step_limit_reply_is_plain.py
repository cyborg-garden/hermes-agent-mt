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


def _live_process(monkeypatch, *, task_id=None, session_key=None):
    """What the registry reports for one running process with these ids."""
    from tools.process_registry import process_registry

    monkeypatch.setattr(process_registry, "has_active_processes", lambda t: t == task_id)
    monkeypatch.setattr(process_registry, "has_active_for_session", lambda k, **_kw: k == session_key)


def test_reply_does_not_deny_a_process_still_running_under_this_turn(monkeypatch):
    agent, _ = _agent()
    agent._current_task_id = "turn-1"
    _live_process(monkeypatch, task_id="turn-1")
    reply = agent._handle_max_iterations(_messages(), 60)
    assert "Nothing is running" not in reply
    assert "still running" in reply


def test_reply_does_not_deny_a_process_from_an_earlier_turn_of_the_session(monkeypatch):
    agent, _ = _agent()
    agent._current_task_id = "turn-2"
    agent._gateway_session_key = "discord:thread-9"
    _live_process(monkeypatch, session_key="discord:thread-9")
    reply = agent._handle_max_iterations(_messages(), 60)
    assert "Nothing is running" not in reply
    assert "still running" in reply


def test_reply_does_not_deny_a_process_found_under_the_session_id(monkeypatch):
    agent, _ = _agent()
    agent._current_task_id = "turn-2"
    agent.session_id = "session-1"
    _live_process(monkeypatch, task_id="session-1")
    assert "Nothing is running" not in agent._handle_max_iterations(_messages(), 60)


def test_another_sessions_process_does_not_change_the_reply(monkeypatch):
    agent, _ = _agent()
    agent._current_task_id = "turn-1"
    agent._gateway_session_key = "discord:thread-9"
    _live_process(monkeypatch, task_id="someone-else", session_key="discord:other")
    assert "Nothing is running now." in agent._handle_max_iterations(_messages(), 60)


def test_failed_process_check_makes_no_claim_either_way(monkeypatch):
    from tools.process_registry import process_registry

    def _boom(_task_id):
        raise RuntimeError("registry broke")

    agent, _ = _agent()
    agent._current_task_id = "turn-1"
    monkeypatch.setattr(process_registry, "has_active_processes", _boom)
    reply = agent._handle_max_iterations(_messages(), 60)
    assert "Nothing is running" not in reply
    assert "can't tell" in reply
    assert "registry broke" not in reply


def test_real_registry_entry_is_seen_by_the_step_limit_reply():
    """No mocked registry: an entry for this turn, as the terminal tool would add it."""
    from tools.process_registry import ProcessSession, process_registry

    agent, _ = _agent()
    agent._current_task_id = "step-limit-real-turn"
    session = ProcessSession(id="proc_step_limit_test", command="sleep 8", task_id="step-limit-real-turn")
    with process_registry._lock:
        process_registry._running[session.id] = session
    try:
        reply = agent._handle_max_iterations(_messages(), 60)
    finally:
        with process_registry._lock:
            process_registry._running.pop(session.id, None)
    assert "Nothing is running" not in reply
    assert "still running" in reply
