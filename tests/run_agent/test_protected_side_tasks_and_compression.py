"""Audit round 3 repros: each test asserts the rule (protected text never reaches a
non-primary model). Title generation and background review are side tasks, skipped
outright in a protected session; compression and auxiliary calls run on the
configured primary only."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.run_agent.test_protected_text_stays_on_primary import (
    FB, _stick_on_fallback,
)
from tests.run_agent.test_routed_turn_escalation_loop import (
    PAPA_REO, PRIMARY, _agent, _reply, _run,
)

AUX = "aux/cheap-side-model"
BODY = "Kia ora koutou. " * 60  # > 500 chars of transcript body, marker only at the end
LONG_TRANSCRIPT = BODY + "\n\n" + PAPA_REO


def _recording_client(sent):
    client = MagicMock()
    client.base_url = "http://127.0.0.1:7/v1"

    def _create(**kw):
        sent.append({"model": kw.get("model"), "messages": kw.get("messages")})
        msg = SimpleNamespace(content="A title", tool_calls=None, reasoning=None,
                              reasoning_content=None, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=None, model=kw.get("model"))

    client.chat.completions.create.side_effect = _create
    return client


# ── B1: title generation truncates the first message, dropping the footer ─────

def test_title_generation_sends_transcript_body_to_title_model():
    """generate_title() cuts user/assistant text to 500 chars. The Papa Reo footer sits
    at the END of a transcript, so it is cut off, _protected_text_pin sees no marker,
    and the body goes to auxiliary.title_generation's model."""
    from agent import auxiliary_client as ac
    from agent import title_generator as tg

    sent = []
    client = _recording_client(sent)

    def _get_cached(provider, model=None, **_kw):
        return client, model

    with patch.object(ac, "_resolve_task_provider_model",
                      side_effect=lambda task, p=None, m=None, b=None, k=None:
                      (p or "openrouter", m or AUX, b, k, None)), \
         patch.object(ac, "_get_cached_client", _get_cached), \
         patch.object(tg, "_auto_title_enabled", return_value=True):
        tg.generate_title(LONG_TRANSCRIPT, "Here is the transcript you sent: " + BODY,
                          main_runtime={"model": PRIMARY, "provider": "custom",
                                        "base_url": "http://127.0.0.1:9/v1", "api_key": "k"})
    leaked = [s for s in sent if s["model"] != PRIMARY and "kia ora koutou" in str(s["messages"]).lower()]
    assert not leaked, f"transcript body sent to {[s['model'] for s in leaked]}"


# ── B2: background review routed to auxiliary.background_review ──────────────

def test_background_review_on_routed_review_model_gets_protected_transcript():
    """finalize_turn only skips review when a ROUTED agent is off its primary. A normal
    protected turn on the primary still spawns the review, and _resolve_review_runtime
    sends the whole (digested) conversation to auxiliary.background_review's model."""
    from agent import background_review as br

    a = _agent()
    snapshot = [
        {"role": "user", "content": "transcribe the voice note"},
        {"role": "tool", "tool_call_id": "c1", "content": LONG_TRANSCRIPT},
        {"role": "assistant", "content": "Done. " + BODY},
    ]
    built = []

    class _FakeAgent:
        def __init__(self, **kw):
            built.append(kw)
            self.model = kw.get("model")
            self._session_messages = []

        def __getattr__(self, name):
            return MagicMock()

        def run_conversation(self, user_message=None, conversation_history=None, **_kw):
            built[-1]["history"] = conversation_history
            return {"final_response": ""}

    cfg = {"auxiliary": {"background_review": {"provider": "openrouter", "model": AUX}}}
    with patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value={"provider": "openrouter", "model": AUX, "api_key": "x",
                             "base_url": "http://127.0.0.1:6/v1", "api_mode": "chat_completions"}), \
         patch("run_agent.AIAgent", _FakeAgent):
        try:
            br._run_review_in_thread(a, snapshot, br._MEMORY_REVIEW_PROMPT)
        except Exception:
            pass
    assert not built, "review fork built for a protected session"
    leaked = [b for b in built if b.get("model") != PRIMARY
              and "papa reo" in str(b.get("history")).lower()]
    assert not leaked, f"protected transcript handed to review fork on {[b['model'] for b in leaked]}"


def test_finalize_turn_skips_review_for_protected_turn_on_primary():
    """A protected turn on the primary must not reach the background-review fork."""
    from agent.protected_session import skip_side_task
    a = _agent()
    msgs = [{"role": "tool", "content": PAPA_REO}]
    assert skip_side_task("background_review", a, *msgs) is True
    assert a._protected_session is True


# ── B3: forced restore leaves the aux 'main runtime' on the fallback ─────────

def test_forced_primary_restore_leaves_aux_main_runtime_on_fallback():
    """Turn 1 sticks on FB (cooldown). Turn 2: turn start set_runtime_main(FB); a tool
    result brings a transcript; the loop gate force-restores the primary but never
    re-publishes set_runtime_main. A later auxiliary call (MoA advisor, plugin
    ctx.llm, MCP sampling, query rewrite - anything with main_runtime=None) is pinned
    to the 'main' runtime, which is still FB."""
    from agent import auxiliary_client as ac

    a = _agent()
    _stick_on_fallback(a)
    tc = SimpleNamespace(id="call_1", type="function",
                         function=SimpleNamespace(name="memory", arguments="{}"))
    aux_sent = []
    client = _recording_client(aux_sent)
    state = {"n": 0}

    def replies(model, i):
        return _reply("", tool_calls=[tc]) if i < 2 else _reply("done")

    def fake_tools(assistant_message, messages, *_a, **_kw):
        state["n"] += 1
        if state["n"] == 1:
            messages.append({"role": "tool", "tool_call_id": "call_1", "content": PAPA_REO})
        else:
            # second tool batch runs after the gate moved the agent to the primary
            with patch.object(ac, "_get_cached_client",
                              lambda provider, model=None, **_kw: (client, model)):
                try:
                    ac.call_llm(task=None, messages=[{"role": "user", "content": PAPA_REO}])
                except Exception as exc:  # noqa: BLE001
                    aux_sent.append({"model": f"raised {type(exc).__name__}", "messages": []})
            messages.append({"role": "tool", "tool_call_id": "call_1", "content": "ok"})

    a.valid_tool_names = set(a.valid_tool_names or ()) | {"memory"}
    with patch.object(a, "_create_openai_client", return_value=MagicMock()), \
         patch.object(a, "_execute_tool_calls", fake_tools):
        result, calls = _run(a, replies, hook_results=[], message="hi")
    models = [c["model"] for c in calls]
    assert models[0] == FB and all(m == PRIMARY for m in models[1:]), models
    assert aux_sent, "aux call not reached"
    assert aux_sent[0]["model"] == PRIMARY, \
        f"aux call carrying protected text pinned to {aux_sent[0]['model']} (main loop on {models})"


# ── B4: manual /compress (CLI, TUI, ACP) between turns on a sticky fallback ───

def test_manual_compress_on_sticky_fallback_sends_transcript_to_fallback():
    """Between turns the agent can still sit on FB (rate-limit cooldown). CLI/TUI/ACP
    /compress call agent._compress_context directly. Its guard (escalate_if_protected)
    only acts on a routed agent, and call_llm pins compression to the compressor's
    model, which is FB."""
    from agent import auxiliary_client as ac

    a = _agent()
    _stick_on_fallback(a)
    assert a.model == FB
    sent = []
    client = _recording_client(sent)
    msgs = [{"role": "user", "content": "transcribe"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "here"},
            {"role": "assistant", "content": "t", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "memory", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": LONG_TRANSCRIPT}] + [
            {"role": "user", "content": f"q{i}"} if i % 2 == 0 else
            {"role": "assistant", "content": f"a{i}"} for i in range(12)]
    with patch.object(ac, "_get_cached_client",
                      lambda provider, model=None, **_kw: (client, model)):
        try:
            a._compress_context(msgs, None, approx_tokens=90_000, force=True)
        except Exception:
            pass
    assert sent, "compression made no model call"
    leaked = [s for s in sent if s["model"] != PRIMARY and "papa reo" in str(s["messages"]).lower()]
    assert not leaked, f"compression of a protected session sent to {[s['model'] for s in leaked]}"


def test_flag_survives_manual_compression_and_is_carried_to_the_continuation():
    """Compression drops the transcript (and its marker) from the history; the session
    stays protected and the continuation's record is created with the flag."""
    from agent import auxiliary_client as ac
    from agent.protected_session import is_protected

    a = _agent()
    sent = []
    client = _recording_client(sent)
    msgs = [{"role": "user", "content": "transcribe"},
            {"role": "tool", "tool_call_id": "c1", "content": LONG_TRANSCRIPT}] + [
            {"role": "user", "content": f"q{i}"} if i % 2 == 0 else
            {"role": "assistant", "content": f"a{i}"} for i in range(12)]
    with patch.object(ac, "_get_cached_client",
                      lambda provider, model=None, **_kw: (client, model)):
        try:
            a._compress_context(msgs, None, approx_tokens=90_000, force=True)
        except Exception:
            pass
    assert is_protected(a)
    assert a._session_init_model_config.get("_protected") is True
    assert all(s["model"] == PRIMARY for s in sent)


def test_goal_judge_title_and_query_rewrite_skip_protected_text():
    from hermes_cli.goals import judge_goal
    from plugins.memory.query_rewrite import rewrite_memory_query
    with patch("agent.auxiliary_client.call_llm") as call:
        assert judge_goal("summarise it", LONG_TRANSCRIPT)[0] == "skipped"
        assert rewrite_memory_query(LONG_TRANSCRIPT) == ""
    call.assert_not_called()


def test_moa_advisors_skip_protected_text():
    from agent.moa_loop import _run_references_parallel
    with patch("agent.auxiliary_client.call_llm") as call:
        out = _run_references_parallel([{"provider": "openrouter", "model": AUX}],
                                       [{"role": "user", "content": LONG_TRANSCRIPT}])
    assert out == [] and not call.called
