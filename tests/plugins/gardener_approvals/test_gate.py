from plugins.gardener_approvals.gate import call_gate


class _Proc:
    def __init__(self, out): self.stdout = out; self.stderr = ""; self.returncode = 0


def test_parses_token_and_message():
    captured = {}

    def runner(cmd, **kw):
        captured["cmd"] = cmd
        captured["env"] = kw.get("env")
        return _Proc("OK ✓ noted, approved — D-2026-06-09-01 resolved\n")

    token, msg = call_gate(
        gate_path="/x/decision-approve.sh", uuid="U-1",
        decision_id="D-2026-06-09-01", action="resolve", value="approve",
        approver="Juniper", decisions_file="/d/decisions.md", runner=runner,
    )
    assert token == "OK"
    assert msg.startswith("✓ noted, approved")
    assert captured["cmd"][0] == "/x/decision-approve.sh"
    assert "--requester-uuid" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--requester-uuid") + 1] == "U-1"
    assert captured["env"]["DECISIONS_FILE"] == "/d/decisions.md"


def test_empty_output_is_error_token():
    token, _msg = call_gate(
        gate_path="/x/g.sh", uuid="U", decision_id="D-2026-06-09-01",
        action="resolve", value="approve", approver="J",
        runner=lambda cmd, **kw: _Proc(""),
    )
    assert token == "ERROR"


def test_runner_exception_returns_error():
    def boom(cmd, **kw): raise OSError("no such file")
    token, _msg = call_gate(
        gate_path="/x/g.sh", uuid="U", decision_id="D-2026-06-09-01",
        action="resolve", value="approve", approver="J", runner=boom,
    )
    assert token == "ERROR"
