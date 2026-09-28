"""Integration tests — ChatAgent E2E（released AgentCore SDK v1.0.0）.

released invoke 署名 `invoke(user_input, session_id, ctx, input_context)` で full backbone
（initialize → pre_process → main(LLM) → post_process(整形 + S-3 + 永続化) → finalize）を
検証する。LLM は依存性注入の Fake を使い決定論化する。

S-1（ADR-006）: `BaseNode.__call__()` が entry ノードの required_trust_level(VERIFIED_EXTERNAL)
を `caller_trust_level`（ctx 由来）と照合し、不足時は status=error で停止する（例外ではない）。
"""

import json
import logging

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.agent import ChatAgent

CONFIG = {
    "system_prompt_template": "You are helpful.",
    "history_window_turns": 10,
    "history_token_budget": 3000,
    "model": "claude-sonnet-4-6",
    "temperature": 0.7,
    "max_tokens": 256,
}


class _Fake:
    """会話履歴を反映した Fake LLM（応答にエコーを含めない＝S-3 安全）。"""

    def generate(self, prompt):
        return "assistant reply"


def _agent(llm=None):
    return ChatAgent(config={**CONFIG, "llm": llm or _Fake()})


def _run(agent, user_input="hello", session_id="s", level=TrustLevel.VERIFIED_EXTERNAL, with_ctx=True):
    ctx = InvocationContext(caller_trust_level=level, session_id=session_id) if with_ctx else None
    return agent.invoke(user_input=user_input, session_id=session_id, ctx=ctx)


def test_invoke_single_turn_produces_output():
    out = _run(_agent())
    assert out.get("error") is None
    assert out["formatted_output"] == "assistant reply"
    assert out["history_persisted"] is True
    assert out["status"] == "success"


def test_multi_turn_history_accumulates():
    agent = _agent()
    _run(agent, user_input="turn 1", session_id="sess-a")
    out2 = _run(agent, user_input="turn 2", session_id="sess-a")
    msgs = json.loads(out2["windowed_messages"])
    contents = [m["content"] for m in msgs]
    assert "turn 1" in contents
    assert "turn 2" in contents


def test_multi_session_isolation():
    agent = _agent()
    _run(agent, user_input="SECRET-A", session_id="sess-a")
    out_b = _run(agent, user_input="hi-b", session_id="sess-b")
    msgs_b = json.loads(out_b["windowed_messages"])
    contents_b = [m["content"] for m in msgs_b]
    assert "SECRET-A" not in contents_b  # session A の履歴が B に漏れない
    assert contents_b == ["hi-b"]


def test_invoke_denied_for_anonymous():
    out = _run(_agent(), level=TrustLevel.ANONYMOUS)
    assert out["status"] == "error"
    assert out.get("formatted_output") is None
    assert any("trust gate denied" in e.lower() for e in out["error_log"])


def test_invoke_denies_absent_context():
    # absent context = ANONYMOUS の default-deny（invoke() の caller_trust_level baseline=ANONYMOUS）
    out = ChatAgent(config={**CONFIG, "llm": _Fake()}).invoke(user_input="hi")
    assert out["status"] == "error"
    assert out.get("formatted_output") is None
    assert any("trust gate denied" in e.lower() for e in out["error_log"])


def test_internal_trust_allowed():
    out = _run(_agent(), level=TrustLevel.INTERNAL)
    assert out.get("error") is None
    assert out["formatted_output"] == "assistant reply"


def test_invoke_lifecycle_trace_emitted(caplog):
    # S-4: ドメインイベント chat_complete が agentcore.audit ロガーに 1 件出る。
    #   1.0.0 wheel は backbone の node_start/node_complete も同ロガーへ自動 emit するため、
    #   ドメインイベント（chat_complete）だけを抽出して検証する。
    with caplog.at_level(logging.INFO, logger="agentcore.audit"):
        _run(_agent(), session_id="sess-x")
    audit = [r for r in caplog.records if r.name == "agentcore.audit"]
    domain = [json.loads(r.getMessage()) for r in audit if '"event_type": "chat_complete"' in r.getMessage()]
    assert len(domain) == 1
    record = domain[0]
    assert record["event_type"] == "chat_complete"
    # 監査イベントに発話本文が含まれていない
    assert "hello" not in json.dumps(record["payload"], ensure_ascii=False)


def test_s2_credential_in_input_blocked():
    leak = "my key is sk-" + "a" * 25
    out = _run(_agent(), user_input=leak)
    assert out["error_code"] == "S2_INJECTION_DETECTED"
    assert out["status"] == "error"


def test_independent_runs_do_not_share_state():
    agent = _agent()
    out1 = _run(agent, user_input="A", session_id="a")
    out2 = _run(agent, user_input="B", session_id="b")
    assert out1["formatted_output"] == "assistant reply"
    assert out2["formatted_output"] == "assistant reply"
    # 各セッションは自分の履歴のみ
    assert json.loads(out2["windowed_messages"]) == [{"role": "user", "content": "B"}]
