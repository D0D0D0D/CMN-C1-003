"""Unit tests — ChatFinalizeNode (S-4 監査トレース・成功/失敗いずれも記録).

S-4 は finalize スロットで発火する（post_process は status==success のときしか到達しない
ため、エラー経路の監査漏れを避ける。レビュー指摘 F-01/F-02 対応）。発話 / 応答本文は記録しない。
"""

import json
import logging

from framework.schemas.agent_status import AgentStatus

from src.nodes.finalize import ChatFinalizeNode


def _emit(state):
    node = ChatFinalizeNode(model="claude-sonnet-4-6")
    return node.on_finalize(state)


def test_audit_emitted_on_success(caplog):
    with caplog.at_level(logging.INFO, logger="agentcore.audit"):
        _emit(
            {
                "status": AgentStatus.SUCCESS.value,
                "session_id": "s1",
                "history_turn_count": 2,
                "history_persisted": True,
                "validated_input": "SECRET_PROMPT",
                "llm_response": "SECRET_REPLY",
            }
        )
    audit = [r for r in caplog.records if r.name == "agentcore.audit"]
    assert len(audit) == 1
    record = json.loads(audit[0].getMessage())
    assert record["event_type"] == "chat_complete"
    # session_id / status は記録し incident 相関を可能にする（F-02）
    assert record["session_id"] == "s1"
    assert record["payload"]["status"] == "success"
    # 発話 / 応答本文は payload に含めない（メタのみ）
    body = json.dumps(record["payload"], ensure_ascii=False)
    assert "SECRET_PROMPT" not in body
    assert "SECRET_REPLY" not in body


def test_audit_emitted_on_error_path(caplog):
    # S-2 ブロック等で post_process に到達しなくても finalize は必ず監査する（F-01）
    with caplog.at_level(logging.INFO, logger="agentcore.audit"):
        _emit({"status": AgentStatus.ERROR.value, "session_id": "s1", "error_code": "S2_INJECTION_DETECTED"})
    audit = [r for r in caplog.records if r.name == "agentcore.audit"]
    assert len(audit) == 1
    record = json.loads(audit[0].getMessage())
    assert record["payload"]["status"] == "error"
    assert record["payload"]["error_code"] == "S2_INJECTION_DETECTED"
