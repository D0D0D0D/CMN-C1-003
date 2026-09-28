"""Unit tests — ResponseNode (S-3 出力ゲート + 履歴永続化).

released SDK v1.0.0: ノードは status を含む部分 dict を返す。S-4 監査は finalize
スロット（ChatFinalizeNode）が担うため本ノードでは emit しない（test_finalize.py で検証）。
"""

import json

from framework.schemas.agent_status import AgentStatus

from src.nodes.response import ResponseNode
from src.services.conversation_repository import ConversationRepository


def test_s3_blocks_credential_in_output():
    repo = ConversationRepository()
    node = ResponseNode(repo)
    leak = "here: sk-" + "b" * 25
    out = node.execute({"session_id": "s1", "validated_input": "q", "conversation_history": "[]", "llm_response": leak})
    assert out["error_code"] == "S3_BLOCKED"
    assert out["status"] == AgentStatus.ERROR.value
    assert out.get("formatted_output") is None
    assert out["history_persisted"] is False
    assert repo.load("s1") == []  # ブロック時は永続化しない


def test_s3_masks_pii_in_output():
    repo = ConversationRepository()
    node = ResponseNode(repo, mask_patterns=["mynumber"])
    out = node.execute(
        {
            "session_id": "s1",
            "validated_input": "q",
            "conversation_history": "[]",
            "llm_response": "あなたの番号は 1234-5678-9012 です",
        }
    )
    assert out.get("error") is None
    assert out["status"] == AgentStatus.SUCCESS.value
    assert "1234-5678-9012" not in out["formatted_output"]
    assert "[REDACTED:mynumber]" in out["formatted_output"]
    # 永続化もマスク後
    saved = repo.load("s1")
    assert "1234-5678-9012" not in json.dumps(saved, ensure_ascii=False)


def test_happy_path_persists_full_history():
    repo = ConversationRepository()
    node = ResponseNode(repo)
    out = node.execute(
        {
            "session_id": "s1",
            "validated_input": "hello",
            "conversation_history": json.dumps([{"role": "user", "content": "earlier"}]),
            "llm_response": "hi there",
        }
    )
    assert out.get("error") is None
    assert out["formatted_output"] == "hi there"
    assert out["history_persisted"] is True
    saved = repo.load("s1")
    assert saved == [
        {"role": "user", "content": "earlier"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_upstream_error_is_passthrough():
    repo = ConversationRepository()
    node = ResponseNode(repo)
    out = node.execute({"error": "boom", "status": AgentStatus.ERROR.value})
    assert out["status"] == AgentStatus.ERROR.value
    assert "formatted_output" not in out
