"""Unit tests — ChatPreProcessNode (S-1 / S-2 / 入力検証 / 履歴ロード + ウィンドウイング).

released AgentCore SDK v1.0.0: S-1 は BaseNode.__call__ が state["caller_trust_level"]
（TrustLevel 文字列値）で自動強制する。ノードは execute() で部分 dict を返す。
"""

import json

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.pre_process import ChatPreProcessNode
from src.services.conversation_repository import ConversationRepository


def _node(repo=None, system_prompt_template="SYS", **kw):
    return ChatPreProcessNode(repo or ConversationRepository(), system_prompt_template=system_prompt_template, **kw)


# ── S-1（__call__ が自動強制）──────────────────────────────────────────────
def test_s1_absent_trust_is_denied():
    # caller_trust_level 不在 = ANONYMOUS → __call__ が S-1 で拒否（execute に到達しない）
    out = _node()({"user_input": "hi", "session_id": "s1"})
    assert out["status"] == AgentStatus.ERROR.value
    assert any("trust gate denied" in e.lower() for e in out["error_log"])


def test_s1_anonymous_is_denied():
    out = _node()({"user_input": "hi", "session_id": "s1", "caller_trust_level": TrustLevel.ANONYMOUS.value})
    assert out["status"] == AgentStatus.ERROR.value
    assert any("trust gate denied" in e.lower() for e in out["error_log"])


def test_s1_verified_external_passes_gate():
    out = _node()({"user_input": "hi", "session_id": "s1", "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value})
    assert out.get("error") is None
    assert out["validated_input"] == "hi"


# ── 入力検証 / S-2（execute 直接呼び出し）────────────────────────────────────
def test_empty_input_rejected():
    out = _node().execute({"user_input": "   ", "session_id": "s1"})
    assert out["error_code"] == "EMPTY_INPUT"
    assert out["status"] == AgentStatus.ERROR.value


def test_s2_credential_in_input_rejected():
    leak = "my key is sk-" + "a" * 25
    out = _node().execute({"user_input": leak, "session_id": "s1"})
    assert out["error_code"] == "S2_INJECTION_DETECTED"


def test_missing_session_id_rejected():
    out = _node().execute({"user_input": "hi", "session_id": ""})
    assert out["error_code"] == "MISSING_SESSION_ID"


def test_happy_path_loads_and_resolves():
    repo = ConversationRepository()
    repo.save("s1", [{"role": "user", "content": "earlier"}])
    out = _node(repo).execute({"user_input": "  hello  ", "session_id": "s1"})
    assert out.get("error") is None
    assert out["validated_input"] == "hello"
    assert out["system_prompt"] == "SYS"
    assert json.loads(out["conversation_history"]) == [{"role": "user", "content": "earlier"}]
    assert out["status"] == AgentStatus.SUCCESS.value


# ── 履歴ウィンドウイング（旧 HistoryManagerNode 相当）────────────────────────
def test_appends_current_user_turn():
    out = _node(history_window_turns=10, history_token_budget=10000).execute(
        {"user_input": "hello", "session_id": "s1"}
    )
    msgs = json.loads(out["windowed_messages"])
    assert msgs[-1] == {"role": "user", "content": "hello"}
    assert out["history_turn_count"] == 1


def test_turn_cap_keeps_most_recent():
    repo = ConversationRepository()
    repo.save("s1", [{"role": "user", "content": f"m{i}"} for i in range(5)])
    out = _node(repo, history_window_turns=2, history_token_budget=10000).execute(
        {"user_input": "latest", "session_id": "s1"}
    )
    msgs = json.loads(out["windowed_messages"])
    assert len(msgs) == 2
    assert msgs[-1] == {"role": "user", "content": "latest"}


def test_token_budget_trims_oldest():
    # 各 content ~ 40 文字 ≈ 10 tokens。budget 15 なら直近 user turn のみ残る
    repo = ConversationRepository()
    repo.save("s1", [{"role": "user", "content": "x" * 40} for _ in range(5)])
    out = _node(repo, history_window_turns=10, history_token_budget=15).execute(
        {"user_input": "y" * 40, "session_id": "s1"}
    )
    msgs = json.loads(out["windowed_messages"])
    assert len(msgs) == 1
    assert msgs[-1]["content"] == "y" * 40
