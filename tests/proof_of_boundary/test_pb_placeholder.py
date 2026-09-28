"""CMN-C1-003 — Proof-of-Boundary Tests (released AgentCore SDK v1.0.0).

docs/02_design.md §12 / §8。フレームワーク境界・セキュリティ境界の検証。
released wheel の固定 5 ノード backbone（initialize→pre_process→main→route→post_process→
finalize）と node 契約に整合させて移行（旧 _invoke_impl / graph レベル gate フックは廃止）。
"""

import ast
import inspect
import json
import logging
import os

import ormsgpack
import pytest

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.agent import ChatAgent
from src.nodes.pre_process import ChatPreProcessNode
from src.nodes.llm_inference import LLMInferenceNode
from src.nodes.response import ResponseNode

CONFIG = {
    "system_prompt_template": "SYS",
    "history_window_turns": 10,
    "history_token_budget": 3000,
    "model": "claude-sonnet-4-6",
    "temperature": 0.7,
    "max_tokens": 256,
}
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

_VALID_TRUST = {TrustLevel.VERIFIED_EXTERNAL, TrustLevel.INTERNAL}


class _Fake:
    def generate(self, prompt):
        return "assistant reply"


def _agent():
    return ChatAgent(config={**CONFIG, "llm": _Fake()})


def _run(agent, user_input="hi", session_id="s", level=TrustLevel.VERIFIED_EXTERNAL):
    ctx = InvocationContext(caller_trust_level=level, session_id=session_id)
    return agent.invoke(user_input=user_input, session_id=session_id, ctx=ctx)


# ── PB-1: BaseNode → AuditLogger（S-4 trace 必発火）──
#   1.0.0 wheel では backbone が node_start/node_complete を agentcore.audit ロガーへ
#   自動 emit する。PB-1 の主旨（TC-05）はドメインイベントが silent failure せず発火すること。
#   そのため「総 audit 件数」は固定せず（backbone 分が混ざる）、ドメインイベント chat_complete を
#   抽出して ちょうど 1 件であることを検証する（重複・欠落いずれも FAIL）。
def test_pb1_trace_event_fires(caplog):
    with caplog.at_level(logging.INFO, logger="agentcore.audit"):
        _run(_agent())
    audit_records = [r for r in caplog.records if r.name == "agentcore.audit"]
    assert audit_records, "no agentcore.audit trace emitted (silent failure)"
    domain_events = [r for r in audit_records if '"event_type": "chat_complete"' in r.getMessage()]
    assert len(domain_events) == 1, f"expected exactly 1 chat_complete domain event, got {len(domain_events)}"


# ── PB-2: State serialization（出力は msgpack 安全 = checkpoint 安全）──
def test_pb2_output_is_msgpack_safe():
    out = _run(_agent())
    ormsgpack.packb(out)
    for key, value in out.items():
        assert isinstance(
            value, (str, int, float, bool, type(None), list, dict)
        ), f"out['{key}'] is {type(value).__name__} — not msgpack-safe"


# ── PB-3: LLMInferenceNode → 外部 LLM service（実呼び出し境界）──
def test_pb3_llm_service_boundary_returns_text():
    node = LLMInferenceNode(llm_client=_Fake(), **{k: CONFIG[k] for k in ("model", "temperature", "max_tokens")})
    out = node.execute({"system_prompt": "SYS", "windowed_messages": json.dumps([{"role": "user", "content": "x"}])})
    assert isinstance(out["llm_response"], str) and out["llm_response"] != ""


# ── PB-4: Import isolation（src/ が Level 0 agenticstar を import しない）──
def test_pb4_import_isolation_no_agenticstar():
    violations = []
    for root, _, files in os.walk(_SRC):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            tree = ast.parse(open(path).read(), filename=path)
            for n in ast.walk(tree):
                mod = ""
                if isinstance(n, ast.Import):
                    mod = n.names[0].name
                elif isinstance(n, ast.ImportFrom):
                    mod = n.module or ""
                if mod.startswith("agenticstar"):
                    violations.append(f"{path}: {mod}")
    assert violations == [], "Level 0 import(s) found: " + "; ".join(violations)


# ── PB-5: Checkpoint safety（出力に invocation_context / credential なし）──
def test_pb5_checkpoint_safety():
    out = _run(_agent())
    assert "invocation_context" not in out
    blob = json.dumps(out, ensure_ascii=False, default=str)
    assert "sk-" not in blob


# ── PB-6: backbone 実行順序（initialize → pre → main → post → finalize）──
def test_pb6_node_execution_order():
    out = _run(_agent())
    assert out["node_history"] == [
        "InitializeNode",
        "ChatPreProcessNode",
        "LLMInferenceNode",
        "ResponseNode",
        "ChatFinalizeNode",
    ]


# ── PB-7（本テンプレ固有）: マルチセッション履歴漏洩なし ──
def test_pb7_multi_session_no_leak():
    agent = _agent()
    _run(agent, user_input="SECRET-A", session_id="A")
    out_b = _run(agent, user_input="hi-b", session_id="B")
    assert "SECRET-A" not in out_b["windowed_messages"]


# ── 契約: ノードは execute() のみ。_invoke_impl を実装しない（ノード契約）──
@pytest.mark.parametrize("cls", [ChatPreProcessNode, LLMInferenceNode, ResponseNode])
def test_node_contract_execute_only(cls):
    assert "execute" in cls.__dict__, f"{cls.__name__} must implement execute()"
    assert (
        "_invoke_impl" not in cls.__dict__
    ), f"{cls.__name__} must NOT implement _invoke_impl() (node contract)"


# ── S-1 多層防御: 全ノードが required_trust_level を VERIFIED_EXTERNAL 以上で宣言（ADR-006）──
#   TrustLevel は文字列 enum（int 比較不可）のため有効値集合で判定する。
@pytest.mark.parametrize("cls", [ChatPreProcessNode, LLMInferenceNode, ResponseNode])
def test_all_nodes_declare_required_trust_level(cls):
    assert (
        "required_trust_level" in cls.__dict__
    ), f"{cls.__name__} must explicitly declare required_trust_level (no implicit ANONYMOUS)"
    assert cls.required_trust_level in _VALID_TRUST, (
        f"{cls.__name__} required_trust_level must be VERIFIED_EXTERNAL or INTERNAL, " f"got {cls.required_trust_level}"
    )


# ── execute() シグネチャ契約（released AgentCore SDK v1.0.0）: execute(self, state) ──
#   1.0.0 wheel の BaseNode.__call__ は execute(state) を state 1 引数のみで呼ぶ。
#   config を受ける旧 rc1 シグネチャは未使用の余計な引数になるため取らないのが正。
@pytest.mark.parametrize("cls", [ChatPreProcessNode, LLMInferenceNode, ResponseNode])
def test_execute_method_signature(cls):
    sig = inspect.signature(cls.execute)
    params = list(sig.parameters)
    assert params == [
        "self",
        "state",
    ], f"{cls.__name__}.execute must be execute(self, state) (no config — 1.0.0 contract), got {params}"
