"""ResponseNode — CMN-C1-003 post_process スロット（整形 + S-3 + 永続化 + S-4）.

released AgentCore SDK (v1.0.0) 契約: FunctionNode を継承し execute(self, state) で
**部分 dict** を返す（__call__ は state 1 引数のみ渡す）。post_process は route() が status==SUCCESS のときのみ到達する。

責務（docs/02_design.md §3, §8）— 旧 ResponseNode + graph レベル S-3 `_security_gate_output`
を固定 post_process スロットへ統合する:
- S-3 出力ゲート: llm_response を PII マスク → credential 残存チェック → 残れば S3_BLOCKED で
  応答ブロック（永続化も返却もしない）
- 通過時: formatted_output を確定し、full 履歴（loaded + user + assistant）を
  ConversationRepository.save() で永続化

S-4 は二層構成（framework の監査契約）: 本ノードも execute() 冒頭で per-node
ドメインイベントを 1 件発火し、加えて invoke 単位の集約イベント chat_complete は finalize
スロット（ChatFinalizeNode）が担う。finalize は backbone の**必ず最後に実行される**ノードの
ため、成功・失敗いずれの経路でも 1 件記録できる（post_process は status==success のときしか
到達しないため、エラー経路の監査漏れを避ける目的）。

旧 graph レベルの _security_gate_output は released wheel に存在しないため本ノード内へインライン
化した（移行: SDK v1.0.0）。
"""

from __future__ import annotations

import json

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from shared.utils.audit_logger import emit_trace_event

from src.masking import apply_masks
from src.security import detect_credentials

_TEMPLATE_ID = "CMN-C1-003"


class ResponseNode(FunctionNode):
    # S-1 多層防御: post ノードも明示宣言（ADR-006）
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, repository, mask_patterns=None) -> None:
        self._repo = repository
        self._mask_patterns = list(mask_patterns or [])

    def execute(self, state: dict) -> dict:
        # S-4: at least one domain event per node — framework の監査契約
        emit_trace_event(
            "chat_response",
            {
                "template_id": _TEMPLATE_ID,
                "llm_response_chars": len(str(state.get("llm_response") or "")),
                "upstream_error": bool(state.get("error")),
            },
            state,
        )

        if state.get("error") or state.get("status") == AgentStatus.ERROR.value:
            return {"error": state.get("error"), "status": AgentStatus.ERROR.value}

        llm_response = state.get("llm_response") or ""

        # ── S-3: 出力ゲート ──
        # (a) PII を mask_patterns で redact して通過（マスク可能なものはマスク）
        masked, _applied = apply_masks(str(llm_response), self._mask_patterns)
        # (b) credential（sk-/JWT 等）が残れば全停止（ブロック必須）
        if detect_credentials(masked):
            return {
                "error": "S-3 blocked: credential detected in output",
                "error_code": "S3_BLOCKED",
                "error_message": "credential pattern detected in llm_response",
                "formatted_output": None,
                "history_persisted": False,
                "status": AgentStatus.ERROR.value,
            }

        # ── 履歴永続化（full = loaded history + user turn + assistant turn）──
        session_id = state.get("session_id")
        try:
            full = json.loads(state.get("conversation_history") or "[]")
        except (ValueError, TypeError):
            full = []
        full = list(full) + [
            {"role": "user", "content": state.get("validated_input", "")},
            {"role": "assistant", "content": masked},   # PII マスク後を保存
        ]
        try:
            self._repo.save(session_id, full)
            history_persisted = True
        except Exception:
            history_persisted = False   # 永続化失敗は非致命（応答は返す）

        # S-4 集約イベント chat_complete は finalize スロット（ChatFinalizeNode）が
        # 成功・失敗いずれの経路でも 1 件記録する（per-node イベントは本 execute 冒頭で発火済み）。
        return {
            "formatted_output": masked,
            "history_persisted": history_persisted,
            "status": AgentStatus.SUCCESS.value,
        }
