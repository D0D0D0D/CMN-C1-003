"""ChatPreProcessNode — CMN-C1-003 pre_process スロット（entry node）.

released AgentCore SDK (v1.0.0) のノード契約:
- `FunctionNode` を継承し `execute(self, state) -> dict` のみ実装（__call__ は state 1 引数のみ渡す）
- **変更したキーだけ**を返す（full state を返さない。node_history は reducer=operator.add の
  ため full state コピー返却は履歴を二重化する）
- node_history は返さない（BaseNode.__call__ が 1 件追記する）
- S-1 は `required_trust_level` を `BaseNode.__call__` が自動強制（pre_process=entry 相当）

責務（docs/02_design.md §3, §6）— 旧 SessionInitNode + HistoryManagerNode の 2 段を
固定 3 スロットの pre_process に統合する:
1. 入力検証（空文字拒否 / session_id 必須）
2. S-2 入力ゲート（credential pattern scan）
3. session_id を key に ConversationRepository.load() で履歴ロード（失敗は非致命=新規会話扱い）
4. system_prompt_template（__init__ 注入）を解決
5. 履歴に今回 user turn を append → 直近 N ターン上限 + token 予算で切り詰め → windowed_messages

旧 graph レベルの S-2 `_security_gate_input` は released wheel に無いため本ノード内へインライン化
した（移行: SDK v1.0.0）。
"""

from __future__ import annotations

import json

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from shared.utils.audit_logger import emit_trace_event

from src.security import detect_credentials

_TEMPLATE_ID = "CMN-C1-003"


def _est_tokens(text: str) -> int:
    """トークン概算（文字数ベースの軽量推定。厳密 tokenizer は不要 — §6）。"""
    return (len(str(text)) + 3) // 4


class ChatPreProcessNode(FunctionNode):
    # S-1: pre_process(entry 相当) の trust gate を BaseNode.__call__ が強制（ADR-006）。
    #   caller_trust_level 不足 → execute() に到達せず status=error で停止。
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        repository,
        system_prompt_template: str = "",
        history_window_turns: int = 10,
        history_token_budget: int = 3000,
    ) -> None:
        self._repo = repository
        self._system_prompt_template = system_prompt_template or ""
        self._max_turns = int(history_window_turns)
        self._token_budget = int(history_token_budget)

    @staticmethod
    def _fail(code: str, message: str) -> dict:
        return {
            "error": message,
            "error_code": code,
            "error_message": message,
            "status": AgentStatus.ERROR.value,
        }

    def execute(self, state: dict) -> dict:
        # S-4: at least one domain event per node — framework の監査契約
        emit_trace_event(
            "chat_pre_process",
            {
                "template_id": _TEMPLATE_ID,
                "input_chars": len(str(state.get("user_input") or "")),
                "has_session_id": bool(state.get("session_id")),
            },
            state,
        )

        # ── 入力検証 ──
        user_input = state.get("user_input") or ""
        if not str(user_input).strip():
            return self._fail("EMPTY_INPUT", "user_input is empty or missing")

        # ── S-2: 入力ゲート（credential scan）──
        if detect_credentials(str(user_input)):
            return self._fail("S2_INJECTION_DETECTED", "credential pattern detected in user_input")

        # ── session_id 必須（マルチセッション分離）──
        session_id = state.get("session_id")
        if not session_id:
            return self._fail("MISSING_SESSION_ID", "session_id is required (history isolation)")

        # ── 履歴ロード（失敗は非致命: 新規会話として続行）──
        try:
            history = self._repo.load(session_id)
        except Exception:
            history = []   # HISTORY_LOAD_FAILED は warning 相当・会話は継続

        validated_input = str(user_input).strip()

        # ── 履歴ウィンドウイング（今回 user turn を append → 上限/予算で切り詰め）──
        messages = list(history) + [{"role": "user", "content": validated_input}]
        if self._max_turns > 0 and len(messages) > self._max_turns:
            messages = messages[-self._max_turns:]

        def total_tokens(msgs):
            return sum(_est_tokens(m.get("content", "")) for m in msgs)

        while len(messages) > 1 and total_tokens(messages) > self._token_budget:
            messages.pop(0)

        return {
            "validated_input": validated_input,
            "system_prompt": self._system_prompt_template,
            "conversation_history": json.dumps(history, ensure_ascii=False),
            "windowed_messages": json.dumps(messages, ensure_ascii=False),
            "history_turn_count": len(messages),
            "status": AgentStatus.SUCCESS.value,
        }
