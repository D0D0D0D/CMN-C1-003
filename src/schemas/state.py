"""State schema for CMN-C1-003 ChatAgent.

Released AgentCore SDK (v1.0.0) 準拠: `framework.schemas.agent_state.AgentState` を
直接継承する flat TypedDict。共有フィールド（user_input / input_context / status /
caller_trust_level / node_history / error_log / formatted_output / execution_time /
session_id / trace_id / correlation_id / llm_response 等）は AgentState から継承する。

ここにはドメイン固有フィールドのみ追加する。primitives + JSON 文字列のみ（msgpack 安全
= LangGraph checkpoint 安全。docs/02_design.md §4）。
conversation_history / windowed_messages は JSON STRING として保持し、list/dict を直接
state に入れない（PB-2 安全）。

NOTE: InvocationContext は State に保存しない（checkpoint に載せられない = PB-5）。trust は invoke() が
`caller_trust_level`（AgentState 継承フィールド・TrustLevel 文字列値）として書き込み、
各ノードの `required_trust_level` を `BaseNode.__call__` が S-1 自動強制する（ADR-006）。
"""

from __future__ import annotations

from typing import Optional

from framework.schemas.agent_state import AgentState


class ChatState(AgentState):
    """ChatAgent のドメイン State（AgentState 拡張）。"""

    # ── PreProcess（session_init + history_manager 統合スロット）出力 ──
    validated_input: Optional[str]        # 検証・正規化済み入力
    system_prompt: Optional[str]          # config.system_prompt_template を解決した system prompt
    conversation_history: Optional[str]   # JSON 配列文字列 [{"role","content"}, ...]（ロード済み full 履歴）
    windowed_messages: Optional[str]      # JSON 配列文字列 — LLM 入力 messages（直近 N ターン + 今回）
    history_turn_count: Optional[int]     # ウィンドウ後のターン数

    # ── Main（llm_inference スロット）出力 ──
    #   llm_response は AgentState 継承フィールドのため再宣言しない

    # ── PostProcess（response スロット）出力 ──
    #   formatted_output は AgentState 継承フィールド
    history_persisted: Optional[bool]     # 履歴永続化の成否

    # ── エラー伝播（status とは別にドメイン診断コードを保持）──
    error: Optional[str]                  # 真値が立つと後続ノードはパススルー
    error_code: Optional[str]             # EMPTY_INPUT / MISSING_SESSION_ID / S2_INJECTION_DETECTED /
                                          # LLM_CALL_FAILED / S3_BLOCKED
    error_message: Optional[str]          # 人間可読の詳細
