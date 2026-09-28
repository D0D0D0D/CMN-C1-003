"""ChatGraph — CMN-C1-003 (released AgentCore SDK v1.0.0).

released L1 `AgentBaseGraph` は **固定 5 ノード backbone** を強制する:
    START → initialize → pre_process → main → {route} → post_process → finalize → END
                                            ↓ (RETRY, max 3)
                                         pre_process
`super().register_nodes()` が initialize / finalize を自動注入し、テンプレは
pre_process / main / post_process の 3 スロットを埋める（compile() が None を検出すると
MissingNodeError）。route() は state["status"] を読み、SUCCESS のときのみ post_process に
到達する。`add_edges()` / `route()` は override しない（L1 が配線する）。

旧 4 ノード（session_init → history_manager → llm_inference → response）を 3 スロットへ
マッピングする（docs/02_design.md §2, §3）:
    pre_process  : ChatPreProcessNode  (session_init + history_manager 統合 / S-1 entry / S-2 入力ゲート)
    main         : LLMInferenceNode      (単一 LLM 推論 + リトライ)
    post_process : ResponseNode          (整形 + S-3 出力ゲート + 履歴永続化)
    finalize     : ChatFinalizeNode      (S-4 監査トレース — 成功・失敗いずれの経路でも 1 件記録)

ChatPreProcessNode と ResponseNode は **同一の ConversationRepository インスタンス**を共有
する（in-memory ストアを跨ノードで一貫させるため）。

旧 graph レベルの _security_gate_input/output / _emit_trace_event / _invoke_impl フックは
released wheel に無いため廃止し、S-2/S-3/S-4 を各ノード内へインライン実装した（移行: SDK
v1.0.0）。LLM クライアントは `config["llm"]` で注入する（同系テンプレート共通の注入シーム）。
"""

from __future__ import annotations

from framework.graph.agent_base_graph import AgentBaseGraph

from src.nodes.pre_process import ChatPreProcessNode
from src.nodes.llm_inference import LLMInferenceNode
from src.nodes.response import ResponseNode
from src.nodes.finalize import ChatFinalizeNode
from src.services.conversation_repository import ConversationRepository
from src.services.llm_client import resolve_llm_client
from src.schemas.state import ChatState


class ChatGraph(AgentBaseGraph):
    """固定 3 スロット: pre_process(pre_process) → main(llm_inference) → post_process(response)。"""

    def __init__(self, config: dict | None = None, repository=None) -> None:
        cfg = dict(config or {})
        super().__init__(config=cfg)  # BaseGraph.__init__ が self.config を保持（register_nodes は compile 時）
        # LLM は config["llm"] 経由でのみ受け取る（同系テンプレート共通）。
        # 🔴 旧実装は `llm_client=` という独自コンストラクタ引数を持っていたが、
        # framework の実契約どおり src/api/server.py は ChatAgent(config=...) としか
        # 呼ばないため永久に未設定で、既定ビルドが毎回 LLM_CALL_FAILED を返していた。
        self._llm_client = resolve_llm_client(cfg.get("llm"))
        self._system_prompt_template = cfg.get("system_prompt_template", "")
        self._history_window_turns = int(cfg.get("history_window_turns", 10))
        self._history_token_budget = int(cfg.get("history_token_budget", 3000))
        self._model = cfg.get("model", "claude-sonnet-4-6")
        self._temperature = float(cfg.get("temperature", 0.7))
        self._max_tokens = int(cfg.get("max_tokens", 1024))
        self._max_retry = int(cfg.get("max_retry", 3))
        self._retry_backoff_base = float(cfg.get("retry_backoff_base", 2.0))
        self._mask_patterns = list(cfg.get("mask_patterns", []))
        self._max_stored = cfg.get("history_max_stored_messages", 50)
        # pre_process と post_process で共有する単一リポジトリ（保存上限付き）
        self._repository = repository or ConversationRepository(max_messages=self._max_stored)

    @property
    def name(self) -> str:
        return "ChatAgent"

    @property
    def state_schema(self) -> type:
        return ChatState

    def register_nodes(self) -> None:
        super().register_nodes()  # initialize / finalize 自動注入 + pre/main/post スロット枠
        self._nodes["pre_process"] = ChatPreProcessNode(
            self._repository,
            system_prompt_template=self._system_prompt_template,
            history_window_turns=self._history_window_turns,
            history_token_budget=self._history_token_budget,
        )
        self._nodes["main"] = LLMInferenceNode(
            llm_client=self._llm_client,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            max_retry=self._max_retry,
            retry_backoff_base=self._retry_backoff_base,
        )
        self._nodes["post_process"] = ResponseNode(
            self._repository,
            mask_patterns=self._mask_patterns,
        )
        # finalize スロットを差し替えて S-4 監査を成功・失敗いずれの経路でも発火させる
        # （post_process は status==success のときしか到達しないため。レビュー指摘 F-01 対応）。
        self._nodes["finalize"] = ChatFinalizeNode(model=self._model)

    def get_output(self, state) -> dict:
        """最終 state からクライアント向け出力を構築（primitives + list/dict のみ = msgpack 安全）。"""
        return {
            # AGENTIC STAR runner は success 時に `output` を必須とする
            # （shared/bootstrap/marketplace_app.py: 無いと
            #  "returned status=success but output is missing" で RuntimeError）。
            # 既存キーは変えず追加のみ。
            "output": state.get("formatted_output"),
            "formatted_output": state.get("formatted_output"),
            "history_persisted": state.get("history_persisted"),
            "history_turn_count": state.get("history_turn_count"),
            "windowed_messages": state.get("windowed_messages"),
            "status": state.get("status"),
            "error": state.get("error"),
            "error_code": state.get("error_code"),
            "error_message": state.get("error_message"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "session_id": state.get("session_id"),
            "node_history": state.get("node_history", []),
            "error_log": state.get("error_log", []),
        }
