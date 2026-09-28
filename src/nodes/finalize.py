"""ChatFinalizeNode — CMN-C1-003 finalize スロット（S-4 監査トレース）.

released AgentCore SDK の finalize スロットは backbone の **必ず最後に実行される**ノード
（route() が success→post_process / error→finalize のいずれでも finalize に到達する）。
そのため invoke 単位の **集約イベント chat_complete** は finalize で発火させ、**成功・失敗
いずれの経路でも 1 件記録**する（旧 graph レベル `_emit_trace_event` が invoke ごとに必ず
発火していた挙動の再現）。S-4 は二層構成（framework の監査契約）: 各ノード
（pre_process / main / post_process）も execute() 冒頭で per-node イベントを 1 件発火する。

`FinalizeNode` を継承し `on_finalize()` のみ override する（`_teardown` の response_metadata 集計
はそのまま継承）。発話 / 応答本文は記録しない（メタのみ）。
"""

from __future__ import annotations

from framework.nodes.defaults.finalize_node import FinalizeNode

from shared.utils.audit_logger import emit_trace_event

_TEMPLATE_ID = "CMN-C1-003"


class ChatFinalizeNode(FinalizeNode):
    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self._model = model

    def on_finalize(self, state: dict) -> dict:
        # S-4: 成功・失敗いずれの経路でも 1 件だけ監査記録（メタのみ）。
        emit_trace_event(
            "chat_complete",
            {
                "template_id": _TEMPLATE_ID,
                "status": state.get("status"),
                "error_code": state.get("error_code"),
                "history_turn_count": state.get("history_turn_count"),
                "model": self._model,
                "history_persisted": state.get("history_persisted"),
            },
            state,
        )
        return {}
