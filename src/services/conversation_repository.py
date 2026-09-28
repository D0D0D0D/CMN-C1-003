"""ConversationRepository — 会話履歴の永続化（MVP は in-memory 実装）。

docs/02_design.md 付録 A で凍結した IF。SessionInitNode / ResponseNode は本クラスの
load() / save() シグネチャのみに依存し、内部実装には触れない。本番 platform IF への
差し替えはこのクラス内部で完結させる（ノード改修不要）。

分離キーは必ず session_id（docs/02_design.md §7 マルチセッション分離）。
返り値・引数は primitives + JSON-serialisable のみ（State 安全則 §4）。
"""

from __future__ import annotations

import copy


class ConversationRepository:
    """会話履歴の永続化。MVP はプロセス内 in-memory dict（再起動で消える前提）。

    保存履歴の無制限増加を防ぐため、rolling window cap（直近 max_messages 件のみ保存）を
    適用する（業界標準の buffer window 方式。docs/02_design.md §6 / 運用ガイド §7）。
    None は無制限。TTL/eviction・要約は本番永続 IF / Sprint 2 で扱う。
    """

    def __init__(self, max_messages: int | None = 50) -> None:
        # {session_id: [{"role": str, "content": str}, ...]}
        self._store: dict[str, list[dict]] = {}
        self._max_messages = max_messages

    def load(self, session_id: str) -> list[dict]:
        """session_id に紐づく会話履歴を返す。

        未知 session_id は [] を返す（新規会話扱い、例外を投げない）。
        他セッションのデータは絶対に返さない（マルチセッション分離）。
        返り値はコピー（呼び出し側の変更が内部状態に波及しない）。
        """
        if not session_id:
            return []
        return copy.deepcopy(self._store.get(session_id, []))

    def save(self, session_id: str, messages: list[dict]) -> None:
        """session_id に会話履歴を保存（MVP は上書き）。

        messages は load() と同形 [{"role","content"}, ...]。
        PII を含みうるため本文はログに出さない（呼び出し側の責務）。
        """
        if not session_id:
            return
        msgs = copy.deepcopy(list(messages))
        # rolling window cap: 直近 max_messages 件のみ保存（無制限増加を防ぐ）
        if self._max_messages is not None and len(msgs) > self._max_messages:
            msgs = msgs[-self._max_messages:]
        self._store[session_id] = msgs
