"""LLMInferenceNode — CMN-C1-003 main スロット（単一 LLM 推論）.

released AgentCore SDK (v1.0.0) のノード契約: FunctionNode を継承し
execute(self, state) で **部分 dict** を返す（__call__ が state 1 引数のみで呼ぶ）。pre_process→main は無条件辺の
ため、上流（pre_process）が status=error / state["error"] を立てた場合はパススルーする。

責務（docs/02_design.md §3, §9）:
- system_prompt + windowed_messages で単一 LLM 呼び出し（検索なし・ツールなし）
- 失敗時は max_retry 回まで指数バックオフでリトライ。尽きたら error_code=LLM_CALL_FAILED
- llm_response を生成。model / temperature / max_tokens は construction で注入

LLM クライアントは依存性注入（llm_client）。released wheel に旧 framework.llm.LLMClient は
無いため、src/services/llm_client.py の注入シームを使う（移行: SDK v1.0.0）。

セキュリティ緩和（#6）: ユーザー文中の role マーカー（System:/User:/Assistant:）を無害化し、
プレーンテキスト連結でのロールなりすまし（プロンプトインジェクション）を低減する。
"""

from __future__ import annotations

import json
import re
import time
from typing import ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from shared.utils.audit_logger import emit_trace_event

from src.services.llm_client import LLMClient

_TEMPLATE_ID = "CMN-C1-003"

# 行頭の role マーカーを無害化（なりすまし対策）
_ROLE_MARKER = re.compile(r"(?im)^\s*(system|user|assistant)\s*:")


def _sanitize(content: str) -> str:
    return _ROLE_MARKER.sub(lambda m: m.group(0).replace(":", "："), str(content))


class EmptyLLMResponseError(RuntimeError):
    """LLM が空応答を返した（一時的失敗として扱いリトライ対象にする）。"""


class LLMInferenceNode(FunctionNode):
    # S-1 多層防御: main ノードも明示宣言（ADR-006）
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        llm_client: object | None = None,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retry: int = 3,
        retry_backoff_base: float = 2.0,
    ) -> None:
        self._llm = llm_client
        self._model = model
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._max_retry = int(max_retry)
        self._backoff_base = float(retry_backoff_base)

    def _client(self):
        return self._llm or LLMClient(
            model=self._model, max_tokens=self._max_tokens, temperature=self._temperature
        )

    def _build_prompt(self, system_prompt: str, messages: list[dict]) -> str:
        lines = []
        if system_prompt:
            lines.append(f"System:\n{system_prompt}")
        for m in messages:
            role = m.get("role", "user")
            content = _sanitize(m.get("content", ""))   # role なりすまし無害化
            lines.append(f"{role.capitalize()}: {content}")
        lines.append("Assistant:")
        return "\n".join(lines)

    #: 再試行しても結果が変わらない例外。設定・実装・シグネチャの誤りが該当し、
    #: 指数バックオフで待つと誤設定のデプロイに毎リクエスト数秒の空転が乗るだけになる。
    NON_RETRYABLE: ClassVar[tuple[type[BaseException], ...]] = (
        NotImplementedError,   # 未設定クライアント（番兵）
        TypeError,             # シグネチャ不一致（誤った注入）
        AttributeError,        # 期待するメソッドが無い
    )

    def _call(self, llm: object, system_prompt: str, messages: list, prompt: str) -> tuple[str, str]:
        """`(text, model)` を返す。

        `chat()` を持つクライアントは role 境界を保ったまま呼び、**応答が名乗る
        モデル名**を受け取る。持たないものは従来の flat prompt 経路。
        """
        chat = getattr(llm, "chat", None)
        if callable(chat):
            return chat(system_prompt, messages)
        return llm.generate(prompt), self._model

    def execute(self, state: dict) -> dict:
        # S-4: at least one domain event per node — framework の監査契約
        emit_trace_event(
            "llm_inference",
            {
                "template_id": _TEMPLATE_ID,
                "configured_model": self._model,   # 実際に答えたモデルは llm_response_received で報告
                "history_turn_count": state.get("history_turn_count"),
                "upstream_error": bool(state.get("error")),
            },
            state,
        )

        # 上流のエラー/終了ステータスはパススルー（pre_process→main は無条件辺）
        if state.get("error") or state.get("status") == AgentStatus.ERROR.value:
            return {"error": state.get("error"), "status": AgentStatus.ERROR.value}

        try:
            messages = json.loads(state.get("windowed_messages") or "[]")
        except (ValueError, TypeError):
            messages = []

        prompt = self._build_prompt(state.get("system_prompt", ""), messages)
        llm = self._client()

        system_prompt = state.get("system_prompt", "")
        last_exc: BaseException | None = None
        attempts = 0
        for attempt in range(self._max_retry + 1):   # 初回 + max_retry 回
            attempts = attempt + 1
            try:
                text, model = self._call(llm, system_prompt, messages, prompt)
                if not text.strip():
                    # 空応答を success にすると、空の assistant 発話が履歴に永続化され、
                    # 利用者には「成功したが何も返らない」応答が出る。一時的な失敗として扱う。
                    raise EmptyLLMResponseError("LLM returned an empty response")
                emit_trace_event(
                    "llm_response_received",
                    {"template_id": _TEMPLATE_ID, "model": model, "attempts": attempts},
                    state,
                )
                return {"llm_response": text, "status": AgentStatus.SUCCESS.value}
            except self.NON_RETRYABLE as exc:
                last_exc = exc
                break
            except Exception as exc:          # 一時的とみなしリトライ
                last_exc = exc
                if attempt < self._max_retry:
                    # 指数バックオフ: base^attempt 秒。
                    # 🔴 `base ** 0` は base に関わらず 1.0 なので、base=0 を指定しても
                    # 初回リトライで 1 秒眠っていた（旧コメントの「base<=1 なら実質
                    # スリープなし」は誤り）。base<=0 は明示的に無効化する。
                    wait = 0.0 if self._backoff_base <= 0 else self._backoff_base ** attempt
                    if wait > 0:
                        time.sleep(wait)

        # リトライ尽き → 安全に LLM_CALL_FAILED を伝播
        return {
            "error": f"LLM call failed after {attempts} attempt(s)",
            "error_code": "LLM_CALL_FAILED",
            "error_message": f"LLM call failed after {self._max_retry} retries: {last_exc}",
            "status": AgentStatus.ERROR.value,
        }
