"""LLM クライアント抽象と `config["llm"]` 注入シーム。

ChatAgent は LLM を中核に使う（単一 LLM 会話）。**注入は `config["llm"]` 経由**で、
モデル ID も具体クライアントもここには書かない（同系テンプレート共通）。

🔴 旧実装は外側 Graph に `llm_client=` という独自コンストラクタ引数を持っていた。
framework の実契約が想定しない形であり、書かれているとおりの結果になっていた —
framework の実契約（`BaseGraph.__init__(self, config=None)`）に従う `src/api/server.py` は
`ChatAgent(config=_config)` としか呼ばないため引数は永久に未設定で、既定ビルドは
`LLMClient.generate()` の `NotImplementedError` を踏み、毎回 `LLM_CALL_FAILED` を返していた。

プロトコルの差:
- 注入されるのは `BaseLLM`（`complete(messages) -> dict`）
- 本 AG のノードが呼ぶのは `generate(prompt) -> str`

`BaseLLMAdapter` がこの差を吸収する。**src/ に置く**（テスト側にだけ置くと本番経路に
アダプタが存在せず、テストだけが緑になる — 実際に旧実装がその状態だった）。
"""

from __future__ import annotations

from typing import Any


class LLMClient:
    """未注入を表す番兵。`generate()` は必ず失敗する。

    ChatAgent には決定論フォールバックが無い（会話の生成が中核能力そのもの）ため、
    未注入は縮退でなく失敗として扱う。`LLMInferenceNode` がこれを `LLM_CALL_FAILED`
    として伝播する。
    """

    def __init__(
        self,
        model: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def generate(self, prompt: str) -> str:
        raise NotImplementedError(
            "No LLM is configured. Inject one via config={'llm': <BaseLLM>} "
            "(the platform / entry adapter supplies it at deploy time)."
        )


class BaseLLMAdapter:
    """`BaseLLM` を本 AG のクライアント契約へ合わせる。

    `chat()` が本線（system / user / assistant の role を保ったまま渡し、応答が名乗る
    モデル名も返す）。`generate()` は flat prompt しか受けない旧経路向けの後方互換。
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def chat(self, system_prompt: str, messages: list) -> tuple[str, str]:
        """`(text, model)` を返す。

        role 境界を保つ理由: system 指示と会話履歴を 1 つの `user` メッセージへ潰すと、
        system 指示の優先度が失われ、履歴に混入した文言を指示と誤読しやすくなる
        （プロンプトインジェクション耐性の低下）。
        """
        payload: list[dict] = []
        if system_prompt:
            payload.append({"role": "system", "content": system_prompt})
        for m in messages:
            if isinstance(m, dict) and m.get("role") and m.get("content") is not None:
                payload.append({"role": str(m["role"]), "content": str(m["content"])})
        if not payload:
            payload = [{"role": "user", "content": ""}]
        response = self._llm.complete(payload)
        return _content_of(response), _model_of(response)

    def generate(self, prompt: str) -> str:
        text, _ = self.chat("", [{"role": "user", "content": prompt}])
        return text


def _content_of(response: Any) -> str:
    content = response.get("content") if isinstance(response, dict) else response
    if content is None:
        # `{"content": None}` を str() にかけると文字列 "None" が回答として利用者に返る。
        return ""
    return content if isinstance(content, str) else str(content)


def _model_of(response: Any) -> str:
    """**応答が名乗るモデル**を返す。

    `BaseLLM` の契約に public な `model` は無く（抽象は complete/stream/bind_tools のみ）、
    実クライアントは `_model` を持つ。呼び出し時点でどのモデルが答えたかは canonical
    response の `model` にしか無いので、そこから採る。
    """
    if isinstance(response, dict):
        name = response.get("model")
        if isinstance(name, str) and name:
            return name
    return "injected"


def resolve_llm_client(injected: Any | None) -> Any:
    """`config["llm"]` の値をクライアントへ解決する。

    🔴 **`BaseLLM` の判定を duck typing より先に置く。** SDK の実クライアント
    （`AnthropicClient` 等）は `__getattr__` で内部 LangChain クライアントへ委譲するため、
    `getattr(x, "generate")` が **バッチ用の別シグネチャの `generate`** を拾ってしまう。
    そのまま使うと文字列 prompt を渡して落ちる。
    """
    if injected is None:
        return LLMClient()
    try:
        from shared.services.llm.base_llm import BaseLLM

        if isinstance(injected, BaseLLM):
            return BaseLLMAdapter(injected)
    except ImportError:  # pragma: no cover — SDK 不在環境
        pass
    # BaseLLM 以外は、chat/generate を**自前で定義している**ものだけ受ける。
    # `__getattr__` 委譲を拾わないよう、クラス階層に実体があるかで判定する。
    for name in ("chat", "generate"):
        if any(name in klass.__dict__ for klass in type(injected).__mro__):
            return injected
    if callable(getattr(injected, "complete", None)):
        return BaseLLMAdapter(injected)
    return LLMClient()
