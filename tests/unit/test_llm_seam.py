"""CMN-C1-003 — config["llm"] 注入シームの単体テスト。

🔴 旧実装は外側 Graph の `llm_client=` という独自コンストラクタ引数でのみ LLM を
受け取っていた。framework の実契約に従う `src/api/server.py` は
`ChatAgent(config=_config)` としか呼ばないため引数は永久に未設定で、**本番は毎回
`LLM_CALL_FAILED`** を返していた。テストが緑だったのは conftest が
`agent._llm_client` を private 属性で直接書き換えていたため。

ここで固定するのは「framework の実契約（config）だけで LLM に到達すること」。
"""

from __future__ import annotations

import pytest

from src.graph.graph import ChatGraph
from src.nodes.llm_inference import LLMInferenceNode
from src.services.llm_client import (
    BaseLLMAdapter,
    LLMClient,
    resolve_llm_client,
)


class FakeBaseLLM:
    """本番 BaseLLM と同じ形（`complete(messages) -> dict`）。"""

    def __init__(self, content: str = "こんにちは", model: str = "azure/gpt-x") -> None:
        self._content, self.model = content, model
        self.calls: list[list[dict]] = []

    def complete(self, messages: list) -> dict:
        self.calls.append(messages)
        return {"content": self._content, "model": self.model, "usage": {}}


class FakeGenerateClient:
    """`generate(prompt)` を直接持つクライアント（テスト Fake / 独自実装）。"""

    model = "custom"

    def generate(self, prompt: str) -> str:
        return "direct"


class TestResolveLLMClient:
    def test_base_llm_is_wrapped(self) -> None:
        client = resolve_llm_client(FakeBaseLLM())
        assert isinstance(client, BaseLLMAdapter)
        assert client.generate("hi") == "こんにちは"

    def test_generate_client_is_used_as_is(self) -> None:
        fake = FakeGenerateClient()
        assert resolve_llm_client(fake) is fake

    @pytest.mark.parametrize("bad", [None, object(), "not a client", 42])
    def test_unusable_falls_back_to_sentinel(self, bad: object) -> None:
        client = resolve_llm_client(bad)
        assert isinstance(client, LLMClient)
        with pytest.raises(NotImplementedError):
            client.generate("hi")

    def test_adapter_sends_prompt_as_user_message(self) -> None:
        llm = FakeBaseLLM()
        resolve_llm_client(llm).generate("質問です")
        assert llm.calls == [[{"role": "user", "content": "質問です"}]]

    @pytest.mark.parametrize(
        ("response", "expected"),
        [({"content": "x"}, "x"), ({"content": None}, ""), ({}, ""), ("bare", "bare"), (123, "123")],
    )
    def test_adapter_tolerates_response_shapes(self, response: object, expected: str) -> None:
        class R:
            def complete(self, m: list) -> object:
                return response

        assert BaseLLMAdapter(R()).generate("p") == expected


class TestGraphReadsConfigLlm:
    """🔴 framework の実契約（config だけ）で LLM に到達すること。"""

    def test_config_llm_reaches_the_node(self) -> None:
        llm = FakeBaseLLM("回答本文")
        graph = ChatGraph(config={"llm": llm})
        graph.register_nodes()
        node = graph._nodes["main"]
        assert isinstance(node, LLMInferenceNode)
        assert node._llm.generate("q") == "回答本文"
        assert llm.calls, "config['llm'] がノードまで届いていない"

    def test_no_llm_yields_sentinel_not_crash(self) -> None:
        graph = ChatGraph(config={})
        graph.register_nodes()
        with pytest.raises(NotImplementedError):
            graph._nodes["main"]._llm.generate("q")

    def test_bespoke_kwarg_is_gone(self) -> None:
        """framework の契約に無い独自コンストラクタ引数を復活させない。"""
        import inspect

        params = set(inspect.signature(ChatGraph.__init__).parameters)
        assert "llm_client" not in params, "禁止された独自コンストラクタ引数が復活している"
        assert "llm" not in params


class TestAuditReportsRealModel:
    """監査は**応答が名乗ったモデル**を報告する。

    `BaseLLM` の契約に public な `model` は無い（抽象は complete/stream/bind_tools のみ）。
    実クライアントは `_model` を持ち、呼び出し時点で誰が答えたかは canonical response の
    `model` にしか無い。config の `model` 文字列は注入クライアントの選択を決めない。
    """

    def test_model_comes_from_the_response(self) -> None:
        adapter = resolve_llm_client(FakeBaseLLM(model="azure/gpt-5.6"))
        text, model = adapter.chat("sys", [{"role": "user", "content": "q"}])
        assert (text, model) == ("こんにちは", "azure/gpt-5.6")

    def test_response_without_model_reports_injected(self) -> None:
        class Unnamed:
            def complete(self, m: list) -> dict:
                return {"content": "x"}

        assert resolve_llm_client(Unnamed()).chat("", [])[1] == "injected"

    def test_generate_only_client_falls_back_to_config_model(self) -> None:
        node = LLMInferenceNode(llm_client=FakeGenerateClient(), model="configured-model")
        _, model = node._call(FakeGenerateClient(), "sys", [], "prompt")
        assert model == "configured-model"


class TestAuditEventPayload:
    """`_audit_model()` を直接呼ぶだけでは emit 経路を検証できない。

    実際に発火した監査イベントの payload を見る（node module 自身の binding を patch
    する — 定義元だけ patch しても import 済みの binding には効かない）。
    """

    def test_emitted_event_carries_the_injected_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.nodes.llm_inference as mod

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(mod, "emit_trace_event", lambda n, p, s: events.append((n, p)))

        node = LLMInferenceNode(
            llm_client=resolve_llm_client(FakeBaseLLM(model="azure/gpt-5.6")),
            model="claude-sonnet-4-6",
        )
        node.execute({"system_prompt": "s", "windowed_messages": "[]"})

        received = [p for name, p in events if name == "llm_response_received"]
        assert received, "llm_response_received が発火していない"
        assert received[0]["model"] == "azure/gpt-5.6", "応答が名乗ったモデルでなく config の文字列を監査に載せている"
        # 呼び出し前イベントは「設定値」であることを名前で明示する
        pre = [p for name, p in events if name == "llm_inference"]
        assert pre and "model" not in pre[0], "呼び出し前に確定していない model を報告している"
        assert pre[0]["configured_model"] == "claude-sonnet-4-6"


class TestSDKClientIsNotMisdetected:
    """🔴 SDK 実クライアントは `__getattr__` で内部 LangChain クライアントへ委譲する。

    `getattr(x, "generate")` を先に見ると、バッチ用の別シグネチャの `generate` を拾って
    文字列 prompt を渡し落ちる。`BaseLLM` の判定を先に置くこと。
    """

    def test_getattr_delegating_base_llm_is_wrapped(self) -> None:
        from shared.services.llm.base_llm import BaseLLM

        class _Inner:
            def generate(self, prompts: list, **kw: object) -> object:  # LangChain 風
                raise AssertionError("バッチ用 generate が呼ばれた（誤判定）")

        class DelegatingClient(BaseLLM):
            """SDK の AnthropicClient / AzureOpenAIClient と同じ委譲形。"""

            def __init__(self) -> None:
                self._client = _Inner()

            def complete(self, messages: list) -> dict:
                return {"content": "ok", "model": "sdk/model"}

            def stream(self, messages: list):
                yield "ok"

            def bind_tools(self, tools: list) -> "DelegatingClient":
                return self

            def __getattr__(self, name: str) -> object:
                return getattr(self._client, name)

        client = DelegatingClient()
        assert callable(client.generate), "前提: 委譲で generate が見える"
        resolved = resolve_llm_client(client)
        assert isinstance(resolved, BaseLLMAdapter), "BaseLLM が adapter で包まれていない"
        assert resolved.chat("", [{"role": "user", "content": "q"}]) == ("ok", "sdk/model")

    def test_delegated_attribute_is_not_treated_as_own_method(self) -> None:
        """`__getattr__` 委譲だけの型は自前クライアントとして受け付けない。"""

        class OnlyDelegates:
            def __init__(self) -> None:
                self._inner = type("I", (), {"generate": lambda self, p: "x"})()

            def __getattr__(self, name: str) -> object:
                return getattr(self._inner, name)

        assert isinstance(resolve_llm_client(OnlyDelegates()), LLMClient)


class TestEmptyResponseIsNotSuccess:
    """空応答を success にしない（空の assistant 発話が履歴に永続化される）。"""

    @pytest.mark.parametrize("content", [None, "", "   ", "\n\t"])
    def test_empty_response_fails(self, content: object) -> None:
        class Empty:
            def complete(self, m: list) -> dict:
                return {"content": content, "model": "m"}

        node = LLMInferenceNode(llm_client=resolve_llm_client(Empty()), max_retry=1, retry_backoff_base=0.0)
        out = node.execute({"system_prompt": "s", "windowed_messages": "[]"})
        assert out.get("error_code") == "LLM_CALL_FAILED", "空応答が success になっている"
        assert "llm_response" not in out


class TestRetryPolicy:
    @pytest.mark.parametrize("exc", [NotImplementedError, TypeError, AttributeError])
    def test_permanent_errors_are_not_retried(self, exc: type[BaseException]) -> None:
        calls = {"n": 0}

        class Broken:
            def generate(self, prompt: str) -> str:
                calls["n"] += 1
                raise exc("permanent")

        node = LLMInferenceNode(llm_client=Broken(), max_retry=3, retry_backoff_base=0.0)
        out = node.execute({"system_prompt": "s", "windowed_messages": "[]"})
        assert calls["n"] == 1, f"{exc.__name__} を再試行している"
        assert "1 attempt(s)" in out["error"], "実試行回数と食い違うメッセージ"

    def test_transient_errors_are_retried(self) -> None:
        calls = {"n": 0}

        class Flaky:
            def generate(self, prompt: str) -> str:
                calls["n"] += 1
                if calls["n"] < 3:
                    raise ConnectionError("temporary")
                return "ok"

        node = LLMInferenceNode(llm_client=Flaky(), max_retry=3, retry_backoff_base=0.0)
        assert node.execute({"system_prompt": "s", "windowed_messages": "[]"})["llm_response"] == "ok"


class TestStructuredMessagesPreserveRoles:
    """system / user / assistant の role 境界を潰さない（M-1）。"""

    def test_roles_are_kept_separate(self) -> None:
        llm = FakeBaseLLM()
        resolve_llm_client(llm).chat(
            "あなたは案内役です",
            [{"role": "user", "content": "こんにちは"}, {"role": "assistant", "content": "はい"}],
        )
        assert llm.calls[0] == [
            {"role": "system", "content": "あなたは案内役です"},
            {"role": "user", "content": "こんにちは"},
            {"role": "assistant", "content": "はい"},
        ], "role をまとめて 1 つの user メッセージへ潰している"


class TestMarketplaceOutputContract:
    """AGENTIC STAR runner は success 時に `output` を必須とする。"""

    def test_get_output_includes_output(self) -> None:
        graph = ChatGraph(config={})
        out = graph.get_output({"formatted_output": "本文", "status": "success"})
        assert out.get("output") == "本文", "output が無いと runner が 'status=success but output is missing' で落ちる"
        assert out.get("formatted_output") == "本文", "既存キーを壊さない"
