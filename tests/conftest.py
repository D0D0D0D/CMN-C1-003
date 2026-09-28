"""tests/conftest.py — CMN-C1-003 test bootstrap (released AgentCore SDK v1.0.0).

The L1 framework (``framework.*`` / ``shared.*``) is provided by the installed
``agenticstar-agentcore`` wheel — there are no vendored stubs and no sys.modules stubbing
(migrated from the interim stub model, SDK v1.0.0).

This only ensures the repo root is importable so ``import src...`` works when running
``pytest tests/`` without an editable install.
"""

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture
def agent_config() -> dict:
    # tests/integration/test_chat_agent.py:36 の単一会話 happy path と同じ構築前提。
    return {}


@pytest.fixture
def case_payload() -> str:
    # tests/integration/test_chat_agent.py:46 の happy path が使う既定 user_input と同じ。
    return "hello"


@pytest.fixture
def live_input_context() -> dict:
    # tests/integration/test_chat_agent.py:40 の happy path は追加 input_context を使わない。
    return {}


# live_llm_injector は削除した。
# 🔴 共通ハーネスは構築前に config["llm"] を設定するが、この fixture が直後に
# `agent._llm_client` を上書きしていたため、実 LLM 検証が本番の config 配線を
# 迂回していた（本番が未配線でもテストは緑になる）。config 経路だけを使う。
