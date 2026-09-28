"""Regression tests for the AGENTIC STAR Marketplace boundary adapter."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.base import NullProvider
from framework.secrets.context import bound_secrets


@pytest.fixture
def marketplace_cli(monkeypatch):
    """Load cli.py against the local SDK, which predates Marketplace helpers."""

    bootstrap = types.ModuleType("shared.bootstrap")
    bootstrap.run_agent_marketplace = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "shared.bootstrap", bootstrap)

    import framework.utils.config_loader as config_loader

    monkeypatch.setattr(
        config_loader,
        "load_agent_config",
        lambda _path: {},
        raising=False,
    )
    sys.modules.pop("cli", None)
    return importlib.import_module("cli")


def _set_scope(monkeypatch, cli) -> None:
    monkeypatch.setenv(cli.STG_DEFAULT_TENANT_ENV, "company-1")
    monkeypatch.setenv(cli.STG_DEFAULT_ORG_ENV, "project-1")


def _contains(value, needle: str) -> bool:
    if isinstance(value, dict):
        return any(_contains(key, needle) or _contains(item, needle) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains(item, needle) for item in value)
    return needle in value if isinstance(value, str) else False


def test_invoke_rebuilds_context_and_drops_runner_history(monkeypatch, marketplace_cli):
    cli = marketplace_cli
    _set_scope(monkeypatch, cli)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test-deployment")
    sentinel = "UNSCANNED-RUNNER-HISTORY-111"
    captured = {}

    class FakeAzureOpenAIClient:
        def __init__(self, config):
            self.config = config

        def complete(self, messages):
            return {"content": "safe reply", "model": "test-deployment"}

        def stream(self, messages):
            yield "safe reply"

        def bind_tools(self, tools):
            return self

    azure_module = types.ModuleType("shared.services.llm.azure_openai_client")
    azure_module.AzureOpenAIClient = FakeAzureOpenAIClient
    monkeypatch.setitem(sys.modules, "shared.services.llm.azure_openai_client", azure_module)

    graph = cli.MarketplaceGraph(config={"max_retry": 0})
    graph.compile()
    graph.provision_secrets(NullProvider())
    original_invoke = graph._compiled.invoke

    def capture_state(initial_state, config=None):
        captured["initial"] = initial_state
        final_state = original_invoke(initial_state, config=config)
        captured["final"] = final_state
        return final_state

    graph._compiled.invoke = capture_state
    ctx = InvocationContext(
        caller_trust_level=TrustLevel.VERIFIED_EXTERNAL,
        session_id="session-1",
    )
    with bound_secrets(graph._secrets_provider):
        result = graph.invoke(
            "current turn",
            session_id="session-1",
            ctx=ctx,
            input_context={
                "conversation_history": [{"role": "user", "content": sentinel}],
                "future_runner_key": "must-not-enter-state",
            },
        )

    assert result["status"] == "success"
    assert captured["initial"]["input_context"] == {
        cli.VERIFIED_TENANT_STATE_ATTR: "company-1",
        cli.VERIFIED_ORG_STATE_ATTR: "project-1",
    }
    assert not _contains(captured["initial"], sentinel)
    assert not _contains(captured["final"], sentinel)
    assert not _contains(result, sentinel)


@pytest.mark.parametrize("invalid", [None, "", " ", "company id", "company\n1"])
def test_scope_fails_closed_at_startup(monkeypatch, marketplace_cli, invalid):
    cli = marketplace_cli
    monkeypatch.setenv(cli.STG_DEFAULT_ORG_ENV, "project-1")
    if invalid is None:
        monkeypatch.delenv(cli.STG_DEFAULT_TENANT_ENV, raising=False)
    else:
        monkeypatch.setenv(cli.STG_DEFAULT_TENANT_ENV, invalid)

    graph = cli.MarketplaceGraph(config={})
    with pytest.raises(ValueError, match=cli.STG_DEFAULT_TENANT_ENV):
        graph.provision_secrets(NullProvider())


def test_azure_llm_resolves_fresh_credentials_for_every_call(monkeypatch, marketplace_cli):
    cli = marketplace_cli
    seen_configs = []

    class FakeAzureOpenAIClient:
        def __init__(self, config):
            self.config = dict(config)
            seen_configs.append(self.config)

        def complete(self, messages):
            return {
                "content": self.config["api_key"],
                "model": self.config["azure_deployment"],
                "tool_calls": [],
            }

        def stream(self, messages):
            yield self.config["api_key"]

        def bind_tools(self, tools):
            return self

    azure_module = types.ModuleType("shared.services.llm.azure_openai_client")
    azure_module.AzureOpenAIClient = FakeAzureOpenAIClient
    monkeypatch.setitem(sys.modules, "shared.services.llm.azure_openai_client", azure_module)

    provider = cli._AllowlistedEnvironmentProvider()
    llm = cli._AzureMarketplaceLLM()
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "first-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://first.example")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "first-deployment")
    with bound_secrets(provider):
        first = llm.complete([{"role": "user", "content": "one"}])

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "second-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://second.example")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "second-deployment")
    with bound_secrets(provider):
        second = llm.complete([{"role": "user", "content": "two"}])

    assert first["content"] == "first-key"
    assert second["content"] == "second-key"
    assert [config["api_key"] for config in seen_configs] == [
        "first-key",
        "second-key",
    ]
    assert "api_key" not in llm.__dict__


def test_provision_secrets_rejects_missing_azure_value(monkeypatch, marketplace_cli):
    cli = marketplace_cli
    _set_scope(monkeypatch, cli)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example")
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    graph = cli.MarketplaceGraph(config={})

    with pytest.raises(ValueError, match="AZURE_OPENAI_DEPLOYMENT"):
        graph.provision_secrets(NullProvider())


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview",
        "https://example.services.ai.azure.example/openai",
        "https://example.services.ai.azure.example/v1",
        "https://example.services.ai.azure.example?api-version=2024-10-21",
        "example.services.ai.azure.example",
    ],
    ids=["api-path", "openai-path", "base-path", "query-only", "missing-scheme"],
)
def test_unusable_azure_endpoint_shape_fails_during_startup(monkeypatch, marketplace_cli, endpoint):
    """AzureOpenAIClient appends its own path, so anything but a bare endpoint breaks.

    The resulting error surfaces inside complete(); a node that swallows it
    degrades silently while the invocation still reports status: success, so the
    adapter has to reject the shape at startup instead.
    """

    cli = marketplace_cli
    _set_scope(monkeypatch, cli)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example")
    graph = cli.MarketplaceGraph(config={})
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "gate-test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gate-test-deployment")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", endpoint)

    with pytest.raises(ValueError, match="bare"):
        graph.provision_secrets(NullProvider())


def test_scope_mapping_uses_entry_adapter_vocabulary(marketplace_cli):
    cli = marketplace_cli
    assert cli.SCOPE_KEYS == (
        cli.VERIFIED_TENANT_STATE_ATTR,
        cli.VERIFIED_ORG_STATE_ATTR,
    )
    assert cli._SCOPE_ENV_BY_KEY == {
        cli.VERIFIED_TENANT_STATE_ATTR: cli.STG_DEFAULT_TENANT_ENV,
        cli.VERIFIED_ORG_STATE_ATTR: cli.STG_DEFAULT_ORG_ENV,
    }
