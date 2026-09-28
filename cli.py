"""AGENTIC STAR Marketplace entry adapter for CMN-C1-003 ChatAgent."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from framework.secrets.base import SecretProvider
from framework.secrets.context import current_secrets
from shared.services.llm.base_llm import BaseLLM

try:
    from framework.utils.config_loader import load_agent_config
except ImportError:  # AgentCore 1.0.0 / 1.0.1 compatibility
    from framework.utils.config_loader import load_config

    def load_agent_config(agent_dir: Path) -> dict[str, Any]:
        config_path = agent_dir / "config" / "config.yaml"
        return load_config(str(config_path)) if config_path.exists() else {}


from src.entry_adapter import (
    STG_DEFAULT_ORG_ENV,
    STG_DEFAULT_TENANT_ENV,
    VERIFIED_ORG_STATE_ATTR,
    VERIFIED_TENANT_STATE_ATTR,
)
from src.graph.graph import ChatGraph

_NAMESPACE = "cmn"
_AGENT_NAME = "CMN-C1-003"
_PROVIDER_AGENT_NAME = "cmn_c1_003"

_AZURE_OPENAI_SECRET_KEYS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
)

# entry-adapter v0.2 publishes the domain scope vocabulary through these two
# VERIFIED_* constants (the equivalent of newer parts' SCOPE_KEYS).  Keep the
# key-to-environment relationship declarative so adding a scope is one data
# change rather than another branch in invoke().
SCOPE_KEYS = (VERIFIED_TENANT_STATE_ATTR, VERIFIED_ORG_STATE_ATTR)
_SCOPE_ENV_BY_KEY = dict(zip(SCOPE_KEYS, (STG_DEFAULT_TENANT_ENV, STG_DEFAULT_ORG_ENV), strict=True))
_SCOPE_ENV_KEYS = tuple(_SCOPE_ENV_BY_KEY[key] for key in SCOPE_KEYS)
_REQUIRED_SECRET_KEYS = (*_AZURE_OPENAI_SECRET_KEYS, *_SCOPE_ENV_KEYS)
_ALLOWED_SECRET_KEYS = frozenset(_REQUIRED_SECRET_KEYS)


class _AllowlistedEnvironmentProvider(SecretProvider):
    """Expose only this single-agent Pod's declared environment values."""

    def __init__(self) -> None:
        self._namespace = _NAMESPACE
        self._agent_name = _PROVIDER_AGENT_NAME

    def get(self, key: str, default: str | None = None) -> str | None:
        if key not in _ALLOWED_SECRET_KEYS:
            return default
        value = os.environ.get(key)
        return value if value else default


def _require_clean_environment_value(provider: SecretProvider, key: str) -> str:
    """Resolve one deployment value without trimming or repairing it."""

    value = provider.require(key)
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise ValueError(f"Marketplace environment variable {key!r} must be a clean non-empty string")
    return value


def _require_azure_endpoint(provider: SecretProvider) -> str:
    """Reject ``AZURE_OPENAI_ENDPOINT`` values that are not a bare Azure resource endpoint.

    ``AzureOpenAIClient`` appends its own ``/openai/v1/`` path, so anything
    beyond a scheme and host silently produces a broken request URL, and the
    failure surfaces inside ``complete()``. A node that swallows that exception
    degrades quietly while the invocation still reports ``status: success``,
    which is why this has to be a startup check. The value is reported here,
    never repaired.
    """

    endpoint = _require_clean_environment_value(provider, "AZURE_OPENAI_ENDPOINT")
    parts = urlsplit(endpoint)
    try:
        port = parts.port
    except ValueError:
        port = -1
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or port == -1
        or parts.path.strip("/")
        or parts.query
        or parts.fragment
        or any(character.isspace() for character in endpoint)
    ):
        raise ValueError(
            "Marketplace environment variable 'AZURE_OPENAI_ENDPOINT' must be a bare Azure "
            "resource endpoint -- scheme and host only, with no path (such as "
            "'/openai'), query or fragment"
        )
    return endpoint


def _scope_from_environment(provider: SecretProvider) -> dict[str, str]:
    """Build all declared scope fields from trusted deployment configuration."""

    return {key: _require_clean_environment_value(provider, _SCOPE_ENV_BY_KEY[key]) for key in SCOPE_KEYS}


class _AzureMarketplaceLLM(BaseLLM):  # type: ignore[misc]
    """Azure BaseLLM whose connection values are resolved for every call."""

    def __init__(self, *, tools: tuple[Any, ...] = ()) -> None:
        if tools:
            self._tools = tools

    def _client(self) -> BaseLLM:
        # Keep only this optional provider integration lazy.  Security/context
        # imports above must happen on the main thread before runner to_thread.
        from shared.services.llm.azure_openai_client import AzureOpenAIClient

        secrets = current_secrets()
        client = AzureOpenAIClient(
            {
                "api_key": secrets.require("AZURE_OPENAI_API_KEY"),
                "azure_endpoint": secrets.require("AZURE_OPENAI_ENDPOINT"),
                "azure_deployment": secrets.require("AZURE_OPENAI_DEPLOYMENT"),
            }
        )
        tools = getattr(self, "_tools", ())
        return client.bind_tools(list(tools)) if tools else client

    def complete(self, messages: list[Any]) -> dict[str, Any]:
        return cast(dict[str, Any], self._client().complete(messages))

    def stream(self, messages: list[Any]) -> Iterator[str]:
        yield from self._client().stream(messages)

    def bind_tools(self, tools: list[Any]) -> "_AzureMarketplaceLLM":
        return type(self)(tools=tuple(tools))


class MarketplaceGraph(ChatGraph):
    """Marketplace-only input boundary and Azure dependency wiring."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        merged = dict(config or {})
        # The runner may provision after compile(), so the non-secret wrapper
        # must occupy the config seam before ChatGraph constructs its nodes.
        merged["llm"] = _AzureMarketplaceLLM()
        super().__init__(config=merged)

    def invoke(
        self,
        user_input: str,
        session_id: str = "",
        ctx: Any | None = None,
        input_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Runner conversation_history is untrusted free text outside this AG's
        # S-2 scan.  Never merge it (or future runner keys) into AgentState.
        del input_context
        adapted_context = _scope_from_environment(_AllowlistedEnvironmentProvider())
        return cast(
            dict[str, Any],
            super().invoke(
                user_input,
                session_id=session_id,
                ctx=ctx,
                input_context=adapted_context,
            ),
        )

    def provision_secrets(self, provider: SecretProvider) -> None:
        # Do not inherit the runner provider's wider dotenv/environment view.
        del provider
        environment_provider = _AllowlistedEnvironmentProvider()
        _scope_from_environment(environment_provider)
        for key in _AZURE_OPENAI_SECRET_KEYS:
            _require_clean_environment_value(environment_provider, key)
        _require_azure_endpoint(environment_provider)
        super().provision_secrets(environment_provider)


def main() -> None:
    # Marketplace bootstrap is supplied by newer SDK/runtime distributions.
    # Keep it lazy so SDK 1.0.0 can import and register this graph.
    from shared.bootstrap import run_agent_marketplace

    config = load_agent_config(Path(__file__).resolve().parent)
    run_agent_marketplace(
        MarketplaceGraph,
        agent_name=_AGENT_NAME,
        namespace=_NAMESPACE,
        config=config,
    )


if __name__ == "__main__":
    main()
