"""AgentCore Platform v1.0 — CMN-C1-003 ChatAgent HTTP entry point.

Standalone HTTP entry point for the agent. Entry points are adapters only — no
business logic here. For platform-level routing, AgentGateway calls
agent.invoke() directly.

CMN-C1-003 note: ChatGraph/ChatAgent take config only (LLM は config["llm"] で注入)
(src/agent.py, src/graph/graph.py). Runtime settings are loaded from
config/config.yaml and passed through unchanged.

この standalone server は config["llm"] を供給しない。したがって認証済みの /invoke でも
LLM_CALL_FAILED になるのが仕様（会話生成が中核能力で決定論フォールバックを持たないため）。
実 LLM は AGENTIC STAR の entry adapter か、config へ BaseLLM を注入する呼び出し側が供給する。

Entry authentication is implemented by entry_adapter.auth:
  - a verified middleware trust (request.state.trust_level) is used as-is;
  - otherwise INVOKE_AUTH_TOKEN Bearer auth promotes ANONYMOUS →
    VERIFIED_EXTERNAL (_resolve_trust raises 401 on token mismatch);
  - an unauthenticated caller (no verified middleware trust AND no token
    promotion) is rejected with 401. The old fallback that defaulted such
    callers to ANONYMOUS (then failing the node-level S-1 gate late, or worse
    silently accepting the attribute's presence) is replaced with an explicit
    entry-level 401/403 fail-closed check.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets import factory as secrets_factory

from src.entry_adapter.auth import _require_verified_trust, _resolve_trust
from src.agent import ChatAgent

app = FastAPI(title="ChatAgent")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_YAML = _REPO_ROOT / "config" / "config.yaml"
with open(_CONFIG_YAML, "r", encoding="utf-8") as _f:
    _config = yaml.safe_load(_f) or {}

agent = ChatAgent(config=_config)
agent.compile()
# Replace namespace/agent_name to match the agent's manifest values.
agent.provision_secrets(secrets_factory(namespace="cmn_c1_003", agent_name="chat_agent"))


def _ctx(request: Request, session_id: str) -> InvocationContext:
    """Build the InvocationContext AFTER entry authentication.

    Scope note: the entry_adapter part also exposes a tenant/org scope contract,
    but this ChatAgent has NO scope model. It is a single-LLM multi-turn
    conversation template (no RAG, no domain KB), keyed only by session_id; there
    is no code path that reads request tenant/org scope. Enforcing
    authenticate_scope() would gate on a scope nothing consumes, so only the auth
    core (_resolve_trust / _require_verified_trust) is wired.
    """
    trust, token_authenticated = _resolve_trust(request)
    # Unauthenticated = trust is still ANONYMOUS after resolution and no token
    # promotion. Test the trust VALUE, not attribute presence — a middleware
    # that explicitly stamps ANONYMOUS must still be treated as unauthenticated
    # (→ 401), not silently accepted as "verified because the attr exists".
    if trust is TrustLevel.ANONYMOUS and not token_authenticated:
        raise HTTPException(status_code=401, detail="Authentication is required.")
    _require_verified_trust(trust)
    return InvocationContext(
        session_id=session_id or str(uuid4()),
        caller_trust_level=trust,
        caller_id=getattr(request.state, "caller_id", ""),
    )


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request):
    ctx = _ctx(request, req.session_id)
    with bound_secrets(agent._secrets_provider):
        return agent.invoke(req.input, session_id=ctx.session_id, ctx=ctx)


@app.get("/health")
def health():
    return {"status": "ok", "agent": "chat_agent"}
