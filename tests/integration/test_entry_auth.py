"""Entry-adapter authentication tests for CMN-C1-003.

Verifies the fail-closed entry authentication in src/api/server.py:
  - an unauthenticated caller (no verified middleware trust, no token) → 401;
  - a valid Bearer INVOKE_AUTH_TOKEN promotes ANONYMOUS → VERIFIED_EXTERNAL → 200;
  - a wrong token → 401;
  - /health stays open (no auth).

The agent.invoke() is patched to a stub so the test exercises the entry adapter
only (no LLM / secrets provider needed). Requires the framework wheel to import
src.api.server; runs in CI where agenticstar-agentcore is installed.
"""

from __future__ import annotations


import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

_TOKEN = "mock-invoke-token-for-testing"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)
    # Import after env is set; patch invoke so no real agent run is needed.
    from src.api import server

    monkeypatch.setattr(
        server.agent,
        "invoke",
        lambda user_input, **kwargs: {"status": "success", "output": "ok"},
    )
    return TestClient(server.app)


def test_unauthenticated_is_rejected(client):
    resp = client.post("/invoke", json={"input": "hi", "session_id": "s1"})
    assert resp.status_code == 401


def test_wrong_token_is_rejected(client):
    resp = client.post(
        "/invoke",
        json={"input": "hi", "session_id": "s1"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_valid_token_is_accepted(client):
    resp = client.post(
        "/invoke",
        json={"input": "hi", "session_id": "s1"},
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_health_is_open(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
