# CMN-C1-003 — ChatAgent

> **Category**: Cat 1 (single technical capability)
> **Industry**: CMN

## Overview

ChatAgent is a reusable multi-turn conversation capability built on a single language-model
call. It keeps a bounded window of the recent dialogue for a session, sends that window plus the
new utterance to the model, and returns the reply — so answers follow the thread of the
conversation without letting history grow without limit.

It suits work where conversation is the whole job: an application that needs session-aware
replies, a prompt author trying a system prompt against a real multi-turn exchange, or a caller
continuing an earlier exchange by reusing the same `session_id`.

The template deliberately does one thing. There is no retrieval, no knowledge base and no tool
calling — those belong to other templates, and mixing them in would make this one harder to
reuse. What is configurable is the prompt, the model settings, the retry behaviour and the
history limits; what is replaceable is the language-model client and the conversation store,
both of which are interfaces rather than bundled implementations.

Two properties are worth stating plainly, because they shape how the template behaves in
practice:

- **History is scoped to a session and never crosses one.** Reads and writes are keyed by
  `session_id`, and a request without one is rejected rather than served from a shared bucket.
- **A failed generation is not dressed up as a reply.** When the model cannot be reached, the
  call is retried with exponential backoff up to `max_retry` times and then ends as
  `status=error` / `LLM_CALL_FAILED`. There is no canned fallback text.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Integration

Provide:

- an AgentCore runtime;
- a `BaseLLM` implementation injected through `config["llm"]` — no provider client is bundled.
  Without one, generation raises on the first call and every invocation ends as `status=error` /
  `LLM_CALL_FAILED`;
- a conversation store, if replies must survive a restart. The bundled
  `ConversationRepository` is an in-process dictionary (see *Security and Limitations*);
- caller authentication in front of the entry adapter.

Runtime parameters are read from `config/config.yaml` (defaults in parentheses):

| Key | Meaning |
|---|---|
| `system_prompt_template` | System prompt prepended to every exchange |
| `history_window_turns` (`10`) | Maximum recent messages replayed to the model |
| `history_token_budget` (`3000`) | Token ceiling for the replayed window; trims further when exceeded |
| `history_max_stored_messages` (`50`) | Rolling cap on messages kept per session |
| `model` (`claude-sonnet-4-6`) | Model name passed to the injected client |
| `temperature` (`0.7`) / `max_tokens` (`1024`) / `timeout_s` (`30`) | Generation settings passed through |
| `max_retry` (`3`) / `retry_backoff_base` (`2.0`) | Retry count and exponential backoff base for the model call |
| `mask_patterns` | Output masks applied by the S-3 gate. **Not set by default** — see *Security and Limitations* |

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent is designed to run on the platform, which supplies the language-model client and the caller identity. Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Supplied by the platform's own package registry at build time. It is not published on public PyPI, so `pip install -e .` alone does not fetch it. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run on AGENTIC STAR, and the agent does not fall back to a degraded
mode when it is absent. The standalone server in `src/api/server.py` does start without the
platform — it loads `config/config.yaml`, compiles the graph and serves `/health` — but it
deliberately does **not** supply `config["llm"]`. An authenticated `/invoke` against it therefore
runs the pipeline and ends at the generation step with `status=error` / `LLM_CALL_FAILED` rather
than returning a partially generated answer. Refusing to answer is deliberate: conversation *is*
this template's capability, and there is no deterministic substitute for it. The test suite runs
without a platform connection, using a fake client.

## Input

`POST /invoke` takes a JSON body with `input` (the user utterance) and `session_id` (the
conversation key). The HTTP adapter substitutes a generated UUID when `session_id` is empty, so
such a request simply starts its own fresh conversation; the graph's own `MISSING_SESSION_ID`
guard applies to callers that invoke the graph directly rather than through this adapter.

Callers must be authenticated. Trust stamped by verified upstream middleware is used as-is;
otherwise, when `INVOKE_AUTH_TOKEN` is set on the server environment, a matching
`Authorization: Bearer` header promotes the caller to `VERIFIED_EXTERNAL`. A caller that is
neither middleware-verified nor token-authenticated is rejected with **401** before the graph is
entered, a wrong token is **401**, and a verified caller below `VERIFIED_EXTERNAL` is **403** —
there is no anonymous path into the conversation.

```bash
curl -X POST http://127.0.0.1:8000/invoke \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${INVOKE_AUTH_TOKEN}" \
  -d '{"input": "hello", "session_id": "example-session"}'
```

`GET /health` returns a liveness response and takes no authentication.

Input fails closed with stable error codes: `EMPTY_INPUT` (no utterance), `MISSING_SESSION_ID`
(no conversation key), `S2_INJECTION_DETECTED` (a credential pattern in the utterance),
`S1_TRUST_DENIED` (insufficient caller trust reaching a node). `MISSING_SESSION_ID` is reachable
only for a direct graph caller, as noted above.

## Output

The standard AgentCore envelope is preserved. `output` carries the reply text; the same content
and the intermediate fields are also surfaced individually — `formatted_output`,
`history_persisted`, `history_turn_count`, `windowed_messages`, alongside `status`, `error`,
`error_code`, `error_message`, `trace_id`, `correlation_id`, `session_id`, `node_history` and
`error_log`.

`windowed_messages` is a JSON string holding the window that was actually sent to the model, so a
caller can see what the reply was conditioned on. `history_turn_count` counts the messages in
that window, not the number of exchanges. `history_persisted` reports whether the turn
reached the conversation store — a store failure is non-fatal and the reply is still returned,
with the flag `false`.

```json
{
  "output": "assistant reply",
  "formatted_output": "assistant reply",
  "history_persisted": true,
  "history_turn_count": 1,
  "windowed_messages": "[{\"role\": \"user\", \"content\": \"hello\"}]",
  "status": "success",
  "error": null,
  "error_code": null,
  "error_message": null,
  "session_id": "example-session"
}
```

Values above are illustrative; real output depends on the utterance and the accumulated history.

## Security and Limitations

Each conversation node declares verified-external trust, so an insufficiently trusted caller is
stopped before any node body runs. The utterance is scanned for credential patterns on the way in and
rejected with `S2_INJECTION_DETECTED` if one is found. On the way out, the reply is passed
through the configured output masks and then re-checked: if a credential pattern survives, the
reply is discarded and the response becomes `S3_BLOCKED` rather than being returned or stored.
Audit events record field names and counts only — the utterance and the reply are never written
to the audit trail.

Three limits are worth knowing before deploying this template as-is:

- **The output mask list is empty by default.** `mask_patterns` is declared in
  `config/agent.yaml`, but the graph reads it from the runtime configuration, and nothing loads
  the manifest at run time. Unless `mask_patterns` is added to `config/config.yaml` (or supplied
  by the caller's configuration), no PII masking is applied — only the credential block. The
  bundled registry recognises `mynumber` and a keyword-anchored `account_number`;
  `appi_sensitive` is a deliberately empty extension point for derived templates. Set the list
  explicitly for any deployment that handles personal data.
- **Conversation history does not survive a restart.** The bundled repository is an in-process
  dictionary with a rolling cap of `history_max_stored_messages` per session. Replace it with a
  durable implementation before relying on continuity across restarts or across replicas.
- **The window is a recency window, not a summary.** Once a conversation exceeds
  `history_window_turns` or the token budget, the oldest turns are dropped rather than
  compressed, so long conversations lose their early context.

Output is machine-generated and assistive. It does not substitute for expert review, and any
decision that matters should be verified by a person against the primary source.

## Release

- Version: 1.0.0
- Category: single technical capability (Cat 1), conversational pattern
- Generation mode: `llm`
- HITL: not enabled
- Persistence: in-process conversation store only; no checkpointer is configured
- External API clients: deployment-injected; none bundled

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection, using a fake language-model client. Running the agent
itself does not: a bare `uvicorn src.api.server:app` serves `/health`, but an authenticated
`/invoke` returns `LLM_CALL_FAILED` by design, because the standalone server supplies no
`config["llm"]`. A real model is supplied by the platform entry adapter, or by a caller that
injects a `BaseLLM` into the configuration.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent manifest and runtime configuration
docs/         design and test documentation
```

## Customising

1. Adjust `config/config.yaml` for your own prompt, model settings and history limits.
2. Inject an LLM client through `config["llm"]` and, if continuity matters, a durable
   conversation repository.
3. Set `mask_patterns` for the personal data your deployment actually handles, registering any
   additional pattern in the mask registry.
4. Re-run the test suite.

## Documentation

`docs/02_design.md` carries the node flow, the state schema, the history policy and the security
design. `docs/03_test_spec.md` records the test specification and the verification results.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
