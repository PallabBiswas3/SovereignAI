# Integrated System Architecture

## System boundary

SovereignAI is the only operator-facing trust boundary. It authenticates the caller, applies role permissions, assigns a run identity, orchestrates evidence services, records the outcome, and returns the final policy-controlled response.

The other projects remain independently deployable services:

- Graph-RAG retrieves document chunks and provenance-linked claims. Its integration endpoint does not synthesize a user answer.
- Time-Series Diagnostic Agent executes the existing strict `diagnose_tool` contract and returns a validated `DiagnosticResult`.
- ControlPlane.ai performs both the prompt precheck and final response-release decision.

## Request sequence

```text
Operator
  |
  | authenticated POST /api/integrations/analyze
  v
SovereignAI
  |
  +-- 1. POST ControlPlane /v1/precheck
  |       block/review -> stop and audit
  |
  +-- 2a. POST Graph-RAG /api/integration/retrieve ----+
  |                                                    |
  +-- 2b. POST Diagnostics /v1/diagnose ---------------+
  |                                                    |
  +-- 3. assemble bounded candidate + evidence context <+
  |
  +-- 4. POST ControlPlane /v1/check
  |       allow/warn/redact -> release final_response
  |       review/block      -> hold final_response
  |
  +-- 5. write organization-scoped audit event
  v
Operator
```

Graph retrieval and time-series diagnosis run concurrently after the prompt passes precheck. A service failure stops the workflow by default (`SOVEREIGN_INTEGRATION_FAIL_CLOSED=true`).

## Latency-aware assurance

General Chat stays inside SovereignAI and uses one local generation after the input safety scan. Authorized document questions call Graph-RAG with `verification_mode=standard`, which checks the highest-ranked material claim using a small evidence excerpt without a retrieval retry. Controlled work, diagnostic tasks, and engineering or finance policies use `verification_mode=thorough`, which verifies up to four claims and permits one bounded retry. The Graph-RAG response reports retrieval, verification, and total milliseconds; SovereignAI adds precheck, evidence, release-check, and total timings to the task runtime metrics.

All profiles fail closed. Latency reduction changes the amount of repeated verification, not the meaning of a supported claim.

## Trust rules

1. Browser clients call SovereignAI only. They never receive service credentials.
2. Integration URLs must resolve to loopback, private IPs, or internal single-label service names. Public service URLs are rejected at client construction.
3. Graph-RAG returns raw evidence and supported structured claims. It cannot authorize a user or release a response.
4. The diagnostic API performs strict envelope validation but leaves authorization to SovereignAI.
5. A generated or deterministic candidate is not releasable until ControlPlane returns `allow`, `allow_with_warning`, or `redact`.
6. `human_review`, `block`, detector failures under consequential policy, and required-service failures remain held.
7. The audit event stores service status and policy outcome, not the full sensitive prompt or evidence payload.

## Runtime configuration

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `SOVEREIGN_GRAPHRAG_URL` | `http://127.0.0.1:3100` | Evidence retrieval service |
| `SOVEREIGN_CONTROLPLANE_URL` | `http://127.0.0.1:8100` | Precheck and release gate |
| `SOVEREIGN_DIAGNOSTICS_URL` | `http://127.0.0.1:8200` | Time-series diagnostic tool |
| `SOVEREIGN_INTEGRATION_TIMEOUT_SECONDS` | `180` | Per-service timeout sized for local CPU inference |
| `SOVEREIGN_INTEGRATION_FAIL_CLOSED` | `true` | Stop when required evidence is unavailable |

The supplied PowerShell launcher binds all processes to `127.0.0.1`. Production deployment should place the same contracts on an authenticated internal network and add service identity, TLS, durable queues, and centralized secret management without changing the ownership boundaries above.

## Current limitation

Graph-RAG uses local Ollama generation and embedding models. Supabase remains its configured graph/vector store, so a hosted Supabase URL is still an external dependency. Deployments requiring a strict offline guarantee must point Graph-RAG at an approved local Supabase deployment. Re-ingest existing documents after changing embedding providers so stored and query vectors use the same model.
