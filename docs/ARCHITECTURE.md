# Architecture (frozen baseline)

Status: **frozen after the 2026-09-25 boundary closure**. New agents, novelty
providers, unrelated features, and ControlPlane rewrites are out of scope.
Changes to the boundaries below require an explicit architecture decision record.

```text
Authenticated principal
  → full effective authorization scope
  → ControlPlane prompt precheck
  → Pump-102 sensor/history diagnosis ─┐
  → GraphRAG ACL filter                │ before scoring
      → dense + BM25 → RRF → rerank    │
      → revision/conflict/provenance   ├→ bounded evidence prompt
  → LocalModelProvider                 │
      → vLLM or Ollama                 │
  → ControlPlane evidence/release check
  → released recommendation or held/abstained state
  → scoped artifact → Evidence Capsule
```

The three closed structural gaps are:

1. The flagship orchestrator invokes `LocalModelProvider`; `.env` provider
   selection now reaches vLLM or Ollama instead of being ignored.
2. The authenticated principal’s organization, departments, workspaces, roles,
   user, clearance, cross-department grant, and scope fingerprint reach GraphRAG.
   Unscoped/unauthorized rows are removed before dense or sparse scoring.
3. Readiness is capability-aware. Installed is not equivalent to ready: VISION
   requires vision capability and CODER requires a coder identity/capability.

ControlPlane is the only release gate. A model failure, retrieval abstention,
contradiction, or policy hold cannot be relabeled as a successful answer.

Application-level locality is not a network-level air gap. Network isolation and
egress verification are deployment controls described in `SOVEREIGNTY.md`.
