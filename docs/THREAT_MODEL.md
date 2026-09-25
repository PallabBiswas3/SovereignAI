# Threat model

Protected assets are organizational documents, plant telemetry, model prompts,
generated recommendations, artifacts, approvals, audit records, and capsules.

| Threat | Boundary/control | Failure behavior |
|---|---|---|
| Cross-tenant or cross-department retrieval | Full scope propagated to GraphRAG; fail-closed ACL before scoring | Forbidden/unscoped rows never become candidates |
| Service key bypasses Supabase RLS | Secret remains server-side; application ACL is mandatory before ranking | Missing scope returns HTTP 400 |
| Text model advertised as VISION/CODER | Runtime capability check | `CAPABILITY_MISMATCH`; UI disables role |
| Model outage/timeout | Provider-bounded request | Deterministic unavailable; no candidate or artifact |
| Contradictory SOP revisions | Provenance and verification results | Conflict surfaced; ControlPlane may hold |
| No evidence | GraphRAG abstention | No factual recommendation is synthesized |
| Prompt injection | Pre-generation ControlPlane check | Evidence/model services are not called |
| Docker absent | Docker-only execution seam | Generated code is never executed on host |
| Unsafe or unsupported candidate | Final ControlPlane check | Answer is not released |
| Artifact/capsule enumeration | Resource scope checks | Unauthorized resources return not found |

Residual risks include compromised internal services, poisoned authorized
documents, stale identity data, model-verifier correlation, and a deployment
whose network policy permits egress. Production deployment must independently
enforce service authentication, TLS, backups, monitoring, and network deny rules.
