# Current features

Last consolidated: 30 September 2026

## Project purpose

SovereignAI is a local-first industrial AI workbench for confidential manufacturing and maintenance work. It is being built to help authorized engineers turn company documents, equipment history, and sensor data into evidence-backed recommendations without sending sensitive data to public AI services.

The project aims to prove that a small local AI system can:

- retrieve only evidence the operator is authorized to see;
- combine documentary and sensor evidence without letting the language model perform safety-critical calculations;
- show citations, uncertainty, conflicts, approvals, and a complete audit trail;
- fail safely when evidence, models, or required services are unavailable; and
- remain advisory—qualified people retain authority over maintenance and physical equipment.

The target outcome is a reproducible Pump-102 industrial pilot in which every released recommendation can be traced from authorized inputs through diagnosis, verification, approval, artifact generation, and a tamper-verifiable Evidence Capsule.

## System boundary

The stable system boundary is:

```text
authenticated operator
  -> authorization and ControlPlane precheck
  -> authorized GraphRAG + sensor diagnostics
  -> bounded evidence context + local Ollama/vLLM generation
  -> ControlPlane release decision
  -> recommendation, scoped artifact, audit, Evidence Capsule
```

SovereignAI owns identity, authorization, orchestration, audit, approvals, and artifacts. Graph-RAG retrieves documentary evidence, the Time-Series Diagnostic Agent produces sensor evidence, and ControlPlane is the final release gate. Required-service failures fail closed.

## Implemented capabilities

| Area | Current capability |
|---|---|
| Local inference | Provider-neutral Ollama/vLLM text generation, Ollama vision, role-aware readiness, streaming, cancellation, timeouts, and FAST/STANDARD/DEEP execution |
| Agent execution | Bounded plan/act/observe/verify flow using registered, schema-validated tools only |
| Evidence | Dense + BM25 retrieval, RRF, optional local reranker, revision/conflict handling, query-aware context compression, provenance, and citations |
| Industrial analysis | OCR and file-type dispatch, deterministic unit/threshold calculations, asset context, read-only telemetry, trends, and Pump-102 inspection workflow |
| Diagnostics | Typed integration with the Time-Series Diagnostic Agent and automatic routing when a valid diagnostic envelope is supplied |
| Governance | PII and prompt-injection checks, structured claim support states, ControlPlane precheck/release control, abstention, and fail-closed handling |
| Security | Local sessions, CSRF, RBAC plus contextual ACLs, authorization before retrieval scoring, scoped caches/resources, and requester/approver separation |
| Execution safety | Generated Python runs only in a restricted, networkless Docker sandbox; there is no host fallback |
| Workflows | Versioned declarative Workcell Packs resolved only to trusted handlers; Pump Inspection v1.0.0 is included |
| Organizations | Validated Organization Packs for hierarchy, users, policies, assets, documents, and organization-scoped Workcells |
| Outputs | Real DOCX, XLSX, and PPTX artifacts with task/evidence lineage |
| Audit and provenance | Persisted task/tool/decision events, SSE progress, SHA-256 Evidence Capsules, independent verification, and optional Ed25519 signing |
| UI | Next.js workbench plus metrics, sovereignty, organization, assets, and onboarding surfaces |

## Flagship workflow

The Pump-102 path combines authorized manuals/SOP/history, sensor diagnosis, deterministic engineering checks, local synthesis, factuality/release control, human approval where required, a maintenance artifact, and an Evidence Capsule. Recommendations remain advisory and cannot command PLC, DCS, SCADA, alarms, setpoints, or equipment.

## Runtime and data

- FastAPI backend and Next.js frontend.
- SQLite for prototype state, audit, metadata, chunks, and caches.
- Local MiniLM embeddings with explicit feature-hashing fallback.
- Configured model roles: GENERAL, VISION, and CODER.
- Loopback/private internal service URLs only; public model/service URLs are rejected.
- APEL synthetic demo: 20 assets, 55 files, seven users, five scenarios, and a 50-question evaluation set.

## Important limits

- The project is a prototype, not a production security or safety accreditation.
- GraphRAG needs approved local/self-hosted storage for a strict offline deployment.
- SQLite, in-memory live channels, and local identity are single-node foundations.
- The optional reranker must be staged locally; otherwise retrieval uses RRF fallback.
- OCR, vision, diagnostics, and recommendations require qualified human review.
- Evidence Capsule integrity proves stored-byte integrity, not factual correctness.
- Live enterprise identity, historian, OPC-UA, SAP/Maximo, durable queues, PKI/HSM, malware scanning, encryption/key management, backup/recovery, and signed append-only audit are not yet implemented.
- Plant writes and autonomous maintenance execution are intentionally absent.

See [Past experiments](PAST_EXPERIMENTS.md) for evidence and [Next plan](NEXT_PLAN.md) for priorities.
