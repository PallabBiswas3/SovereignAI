# SovereignAI implementation status

This document describes what the repository now does. It is an implementation inventory, not a production-security accreditation.

## Runtime model roles

| Logical role | Ollama tag | Purpose |
|---|---|---|
| GENERAL | `qwen3-vl:4b` | General responses and registered-tool decisions |
| VISION | `qwen3-vl:4b` | Local image evidence analysis |
| CODER | `qwen2.5-coder:7b` | Source generation and bounded repair |

GENERAL and VISION intentionally share one downloaded model blob. `/api/models/status` queries Ollama and returns `READY`, `MODEL_NOT_INSTALLED`, `OLLAMA_UNAVAILABLE`, or `CONFIGURATION_ERROR` per logical role.

## Working execution paths

- General requests use a local model. Tool-oriented requests use a bounded decide/act/observe loop with registered schemas, policy checks, maximum calls/decisions/time, concise decision summaries, source collection, and artifact collection.
- Coding requests receive a bounded CSV profile, use the coder model to generate Python, execute only in the networkless restricted Docker sandbox, and may repair against stderr up to three attempts. Code versions, execution records, reports, and outputs are registered and audited. A deterministic source template exists only as an honest emergency fallback.
- Inspection requests use OCR or digital extraction, semantic SOP retrieval, and deterministic metric-specific comparisons. The LLM never decides arithmetic thresholds. A management-package request generates a real DOCX, XLSX, and PPTX.
- Multi-file evidence is dispatched by type: scanned PDF to OCR, image to local vision, SOP to ingestion/retrieval, DOCX/Markdown/text to parsers, and CSV/XLSX to structured profiling. File provenance remains attached.
- Generic grounding splits responses into claims and records semantic, lexical, retrieval, optional judge, source, and `SUPPORTED`/`WEAKLY_SUPPORTED`/`UNSUPPORTED` fields. Material unsupported claims enter governance.
- Approval-required registered actions are persisted without execution. Approval revalidates the current policy and exact stored arguments before controlled execution. Disabled destructive actions cannot be enabled by approval.
- `POST /api/tasks/start` plus SSE provides live structured lifecycle events without hidden reasoning. Final runs, tools, code attempts, artifacts, approvals, and decisions remain auditable in SQLite.

## Retrieval and evaluation

Production/demo retrieval uses a fully local MiniLM semantic encoder when its weights are cached; feature hashing remains an explicit offline fallback. Document embedding provider and dimension are persisted to prevent incompatible-vector comparisons.

The offline benchmark contains 20 routing cases, 20 RAG cases, 20 governance cases, and 10 agent-planning cases. It reports routing accuracy/confusion matrix/macro F1; Recall@1/Recall@3/MRR/citation/refusal metrics and hash-vs-semantic comparison; governance precision/recall/F1/error rates/confusion matrices; benchmark and observed agent metrics; and available system resource metrics. These are small benchmark results, not production claims.

## Boundaries

- Ollama, its two model tags, Tesseract, and Docker are external local runtime dependencies and are not silently emulated.
- Docker must be running for verified code execution. Generated code never falls back to host execution.
- The semantic model must be staged locally for true semantic retrieval; otherwise configured fallback is reported through provider metadata.
- Vision and OCR results require human review for consequential engineering decisions.
- SSE channels are in-memory and single-process even though final audit and task events are persisted.
- Application endpoint restrictions are not packet-level proof. Use the internal Compose network and active verifier described in `SOVEREIGNTY.md`.
- Authentication, RBAC, encrypted storage, signed audit logs, malware scanning, durable workers, and formal accreditation remain outside this prototype.
