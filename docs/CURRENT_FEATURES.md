# Current features

Last consolidated: 1 October 2026

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

## Inference experiment tooling

`benchmarks/inference_tradeoff.py --protocol controlled` now runs the version-3
host-local protocol implemented in `benchmarks/inference_controls.py`. The old
warm-only protocol is explicitly `legacy-controlled`; `smoke` remains compatible.
Neither legacy mode can mark an experiment complete.

- Provider API identity plus local listener/process identity: Ollama version,
  digest, format, quantization, template/system hashes and selected runtime
  settings; vLLM/PyTorch/Transformers/Python versions, pinned local HF revision,
  weight/config hashes, explicit dtype and selected runtime settings. vLLM must
  use the same Python environment as its harness. Drift fails the experiment.
- Host OS/CPU/RAM and normalized-source harness hash. Per-batch 50 ms host
  available RAM and provider process-tree RSS samples, including errors and
  limitations; unavailable measurements never become zero-valued substitutes.
- Provider-verified **full prompt-token ceilings**, including template overhead.
  The complete fixture is retained; budgets too small for it fail. Neutral padding
  fills remaining space. vLLM uses `/tokenize`; Ollama uses untimed one-token
  calibration. Measured provider usage must agree exactly with calibration.
  `target_tokens` is retained only as a v3 compatibility alias for
  `prompt_token_budget`; historical v2 values remain whitespace words.
- Seeded shuffled repetition blocks cover requested budgets, concurrency 1/2/4,
  warm fresh/reused prefixes, and unprimed cold batches. Each lane has its own
  nonce. Warm-ups are excluded from summaries. Batch throughput uses the actual
  batch wall-clock interval, not a sum across repetitions.
- Reset after calibration and before every batch: Ollama unload acknowledgement
  plus `/api/ps` absence; vLLM requires an external restart, disappearance of the
  previous process tree, a newly created listener, and matching runtime identity.
  The harness never invokes shell commands, terminates processes, or evicts OS
  caches. Failed reset means no cold request. Cold concurrency describes an
  initially cold **batch**, not an independently cold model for each lane.
- Versioned expected-fact/citation JSON schema and fixture provenance/review
  metadata. The shipped fixture is explicitly synthetic and unreviewed. Final
  execution refuses it; `--plan-only` makes no runtime requests. Lexical output
  scoring remains a proxy, even with reviewed expected labels.
- `summary.json` records separate completion gates for both providers, identity
  and its consistency, cold state, warm-ups, memory, complete condition coverage,
  valid token accounting, reviewed real quality labels and failures. Host-local
  partials remain incomplete until a strict matching-config merge. Raw JSON/CSV,
  prompts, answers, samples and source-report hashes are retained. Existing output
  directories cannot be overwritten by a new run.

No final large live benchmark has been run with this protocol. Windows Ollama
and WSL BF16 vLLM constitute a **runtime + format + host** comparison. Process RSS
is not VRAM, may double-count shared pages, and misses between-sample peaks; WSL
reports guest memory. A process restart is not disk/page-cache cold. vLLM loads
weights before serving, so its measured cold request excludes server startup;
Ollama's post-unload request may include loading. Prefix reuse is a controlled
input condition, not proof of a cache hit. The two budget experiment labels both
use full prompt ceilings with fixed evidence; they do not measure retrieval or
evidence-selection quality. See `NEXT_PLAN.md` for commands and remaining gates.

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
