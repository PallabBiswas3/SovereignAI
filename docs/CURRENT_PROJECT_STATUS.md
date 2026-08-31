# SovereignAI: current project status and behavior

Last verified: 1 September 2026

SovereignAI is a working local-first industrial AI workbench, not merely a chat page. Its Next.js UI sends typed tasks to FastAPI; the backend applies governance, classifies and routes work, uses local Ollama models where probabilistic interpretation is appropriate, executes only registered tools, preserves evidence, generates real office artifacts, and persists task/audit state in SQLite.

For the detailed subsystem inventory and limitations, see [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md). For the distinction between application and network isolation, see [SOVEREIGNTY.md](SOVEREIGNTY.md).

## Verified runtime on this workstation

| Component | Verified state |
|---|---|
| Ollama | Reachable at `http://localhost:11434` |
| GENERAL | `qwen3-vl:4b` — `READY` |
| VISION | `qwen3-vl:4b` — `READY`; shares the GENERAL model blob |
| CODER | `qwen2.5-coder:7b` — `READY` |
| Semantic embeddings | Local `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions |
| OCR | Local Tesseract/PDFium workflow passes scanned-PDF tests |
| Docker sandbox | Docker CLI is installed, but the engine was stopped during final verification; no host execution fallback occurred |
| Backend tests | 37 passed |
| Frontend | TypeScript and optimized production build passed |

## What each task path does

### General and tool tasks

Ordinary questions route to the GENERAL role. Requests that need application capabilities use a bounded local-model decision loop. The model can choose only registered, schema-described tools; the control plane validates arguments and policy before execution. Calls, decisions, elapsed time, and retries are bounded. The agent retains concise action summaries, sources, artifacts, and verification records without storing private chain-of-thought.

If local final synthesis exhausts the execution bound after a tool succeeded, the system returns that verified tool observation directly instead of fabricating a response or falsely marking the tool execution as unsuccessful.

### Coding tasks

A CSV coding task is profiled into columns, inferred types, row count, missing counts, and five safe sample rows. The CODER model generates source code; Docker executes it with no network and strong resource restrictions. stderr and current code can be returned to the coder for at most three total attempts. Code versions, stdout/stderr summaries, output files, exit codes, and final verification are audited. The deterministic script is emergency fallback only. When Docker is unavailable, code is not run on the host and the run remains honestly unverified.

### Inspection and multi-file tasks

The inspection workflow routes scanned PDFs to OCR, images to the local vision role, SOPs to semantic retrieval, office/text documents to parsers, and CSV/XLSX files to structured profiling. Evidence keeps per-file provenance. Engineering thresholds and replacement rules remain deterministic Python comparisons against retrieved, metric-specific SOP evidence.

A management-package request creates and registers:

- `approval_note.docx`
- `inspection_analysis.xlsx`
- `management_briefing.pptx`

All three formats are opened in automated tests and associated with one run and its audit trail.

### Grounding and approvals

Generic answers are split into claims and classified as `SUPPORTED`, `WEAKLY_SUPPORTED`, or `UNSUPPORTED` using semantic, lexical, retrieval, numerical-consistency, and provenance signals. Unsupported material claims enter governance. Scores are called grounding confidence/evidence support, not calibrated hallucination probability.

Approval-required actions are stored without execution. Approval revalidates the exact saved tool and arguments against the current policy and registered schema before execution. Safe `write_file` demonstrates this path. Disabled deletion and arbitrary shell actions remain blocked even after an attempted approval.

### Live UI, sovereignty, and evaluation

The UI accepts multiple attachments and subscribes to SSE task events for governance, classification, routing, planning, steps, tools, sources, warnings, artifacts, completion, and failure. Final task state and events are persisted; live channels are process-local.

The application rejects public and private-LAN model endpoints, permitting only loopback and explicit internal service names. Compose uses an internal network and offers a pinned internal Ollama service. The active air-gap verifier checks that internet egress fails while Ollama remains reachable. This is distinct from host-wide packet monitoring.

The offline benchmark contains 70 labeled cases: 20 routing, 20 RAG, 20 governance, and 10 agent cases. Current small-benchmark results are routing accuracy 0.95/macro-F1 0.947; semantic RAG Recall@1, Recall@3, MRR, and citation correctness 1.0 with refusal correctness 0.75; and PII/injection F1 1.0. These are synthetic regression results, not production-quality claims.

## Fine-tuning status

Fine-tuning is intentionally not implemented or required. The current priorities are local model integration, RAG, deterministic engineering controls, and larger evaluation sets. Consider LoRA only after collecting representative, permissioned examples and measuring a reproducible gap that prompting/RAG cannot solve.
