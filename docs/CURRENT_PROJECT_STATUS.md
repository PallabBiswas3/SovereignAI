# SovereignAI: current project status and behavior

Last verified: 18 September 2026

SovereignAI is a working local-first industrial AI workbench, not merely a chat page. Its Next.js UI sends typed tasks to FastAPI; the backend applies governance, classifies and routes work, uses local Ollama models where probabilistic interpretation is appropriate, executes only registered tools, preserves evidence, generates real office artifacts, and persists task/audit state in SQLite.

For the detailed subsystem inventory and limitations, see [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md). For the distinction between application and network isolation, see [SOVEREIGNTY.md](SOVEREIGNTY.md). For the complete machine-measured local-inference optimization history, including negative results and current runtime choices, see [LOCAL_AI_RUNTIME_RESEARCH_LOG.md](LOCAL_AI_RUNTIME_RESEARCH_LOG.md).

## Verified runtime on this workstation

| Component | Verified state |
|---|---|
| Ollama | Reachable at `http://localhost:11434`; selected general-inference backend after paired Ollama-vs-llama.cpp measurement |
| GENERAL | `qwen3:4b-instruct`, `Q4_K_M` — current measured local choice |
| GENERAL CPU tuning | 5 threads / batch 128; measured tuning profile |
| GENERAL context | adaptive; 2048 preferred for short requests, escalates only when required |
| GENERAL KV cache | `f16` remains the safe persisted choice pending a cleaner controlled compressed-cache rerun |
| VISION | `qwen3-vl:4b-instruct` — `READY` |
| CODER | `qwen2.5-coder:7b` — `READY` |
| Semantic embeddings | Local `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions |
| OCR | Local Tesseract/PDFium workflow passes scanned-PDF tests |
| Docker sandbox | Docker CLI is installed; generated code never falls back to host execution |
| Local runtime CI | Phase 20-27 runtime tests and benchmark-script compilation are covered by `Local AI Runtime checks` |
| Frontend | TypeScript and optimized production build passed in the established verification workflow |

## Measured local-inference status

The CPU-laptop optimization track is now a first-class engineering subsystem rather than a collection of ad-hoc benchmark commands.

Current measured choices for the 16 GB, 10-physical/12-logical-core target laptop are:

- backend: **Ollama**;
- model: **Qwen3 4B Instruct Q4_K_M**;
- CPU runner: **5 threads / batch 128**;
- preferred short-request context: **2048**;
- context behavior: **adaptive escalation** rather than a fixed large context;
- normal model concurrency: **one generation at a time**;
- approximate keep-alive target: **60 seconds**;
- KV cache: **f16 for now**.

The most important verified backend result is the paired interleaved Ollama-vs-llama.cpp experiment. With the same logical model/weights and quality score `1.0` on both paths, Ollama measured `10.680 tok/s` median versus `9.107 tok/s` for llama.cpp. The median paired llama.cpp/Ollama ratio was `0.8871`, so direct llama.cpp is implemented but is not the selected runtime on this workstation.

Context measurement also showed that larger fixed windows are expensive in RAM. Phase 8 measured resident sizes of approximately 2685 MB at context 2048, 2973 MB at 4096, and 3550 MB at 8192, while decode throughput changed only modestly in that run. The runtime therefore uses the smallest safe context bucket that fits the request.

Phase 10 added isolated KV-cache experiments for `f16`, `q8_0`, and `q4_0` at context 2048 and 4096. All current smoke-quality canaries passed, but the OS available-RAM baselines differed materially between isolated runs and the long-context canary introduces heavy thermal load before some performance samples. The benchmark safely persisted `f16`; q8_0 remains a candidate for a more tightly controlled rerun, especially at context 4096. No strong RAM-saving claim is made from the Phase 10 v1 OS-delta numbers.

See [LOCAL_AI_RUNTIME_RESEARCH_LOG.md](LOCAL_AI_RUNTIME_RESEARCH_LOG.md) for the full Phase 1-10 chronology, exact metrics, commands, limitations, negative results, and next experiments.

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

## Reusable organization onboarding

Versioned Organization Packs now onboard organization structure, departments, workspaces, local or externally managed identities, access policies, assets, approved documents, and organization-scoped Workcells. Validation is fail-closed; dry-run is non-mutating; actual import is transactional, idempotent, and audited. A same-organization admin interface is available at `/admin/onboarding`, while first-time organization bootstrap remains a local CLI operation. See [ORGANIZATION_PACKS.md](ORGANIZATION_PACKS.md) and [SOVEREIGNAI_2_BATCH6.md](SOVEREIGNAI_2_BATCH6.md).

Real corporate identity and plant-system adapters are not generic configuration: they require the target company's endpoints, certificates, account/group mappings, network zones, and acceptance approval. The existing plant connector boundary remains read-only.
