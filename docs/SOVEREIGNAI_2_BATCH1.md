# SovereignAI 2.0 — Batch 1 implementation

Completed on 1 September 2026. This batch implements phases 0–5 only. Hybrid retrieval,
GraphRAG, and later architecture phases have deliberately not been started.

## What changed

### True local streaming

The provider-neutral LLM interface now exposes incremental `GenerationChunk` values. The
Ollama adapter consumes `/api/generate` as NDJSON with `stream: true`; each text fragment is
published as an SSE `model_token` event and rendered immediately in the chat bubble. The
terminal completion record supplies time to first token, token count, prompt-token count,
throughput, load duration, total duration, warm/cold observation, and scheduler wait time.

Generation supports a total deadline, explicit `DELETE /api/tasks/{task_id}` cancellation,
and cancellation when the last SSE client disconnects. Final task state replaces the partial
display instead of appending it, so token text is not duplicated.

### Execution modes

`CreateTaskRequest.execution_mode` accepts `AUTOMATIC`, `FAST`, `STANDARD`, or `DEEP`.
Automatic selection is deterministic and auditable: short factual lookups use FAST; coding,
multiple evidence inputs, and consequential multi-step requests use DEEP; ordinary requests
use STANDARD. FAST has one generation step, STANDARD keeps preparation/generation/
verification, and DEEP adds a completeness review. Requested/selected mode and reason appear
in persisted state, audit records, SSE, API output, and the workbench.

### Resource admission and model lifecycle

The process-local scheduler admits one generative model job at a time by default and exposes
priority, role, memory requirement, mode, queue depth, and queue wait. CPU job capacity is a
separate configurable semaphore. `/api/models/status` now reports observed CPU/RAM, optional
NVIDIA VRAM, queue state, model warmth, lifecycle state, and recent generation metrics.

Lifecycle states are `NOT_INSTALLED`, `COLD`, `LOADING`, `READY`, `BUSY`, `IDLE`, and `ERROR`.
Ollama still owns model loading and GPU memory; SovereignAI only observes `/api/ps`, controls
`keep_alive`, and records transitions. `SOVEREIGN_MODEL_IDLE_TIMEOUT_SECONDS` supplies the
default keep-alive duration; `SOVEREIGN_MODEL_KEEP_ALIVE` can override it with Ollama syntax.

### Versioned local cache

A replaceable `CacheBackend` now has a SQLite implementation in `cache_entries`. Keys include
the identities required for safe invalidation:

- OCR: file hash, language, confidence threshold, and preprocessing pipeline version.
- Embeddings: text hash, provider/model identity, dimension, and pipeline version.
- Retrieval: query hash, collection version, ACL scope, retriever version, and result limit.
- Vision: image hash, model, prompt hash, and result-schema version.
- Deterministic inspection: all input/collection hashes, workflow version, and rule version.

Cache hit/miss/entry counts are returned by `/api/models/status`. Final user-specific model
answers are intentionally not cached.

## Runtime configuration

The new environment variables are documented in `.env.example`:

- `SOVEREIGN_MAX_GPU_MODEL_JOBS`
- `SOVEREIGN_MAX_CPU_JOBS`
- `SOVEREIGN_MODEL_IDLE_TIMEOUT_SECONDS`
- optional `SOVEREIGN_MODEL_KEEP_ALIVE`
- `SOVEREIGN_MODEL_GENERATION_TIMEOUT_SECONDS`
- `SOVEREIGN_CACHE_ENABLED`

The defaults remain local-only and suitable for a memory-constrained workstation.

## Verification

| Check | Result |
|---|---|
| Focused streaming/mode/scheduler/lifecycle/cache and SSE tests | 7 passed |
| Full backend suite | 43 passed |
| Python compilation | Passed |
| Frontend TypeScript | Passed |
| Next.js production build | Passed; `/`, `/metrics`, `/sovereignty`, `/_not-found` |
| Git whitespace validation | Passed |

The complete regression run used a two-second **test-process-only** generation timeout so a
running Ollama instance could not make unit tests wait for full CPU inference. Production
configuration remains 300 seconds. Direct NDJSON behavior and timing metrics are tested with
a deterministic mock Ollama transport.

## Honest limitations

- Scheduler and live task channels are process-local. Multiple API workers require a shared
  queue/event backend in a later deployment phase.
- Cancellation closes SovereignAI's HTTP stream; Ollama ultimately controls how promptly its
  underlying generation work is released.
- GPU memory is observed through `nvidia-smi` when available; no direct unload or VRAM claim
  is made.
- SQLite cache is appropriate for the single-node modular monolith, not a multi-node cluster.
- Retrieval remains the existing vector cosine path. Hybrid retrieval and GraphRAG are outside
  this batch and were not partially implemented.
