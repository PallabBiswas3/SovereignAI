# SovereignAI 2.0 — Batch 1 implementation checklist

## Phase 0 — Baseline

- [x] Inspect repository and configuration.
- [x] Locate API, orchestration, provider, router, tools, sandbox, RAG, OCR, vision,
  governance, artifacts, audit, SSE, and frontend boundaries.
- [x] Run backend tests.
- [x] Run Python compilation.
- [x] Run frontend typecheck.
- [x] Run frontend production build.
- [x] Record exact results and migration risks.

## Phase 1 — True streaming

- [x] Provider-neutral incremental chunk contract.
- [x] True Ollama NDJSON streaming.
- [x] Cancellation, total-timeout, and disconnected-client handling.
- [x] SSE token and generation-metric events without duplicate final output.
- [x] Available-model and runtime-stat provider methods.

## Phase 2 — Execution depth

- [x] Automatic/Fast/Standard/Deep request option.
- [x] Deterministic automatic mode selection.
- [x] Selected mode in state, audit, SSE, API, and frontend.
- [x] Fast path avoids an unnecessary full plan.

## Phase 3 — Resource scheduler foundation

- [x] One configurable GPU-model job at a time by default.
- [x] Configurable CPU capacity and model idle timeout.
- [x] Priority, role, mode, memory requirement, queue wait, and queue depth state.
- [x] Honest CPU/RAM and optional `nvidia-smi` VRAM observation.

## Phase 4 — Model lifecycle

- [x] NOT_INSTALLED/COLD/LOADING/READY/BUSY/IDLE/ERROR states.
- [x] Last use, load duration, TTFT, throughput, failure rate, memory, and queue fields.
- [x] Configurable Ollama keep-alive without claiming direct GPU control.

## Phase 5 — Cache foundation

- [x] Replaceable cache interface with SQLite implementation.
- [x] Versioned OCR, embedding, retrieval, vision, and deterministic-computation keys.
- [x] No caching of user-specific final model answers.
- [x] Cache hit/miss observability and focused correctness tests.

## Batch gate

- [x] Focused Batch 1 tests pass (7 tests including the prior SSE regression).
- [x] Full backend regression passes (43 tests).
- [x] Python compilation passes.
- [x] Frontend typecheck and production build pass.
- [x] Documentation states limitations honestly.
