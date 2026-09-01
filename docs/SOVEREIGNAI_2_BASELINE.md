# SovereignAI 2.0 Batch 1 baseline

Verified on 1 September 2026 before Batch 1 implementation.

## Existing architecture

SovereignAI is a modular-monolith FastAPI backend with a Next.js frontend. SQLite stores
conversations, task state, knowledge chunks, artifacts, approvals, audit records, network
events, evaluation snapshots, and persisted SSE events. Local Ollama roles are configured in
YAML and selected by a deterministic task classifier and capability-scored router. Bounded
agent, inspection, coding, OCR, vision, RAG, artifact, governance, and approval paths remain
separate modules behind typed interfaces.

The boundaries preserved by Batch 1 are:

- `app.llm` owns provider-neutral inference contracts and the local Ollama adapter.
- `app.router` owns task classification, model definitions, readiness, and routing.
- `app.agent` owns transparent plans, state, and bounded execution.
- `app.api.tasks` owns task admission, persistence, audit, and SSE lifecycle publication.
- `app.rag`, `app.multimodal`, `app.workflows`, and `app.sandbox` retain their current
  deterministic and security boundaries.
- The backend remains a modular monolith; no cloud dependency or new large model is added.

## Exact baseline results

| Check | Command | Result |
|---|---|---|
| Python compilation | `.\.venv\Scripts\python.exe -m compileall -q backend\app tests` | Passed |
| Backend regression | `.\.venv\Scripts\python.exe -m pytest -q` | 37 passed |
| Frontend types | `npm run typecheck` | Passed |
| Frontend production build | `npm run build` | Passed; `/`, `/metrics`, `/sovereignty`, and `/_not-found` generated |

The backend run emitted two pre-existing warnings: Starlette's deprecated `httpx` test-client
bridge and a Torch/NumPy `_ARRAY_API` initialization warning. Neither failed the baseline.

## Pre-existing working-tree state

Before Batch 1, `frontend/tsconfig.tsbuildinfo` was already modified and
`workspace/approval-tests/af59edf3-1934-49eb-9863-9327eef70241.md` was already untracked.
They are treated as user-owned files and are not intentionally reverted or removed.

## Batch 1 migration risks

- Streaming can accidentally publish a token and then duplicate it in the final event.
- Client disconnect cancellation can leave an Ollama response or scheduler slot active.
- New execution-mode fields can break persisted-state and API compatibility.
- A process-local scheduler can deadlock if a permit is not released on error/cancellation.
- Lifecycle state must not claim that Ollama unloaded a model when only observed state exists.
- Cache keys must include pipeline/model/version identity and must not cache final user output.

