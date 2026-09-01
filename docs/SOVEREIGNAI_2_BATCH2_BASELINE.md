# SovereignAI 2.0 Batch 2 baseline

Verified on 2 September 2026 before Batch 2 implementation.

## Exact verification results

| Check | Result |
|---|---|
| Backend regression | 43 passed |
| Python compilation | Passed |
| Frontend TypeScript | Passed |
| Next.js production build | Passed; `/`, `/metrics`, `/sovereignty`, `/_not-found` |

The backend regression used a two-second test-process-only generation timeout so the two
legacy API tests do not wait on live Ollama CPU inference. Production remains configured for
300 seconds. Existing warnings are Starlette's deprecated `httpx` TestClient bridge and the
known Torch/NumPy `_ARRAY_API` initialization warning.

## Existing evidence path

- `KnowledgeIngestionService` extracts and provenance-chunks local documents, embeds every
  chunk, and stores text, page, section, metadata, and vectors in SQLite.
- `LocalRetriever` is a dense cosine retriever. Its cache identity currently includes query,
  corpus checksum state, access-scope string, dense retriever version, and limit.
- Retrieval enters the inspection workflow, knowledge endpoints, knowledge-search tool, demo,
  grounding checker, and evaluation runner. The general free-form agent does not currently
  compile retrieved evidence into its prompt.
- The inspection workflow extracts readings, retrieves SOP chunks, applies pump-specific
  Python comparisons, creates provenance sources, and generates artifacts. Evidence is mostly
  dictionaries rather than reusable typed entities.
- `GroundingChecker` sentence-splits generated output and combines lexical, semantic, numeric,
  and retrieval signals. It does not provide centralized schema/numerical/unit/rule verifiers.
- Batch 1 caches embeddings, dense retrieval, OCR, vision, and deterministic inspection results.
  Hybrid retrieval must use a new identity so dense-only entries cannot be reused.
- Revision and access-scope values can be carried in existing document/chunk metadata, but
  neither revision conflicts nor actual authorization filtering are implemented yet.

## Batch 2 implementation plan

1. Keep `LocalRetriever` as the compatible dense boundary; add BM25, RRF, hybrid DTOs,
   access-scope filtering, versioned cache identity, and optional reranking beside it.
2. Add a CPU-local cross-encoder abstraction with no-download loading and fusion fallback.
3. Introduce generic evidence types and a token-budgeted `ContextCompiler`.
4. Refactor the flagship inspection path to acquire requirements, normalize measurements,
   create deterministic calculations/claims, detect revision conflicts, and run pluggable
   verification before recommendations are exposed.
5. Add bounded DEEP query decomposition and measured dense/hybrid/reranked benchmarks.
6. Surface claims, calculations, evidence, and conflicts in the existing workbench, then run
   all focused and regression gates. No Batch 3 component will be started.

## Dependency observation

The environment already contains Transformers and a local MiniLM embedding model. The preferred
MS MARCO cross-encoder, Pint, `sentence-transformers`, and `rank_bm25` are not installed locally.
Batch 2 therefore requires an honest reranker fallback and either an offline-prepared reranker
directory or later package/model staging. No runtime download may occur.
