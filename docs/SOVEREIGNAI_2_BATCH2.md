# SovereignAI 2.0 — Batch 2

Evidence Intelligence and Small-Model Amplification, completed 2 September 2026.

Batch 2 improves retrieval and deterministic evidence handling without changing the configured
4B general/vision model or 7B coder model. It adds no cloud service, fine-tuning, GraphRAG,
PostgreSQL, LDAP, full RBAC, or Batch 3 component.

## Architecture

```text
Task / query
  ├─ deterministic DEEP decomposition (DEEP only, max 4)
  ├─ dense cosine retrieval (existing local embedding provider)
  ├─ local BM25 retrieval
  └─ Reciprocal Rank Fusion
       └─ CPU cross-encoder when staged locally
            └─ access-scope filter (performed before ranking)
                 └─ typed retrieval candidates
                      └─ ContextCompiler
                           ├─ EvidenceSource / EvidenceFragment
                           ├─ Measurement / Rule / Calculation
                           ├─ conflict and requirement checks
                           └─ bounded structured context
                                └─ small local LLM for synthesis where needed
                                     └─ Claim objects
                                          └─ ordered VerificationEngine
```

The operational rule is: models propose, evidence supports, Pint/Python normalizes and
calculates, policies constrain, and humans approve consequential action.

## Hybrid retrieval

`LocalRetriever` remains the backwards-compatible dense retriever. `BM25Retriever` implements
fully local Okapi BM25 against SQLite chunks and retains document/chunk IDs, file, page,
section, revision, hash, classification, department, text, and access scope.

`HybridRetriever` runs both branches and applies Reciprocal Rank Fusion:

```text
RRF(d) = sum(1 / (k + rank_i(d)))
```

The default `k` is 60. Raw cosine and BM25 values are never directly added. A result contains
separate dense, sparse, fusion, and optional reranker scores plus retrieval methods. Stable
source fields remain compatible with existing consumers.

Scope is currently a prototype metadata filter, not full authorization. Filtering happens
before scoring, and scope is included in cache identity so later RBAC can replace the policy
without redesigning retrieval.

## Reranker

`Reranker` is provider-neutral. `LocalCrossEncoderReranker` uses Transformers directly on CPU,
under Batch 1 CPU admission. Its default identity is
`cross-encoder/ms-marco-MiniLM-L-6-v2`. Loading is lazy and accepts only a local directory or an
existing Hugging Face cache snapshot. It checks local files before importing/model loading;
runtime downloads are disabled.

The model is not present on this development machine. Current runtime behavior is therefore:

```text
RERANKER_UNAVAILABLE → preserve RRF order → return successful hybrid retrieval
```

`/api/models/status` reports the reranker identity, version, CPU device, local availability,
offline policy, and fallback. Retrieval telemetry records candidate count, output count,
duration, availability, and warning.

### Offline preparation

On a controlled connected staging machine, download the cross-encoder to a directory, verify
its files/hashes, and transfer it through the approved media process. One possible staging
command is:

```powershell
huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 --local-dir models/reranker/ms-marco-MiniLM-L-6-v2
```

On the offline machine set:

```powershell
$env:SOVEREIGN_RERANKER_MODEL="C:\approved-models\ms-marco-MiniLM-L-6-v2"
```

Pint and all Python wheels must likewise be staged before air-gap transfer. No runtime network
access is required.

## Hybrid cache identity

Hybrid results cannot reuse Batch 1 dense-only entries. The new identity contains:

- query hash and collection/index version;
- embedding provider/model identity;
- dense and BM25 versions;
- fusion strategy/version and RRF limits;
- reranker identity, version, availability, and local `config.json` hash when installed;
- normalized access scope;
- dense, sparse, fusion, rerank, and final limits.

Installing or changing reranker weights therefore invalidates fallback cache results.

## Context Compiler

`ContextCompiler` does not concatenate raw chunks. It:

1. orders by revision and retrieval strength;
2. removes exact duplicates;
3. removes near duplicates only within the same document;
4. preserves conflicting revisions;
5. retains technical identifiers, values, and engineering units;
6. selects the most relevant sentences when a chunk is too large;
7. enforces model-window, output-reserve, evidence-token, and evidence-count budgets;
8. emits typed structured context and metrics without hidden reasoning.

Metrics include raw/reranked/final counts, raw and compiled token estimates, deduplicated and
dropped counts, compression ratio, and compilation duration. FAST is limited to three evidence
chunks and 1,000 evidence tokens. STANDARD uses the normal configured budget. DEEP may add up
to four bounded retrieval subqueries before candidate deduplication.

## Structured evidence and evidence-first execution

Domain-neutral Pydantic models now represent:

- `EvidenceSource` and `EvidenceFragment`;
- `Measurement` retaining original and normalized values;
- `Rule` and revision-aware `RuleSource`;
- deterministic `Calculation`;
- material `Claim` and support status;
- `Finding`, `Recommendation`, `EvidenceConflict`, and `EvidenceRequirement`;
- `EvidenceBundle` and typed failure codes.

The flagship inspection workflow now acquires measurements and relevant requirements first,
extracts explicit rules, normalizes units, performs Python comparisons, constructs claims,
verifies them, compiles context, and then exposes the recommendation. Threshold comparisons
are not delegated to Qwen.

If a required rule or measurement is absent, the bundle contains `INSUFFICIENT_EVIDENCE` and
the recommendation requests the missing evidence. Different numerical limits across stated
revisions produce `DOCUMENT_REVISION_CONFLICT`; no revision is silently selected.

## Verification

`VerificationEngine` runs small pluggable verifiers in this order:

1. schema;
2. evidence references;
3. numerical recomputation;
4. dimensional-unit compatibility;
5. rule/calculation linkage;
6. semantic lineage.

The existing `GroundingChecker` remains for free-form semantic assessment. Specialized claims
are constructed directly from measurements, rules, and calculations. Material unsupported
claims are represented as `UNSUPPORTED` or `INSUFFICIENT_EVIDENCE`, rather than silently
passing as facts. The reported score is support/grounding confidence, never “hallucination
probability.” No optional second Qwen verification pass is enabled in this batch because the
flagship calculations resolve deterministically.

## Unit normalization

`UnitService` uses local Pint dimensional analysis. It supports the requested pressure,
temperature, length, velocity, power, force, mass, frequency, and rotational-speed units.
Original values/units are retained and normalized values/units are added. Examples:

```text
500 kPa → 5.0 bar
293.15 K → 20.0 °C
60 rpm → 1.0 Hz
```

Missing units return `UNIT_AMBIGUOUS`; incompatible dimensions return `INCOMPATIBLE_UNITS`.
Rounding is bounded to source-appropriate precision to avoid floating-point display artifacts.

## APIs, audit, SSE, and frontend

Existing task endpoints and event types remain. `AgentRunState` adds optional-compatible lists
for measurements, rules, calculations, claims, conflicts, retrieval metrics, and context
metrics. New SSE events are `calculation_completed`, `claim_verified`, and
`evidence_conflict`. Audit records capture the same structured objects.

The workbench inspector adds “Why this answer?”, Measurements, Rules & Calculations, and
Evidence Conflicts. It renders backend objects and never parses model prose. The metrics page
shows the versioned Batch 2 comparison and context preservation result.

## Configuration

| Setting | Default |
|---|---:|
| `SOVEREIGN_HYBRID_DENSE_TOP_K` | 30 |
| `SOVEREIGN_HYBRID_SPARSE_TOP_K` | 30 |
| `SOVEREIGN_HYBRID_FUSION_CANDIDATE_LIMIT` | 50 |
| `SOVEREIGN_HYBRID_RRF_K` | 60 |
| `SOVEREIGN_HYBRID_RERANK_TOP_K` | 10 |
| `SOVEREIGN_HYBRID_FINAL_CONTEXT_K` | 5 |
| `SOVEREIGN_DENSE_RETRIEVER_VERSION` | `cosine-v2` |
| `SOVEREIGN_BM25_INDEX_VERSION` | `bm25-v1` |
| `SOVEREIGN_FUSION_STRATEGY_VERSION` | `rrf-v1` |
| `SOVEREIGN_RERANKER_ENABLED` | `true` |
| `SOVEREIGN_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `SOVEREIGN_RERANKER_LOCAL_FILES_ONLY` | `true` |
| `SOVEREIGN_RERANKER_VERSION` | `cross-encoder-v1` |
| `SOVEREIGN_CONTEXT_MAX_FRACTION_OF_WINDOW` | 0.60 |
| `SOVEREIGN_CONTEXT_OUTPUT_RESERVE_TOKENS` | 1024 |
| `SOVEREIGN_CONTEXT_MAX_EVIDENCE_CHUNKS` | 8 |
| `SOVEREIGN_CONTEXT_MAX_EVIDENCE_TOKENS` | 3000 |
| `SOVEREIGN_CONTEXT_NEAR_DUPLICATE_THRESHOLD` | 0.90 |
| `SOVEREIGN_MAX_RETRIEVAL_SUBQUERIES` | 4 |

## Measured evaluation

Benchmark `2026.09-batch2-v1` uses the same six-case synthetic corpus for exact identifier,
paraphrase, numeric, section/standard, unanswerable, and conflicting-revision queries.

One recorded run produced:

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Citation precision | Unsupported refusal |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Hybrid | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Hybrid + configured reranker | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

The third row used honest RRF fallback because reranker weights were unavailable, so reranker
improvement is `null`, not zero or a fabricated gain. The small fixture is intentionally not a
production relevance claim.

Context evaluation reduced 20 candidates/444 estimated evidence tokens to 5 fragments and 116
compiled tokens, deduplicated 15 repeated chunks, achieved a 0.2584 token ratio, and retained
the required limit fact. Deterministic claim-verification correctness was 1.00 on its two-case
fixture. Compression ratio alone is not treated as semantic quality.

## Verification gate

- Complete backend regression: **68 passed**.
- Python compilation: passed.
- Frontend TypeScript: passed.
- Next.js production build: passed for `/`, `/metrics`, `/sovereignty`, `/_not-found`.
- No cloud runtime dependency added.

## Known limitations and migration concerns

- The configured cross-encoder is not staged on this machine; current production behavior is
  hybrid RRF fallback until weights are transferred locally.
- Access scope is metadata filtering, not identity-based authorization or full RBAC.
- BM25 is rebuilt from the SQLite corpus for each search; this is appropriate for the current
  small single-node corpus, not a large enterprise index.
- Token counts are deterministic estimates, not the exact tokenizer count of every model.
- Generic free-form claims still use the existing grounding path; full schema-constrained local
  claim extraction is not forced for every chat response.
- Document revision applicability cannot be inferred when metadata does not state it; the
  system surfaces a conflict for review.
- SQLite and process-local admission remain single-node foundations.
- Pint is a new Python dependency; offline deployment bundles must include Pint, flexcache,
  flexparser, and platformdirs wheels.

Batch 3 was not started. The recommended next step is to stage and benchmark the configured
cross-encoder on target CPU hardware, expand the labeled corpus, and only then design Batch 3
Workcell Pack/Evidence Capsule concepts against measured needs.
