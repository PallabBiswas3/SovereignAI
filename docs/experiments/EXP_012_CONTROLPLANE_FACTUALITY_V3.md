# EXP-012 — ControlPlane factuality v3

**Date:** 2026-09-21  
**Status:** Implemented-unverified  
**Area:** ControlPlane / factuality / GraphRAG evidence verification  
**Branch:** `controlplane-factuality-v3`  
**Base:** `controlplane-factuality-v2`

## Problem

Factuality v2 removed duplicated Phase-5 work and substantially reduced latency, but the 1,000-case integrated validation still showed an over-conservative factuality layer:

```text
action accuracy:        65.60%
unsafe-release rate:     0.67%
over-intervention rate: 49.02%
human-review rate:      26.10%
hallucination F1:       63.82%
claim_unknown precision:32.52%
claim_unknown recall:   80.71%
claim_unknown FP:       469
p50 latency:           ~102 ms
p95 latency:           ~498 ms
```

The primary quality problem was not contradiction recall. It was treating too many unresolved claims as one broad `UNKNOWN` class and allowing that ambiguity to drive policy interventions.

## Research-grounded design principles

Factuality v3 follows several recurring ideas in claim-verification research:

1. verify fine-grained claims rather than whole responses;
2. decontextualize extracted claims so they remain self-contained;
3. keep claim structure useful for retrieval/audit;
4. distinguish explicit contradiction from absence of support;
5. abstain when evidence is insufficient rather than converting low support into a contradiction;
6. use specialized support/alignment scorers as support signals, not universal contradiction detectors;
7. keep evidence provenance and evidence sufficiency visible to policy.

These principles motivate the implementation but do not imply that the current heuristic extractor or thresholds reproduce any paper exactly.

## Architecture

```text
LLM response
    ↓
atomic + decontextualized claim extraction
    ↓
structured claim
(subject / predicate / object / qualifiers)
    ↓
cheap deterministic Phase 5
    ↓
risk prior → routing only
    ├─ QUICK
    ├─ STANDARD: retrieval + DeBERTa NLI
    └─ DEEP: bounded evidence-search agent
    ↓
GraphRAG structured evidence + provenance
    ↓
evidence quality gate
    ↓
optional batched support scorer
(MiniCheck or Align-style)
    ↓
SUPPORTED / CONTRADICTED / UNSUPPORTED / UNDECIDABLE / CONFLICTING
    ↓
policy
```

## Implementation

### 1. Atomic and decontextualized claims

`ClaimExtractor` now:

- splits sentence/clause-sized factual units more aggressively;
- keeps original and parent spans;
- resolves simple leading pronouns using the preceding subject/entity;
- records a lightweight structured representation:

```text
subject
predicate
object
qualifiers
```

The verifier retrieves and checks `claim.verification_text`, which prefers the self-contained decontextualized form.

This is a deterministic low-latency extractor. It is intentionally not described as perfect semantic decomposition.

### 2. Richer factuality state space

Added explicit claim states:

```text
SUPPORTED
CONTRADICTED
UNSUPPORTED
UNDECIDABLE
CONFLICTING
```

Legacy `UNKNOWN` remains in the internal schema for backward compatibility with older data/pipelines.

Semantics:

- `CONTRADICTED`: evidence explicitly contradicts the claim.
- `UNSUPPORTED`: evidence is sufficiently complete/provenanced to support absence-of-support reasoning, and a support checker gives low support.
- `UNDECIDABLE`: evidence is missing, irrelevant, incomplete, or verification remains inconclusive.
- `CONFLICTING`: strong evidence supports materially different outcomes.

### 3. GraphRAG provenance handoff

The SovereignAI industrial integration boundary now translates GraphRAG claims/chunks into typed ControlPlane `grounding_evidence`.

Available fields are preserved when actually returned:

```text
evidence_id
document_id
chunk_id
source_name
page
revision
retrieval_score
authorization_scope
```

Missing provenance is omitted rather than invented. The existing serialized context is retained as a backward-compatible fallback.

GraphRAG internals were not changed.

### 4. Evidence-quality gate

Evidence is classified as:

```text
no_evidence
irrelevant_evidence
legacy_context
partial_structured
strong_structured
```

Legacy free-text context may support or contradict a claim, but it is not considered complete enough to infer `UNSUPPORTED` from absence of support.

Structured evidence becomes eligible for unsupportedness reasoning only when there is an explicit completeness signal or enough strong/provenanced evidence under the configured gate.

### 5. DeBERTa + support scorer separation

DeBERTa NLI remains the contradiction-capable path.

Optional support backends:

```text
MiniCheck
Align-style text alignment
```

are support-only. A low support score does **not** mean contradiction.

Support inference is batched once per response rather than invoked once per claim.

### 6. Adaptive routing

The lightweight risk model is now a compute-routing prior only.

By default it does not emit a normal hallucination finding.

For non-consequential interactions:

```text
low risk    -> QUICK
medium risk -> STANDARD
high risk   -> DEEP (capped by profile maximum)
```

A deterministic value/date conflict forces at least STANDARD verification.

Consequential interactions use the profile's configured maximum depth.

### 7. Consequence-aware policy

Ordinary ambiguity does not automatically require human review.

- confirmed contradictions remain strict;
- unsupported claims can warn/review depending on profile;
- conflicting/undecidable claims escalate to human review when consequential;
- if required deeper factuality verification is unavailable, the checker fails closed through the existing detector-failure policy.

## Optional backend APIs

MiniCheck is optional and loaded only when enabled:

```text
CONTROLPLANE_SUPPORT_BACKEND=minicheck
```

Align-style support checking is optional and requires the official checkpoint path:

```text
CONTROLPLANE_SUPPORT_BACKEND=alignscore
CONTROLPLANE_ALIGNSCORE_CHECKPOINT=<path>
```

The default remains DeBERTa-only so the base system does not silently acquire a new large runtime dependency.

## Evaluation caveat

The existing 15K synthetic ControlPlane corpus was created before the v3 state taxonomy. Its gold claim state is primarily `supported`, `contradicted`, or `unknown`.

Therefore it can evaluate:

- unsafe release;
- action accuracy;
- over-intervention;
- contradiction detection;
- unresolved-vs-resolved behavior;
- latency;

but **cannot establish semantic precision separately for `UNSUPPORTED`, `UNDECIDABLE`, and `CONFLICTING`**.

The evaluator therefore reports new state counts while mapping the old `unknown` label only to an aggregate unresolved metric. Independent state-level evaluation requires a new manually adjudicated/structured-evidence set.

## Acceptance criteria

V3 is not accepted until measurement shows:

1. unsafe release does not materially regress from the v2 ~0.67% validation result;
2. over-intervention materially improves from ~49.02%;
3. action accuracy recovers toward or above the earlier 68.70% integrated baseline;
4. contradiction recall remains strong;
5. ordinary non-consequential ambiguity no longer dominates warnings/reviews;
6. consequential unresolved/conflicting claims remain reviewable;
7. latency is measured separately for DeBERTa-only, MiniCheck-assisted and Align-assisted configurations;
8. GraphRAG provenance survives the integration boundary.

## Regression commands

From `Controlplane.ai`:

```powershell
..\.venv\Scripts\python.exe -m pytest -q `
  tests\test_factuality_v3.py `
  tests\test_factuality_v2.py `
  tests\test_controlplane_adaptive_integration.py `
  tests\test_controlplane.py
```

From `backend`:

```powershell
..\.venv\Scripts\python.exe -m pytest -q tests\test_grounding_evidence_adapter.py
```

## Validation

DeBERTa-only first:

```powershell
$env:CONTROLPLANE_ADAPTIVE_VERIFICATION='1'
$env:CONTROLPLANE_SUPPORT_BACKEND='none'
$env:CONTROLPLANE_NLI_MODEL='cross-encoder/nli-deberta-v3-small'

..\.venv\Scripts\python.exe scripts\evaluate_controlplane.py `
  --scenarios data\controlplane\generated\completed_pack_v1\controlplane_batch_generation\validation.jsonl `
  --output results\controlplane\factuality_v3_validation_deberta.json
```

Backend comparison:

```powershell
..\.venv\Scripts\python.exe scripts\benchmark_factuality_backends.py `
  --backends deberta,minicheck `
  --output results\controlplane\factuality_v3_backend_comparison.json
```

Add `alignscore` only after installing the official Align package and providing the local checkpoint:

```powershell
..\.venv\Scripts\python.exe scripts\benchmark_factuality_backends.py `
  --backends deberta,minicheck,alignscore `
  --align-checkpoint <path-to-Align-base.ckpt>
```

## Next independent evaluation work

Build an adjudicated industrial factuality set containing examples with explicit labels for:

```text
SUPPORTED
CONTRADICTED
UNSUPPORTED
UNDECIDABLE
CONFLICTING
```

Each example should include the evidence bundle that was actually available to the system, source provenance, and two-reviewer/adjudication metadata. Until that exists, the v3 state taxonomy should be treated as an engineering design under evaluation rather than publication-grade accuracy evidence.

## Decision

**Implemented-unverified. Do not merge until regression tests and validation measurements are recorded.**
