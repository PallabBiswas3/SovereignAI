# EXP-013 — Authorized/RAG evidence distillation

**Date:** 2026-09-21  
**Status:** Implemented-unverified  
**Area:** GraphRAG / authorized generation / local inference latency  
**Branch:** `optimize-rag-evidence-distillation-v2`  
**PR:** #10

## Problem

The first Authorized/RAG latency experiment reduced mean total latency from roughly 185 s to ~139 s, but the routine 384-token output cap truncated all 3/3 measured answers and TTFT remained around 63 s. The dominant remaining problem is therefore the large model-facing authorized prompt/prefill path, not only decode length.

The previous measured authorized prompt was roughly 1,286 tokens versus about 85 tokens for general chat.

## Hypothesis

Keep retrieval, authorization, ranking, provenance and ControlPlane unchanged, but replace raw/per-chunk context exposure with a query-aware evidence packet. A smaller evidence packet should reduce prompt evaluation/TTFT without reducing factual support.

## Implementation

`EvidenceDistiller` runs after authorized retrieval/reranking and before model context compilation. It is deterministic and local.

It prioritizes:

- query-term overlap;
- asset/technical identifier overlap;
- numerical values and engineering units;
- limits, thresholds, alarm/trip/setpoint language;
- requirements and negation;
- source diversity.

It also:

- removes near-duplicate sentences;
- preserves source/document provenance because the original `RetrievedChunk` remains attached to every selected excerpt;
- detects differing numerical evidence across revisions of the same file/section and reserves space for the conflicting revision instead of silently hiding it through top-k truncation;
- exposes selection metrics through `ContextCompilationMetrics`.

GraphRAG retrieval internals are intentionally unchanged.

## Feature controls

The feature is **off by default** until measurement accepts a concrete configuration:

```text
SOVEREIGN_CONTEXT_DISTILLATION_ENABLED=false
SOVEREIGN_CONTEXT_DISTILLATION_TARGET_TOKENS=700
SOVEREIGN_CONTEXT_DISTILLATION_SENTENCE_REDUNDANCY_THRESHOLD=0.82
```

`700` is an experiment target, not an accepted production policy.

## Deterministic regression gate

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_evidence_distillation.py
```

The tests cover:

1. preservation of a query-relevant technical threshold;
2. preservation of conflicting revisions even when one falls below the normal chunk cutoff;
3. enforcement of the configured distilled evidence-token budget;
4. default-off compatibility with the existing context path.

## Local A/B protocol

Keep the same model, retrieval corpus, prompt set, machine state and output budget for every run. Do not re-use the rejected 384-token output cap as the final configuration.

### Control

```powershell
$env:SOVEREIGN_CONTEXT_DISTILLATION_ENABLED='false'
.\scripts\Stop-Workspace.ps1
.\scripts\Start-Workspace.ps1
.\.venv\Scripts\python.exe scripts\benchmark_system.py --runs 3
```

### 900-token evidence target

```powershell
$env:SOVEREIGN_CONTEXT_DISTILLATION_ENABLED='true'
$env:SOVEREIGN_CONTEXT_DISTILLATION_TARGET_TOKENS='900'
.\scripts\Stop-Workspace.ps1
.\scripts\Start-Workspace.ps1
.\.venv\Scripts\python.exe scripts\benchmark_system.py --runs 3
```

Repeat the same procedure for `700` and `500` evidence-token targets.

## Record for each configuration

- actual model prompt tokens;
- selected evidence tokens and source count;
- TTFT p50/p95;
- total latency p50/p95;
- output tokens and termination reason;
- CPU/RAM before and after each run;
- citation/source correctness;
- retention of numerical values, units, exceptions and conflicting revisions;
- answer completeness/truncation;
- unsupported claims;
- ControlPlane action.

## Acceptance rule

A distilled configuration is accepted only if it materially lowers model-facing prompt size and TTFT while:

- avoiding systematic answer truncation;
- preserving citations/provenance;
- retaining the evidence needed for the benchmark questions;
- preserving conflict evidence;
- not increasing unsupported claims;
- not degrading ControlPlane release behavior.

Do not choose a target solely because it is fastest.

## Next step after acceptance

Only after the evidence-distillation experiment is accepted should we consider deeper retrieval changes such as adaptive retrieval depth, query decomposition changes, or reranker/top-k changes. This keeps the experiment attributable: first optimize representation, then retrieval if needed.
