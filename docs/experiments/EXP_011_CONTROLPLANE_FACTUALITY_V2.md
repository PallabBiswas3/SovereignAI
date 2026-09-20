# EXP-011 — ControlPlane factuality v2 hardening

**Date:** 2026-09-21  
**Status:** Implemented-unverified at system-evaluation level; focused regression suite passed locally  
**Area:** ControlPlane / hallucination detection / factuality verification  
**Branch:** `controlplane-factuality-v2`  
**PR:** #7

## Problem

The existing ControlPlane factuality architecture was conceptually sound but had two important engineering problems:

1. the lightweight hallucination detector and `AdaptiveFactVerificationService` both repeated Phase-5 claim extraction and deterministic verification;
2. the `unsupported_entity_claim` heuristic was a major source of false-positive hallucination findings and contributed to over-intervention.

The previous completed-pack integrated evaluation showed:

```text
validation cases: 1,000
action accuracy: 68.70%
unsafe-release rate: 0.67%
over-intervention rate: 49.02%
human-review rate: 23.30%
hallucination F1: 62.63%
median latency: 151.5 ms
p95 latency: 760.4 ms
```

The frozen 2,000-case test showed:

```text
action accuracy: 68.00%
unsafe-release rate: 0.67%
over-intervention rate: 50.29%
hallucination precision: 42.19%
hallucination recall: 99.87%
hallucination F1: 59.32%
median latency: 247.3 ms
p95 latency: 1,267.6 ms
```

These synthetic labels are engineering evidence, not publication-grade independent ground truth.

## Hypothesis

Reusing one canonical Phase-5 state and demoting noisy entity-mismatch detection to a diagnostic signal should reduce duplicated work and false interventions without materially increasing unsafe release.

## Constraints / invariants

- Do not redesign GraphRAG internals.
- Preserve QUICK / STANDARD / DEEP verification architecture.
- Preserve deterministic checks, NLI verification and bounded evidence-agent escalation.
- Preserve policy-visible `claim_unknown` subtype so existing policy rules keep working.
- Preserve fail-closed behavior for consequential unresolved claims.
- Treat the trained hallucination model as a verification-routing prior, not an authoritative truth decision.

## Implementation

### Shared factuality state

Added `PreparedFactuality`, containing the canonical `ResponseRecord` and Phase-5 runtime.

The hallucination detector now prepares Phase 5 once and both the lightweight detector and AdaptiveFact verifier can reuse the exact same claims/evidence state.

Direct callers of `AdaptiveFactVerificationService.verify(...)` still work without a prepared artifact; the verifier falls back to constructing and running Phase 5 itself.

### Entity mismatch calibration

`unsupported_entity_claim` is now diagnostic-only by default.

The finding can still be explicitly restored with:

```text
emit_unsupported_entity_findings = true
```

This retains the heuristic for debugging/ablation while preventing it from automatically driving ordinary policy outcomes.

### UNKNOWN reason taxonomy

Unresolved claim findings now retain a reason such as:

```text
no_evidence
conflicting_evidence
verifier_uncertain
unverified
deterministic_conflict_candidate
insufficient_support
```

The policy-facing subtype remains `claim_unknown` for backward compatibility.

### Structured grounding evidence

Added a typed `GroundingEvidence` contract carrying:

```text
evidence_id
text
source_name
document_id
chunk_id
page
revision
retrieval_score
authorization_scope
metadata
```

The current AdaptiveFact retrieval path remains text-compatible by rendering typed evidence into the grounding context. This establishes a provenance-aware GraphRAG → ControlPlane contract without redesigning GraphRAG retrieval.

## Focused local regression result

Command:

```powershell
..\.venv\Scripts\python.exe -m pytest -q `
    tests\test_factuality_v2.py `
    tests\test_controlplane_adaptive_integration.py `
    tests\test_controlplane.py
```

Result after updating the legacy entity test to opt into the historical heuristic explicitly:

```text
26 passed
```

The first run produced one expected legacy-test mismatch because that test assumed `unsupported_entity_claim` was always emitted. The production behavior was intentionally changed; the legacy test was corrected to opt into the heuristic so it continues testing entity parsing rather than the old default policy behavior.

## Current interpretation

The code-level compatibility/regression gate is passed.

This does **not** yet prove that factuality v2 improves safety/calibration. The next gate is a before/after evaluation on the existing 1,000-case validation split.

## Next experiment / measurement gate

Run the same validation workload with separate output filenames so the previous baseline artifacts are preserved.

### Lightweight factuality v2 validation

```powershell
Remove-Item Env:CONTROLPLANE_ADAPTIVE_VERIFICATION -ErrorAction SilentlyContinue
..\.venv\Scripts\python.exe scripts\evaluate_controlplane.py `
  --scenarios data\controlplane\generated\completed_pack_v1\controlplane_batch_generation\validation.jsonl `
  --output results\controlplane\factuality_v2_validation_lightweight.json
```

### Integrated DeBERTa factuality v2 validation

```powershell
$env:CONTROLPLANE_ADAPTIVE_VERIFICATION='1'
$env:CONTROLPLANE_NLI_MODEL='cross-encoder/nli-deberta-v3-small'
..\.venv\Scripts\python.exe scripts\evaluate_controlplane.py `
  --scenarios data\controlplane\generated\completed_pack_v1\controlplane_batch_generation\validation.jsonl `
  --output results\controlplane\factuality_v2_validation_integrated.json
```

## Metrics to compare

Primary safety/calibration metrics:

```text
unsafe-release rate
over-intervention rate
human-review rate
action accuracy
hallucination precision
hallucination recall
hallucination F1
unknown-claim precision / recall
unsupported-entity false positives
```

Performance metrics:

```text
median latency
p95 latency
p99 latency
```

## Acceptance criteria

The change should not be accepted merely because tests pass.

Preferred outcome:

1. unsafe release remains near the current integrated baseline and does not materially regress;
2. over-intervention decreases from the previous ~49% validation result;
3. hallucination precision improves without collapsing recall;
4. human-review burden decreases or remains justified by real unresolved evidence;
5. shared Phase-5 reuse does not break STANDARD/DEEP verification;
6. latency should be neutral or better after removing duplicated Phase-5 work.

If the validation comparison is favorable, run the frozen 2,000-case test once and record that as the final pre-merge evaluation. Do not repeatedly tune against the frozen test split.

## Decision

**Pending validation-set measurement.**
