# EXP-011 — ControlPlane factuality v2 hardening

**Date:** 2026-09-21  
**Status:** Measured; partial success, calibration work still required  
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

Result:

```text
26 passed
```

The first run produced one expected legacy-test mismatch because that test assumed `unsupported_entity_claim` was always emitted. The production behavior was intentionally changed; the legacy test was corrected to opt into the heuristic so it continues testing entity parsing rather than the old default policy behavior.

## Integrated 1,000-case validation result

Command:

```powershell
$env:CONTROLPLANE_ADAPTIVE_VERIFICATION='1'
$env:CONTROLPLANE_NLI_MODEL='cross-encoder/nli-deberta-v3-small'
..\.venv\Scripts\python.exe scripts\evaluate_controlplane.py `
  --scenarios data\controlplane\generated\completed_pack_v1\controlplane_batch_generation\validation.jsonl `
  --output results\controlplane\factuality_v2_validation_integrated.json
```

Measured result:

```text
n: 1000
action accuracy: 65.60%
unsafe-release rate: 0.671%
over-intervention rate: 49.02%
human-review rate: 26.10%
hallucination precision: 46.86%
hallucination recall: 100.00%
hallucination F1: 63.82%
claim_contradicted precision: 59.61%
claim_contradicted recall: 98.37%
claim_contradicted F1: 74.23%
claim_unknown precision: 32.52%
claim_unknown recall: 80.71%
claim_unknown F1: 46.36%
median latency: 103.7 ms
p95 latency: 520.2 ms
p99 latency: 838.2 ms
mean latency: 147.0 ms
```

Comparison with the previous integrated validation baseline:

| Metric | Previous | Factuality v2 | Interpretation |
|---|---:|---:|---|
| Action accuracy | 68.70% | 65.60% | regressed |
| Unsafe-release rate | 0.67% | 0.67% | effectively unchanged |
| Over-intervention | 49.02% | 49.02% | unchanged |
| Human-review rate | 23.30% | 26.10% | regressed |
| Hallucination F1 | 62.63% | 63.82% | modest improvement |
| Median latency | 151.5 ms | 103.7 ms | improved ~31.6% |
| p95 latency | 760.4 ms | 520.2 ms | improved ~31.6% |

The entity-rule change did not materially reduce over-intervention because the dominant remaining source is `claim_unknown`, which produced 469 false-positive scenarios on this validation run.

Contradiction handling is substantially stronger than unknown handling: `claim_contradicted` reached ~74.2% F1 with 98.4% recall, while `claim_unknown` remained noisy at ~46.4% F1.

## Interpretation

The experiment is a **partial success**:

- shared Phase-5 reuse delivers a clear latency improvement;
- unsafe-release behavior remains essentially unchanged;
- hallucination F1 improves modestly;
- over-intervention does not improve;
- action accuracy and human-review burden regress;
- the next bottleneck is UNKNOWN calibration, not contradiction detection.

Therefore PR #7 is not yet ready to be accepted or merged solely on this result.

## New instrumentation

The evaluator now records `unknown_reason_metrics` so the next rerun can attribute `claim_unknown` false positives to reasons such as:

```text
no_evidence
conflicting_evidence
verifier_uncertain
insufficient_support
```

This is required before tuning thresholds or policy because aggregate `claim_unknown` counts are not specific enough to identify the correct fix.

## Next experiment

1. Pull the updated evaluator.
2. Rerun the same 1,000-case integrated validation set.
3. Inspect `unknown_reason_metrics`.
4. Change only the dominant noisy reason(s), preserving strict handling for consequential conflicting evidence and genuine no-evidence cases.
5. Rerun validation before touching the frozen 2,000-case test.

Do **not** tune directly against the frozen test split.

## Decision

**Keep the architectural cleanup and shared Phase-5 reuse. Do not yet accept the current UNKNOWN-policy behavior. Continue calibration on the validation set.**
