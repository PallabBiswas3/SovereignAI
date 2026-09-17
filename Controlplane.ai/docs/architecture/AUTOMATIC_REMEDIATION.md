# Phase 9 automatic remediation

## Goal

Phase 9 reduces human-review burden after Phase 7/8 verification without allowing an uncertain generative model to approve its own output. It assigns one auditable operational action:

- `release`: return the original response because every required claim is supported
- `corrected_release`: remove unresolved claims and return only exact, high-confidence supported claims
- `block`: suppress a response containing confirmed contradictions
- `safe_abstain`: return a standard evidence-unavailable response for a non-consequential unresolved case
- `human_review`: escalate unresolved consequential decisions

## Safety boundary

The default remediator is extractive, not generative. It cannot invent a correction. A corrected response contains only the exact text of claims already marked `SUPPORTED` at or above `remediation_min_supported_confidence`. Removed and retained claim IDs are recorded in `metadata.adaptive_pipeline.remediation`.

Safe abstention and blocking count as automated handling, but they are not useful answers. The benchmark therefore reports all of the following:

- `automation_rate`
- `human_review_rate`
- `useful_answer_rate`
- `supported_answer_retention_rate`
- `hallucination_containment_rate`
- `unsafe_original_release_rate`
- `safe_abstention_rate` and `block_rate`

This prevents a system from claiming success by automatically refusing every request.

## Policy behavior

Low-impact unresolved responses can be corrected extractively or safely declined. Consequential unresolved responses still require accountable human review. Confirmed contradictions are automatically blocked in both cases.

Configuration is under `dispatcher` in `configs/phase7_adaptive.yaml`:

```yaml
auto_remediation_enabled: true
remediation_min_supported_confidence: 0.90
auto_abstain_nonconsequential: true
```

## Evaluation

Run:

```bash
python scripts/run_phase7_adaptive.py --max-records 100
```

For an immediate controlled demonstration of correction, abstention, and review:

```bash
python scripts/run_phase9_remediation_demo.py
```

Read `metrics.automatic_remediation` in `results/phase7/phase7_summary.json`. RAGTruth is non-consequential benchmark content, so its automation rate measures the correction/block/abstention policy. A regulated-use-case evaluation must set `metadata.consequential: true`; uncertainty will then remain routed to a human.

The 90% automation objective is valid only alongside the usefulness and safety metrics. It is not a claim that 90% of original answers were proven factually correct.

## Pre-evidence-guard prototype baseline

Before the later Phase 6 evidence-relevance and contradiction-corroboration changes, the fixed seed-42 stratified sample of 100 RAGTruth test responses across 90 source documents produced:

| Metric | Result |
|---|---:|
| Automated handling | 100% |
| Human review | 0% |
| Useful unchanged/corrected answer | 52% |
| Hallucination containment | 100% |
| Unsafe original release | 0% |
| Supported-answer retention | 60% |
| Safe handling | 74% |
| Automatic block | 32% |
| Safe abstention | 16% |

This meets the operational goal of more than 90% handling without a human for non-consequential traffic. It does **not** meet a hypothetical goal of proving more than 90% of answers correct. The low useful-answer and supported-retention rates show that retrieval, claim verification, and false-positive reduction remain the next research priorities.

This test sample has since been inspected and must be treated as a historical baseline, not an unseen final evaluation of the current evidence-aware configuration. Current-config diagnostics are validation-only and documented in [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md).
