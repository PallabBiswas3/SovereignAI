# Architecture review

## Overall assessment

The project has a sound research-prototype structure, but it is not a finished production architecture. Its strongest design choice is the separation of factual status (`SUPPORTED`, `CONTRADICTED`, `UNKNOWN`) from operational action (release, correct, block, abstain, or review). Unknown evidence is never silently converted into factual certainty.

## What is designed well

- Provider-independent `ResponseRecord` and claim schemas
- Separate risk routing, deterministic verification, retrieval/NLI, agentic search, aggregation, and remediation stages
- Policy-specific ControlPlane profiles rather than one global threshold
- Bounded agent iterations, search, NLI, latency, and cost
- Approved evidence tools and source citations
- Audit traces and human-feedback storage
- Explicit abstention and consequential-use-case review
- Separate automation, usefulness, containment, and unsafe-release metrics
- Automated component and end-to-end tests

## Accuracy safeguards added after evaluation

- Retrieval now removes near-zero-similarity context chunks before NLI.
- A contradiction requires stronger retrieval relevance than ordinary evidence eligibility.
- Strong entailment and contradiction from different chunks is treated as conflicting evidence and remains `UNKNOWN`.
- Runtime and Phase 6.1 threshold tuning use the same evidence-aware decision function.
- Agent evidence must satisfy both authority and retrieval-relevance requirements.

These changes target false contradiction findings. They do not constitute a new accuracy result until thresholds are retuned on validation data and evaluated on a new source-separated holdout.

### Validation-only effect

The compatible Phase 6.1 cache was rebuilt on 300 source-safe validation responses after the retrieval change, and the expanded threshold sweep was rerun after contradiction corroboration:

| Diagnostic | Earlier artifact | Evidence-aware validation |
|---|---:|---:|
| NLI evidence pairs | 5,663 | 3,638 |
| NLI contradiction predictions | 245 | 14 |
| Predicted-contradiction conflict-span alignment | 7.76% | 42.86% |
| Baseless claims left unknown | 80.84% | 92.81% |
| NLI claim resolution | 29.42% | 18.67% |

This is a substantial precision-oriented improvement for contradiction diagnostics, paid for with lower coverage. No threshold candidate met every safety constraint: supported predictions still overlap hallucination spans at 4.47% against the configured 3% maximum. The selected artifact is therefore explicitly labelled as a fallback, not a fully safe operating point.

## Remaining limitations

1. **Claim atomicity:** the rule-based extractor can leave multiple facts in one sentence-sized claim.
2. **Retrieval quality:** TF-IDF is fast and auditable but lacks semantic reranking and multi-hop retrieval. Relevance filtering reduced noise but did not solve support errors.
3. **Model domain fit:** the small NLI model is generic and has not been fine-tuned on enterprise claim/evidence pairs.
4. **Ground truth:** RAGTruth has response hallucination spans, not complete atomic-claim labels. Exact agent claim accuracy therefore cannot be claimed.
5. **Evaluation reuse:** the seed-42 100-response set has already been inspected and must not be used for further tuning. Final claims need a new source-separated holdout.
6. **Latency:** deep CPU NLI and agent search can take tens of seconds; production use needs asynchronous review, GPU inference, caching, or smaller route budgets.
7. **Product integration:** the low-latency ControlPlane gateway and the heavier adaptive research pipeline are not yet one default runtime path.
8. **Remediation usefulness:** extractive correction is safe but can remove useful details; a generative rewrite would require independent re-verification.

## Recommended next experiments

1. Build a manually reviewed claim/evidence dataset with `SUPPORTED`, `CONTRADICTED`, and `UNKNOWN` labels.
2. Retune Phase 6.1 thresholds with the evidence-aware rules on validation data.
3. Add semantic retrieval/reranking and measure retrieval recall before changing the NLI model.
4. Evaluate a larger NLI model only on uncertain important claims.
5. Fine-tune only after enough domain-labelled hard examples exist.
6. Lock configuration and evaluate once on an unseen source-separated holdout.

The architecture is appropriate for a responsible-AI prototype. It should be presented as an auditable safety and governance system with measured limitations, not as a production-certified hallucination oracle.
