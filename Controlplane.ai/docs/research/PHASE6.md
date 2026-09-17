# Phase 6 — Evidence Retrieval + NLI

Phase 6 receives claims that Phase 5 intentionally left `UNKNOWN`. It retrieves a small amount of relevant context and uses NLI to decide whether the evidence supports, contradicts, or remains insufficient for the claim.

## Decision policy

For every Phase-5 UNKNOWN claim:

1. Reuse a Phase-5 numeric/date conflict candidate sentence when available.
2. Retrieve the top-k TF-IDF evidence chunks from the response context.
3. Score evidence/claim pairs with the compact NLI model.
4. Return `SUPPORTED` only when entailment passes its threshold and exceeds contradiction by the configured margin.
5. Return `CONTRADICTED` only when contradiction passes its threshold and exceeds entailment by the configured margin.
6. Otherwise preserve `UNKNOWN` for later phases.

The first smoke test should use `max_records: 100`. Full-dataset NLI can be expensive on CPU, so inspect the 100-record result before increasing the run size.

## Run

```powershell
python scripts/run_phase6_retrieval_nli.py --config configs/phase6_ragtruth.yaml
```

The default NLI model is the same compact model used by the Phase-1 retrieval+NLI baseline:

```text
cross-encoder/nli-deberta-v3-small
```

## Main metrics

- claims sent to NLI
- NLI pairs
- NLI resolution rate among Phase-5 UNKNOWN claims
- remaining UNKNOWN rate
- overall claim resolution rate
- conflict-span detection recall
- predicted contradiction/conflict alignment
- supported-claim hallucination overlap rate
- retrieval and NLI latency

RAGTruth span metrics remain diagnostics only. Hallucination spans are not treated as gold atomic-claim boundaries.
