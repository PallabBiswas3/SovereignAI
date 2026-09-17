# Phase 5 — Claim Extraction + Deterministic Verification

## Goal
Turn responses that need verification into sentence/semicolon-sized factual units, extract structured facts (numbers, dates, entities), and resolve the safest cases without a transformer call.

Phase 5 is deliberately conservative:

- exact, anchored numeric/date matches -> `SUPPORTED`
- strongly anchored numeric/date mismatches -> `CONTRADICTED`
- entity presence can support an entity signal
- missing evidence or uncertain relation -> `UNKNOWN`
- `UNKNOWN` is the handoff to Phase 6 retrieval + NLI

RAGTruth hallucination spans are used only for localization/alignment diagnostics. They are not treated as gold atomic-claim boundaries.

## Added files

```text
configs/phase5_ragtruth.yaml
scripts/run_phase5_claims.py
src/adaptivefact/extraction/__init__.py
src/adaptivefact/extraction/claim_extractor.py
src/adaptivefact/extraction/ner.py
src/adaptivefact/extraction/numeric_date.py
src/adaptivefact/verification/deterministic.py
src/adaptivefact/verification/phase5.py
tests/test_phase5_claims.py
```

`src/adaptivefact/verification/__init__.py` is also updated.

## Install / update

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
```

The default config does not require the spaCy English model because `use_spacy: false`.
If you later enable spaCy:

```powershell
python -m spacy download en_core_web_sm
```

## Test

```powershell
pytest tests/test_phase5_claims.py -v
```

Expected: 8 tests pass.

## Run a quick smoke test

Edit `configs/phase5_ragtruth.yaml` temporarily:

```yaml
experiment:
  max_records: 100
```

Then run:

```powershell
python scripts/run_phase5_claims.py --config configs/phase5_ragtruth.yaml
```

## Run full RAGTruth test split

Set:

```yaml
experiment:
  max_records: null
```

Then:

```powershell
python scripts/run_phase5_claims.py --config configs/phase5_ragtruth.yaml
```

Outputs:

```text
results/phase5/phase5_summary.json
results/phase5/enriched_records_sample.jsonl
```

## Metrics to inspect

The most important Phase 5 metrics are:

- `deterministic_resolution_rate`
- `phase6_unresolved_rate`
- `claim_type_counts`
- `status_counts`
- `hallucination_span_claim_coverage`
- `conflict_span_detection_recall`
- `predicted_contradiction_conflict_alignment`
- `supported_claim_hallucination_overlap_rate`
- P50/P95/P99 latency for extraction and deterministic verification

The key research question is not whether Phase 5 solves all hallucinations. It is:

> What fraction of claims can be resolved safely without paying for NLI or an LLM verifier?

Everything unresolved is intentionally passed to Phase 6.
