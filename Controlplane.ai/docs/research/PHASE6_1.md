# Phase 6.1 — Validation-only NLI threshold tuning

Phase 6.1 tunes the Phase-6 entailment threshold, contradiction threshold, and decision margin without using the official RAGTruth test labels.

The script creates a source-safe validation subset from RAGTruth train data, excluding any train source IDs that occur in the official test split. It runs Phase 5, retrieves top-k evidence for UNKNOWN claims, scores each evidence/claim pair with NLI once, caches those probabilities, and then evaluates many threshold combinations without rerunning the transformer.

Run:

```powershell
python scripts/run_phase6_1_tuning.py --config configs/phase6_1_tuning.yaml
```

The first run writes `results/phase6_1/nli_score_cache.json`. Later threshold sweeps reuse this cache automatically. Use `--rebuild-cache` only when the validation set, retrieval configuration, model, or Phase-5 preprocessing changes.

Outputs:

- `results/phase6_1/phase6_1_summary.json`
- `results/phase6_1/threshold_sweep.csv`
- `results/phase6_1/nli_score_cache.json`
- `models/phase6/selected_nli_thresholds.json`

`configs/phase6_ragtruth.yaml` points to the selected-threshold artifact. After tuning, set its `experiment.max_records` to the desired final evaluation size and run Phase 6 normally.

The selection diagnostics are intentionally span-based proxies. RAGTruth hallucination spans localize hallucinated text; they are not gold atomic-claim labels. The tuner therefore reports conflict-span recall, baseless-span conservative handling, NLI contradiction/conflict alignment, and NLI-supported hallucination overlap rather than calling them claim-level accuracy.
