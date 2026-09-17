# Phase 1 and Phase 2 implementation

## Phase 1: always-on baselines

The implementation provides four benchmark rows:

1. `NoVerificationBaseline` — always returns the answer and establishes the verification-latency floor.
2. `RetrievalNLIBaseline` — splits the response into sentence-level factual units, retrieves the most relevant context chunks with TF-IDF, then applies a configurable NLI model. Sentence units are intentional here; atomic claim extraction belongs to Phase 5/7.
3. `LLMJudgeBaseline` — sends query, context, and answer to a configurable local Hugging Face instruct model and requires a structured hallucination judgment.
4. `SelfConsistencyBaseline` — draws fresh stochastic samples only when the configured generator matches the model that produced the RAGTruth record. Mismatched records are skipped rather than silently turning cross-model agreement into a fake SelfCheckGPT result. Because RAGTruth contains outputs from several original generators, a rigorous self-consistency benchmark must be run model-by-model for generators you can actually resample, or on a separately regenerated benchmark subset.

The default config enables only no-verification + retrieval/NLI. LLM judge and self-consistency are disabled by default because they are intentionally expensive. Self-consistency also has a stricter methodological requirement: the sampling model must match the generator of the evaluated record.

Run:

```bash
python scripts/run_phase1_baselines.py --config configs/phase1_ragtruth.yaml
```

Recommended first run:

```yaml
experiment:
  max_records: 50
```

After that, enable `llm_judge` and then `self_consistency` separately so each baseline's latency/cost is interpretable.

## Phase 2: Risk Estimator V1

V1 is provider-independent. It does not require generator logits.

Feature groups:

- response length and sentence statistics,
- numeric, percentage, currency, date, URL and citation counts,
- lightweight entity counts,
- query specificity/currentness/citation flags,
- lexical response-context support,
- minimum/mean/maximum sentence support,
- exact entity support ratio,
- exact number support ratio,
- exact date support ratio,
- optional spaCy NER,
- optional Sentence-Transformer semantic support.

Two models are trained:

- Logistic Regression with standardization and balanced class weights,
- XGBoost with class-imbalance weighting.

The RAGTruth official train split is divided into fit/validation subsets using `GroupShuffleSplit` on `source_id`, so the same source document cannot occur in both fit and validation. Among candidate source-disjoint splits, the code selects one whose hallucination prevalence is close to the complete training split.

XGBoost becomes the selected Phase 2 model only if its validation F1 improves over Logistic Regression by at least the configurable `min_f1_gain_for_xgboost` value. The default is `0.03` absolute F1.

The Phase 2 manifest contains:

- fit/validation/test sizes,
- hallucination prevalence,
- validation and test Precision/Recall/F1/AUROC/AUPRC,
- P50/P95/P99 risk-estimation latency,
- Logistic Regression coefficients,
- XGBoost feature importances,
- selected-model decision.

CPU-friendly run:

```bash
python scripts/run_phase2_risk.py --config configs/phase2_ragtruth.yaml
```

Full V1 with spaCy + semantic support:

```bash
python -m spacy download en_core_web_sm
python scripts/run_phase2_risk.py --config configs/phase2_ragtruth_semantic.yaml
```

Outputs are written to `models/phase2/` or `models/phase2_semantic/`.

## Important boundary with Phase 3

Phase 2 uses the raw model probability and a fixed 0.5 decision threshold only for model comparison. It does **not** claim this probability is calibrated. Platt/isotonic calibration and routing-threshold optimization remain Phase 3, exactly as planned.
