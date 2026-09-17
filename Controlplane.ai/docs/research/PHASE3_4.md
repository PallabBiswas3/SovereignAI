# Phase 3 and Phase 4

## What was added

Phase 3 adds probability calibration and provisional routing-threshold optimization. It re-creates the Phase-2 source-disjoint validation set, then splits that validation set again by `source_id` into three non-overlapping subsets:

1. calibration-fit: fits Platt and isotonic calibrators;
2. calibration-selection: chooses identity/Platt/isotonic using Brier score, with ECE as tie-breaker;
3. routing-tune: chooses `tau1` and `tau2` without touching the official RAGTruth test split.

`tau1` is selected to maximize the fast-path fraction while satisfying the configured maximum hallucination rate in the fast bucket and minimum hallucination recall outside it. `tau2` creates a high-risk bucket with at least the configured hallucination rate. These thresholds are intentionally provisional: once Phases 5-8 provide real downstream latency/cost/accuracy, they should be optimized again with those real route costs.

Phase 4 adds the actual three-way router and evaluates routing behavior on the untouched RAGTruth test set plus HaluEval QA as an external generalization check. The HaluEval QA loader constructs one supported and one hallucinated ResponseRecord from each QA source row, preserving its `knowledge` field as context.

## New files

```text
configs/phase3_ragtruth.yaml
configs/phase4_router.yaml
scripts/run_phase3_calibration.py
scripts/run_phase4_router.py
scripts/download_halueval.py
src/adaptivefact/risk/calibration.py
src/adaptivefact/risk/threshold_optimizer.py
src/adaptivefact/routing/__init__.py
src/adaptivefact/routing/router.py
src/adaptivefact/routing/policy.py
src/adaptivefact/routing/evaluation.py
src/adaptivefact/data/loaders/halueval.py
tests/test_phase3_calibration.py
tests/test_phase4_router.py
```

## Run order

From the repository root, with `.venv` activated:

```powershell
pip install -e .
pytest tests/test_phase3_calibration.py tests/test_phase4_router.py -v
python scripts/run_phase3_calibration.py --config configs/phase3_ragtruth.yaml
python scripts/download_halueval.py
python scripts/run_phase4_router.py --config configs/phase4_router.yaml
```

Phase 3 requires these existing Phase-2 artifacts:

```text
models/phase2/selected_model.joblib
models/phase2/manifest.json
```

Phase 3 creates:

```text
models/phase3/selected_calibrator.joblib
models/phase3/thresholds.json
models/phase3/manifest.json
```

Phase 4 creates:

```text
results/phase4/router_evaluation.json
```

The default Phase-4 config evaluates all RAGTruth test responses and the first 1,000 HaluEval QA source rows, which become 2,000 ResponseRecords. Set `max_source_records: null` for the full HaluEval QA evaluation.
