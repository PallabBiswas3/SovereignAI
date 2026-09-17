# Overall system validation report

## Scope

This report compares the lightweight ControlPlane path and the integrated
ControlPlane + AdaptiveFact path on 100 synthetic, unreviewed validation
candidates. It also records local CPU concurrency stress tests. These runs are
engineering diagnostics, not independent production-accuracy evidence.

## Accuracy and policy validation

| Metric | Lightweight | Integrated DeBERTa |
|---|---:|---:|
| Cases | 100 | 100 |
| Action accuracy | 81.0% | 79.0% |
| Unsafe allow rate | 11.59% | 5.80% |
| Over-intervention on expected allows | 35.48% | 54.84% |
| Human-review rate | 33.0% | 37.0% |
| Hallucination precision | 62.32% | 57.32% |
| Hallucination recall | 84.31% | 92.16% |
| Hallucination F1 | 71.67% | 70.68% |
| Privacy/Bias/Policy category F1 | 100% | 100% |
| Mean latency | 6.30 ms | 120.09 ms |
| P95 latency | 10.61 ms | 263.40 ms |
| Maximum latency | 47.30 ms | 6,112.30 ms |

The integrated route is safer but more conservative: it catches more expected
hallucination cases and halves unsafe allows, while producing more false alarms,
warnings, and reviews. Claim contradiction precision is 30% with 50% recall;
unknown-claim precision is 18.06% with 61.90% recall. Those claim-level results
are not sufficient for automatic production correction.

## Action confusion

The integrated system correctly handled every expected block (10/10), redaction
(12/12), and warning (9/9). It reviewed 34/38 expected review cases and allowed
the other four. Of 31 expected allows, it allowed 14, warned on 14, and sent
three for human review.

## Stress tests

| Route | Requests | Concurrency | Errors | Throughput | P50 | P95 | P99 | Budget exceeded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Lightweight | 1,000 | 25 | 0 | 125.65 req/s | 23.31 ms | 1,313.33 ms | 1,403.60 ms | 0% |
| Integrated | 100 | 4 | 0 | 19.42 req/s | 107.63 ms | 876.59 ms | 1,130.95 ms | 0% |
| Integrated | 200 | 8 | 0 | 14.11 req/s | 203.30 ms | 2,357.83 ms | 3,567.23 ms | 13.5% |

Concurrency 8 saturates the local CPU NLI path: throughput decreases while tail
latency and budget violations increase. Production deep verification should use
bounded worker concurrency, batching, caching, GPU inference, or an asynchronous
queue instead of sharing the synchronous API worker pool.

## Current decision

The system is suitable for demonstration, regression testing, and shadow-mode
pilots. It is not ready for autonomous production enforcement because unsafe
allows remain above zero, factual false positives are high, human-review burden
is high, and the benchmark labels are not yet independently reviewed.

## Reproduction

Lightweight validation:

```powershell
Remove-Item Env:CONTROLPLANE_ADAPTIVE_VERIFICATION -ErrorAction SilentlyContinue
.venv\Scripts\python.exe scripts\evaluate_controlplane.py `
  --scenarios data\controlplane\benchmark_v1\validation_candidates.jsonl `
  --output results\controlplane\validation_100_lightweight.json
```

Integrated validation:

```powershell
$env:CONTROLPLANE_ADAPTIVE_VERIFICATION = "1"
$env:CONTROLPLANE_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
.venv\Scripts\python.exe scripts\evaluate_controlplane.py `
  --scenarios data\controlplane\benchmark_v1\validation_candidates.jsonl `
  --output results\controlplane\validation_100_integrated.json
```

Stress test:

```powershell
.venv\Scripts\python.exe scripts\stress_controlplane.py `
  --requests 1000 --concurrency 25 --warmup 10 `
  --output results\controlplane\stress_lightweight_1000_c25.json
```

Run a ramp at concurrency 1, 2, 4, 8, 16, and 25. Add a multi-hour soak test,
traffic spikes, maximum-size inputs, malformed Unicode, missing/corrupt model
artifacts, NLI timeouts, audit-store failures, adversarial evidence, conflicting
sources, and forced detector exceptions. Define release gates for unsafe allows,
false intervention, human-review capacity, errors, p95/p99 latency, throughput,
and cost before interpreting any run as a production pass.
