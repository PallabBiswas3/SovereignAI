# Past experiments and results

Last consolidated: 30 September 2026

This file keeps the project history and measured outcomes in one place. Results are local engineering evidence, usually from synthetic or small fixtures; they are not production accuracy, safety, or performance claims.

## Delivery history

| Stage | What was tested or added | Result |
|---|---|---|
| Baseline | Local FastAPI/Next.js workbench, Ollama routing, OCR, vision, tools, sandboxed code, artifacts, governance, audit | Working prototype established |
| Batch 1 | Streaming, FAST/STANDARD/DEEP modes, resource admission, model lifecycle, versioned cache | 43 backend tests passed |
| Batch 2 | Dense + BM25 retrieval, RRF, optional offline reranking, context compilation, typed evidence, unit checks | 68 backend tests passed |
| Batch 3 | Declarative Workcells and tamper-verifiable Evidence Capsules | 87 passed, 1 Windows symlink test skipped |
| Batch 4 | Local identity, RBAC/ACL, pre-ranking authorization, separation of duties, APEL demo | 103 passed, 1 skipped |
| Batch 5 | Asset Passports, read-only simulated telemetry, deterministic trends, draft-only CMMS | 114 passed, 1 skipped |
| Batch 6 | Validated, transactional Organization Packs | 134 passed, 1 skipped |
| Integrated system | GraphRAG + Time-Series Diagnostic Agent + local model + ControlPlane release gate | Contract and failure-path tests passed; live networked timing still pending |

## ControlPlane factuality

The v2 experiment reused one factuality state and reduced median and p95 latency by about 32%, but did not reduce over-intervention. On the 1,000-case synthetic validation set it produced 65.6% action accuracy, 0.67% unsafe release, 49.02% over-intervention, 63.82% hallucination F1, 103.7 ms median latency, and 520.2 ms p95 latency.

Factuality v3 separated `SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`, `UNDECIDABLE`, and `CONFLICTING`, made the lightweight model a routing signal, and added an evidence-quality gate. The accepted DeBERTa-only run reached:

| Metric | Result |
|---|---:|
| Action accuracy | 76.9% |
| Over-intervention on legacy labels | 0.0% |
| Contradiction precision / recall / F1 | 66.3% / 97.6% / 78.9% |
| Factuality latency p50 / p95 | ~12 ms / ~452 ms |

The corpus predates the v3 taxonomy, so these results do not establish separate real-world precision for unsupported, undecidable, or conflicting claims.

## Retrieval and grounding

The September 26 offline fixture reported semantic Recall@1, Recall@3, MRR, and citation correctness of 1.0. Unanswerable refusal accuracy was 0.75 on four cases. The hash baseline reached Recall@1 0.812, Recall@3 0.938, and MRR 0.877. Context compilation reduced 444 estimated evidence tokens to 116 while retaining the required fact.

The local cross-encoder was unavailable, so no reranker improvement is claimed. Live GraphRAG evaluation was blocked because the configured Supabase endpoint was external and unreachable; a local/self-hosted corpus is still required.

## Local inference

The first matched Qwen3 0.6B smoke run completed 26/26 observations across Ollama and CPU vLLM. It was a runtime-plus-format comparison, not an identical-precision runtime comparison.

| Runtime | 256-token context TTFT / total | 512-token context TTFT / total |
|---|---:|---:|
| Ollama | 7.756 s / 9.419 s | 4.655 s / 5.599 s |
| vLLM | 12.867 s / 15.513 s | 23.037 s / 25.785 s |

At evidence budgets of 256 and 512 tokens, the simple quality proxy was unchanged, so 256 tokens was Pareto-optimal in this smoke run. Repeated-prefix TTFT fell below 300 ms for both runtimes. Concurrency observations were confounded by cold/warm order and are not a fair throughput comparison.

Query-aware evidence selection later reduced the authorized-answer prompt from about 1,356 to 708 tokens while preserving full provenance outside the generation prompt.

## Integration and security checks

- Seven Pump-102 integration contract tests covered authorized evidence, diagnostics, governance, outage, denial, provenance, artifact, and capsule behavior.
- Fifteen additional governance, resource-security, and automatic-routing negative tests passed.
- PII and prompt-injection fixtures each reported precision, recall, and F1 of 1.0 over 20 synthetic cases.
- A live, fully networked Pump-102 run was not completed because all required local services were not running together.

## 30 September 2026: controlled inference harness prerequisite

Implemented and checked on base revision `8062b29d0a25df5b2e63c59ac6a927cf6d4afd1c`.
This is harness validation, not a completed model/runtime comparison.

- Focused synthetic protocol/HTTP fixtures plus existing system-benchmark,
  retrieval-ACL, integration, automatic-routing, and retry regressions: **58 passed**.
  Command:
  `python -m pytest tests/test_inference_tradeoff.py tests/test_system_benchmark.py tests/test_phase31_retrieval_acl.py tests/test_phase38_system_integration.py tests/test_phase39_automatic_integration_routing.py tests/test_integration_retry.py`.
- Fixtures check randomized balanced repetition blocks, warm-up exclusion, failure
  suppression, invalid configuration before requests, private endpoint restrictions,
  no proxy/redirect use, truncated/empty stream rejection, raw provenance, unknown
  citation penalties, and legacy smoke compatibility. Synthetic timings are test
  inputs, not benchmark results.
- Actual unavailable-service smoke check on loopback ports 8001 and 11434:
  **20 attempted warm-ups, 20 failures, 0 measured successes, exit code 2**.
  Command:
  `python benchmarks/inference_tradeoff.py --protocol controlled --vllm-model Qwen/Qwen3-0.6B --ollama-model qwen3:0.6b --vllm-format unavailable --ollama-format unavailable --context-lengths 64 --evidence-budgets 128 --repetitions 5 --timeout 0.2 --output /tmp/sovereign-inference-unavailable`.
  Neither inference runtime was installed/running in this execution environment.
  No live inference latency, throughput, memory, or quality result is claimed.
- Full repository regression suite and networked Pump-102 workflow were not run.
  Verified cold starts, memory instrumentation, exact comparable model formats,
  tokenizer-aware budgets, and manually adjudicated quality remain outstanding.

The harness now reports end-to-end request throughput, unlike the older
post-first-token calculation; do not compare those fields directly with the
September 26 smoke results. Historical observations above remain unchanged.

## Reproducible evidence

- Runtime results: [`../experiments/results/2026-09-26/`](../experiments/results/2026-09-26/)
- Inference harness: [`../benchmarks/inference_tradeoff.py`](../benchmarks/inference_tradeoff.py)
- System harness: [`../scripts/benchmark_system.py`](../scripts/benchmark_system.py)
- Root regression tests: [`../tests/`](../tests/)

See [Current features](CURRENT_FEATURES.md) for the implemented system and [Next plan](NEXT_PLAN.md) for the remaining work.
