# Engineering closeout results — 2026-09-26

These are local measurements collected from the worktree based on commit
`0aefefe7a1ca6c0bac6dcaea3402a9e56ae225e2`. They are research fixtures and
runtime observations, not production performance claims.

## Environment

- Windows 10 build 26200
- 10 physical / 12 logical CPU cores
- 15.68 GiB system RAM
- Ollama model: `qwen3:4b-instruct`
- vLLM model configured: `Qwen/Qwen3-0.6B`
- vLLM endpoint: unavailable during this run
- GraphRAG live Supabase endpoint: external and unreachable from the sovereign
  test environment

## GraphRAG and grounding

The deterministic TypeScript metric, authorized-retrieval, adaptive-policy,
verification-policy, and bounded claim-verifier suites all passed.

The backend offline fixture produced:

| Measure | Result |
|---|---:|
| Semantic retrieval Recall@1 | 1.000 |
| Semantic retrieval Recall@3 | 1.000 |
| Semantic retrieval MRR | 1.000 |
| Citation correctness | 1.000 |
| Unanswerable refusal accuracy | 0.750 |
| Hash baseline Recall@1 | 0.812 |
| Hash baseline Recall@3 | 0.938 |
| Hash baseline MRR | 0.877 |
| Batch-2 hybrid Recall@1 / MRR | 1.000 / 1.000 |
| Context compression | 444 → 116 tokens (0.2584) |
| Required fact preserved | yes |
| Deterministic claim verification | 2/2 correct |

The most important quality gap is abstention: one of four unanswerable semantic
queries crossed the current similarity threshold. The local cross-encoder was
not cached, so no reranker-improvement claim is made.

The live `eval:retrieval` command was attempted and failed because it targeted
an external Supabase hostname that was not resolvable. No remote retrieval result
was substituted. Reproduction requires a self-hosted/local Supabase dataset with
real node/chunk labels and authorization scope.

## Inference Pareto experiment

The compact run produced 28 observations: 14 successful Ollama measurements and
14 explicit vLLM connection failures. Raw observations and the derived summary
are in [`inference/`](inference/).

### Evidence budget

| Approx. evidence tokens | TTFT | Total latency | Decode tok/s | Groundedness | Citation recall |
|---:|---:|---:|---:|---:|---:|
| 256 | 17.065 s | 26.156 s | 9.02 | 0.833 | 0.750 |
| 512 | 18.837 s | 31.710 s | 7.53 | 0.833 | 0.750 |
| 1024 | 45.786 s | 61.993 s | 5.99 | 0.833 | 0.750 |

Within these tested points, 256 tokens is the only Pareto-optimal budget because
quality remained unchanged while TTFT increased at larger budgets. This is a
small prompt-based proxy, not a labeled domain-quality verdict.

### Prefix reuse

Repeating the same 512-token prefix yielded TTFT of 414 ms and then 165 ms,
compared with 23.670 s for the separate 512-token context-length observation.
This strongly supports caching repeated evidence prefixes, though a larger
randomized run is required for a confidence interval.

### Concurrency caveat

Aggregate throughput was 0.88, 5.46, and 5.18 output tok/s at concurrency 1, 2,
and 4. The concurrency-1 request paid a 55.7 s cold/prefill cost while later
requests benefited from warm state, so these points must not be interpreted as
a fair batching comparison. A future run must randomize concurrency order or
restart the runtime before every condition.

## Pump-102 and negative/security evaluation

Seven Pump-102 system-integration contract tests passed, covering the complete
authorized evidence/diagnostic/governance path, model outage, denial, provenance,
artifact, and capsule behavior. Fifteen additional governance, auth/resource,
and automatic-routing negative tests passed. The PII and prompt-injection
fixtures each measured precision, recall, and F1 of 1.000 over 20 cases.

A live networked Pump-102 timing result is not published because GraphRAG,
ControlPlane, the diagnostic service, backend API, and vLLM were not running as
one local stack. The passing result is an in-process contract evaluation, not a
claim about deployed end-to-end latency.

## Reproduction

```powershell
# GraphRAG deterministic/security evaluation
Set-Location Graph-RAG\server
npm run test:metrics
npm run test:retrieval-core
npm run test:agent-policy
npm run test:verification-policy
npm run test:claim-verifier

# Negative and Pump-102 integration contracts
Set-Location ..\..
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_phase9_governance.py `
  tests/test_phase33_auth_approval_security.py `
  tests/test_phase34_resource_security.py `
  tests/test_phase38_system_integration.py `
  tests/test_phase39_automatic_integration_routing.py

# Compact inference experiment
.\.venv\Scripts\python.exe benchmarks\inference_tradeoff.py `
  --vllm-model Qwen/Qwen3-0.6B `
  --ollama-model qwen3:4b-instruct `
  --context-lengths 256 512 `
  --evidence-budgets 256 512 1024 `
  --repetitions 2 --timeout 300 `
  --output experiments\results\2026-09-26\inference
```
