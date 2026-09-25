# Matched Qwen3 0.6B runtime smoke benchmark

Date: 2026-09-26

This records the first successful local run in which both runtimes completed the same benchmark harness without connection failures.

## Runtime setup

- vLLM: `Qwen/Qwen3-0.6B`, CPU build, BF16, `--max-model-len 4096`, `--max-num-seqs 4`
- Ollama: `qwen3:0.6b`
- Harness: `benchmarks/inference_tradeoff.py`
- Smoke context lengths: 256, 512
- Smoke evidence budgets: 256, 512
- Prefix repetitions: 2
- Concurrency conditions: 1, 2, 4
- Total observations: 26
- Successful observations: 26

This is a **runtime-plus-format** comparison, not a pure runtime-only comparison, because the vLLM and Ollama model formats/precisions are not identical.

## Single-request context results

| Runtime | Context tokens | TTFT | Total latency | Decode throughput |
|---|---:|---:|---:|---:|
| Ollama | 256 | 7.756 s | 9.419 s | 19.25 tok/s |
| Ollama | 512 | 4.655 s | 5.599 s | 25.42 tok/s |
| vLLM | 256 | 12.867 s | 15.513 s | 10.96 tok/s |
| vLLM | 512 | 23.037 s | 25.785 s | 10.55 tok/s |

The non-monotonic Ollama context result indicates warm/cache/order effects, so these values are smoke observations rather than a final scaling curve.

## Evidence-budget results

| Runtime | Evidence tokens | TTFT | Total latency | Decode throughput | Groundedness | Citation recall |
|---|---:|---:|---:|---:|---:|---:|
| Ollama | 256 | 3.110 s | 5.768 s | 31.23 tok/s | 0.667 | 0.500 |
| Ollama | 512 | 3.669 s | 6.927 s | 25.78 tok/s | 0.667 | 0.500 |
| vLLM | 256 | 16.745 s | 23.435 s | 7.03 tok/s | 0.667 | 0.500 |
| vLLM | 512 | 19.173 s | 23.624 s | 10.56 tok/s | 0.667 | 0.500 |

Within this smoke run, 256 evidence tokens was Pareto-optimal for both runtimes because the larger 512-token budget did not improve the simple groundedness/citation proxy.

## Prefix reuse

| Runtime | Repetition 1 TTFT | Repetition 2 TTFT |
|---|---:|---:|
| Ollama | 201.25 ms | 93.34 ms |
| vLLM | 279.19 ms | 223.65 ms |

Both runtimes benefited strongly from repeated-prefix reuse.

## Concurrency smoke results

| Runtime | Concurrency | TTFT p50 | Latency p50 | Aggregate throughput |
|---|---:|---:|---:|---:|
| Ollama | 1 | 11.545 s | 13.363 s | 2.24 tok/s |
| Ollama | 2 | 1.105 s | 2.910 s | 15.11 tok/s |
| Ollama | 4 | 3.111 s | 5.201 s | 13.02 tok/s |
| vLLM | 1 | 50.775 s | 53.933 s | 0.56 tok/s |
| vLLM | 2 | 0.641 s | 5.912 s | 10.10 tok/s |
| vLLM | 4 | 1.561 s | 10.707 s | 11.12 tok/s |

These concurrency numbers are **not a fair final comparison** because concurrency 1 paid a cold/prefill cost while later conditions ran with warmer runtime/cache state. They demonstrate that the harness and concurrent execution paths work, not that one concurrency level is definitively superior.

## Current conclusions

1. The matched-size benchmark infrastructure now works end-to-end for both runtimes: 26/26 observations succeeded.
2. On this CPU laptop, Ollama had lower single-request latency in this smoke run.
3. vLLM showed large throughput recovery at concurrency 2/4 relative to its cold concurrency-1 observation, but a controlled warm/cold experiment is required before drawing a batching conclusion.
4. Prefix reuse is highly valuable for both runtimes.
5. Increasing the evidence budget from 256 to 512 tokens produced no improvement in the current simple quality proxy, so compact evidence remains the strongest direction to investigate.

## What this result does not establish

- It does not prove Ollama is universally faster than vLLM.
- It does not provide a runtime-only comparison under identical quantization/precision.
- It does not provide a fair final concurrency comparison because cache/warm state was not controlled across conditions.
- It does not establish production GraphRAG quality; the quality score here is a small prompt-based proxy.
