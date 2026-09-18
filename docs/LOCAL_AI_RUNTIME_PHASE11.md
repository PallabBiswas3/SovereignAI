# Local AI Runtime — Phase 10.1 result and Phase 11

Verified on the target Windows CPU laptop on 18 September 2026.

## Phase 10.1 — controlled f16 vs q8_0 confirmation at 4096

The earlier Phase 10 isolated-block result suggested q8_0 might be faster than f16 at context 4096. Because that result had unequal memory baselines and substantial thermal load, it was treated only as a hypothesis.

The confirmation benchmark used four paired rounds with alternating order:

1. f16 -> q8_0
2. q8_0 -> f16
3. f16 -> q8_0
4. q8_0 -> f16

It also measured performance before the expensive long-context canary, explicitly unloaded each isolated model, terminated the isolated Ollama process tree, waited for memory recovery between trials, limited Ollama to one loaded model / one parallel request, and retained per-run server logs.

Measured q8_0/f16 throughput ratios:

- round 1: 0.8043
- round 2: 0.9890
- round 3: 1.0130
- round 4: 0.7838
- median paired ratio: **0.8966**
- q8_0 win rate: **0.25**

Memory / quality:

- f16 median resident size: **2973.49 MB**
- q8_0 median resident size: **2703.57 MB**
- q8_0 resident saving: **269.92 MB**
- f16 short-quality score: **1.0**
- q8_0 short-quality score: **1.0**

Conclusion:

- q8_0 genuinely reduces resident size at context 4096 by about 270 MB;
- however, it was about 10.3% slower by the paired median and won only one of four rounds;
- the earlier apparent q8_0 speed advantage was not reproduced under the stronger paired design;
- **f16 remains the runtime KV-cache choice at both 2048 and 4096 on this machine**;
- the expensive long-context confirmation is not justified because q8_0 already fails the throughput and win-rate guardrails.

This negative result is retained intentionally. q8_0 may still be useful on a machine where memory capacity is more important than generation speed, but it is not the performance-optimal default for this target laptop.

## Current measured general-inference configuration

- backend: Ollama
- model: `qwen3:4b-instruct`
- weight quantization: `Q4_K_M`
- CPU threads: 5
- batch: 128
- adaptive context: 2048 for short prompts, larger only when required
- KV cache: `f16`
- normal model concurrency: 1 generation
- keep-alive: approximately 60 s
- local Ollama loopback endpoint: `http://127.0.0.1:11434`
- llama.cpp: implemented but not selected (paired backend benchmark favored Ollama)

## Phase 11 — end-to-end SovereignAI latency

Phase 11 asked a narrower systems question: after provider/runtime tuning, how much latency does a complete SovereignAI `/api/chat` request add over invoking the same local Ollama model directly?

### Instrumented `/api/chat` phases

Every `/api/chat` response includes `runtime_metrics` containing:

- `endpoint_total_seconds`
- `request_setup_seconds`
- `routing_seconds`
- `provider_setup_seconds`
- `generation_seconds`
- `persistence_commit_seconds`
- `measured_phase_seconds`
- `application_overhead_excluding_model_seconds`
- nested provider-generation telemetry including TTFT, queue wait, load time, decode throughput, warm/cold status, CPU tuning and adaptive-context plan.

### Benchmark evolution

The first paired benchmark was not sufficient for a strong latency claim because the first measured request after each reset could pay an Ollama runner reload while the second request reused that exact runner configuration. This produced order-dependent ratios above and below 1.0.

The final `benchmark_end_to_end_latency_v4.py` design therefore:

1. unloads/reloads the model between comparison rounds;
2. performs runner warm-up outside the measured calls;
3. enforces a benchmark-only RAM headroom gate without weakening the production scheduler reserve;
4. rejects a measured pair if either target incurs material model load;
5. alternates SovereignAI/direct order;
6. uses the effective model, system prompt, adaptive context, output budget, temperature, CPU thread count and batch size emitted by the real SovereignAI calibration request.

### Windows loopback hostname finding

A controlled hostname A/B benchmark exposed a real Windows-local latency penalty in the model registry configuration.

Before the fix, `config/models.yaml` used `http://localhost:11434` while standalone runtime benchmarks used `http://127.0.0.1:11434`. Twelve alternating trials against the same warm Ollama server showed:

- `/api/ps` median request time via `localhost`: **269.768 ms**
- `/api/ps` median request time via `127.0.0.1`: **5.385 ms**
- median `/api/ps` lifecycle penalty: **263.498 ms**
- one-token `/api/generate` median request time via `localhost`: **382.034 ms**
- one-token `/api/generate` median request time via `127.0.0.1`: **111.480 ms**
- median generation lifecycle penalty: **272.020 ms**
- median Ollama-reported model time remained close: **108.855 ms** (`localhost`) vs **101.366 ms** (`127.0.0.1`)
- non-model request overhead was approximately **272.311 ms** via `localhost` vs **8.147 ms** via IPv4 loopback.

This isolated the large delay to the hostname/network path on the target Windows environment rather than the model's inference work. The local model registry now uses `http://127.0.0.1:11434` consistently for general, coder and vision Ollama models.

### Final validated Phase 11 result

After the IPv4-loopback fix, the exact-runner/RAM-stable v4 benchmark completed all four requested rounds with **zero invalid warm-path rounds**.

Measured model load during accepted calls remained only about **4.8-9.6 ms**, so the earlier runner-reload confound was removed.

Median results:

| Metric | SovereignAI | Direct Ollama |
|---|---:|---:|
| Client wall time | 6.859 s | 7.226 s |
| Endpoint total | 6.837 s | — |
| Model load | 6.1 ms | 7.4 ms |
| Decode throughput | 11.410 tok/s | 10.934 tok/s |
| TTFT | 0.555 s | 0.674 s |

SovereignAI request-path medians:

- request setup: **0.148 ms**
- routing: **6.801 ms**
- provider construction: **0.623 ms**
- SQLite persistence/commit: **9.267 ms**
- queue wait: **31 ms**
- application overhead excluding Ollama-reported model time: **123.483 ms**
- application-overhead fraction: **1.81%**

The reported median paired wall ratio was `0.903984`, but it must **not** be interpreted as evidence that SovereignAI intrinsically makes Ollama faster. The responses do not contain identical output-token counts and the laptop exhibits normal TTFT/CPU/thermal variability. The robust conclusion is that SovereignAI's request wrapper adds only a small amount of latency relative to model generation and does not measurably degrade decode throughput.

### Phase 11 conclusion

Phase 11 is complete for the plain `/api/chat` path.

Evidence supports the following claims:

- the major application-level latency bug was the Windows `localhost` path and has been removed by using IPv4 loopback;
- the production RAM safety reserve remains intact;
- warm-path model-load contamination was eliminated in the final benchmark;
- routing and persistence costs are each only single-digit milliseconds to roughly 10 ms;
- normal queue/admission overhead is only a few tens of milliseconds;
- total application overhead is about **123 ms median / 1.8% of endpoint latency** for this workload;
- model decode throughput is effectively unchanged by the SovereignAI wrapper;
- further micro-optimization of the basic `/api/chat` path is not justified by current evidence.

The next performance target should therefore move outward to the real interactive path:

`AgentExecutor -> model-token batching/persistence -> callbacks -> SSE -> browser first frame`.

That path should be measured using realistic workloads such as GraphRAG Q&A, industrial diagnostics, tool calls, code generation and multi-step agents rather than another synthetic micro-optimization of the already-small chat wrapper.

### Reproduce final Phase 11 benchmark

Start Ollama using the validated f16 profile, restart the SovereignAI backend, then run from the repository root:

```powershell
python backend/scripts/benchmark_end_to_end_latency_v4.py `
  --repeats 4 `
  --output workspace/e2e_latency_v4.json
```

A valid result requires every accepted pair to remain below the configured measured-load threshold and the production RAM admission reserve must not be weakened to make the benchmark pass.
