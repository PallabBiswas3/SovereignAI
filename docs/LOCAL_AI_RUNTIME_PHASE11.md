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
- llama.cpp: implemented but not selected (paired backend benchmark favored Ollama)

## Phase 11 — end-to-end SovereignAI latency

Provider-side tuning is now sufficiently explored. The next question is application overhead: how much slower is a complete SovereignAI chat request than calling the same local model directly?

### Instrumented `/api/chat` phases

Every `/api/chat` response now includes `runtime_metrics` containing:

- `endpoint_total_seconds`
- `request_setup_seconds`
- `routing_seconds`
- `provider_setup_seconds`
- `generation_seconds`
- `persistence_commit_seconds`
- `measured_phase_seconds`
- `application_overhead_excluding_model_seconds`
- nested provider-generation telemetry, including TTFT, queue wait, load time, decode throughput, warm/cold status, CPU tuning and adaptive-context plan.

### Paired benchmark

`backend/scripts/benchmark_end_to_end_latency.py` performs a warm-path paired comparison between:

1. `POST /api/chat`
2. direct Ollama `/api/generate`

The benchmark first calibrates from a real SovereignAI response. The direct Ollama request then uses the same:

- effective model
- system prompt
- selected adaptive context
- output-token budget
- temperature
- `num_thread`
- `num_batch`

Rounds alternate order to reduce sustained-load/order bias.

The benchmark reports:

- SovereignAI client wall time
- SovereignAI endpoint total time
- direct Ollama wall time
- paired SovereignAI/direct wall ratio
- routing time
- provider construction time
- SQLite persistence/commit time
- model generation time
- application overhead excluding Ollama-reported model time
- TTFT
- queue wait
- load time
- decode throughput

### Windows loopback hostname finding

A controlled hostname A/B benchmark exposed a real Windows-local latency penalty in the model registry configuration.

Before this fix, `config/models.yaml` used `http://localhost:11434` while the standalone runtime benchmarks used `http://127.0.0.1:11434`. Twelve alternating trials against the same warm Ollama server showed:

- `/api/ps` median request time via `localhost`: **269.768 ms**
- `/api/ps` median request time via `127.0.0.1`: **5.385 ms**
- median `/api/ps` lifecycle penalty: **263.498 ms**
- one-token `/api/generate` median request time via `localhost`: **382.034 ms**
- one-token `/api/generate` median request time via `127.0.0.1`: **111.480 ms**
- median generation lifecycle penalty: **272.020 ms**
- median Ollama-reported model time remained close: **108.855 ms** (`localhost`) vs **101.366 ms** (`127.0.0.1`)
- non-model request overhead was approximately **272.311 ms** via `localhost` vs **8.147 ms** via IPv4 loopback.

This isolates the large delay to the hostname/network path on the target Windows environment rather than the model's inference work. The local model registry now uses `http://127.0.0.1:11434` consistently for general, coder and vision Ollama models.

This also explains why the earlier metric `endpoint_total - Ollama total_duration` exaggerated apparent SovereignAI application overhead: part of that difference was transport/hostname overhead outside Ollama's reported model duration.

### Run

Start Ollama using the validated f16 profile, start the SovereignAI backend locally, then from the repository root run:

```powershell
python backend/scripts/benchmark_end_to_end_latency.py `
  --repeats 4 `
  --output workspace/e2e_latency.json
```

This is a warm-path benchmark by design. Cold-load behavior was already measured separately in earlier phases.

### Decision rule

Do not optimize a subsystem merely because it exists. Use the Phase 11 breakdown first:

- if routing/provider setup is meaningful, cache immutable registries/providers where safe;
- if SQLite commit time is meaningful, optimize transaction/event persistence;
- if queue wait dominates, revisit scheduling only with evidence;
- if application overhead is already tiny relative to model generation, stop micro-optimizing the request path and move to real workload quality/evaluation.

The next code change after Phase 11 should therefore be determined by this measurement, not guessed in advance.
