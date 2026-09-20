# Local AI Runtime research log

Last updated: 18 September 2026

This document is the canonical record for the SovereignAI local-inference optimization track. It records what was implemented, what was measured on the target laptop, what conclusions are justified by those measurements, what did not work, and what still needs cleaner validation.

The target is not benchmark theater. The objective is to make SovereignAI genuinely useful on an ordinary CPU laptop with limited RAM while preserving local-only inference, quality, reliability, and auditable behavior.

## Target machine and baseline

Measured target machine:

- Windows laptop
- 16 GB system RAM (`~16052 MB` total)
- 10 physical / 12 logical CPU cores
- CPU-only inference; no GPU assumed
- primary general model: `qwen3:4b-instruct`
- measured quantization: `Q4_K_M`
- Ollama endpoint: `http://127.0.0.1:11434`

Initial cold-load measurement at context 4096 / output 128:

| Metric | Baseline |
|---|---:|
| Available RAM before load | 2896.27 MB |
| Available RAM after cold load | 836.74 MB |
| OS-visible RAM delta | 2059.53 MB |
| Ollama process RSS delta | 3023.17 MB |
| Resident model size | ~3036.5 MiB |
| Cold wall time | 15.021 s |
| Cold TTFT | 7.183 s |
| Model load time | 6.529 s |
| Cold decode throughput | 13.781 tok/s |
| Warm TTFT | ~0.13 s |
| Sustained realistic decode | typically ~8-11 tok/s |

The practical constraint discovered immediately was RAM pressure, not just raw decode throughput.

## Current runtime decision

As of the latest verified experiments, the normal general-inference path should use:

| Setting | Current choice |
|---|---|
| Backend | Ollama |
| Model | `qwen3:4b-instruct` |
| Weight quantization | `Q4_K_M` |
| CPU threads | 5 |
| Batch size | 128 |
| Preferred short-request context | 2048 |
| Context policy | adaptive escalation only when needed |
| Normal model concurrency | 1 generation at a time |
| Keep-alive | ~60 s |
| KV cache | `f16` for now |
| Direct llama.cpp | implemented, but not selected |

The `f16` KV-cache choice is provisional for one reason explained in Phase 10: the current RAM-delta benchmark has a baseline-control problem. The persisted runtime choice is safe, but compressed KV-cache memory savings need a cleaner controlled rerun before being claimed.

---

# Phase-by-phase engineering record

## Phase 1 — Reduce streaming overhead

Implemented bounded streaming aggregation in the agent executor.

Instead of forwarding arbitrarily tiny provider chunks immediately, text is grouped up to approximately:

- 96 characters, or
- 50 ms

before emitting a model-token event.

The exact generated text is preserved. Events are still persisted through the existing task/event path; this phase did not replace durable task events with transient broker-only frames.

Why this matters: tiny callback/event/DB operations can become a meaningful fraction of latency when the model itself generates only around 8-11 tokens/s.

Remaining question: only an end-to-end SSE/persistence benchmark can determine whether durable per-chunk event storage is still a material bottleneck.

## Phase 2 — Runtime latency instrumentation

Added local inference timing/telemetry covering:

- wall time
- queue time
- model load duration
- prompt evaluation
- TTFT
- decode duration
- decode tokens/s
- token counts
- warm/cold status
- UI first-frame timing
- callback count/time/fraction
- residual application overhead

This made later optimization decisions measurement-driven rather than based on subjective UI feel.

## Phase 3 — Execution-mode runtime profiles

Added FAST / STANDARD / DEEP runtime modes.

Representative limits:

- FAST: smaller context/output budget for responsive interaction
- STANDARD: medium context/output budget
- DEEP: larger context/output budget for evidence-heavy reasoning

Structured generation uses its own conservative temperature/output settings.

The execution mode remains an upper bound. Later adaptive-context logic can choose a smaller context window per request.

## Phase 4 — Model residency lifecycle

Added Ollama residency operations:

- list resident models
- determine whether a model is already resident
- ensure a model is warm
- unload a model
- warm a configured pool

This supports short warm follow-ups without assuming unlimited RAM.

## Phase 5 — CPU/RAM admission scheduler

Added resource-aware model admission for CPU laptops.

Important policy decisions:

- default to one active model generation
- reserve RAM before loading another model
- reserve is `max(1536 MB, 15% of total RAM)`
- resident models have zero incremental load estimate for admission
- nonresident known models use measured footprint estimates
- unknown models are handled conservatively

The scheduler exists to prevent the runtime from turning a 16 GB machine into a swap-heavy benchmark artifact.

## Phase 6 — Measured model footprints and adaptive residency

Added machine-local footprint persistence and adaptive residency.

Key correction made during this phase:

- admission must use **OS-visible incremental RAM consumption**, not raw process RSS/model mapped size.

From the cold baseline:

`2059.53 MB × 1.10 safety multiplier ≈ 2265.48 MB`

became the more realistic admission estimate for loading the Qwen model on this machine.

Adaptive residency can evict another resident model to create room. Current eviction is not a full true-LRU implementation; it favors the largest other resident candidate under the current logic.

## Phase 7 — CPU thread/batch tuning

Added a reproducible CPU tuning benchmark and persisted machine-local tuning.

Tested combinations of:

- threads: 4, 5, 6
- batch: 64, 128

using seeded/interleaved measurements to reduce ordering bias.

Final ranking included:

| Threads | Batch | Median tok/s | Mean tok/s |
|---:|---:|---:|---:|
| **5** | **128** | **9.099** | **9.533** |
| 4 | 64 | 8.922 | 9.141 |
| 6 | 128 | 8.866 | 9.452 |
| 4 | 128 | 8.678 | — |
| 5 | 64 | 8.586 | — |
| 6 | 64 | 8.535 | — |

Decision:

- use `num_thread=5`
- use `num_batch=128`

No further thread-count tuning is justified unless hardware, power mode, thermal behavior, or Ollama changes materially.

The interleaved rounds also showed overall performance drift across sustained CPU loading, which is why later experiments explicitly account for thermal/order bias.

## Phase 8 — Adaptive context and model/quantization selection

Implemented an adaptive context planner with buckets:

- 2048
- 4096
- 8192
- 16384

The planner estimates prompt tokens conservatively, adds output reserve, then selects the smallest bucket that fits without exceeding the execution-mode ceiling.

### Measured context results

| Context | Median tok/s | Resident size |
|---:|---:|---:|
| 2048 | 10.306 | 2684.99 MB |
| 4096 | 10.543 | 2973.49 MB |
| 8192 | 10.733 | 3550.49 MB |

The throughput increase from 2048 to 8192 was small while resident memory increased by about 866 MB.

Available RAM after load in that run fell approximately to:

- 2048: 682 MB
- 4096: 619 MB
- 8192: 458 MB

Decision:

- 2048 is the preferred context for normal short requests
- escalate to 4096/8192 only when prompt + output actually require it
- do not use a large fixed context merely because the model supports it

### Model/quantization quality result

Installed candidate set contained only `qwen3:4b-instruct`, so this phase did **not** compare multiple weight quantizations.

Measured installed model:

- model: `qwen3:4b-instruct`
- quantization: `Q4_K_M`
- quality canary score: `1.0`
- selected context profile: `2048`
- selected model remained Qwen3 4B Q4_K_M

The five smoke canaries covered arithmetic, industrial extraction, exact instruction following, numeric comparison, and ordered extraction. They are regression guardrails, not a comprehensive semantic benchmark.

## Phase 9 — Ollama vs direct llama.cpp

Implemented a real llama.cpp provider using the OpenAI-compatible local server API, plus backend-profile persistence and a provider factory.

General AgentOrchestrator and `/api/chat` can honor a validated backend profile. Specialized coder/tool/vision paths remain conservative until separately validated.

A Windows launcher was added that can resolve the exact Qwen GGUF model layer from Ollama's local content-addressed model store. This allowed both runtimes to use the same model weights rather than comparing different downloads.

The launcher also handles occupied ports and can move from 8080 to another free localhost port.

### Initial block benchmark

Both backends passed all quality checks.

| Backend | Median tok/s | Median wall |
|---|---:|---:|
| Ollama | 9.097 | 14.234 s |
| llama.cpp | 8.193 | 15.722 s |

llama.cpp / Ollama throughput ratio: `0.901`.

Because the benchmark ran one backend as a block before the other, thermal/order bias was still possible.

### Paired interleaved benchmark

The benchmark was upgraded to alternate order:

- round 1: Ollama → llama.cpp
- round 2: llama.cpp → Ollama
- round 3: Ollama → llama.cpp
- round 4: llama.cpp → Ollama

Measured:

| Backend | Median tok/s | Median wall |
|---|---:|---:|
| **Ollama** | **10.680** | **12.148 s** |
| llama.cpp | 9.107 | 14.124 s |

Per-round llama.cpp/Ollama ratios:

- 0.8842
- 0.8004
- 0.8920
- 0.8899

Median paired ratio: **0.8871**.

Both backends again scored `1.0` on the quality canaries.

Conclusion:

- llama.cpp was about 11.3% slower on this target laptop under the paired experiment
- the backend selector correctly persisted **Ollama**
- direct llama.cpp remains implemented as an experimental/fallback capability, but there is no evidence-based reason to switch normal inference to it on this machine

This is an important negative result and is intentionally retained in the project.

## Phase 10 — KV-cache / RAM optimization

Implemented:

- cache types `f16`, `q8_0`, `q4_0`
- Flash Attention enabled for the isolated benchmark servers
- isolated local Ollama server per cache configuration
- contexts 2048 and 4096
- CPU tuning reused: 5 threads / batch 128
- quality smoke tests
- added long-context retrieval canary
- throughput measurements
- resident model-size measurements
- OS available-RAM snapshots
- persisted KV-cache profile
- optimized Ollama startup helper

Selection guardrails were intentionally conservative:

- quality >= 0.95
- throughput regression <= 5%
- measured RAM saving >= 128 MB

### Phase 10 results — context 2048

All cache types scored `1.0` on the current quality canaries, including long-context target retrieval.

| Cache | Median tok/s | Resident size | OS RAM delta | Current eligibility |
|---|---:|---:|---:|---|
| **f16** | **10.210** | 2684.99 MB | 1966.43 MB | selected |
| q8_0 | 9.053 | 2550.07 MB | 2533.03 MB | no |
| q4_0 | 7.312 | 2478.07 MB | 2489.83 MB | no |

Observed resident-size reduction versus f16:

- q8_0: ~135 MB smaller
- q4_0: ~207 MB smaller

However, the OS-RAM baselines were very different between isolated runs, so the raw OS-delta comparison reported no trustworthy RAM saving. q8_0 was also about 11% slower than f16 at 2048 in this run, while q4_0 was much slower.

Long-context canary wall times in this run increased substantially as cache compression became more aggressive:

- f16: ~41.7 s
- q8_0: ~53.2 s
- q4_0: ~68.1 s

Current safe choice at 2048: **f16**.

### Phase 10 results — context 4096

Again, all cache modes passed the current quality canaries.

| Cache | Median tok/s | Resident size | OS RAM delta | Current eligibility |
|---|---:|---:|---:|---|
| **f16** | 5.808 | 2973.49 MB | 2612.09 MB | selected by current policy |
| q8_0 | **6.782** | 2703.57 MB | 2757.12 MB | no under current RAM policy |
| q4_0 | 5.478 | 2559.57 MB | 2894.26 MB | no |

Resident-size reduction versus f16:

- q8_0: ~270 MB smaller
- q4_0: ~414 MB smaller

The q8_0 throughput number at 4096 is interesting because it was about 16.8% higher than the f16 measurement while also reporting a smaller resident size. It is **not yet treated as a validated win** because:

1. the OS available-RAM baseline differed heavily between cache runs;
2. the long-context quality canary runs before performance measurement can create major thermal load;
3. cache configurations were not yet paired/interleaved the way Phase 9 backends were;
4. the 4096 f16 throughput in this Phase 10 run was much lower than the earlier Phase 8 4096 measurement, which is another sign that run conditions were different.

Therefore Phase 10 v1 is useful for identifying candidates, but not sufficient for a strong compressed-cache RAM/performance claim.

### Current Phase 10 conclusion

Persisted safe profile:

- cache: `f16`
- context: 2048
- quality: 1.0
- median throughput: 10.21 tok/s in this run
- resident size: 2684.99 MB

Research conclusion:

- q4_0 is not attractive at 2048 because the throughput penalty is large
- q8_0 remains worth one cleaner controlled experiment, especially at 4096
- do not claim OS-RAM savings from Phase 10 v1 because starting free-memory states were not controlled tightly enough

---

# What has actually improved

The local runtime evolved from a fixed Ollama call path into a measured resource-aware inference subsystem:

```text
request
  ↓
execution mode
  ↓
adaptive context planner
  ↓
model/quantization profile
  ↓
backend selection profile
  ↓
resource admission + residency
  ↓
CPU tuning
  ↓
Ollama (current winner) / llama.cpp experimental path
  ↓
stream aggregation + latency telemetry
```

Concrete improvements now in the repository include:

- request-adaptive context sizing
- measured CPU tuning
- RAM-aware model admission
- measured model footprints
- adaptive residency
- model/quantization profile persistence
- backend benchmark and automatic safe selection
- direct llama.cpp provider
- exact Ollama-GGUF reuse for fair backend testing
- KV-cache experiment/profile infrastructure
- optimized local startup helpers
- CI coverage for the runtime optimization modules

# Machine-local profile files

These are intentionally local data, not universal constants:

- `backend/data/cpu_tuning.json`
- `backend/data/model_footprints.json`
- `backend/data/model_optimization.json`
- `backend/data/backend_selection.json`
- `backend/data/kv_cache_optimization.json`

They encode empirical choices for the machine on which the benchmarks ran.

# Reproducible benchmark commands

CPU/model/context optimization:

```powershell
python backend/scripts/benchmark_model_optimization.py `
  --base-model qwen3:4b-instruct `
  --output workspace/qwen3_model_optimization.json
```

Backend comparison:

```powershell
python backend/scripts/benchmark_inference_backends.py `
  --model qwen3:4b-instruct `
  --output workspace/backend_comparison.json
```

KV-cache experiment:

```powershell
python backend/scripts/benchmark_kv_cache.py `
  --model qwen3:4b-instruct `
  --output workspace/kv_cache_benchmark.json
```

Launch Ollama using the persisted validated KV profile:

```powershell
.\backend\scripts\start_ollama_optimized.ps1
```

# Negative results retained intentionally

The project records unsuccessful experiments because they prevent repeated work and make the engineering history honest.

1. **Direct llama.cpp did not outperform Ollama** on this target laptop. The paired result favored Ollama by roughly 11%.
2. **Large fixed context windows are not justified** for normal requests. They consume significantly more RAM for only small decode-throughput differences in the Phase 8 measurement.
3. **q4_0 KV cache is currently unattractive at 2048**, despite smaller resident size, because throughput dropped substantially.
4. **Phase 10 v1 OS-RAM deltas cannot be used as a clean cross-cache memory comparison** because starting free-memory conditions varied materially.

# Known limitations and measurement discipline

When interpreting any benchmark in this file:

- results are machine-specific;
- laptop power mode, background processes, memory pressure and temperature affect CPU inference;
- quality canaries are smoke tests, not comprehensive task evaluations;
- resident model size and OS available RAM measure different things;
- block-order tests can be biased by sustained thermal load;
- a result should not be promoted to a runtime default when the benchmark methodology does not isolate the claimed improvement.

This is why the project sometimes keeps the conservative setting even when an experimental number looks promising.

# Next runtime research directions

Highest-value next steps:

1. **Phase 10 v2 controlled KV benchmark**
   - randomize/interleave cache order where possible;
   - return the machine to a comparable memory state between configurations;
   - separate quality/long-context thermal load from throughput measurement;
   - record process-private/RSS and resident model size alongside OS memory;
   - re-check q8_0 at 4096.

2. **Prompt/prefix cache evaluation**
   - repeated system prompts and stable industrial instruction prefixes;
   - measure prompt-eval savings separately from decode throughput.

3. **Pressure-aware keep-alive/residency**
   - shorten residency automatically under RAM pressure rather than using only a static 60-second policy.

4. **True LRU eviction**
   - current adaptive residency is not a complete usage-history-based LRU implementation.

5. **End-to-end SSE/event persistence profiling**
   - determine whether durable model-token event writes are a material application bottleneck after provider/runtime optimization.

6. **Broader quality suite**
   - replace smoke-only canaries with representative SovereignAI tasks: evidence-grounded synthesis, industrial extraction, tool selection, JSON/schema reliability, and long-context document work.

# Rule for future updates

Every new local-runtime experiment should append:

1. hypothesis;
2. machine/runtime state;
3. exact command/configuration;
4. benchmark design;
5. raw summary metrics;
6. quality result;
7. conclusion;
8. whether the runtime default changed;
9. limitations/confounders.

Do not overwrite negative results just because a later experiment performs better. Preserve them as part of the engineering record and explain why the later methodology is more trustworthy.
