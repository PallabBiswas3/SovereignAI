# SovereignAI Local LLM RAM & Latency Audit

**Audit date:** 2026-09-19  
**Branch reviewed:** `optimize-authorized-rag-latency`  
**Scope:** Current mechanisms that reduce, control, observe, or diagnose system RAM pressure and local-Ollama latency.

This is a **current-state architecture audit**, not a claim that every mechanism has produced a measured speedup. Measured results, implemented controls, and proposed improvements are separated below.

---

## 1. Current local-model topology

SovereignAI currently routes among three local Ollama roles:

| Role | Model | Intended use | Declared resource class |
|---|---|---|---|
| GENERAL | `qwen3:4b-instruct` | normal local text generation / synthesis | medium |
| VISION | `qwen3-vl:4b-instruct` | local visual evidence | medium |
| CODER | `qwen2.5-coder:7b` | code generation / repair | medium |

All three are local-only. The workbench recommends 16 GB system RAM, but that does **not** imply all models can remain resident simultaneously without pressure.

---

# 2. What is already implemented for RAM/resource control

## RAM-001 — Single generative-model admission by default

`ResourceScheduler` defaults to:

```text
max_gpu_model_jobs = 1
max_cpu_jobs = 2
```

Despite the historical `gpu` naming, this model-job admission also protects a CPU-only local setup by preventing multiple expensive generative jobs from being admitted concurrently through the same scheduler.

### Benefit

- reduces simultaneous model-generation memory pressure;
- prevents multiple LLM generations from fighting for CPU/memory bandwidth;
- makes latency more predictable under concurrent requests;
- exposes queue depth and queue wait.

### Important limitation

This is **concurrency admission**, not true memory-aware admission. It does not currently ask:

```text
How much free RAM exists?
How much RAM will this specific model/context require?
Should another loaded model be evicted first?
```

The configured `memory_requirement` is descriptive and used by routing metadata/scoring, but the scheduler does not turn it into a measured-MB admission calculation.

---

## RAM-002 — Separate CPU admission

A second semaphore bounds CPU-heavy non-generative work. This is used by local CPU pipelines such as reranking.

### Benefit

Prevents unlimited CPU-side jobs from competing with Ollama generation.

### Limitation

No dynamic adjustment from current CPU load or free RAM exists yet.

---

## RAM-003 — RAM/VRAM observability

The resource snapshot records:

```text
CPU usage
RAM used / total
optional NVIDIA VRAM used / total
active model jobs
active CPU jobs
queue depth
current model
last model
```

RAM comes from `psutil`; optional NVIDIA memory comes from `nvidia-smi`.

`/api/models/status` combines those resource measurements with model availability and lifecycle information.

### Benefit

This gives the system the measurements required for future adaptive admission.

### Limitation

Today those values are mainly **observed/reported**. They do not yet automatically alter model admission, context size, keep-alive, or model selection.

---

## RAM-004 — Ollama lifecycle / loaded-model observation

SovereignAI observes `/api/ps` and tracks model state as:

```text
NOT_INSTALLED
COLD
LOADING
READY
BUSY
IDLE
ERROR
```

Recorded metrics include load duration, TTFT, throughput, generation duration, output tokens, failure rate, warm/cold status, queue depth, and observed memory when available.

### Benefit

Cold-load cost can be separated from prompt/decode cost.

### Limitation

SovereignAI does not directly control Ollama's memory allocator. It also does not yet proactively unload a model when RAM crosses a threshold.

---

## RAM-005 — Configurable Ollama keep-alive

Current default:

```text
model_idle_timeout_seconds = 300
```

The Ollama provider uses that as the default `keep_alive` unless `SOVEREIGN_MODEL_KEEP_ALIVE` overrides it.

### Latency benefit

Keeping a recently used model resident avoids repeated cold loads.

### RAM trade-off

The exact same mechanism can increase memory pressure on a 16 GB machine because a no-longer-active model may remain loaded for several minutes.

This is therefore a **RAM-vs-cold-latency trade-off**, not a universally beneficial optimization.

A future experiment should make keep-alive adaptive to available RAM and expected next model.

---

# 3. What is already implemented for local-LLM latency

## LAT-001 — True token streaming

Ollama is consumed through streaming `/api/generate` NDJSON and model fragments are forwarded through SSE immediately.

### Benefit

Reduces **perceived latency** because the user sees the first output token instead of waiting for full generation.

### Important distinction

Streaming does not necessarily reduce total compute time or RAM use. Its main user-facing benefit is TTFT/interaction responsiveness.

---

## LAT-002 — Cancellation and total generation deadline

Generation can be cancelled when the task is deleted or the last SSE client disconnects. A total generation timeout also exists.

### Benefit

Stops useless work from consuming CPU/RAM after the user no longer needs it.

### Limitation

Ollama ultimately controls how quickly the underlying generation computation is released after the HTTP stream closes.

---

## LAT-003 — `think: false` / direct-response generation

The Ollama payload explicitly requests:

```text
think = false
```

and Qwen3 direct-prompt handling exists for applicable tags.

### Benefit

Avoids intentionally invoking a long reasoning mode for ordinary industrial responses and reduces unnecessary hidden/output computation where supported by the local model/runtime.

---

## LAT-004 — Deterministic FAST/STANDARD/DEEP execution selection

FAST mode is used for bounded short factual lookups. Its agent plan contains only one local generation step, while STANDARD/DEEP retain more preparation/verification behavior.

### Benefit

Avoids expensive workflow depth when the task does not require it.

### Limitation

The main LLM generation still dominates many workloads, so execution-mode selection alone cannot solve long decode times.

---

## LAT-005 — Role-specific routing instead of always using the largest model

The model router scores capability match while applying latency and resource penalties. Ordinary text/document work normally remains on the 4B GENERAL model; the 7B coder is reserved for coding-oriented tasks and the VLM for vision.

### Benefit

Avoids paying 7B/vision-model cost for requests that the 4B model can handle.

### Current limitation

All current configured models are labelled `memory_requirement: medium`, so resource scoring is not very discriminative. The scoring penalty is static and does not incorporate live free RAM or live measured model speed.

---

## LAT-006 — Versioned local cache

SQLite-backed caches exist for:

```text
OCR
embeddings
retrieval
vision
selected deterministic workflows
```

Cache identity includes the data/model/pipeline versions needed to prevent stale reuse.

Final user-specific LLM answers are intentionally **not** cached.

### Benefit

Avoids repeating expensive preprocessing, embedding, retrieval, OCR, vision, and deterministic work around the LLM.

### Limitation

A repeated conversational answer still requires full model generation by design.

---

# 4. Context-size work already done

Context reduction is currently one of the strongest latency/RAM strategies in the project.

## LAT-CTX-001 — Batch 2 `ContextCompiler`

For evidence workflows, the compiler:

- orders evidence;
- removes exact duplicates;
- removes safe near-duplicates;
- preserves conflicting revisions;
- extracts relevant sentences from large chunks;
- enforces evidence-count and evidence-token budgets;
- reserves output capacity.

One recorded fixture reduced:

```text
20 candidates / 444 estimated evidence tokens
        ↓
5 fragments / 116 compiled tokens
```

with token ratio `0.2584`, while retaining the required fact in that small fixture.

### Benefit

Smaller prompts reduce prompt evaluation work and model context/KV-cache use.

### Important note

This was a small synthetic context-preservation evaluation, not a universal production latency claim.

---

## LAT-CTX-002 — New Authorized/RAG model-facing compaction

The initial local system benchmark exposed an especially slow Authorized/RAG path:

```text
~185 s total latency
~4.8 generated tokens/s
559-805 output tokens
```

while General Chat in the same broader benchmark was about:

```text
~10.6 generated tokens/s
```

and RAM pressure was reported around 92-96% during much of the observed run.

Before the new optimization, Authorized generation could receive:

```text
6 evidence chunks normally
10 for broad briefings
full text
num_ctx = 8192
num_predict = 1280
```

The current experimental branch now compacts only the **model-facing Authorized packet**.

### Routine Authorized query

```text
max evidence chunks = 4
max evidence text/chunk = 1000 characters
num_ctx = 4096
num_predict = 384
```

### Broad management / multi-domain query

```text
max evidence chunks = 6
max evidence text/chunk = 1600 characters
num_ctx = 6144
num_predict = 640
```

### General Chat

Remains:

```text
num_ctx = 8192
num_predict = 1280
```

### Why this is important for RAM

The context window influences KV-cache/context working memory. A smaller `num_ctx` can therefore reduce runtime memory requirements as well as prompt-processing cost.

### Safety invariant

Only the model-facing packet is compacted. Full retrieved evidence, provenance, source records and audit state remain outside that truncation step.

### Current status

**Implemented but not yet accepted as a measured optimization.**

It must still pass the local before/after benchmark and evidence-quality regression tests.

---

# 5. Output-token reduction already done

Authorized generation previously allowed `num_predict = 1280` and observed outputs reached 559-805 tokens.

The experimental budget now limits routine Authorized answers to 384 output tokens and broad briefings to 640.

Why this matters on CPU:

```text
600 output tokens / 4.8 tok/s ≈ 125 s decode time
800 output tokens / 4.8 tok/s ≈ 167 s decode time
```

This arithmetic is only decode-time intuition, not an end-to-end prediction, but it shows why output length can dominate local CPU latency.

The system already records `done_reason` and `output_truncated`, so a speed improvement that simply cuts off required answers can be detected and rejected.

---

# 6. Benchmark / observability work already done

## System benchmark

`scripts/benchmark_system.py` separates:

```text
chat_sync
task_general_stream
task_authorized_stream
integration_graph_controlplane
integration_full_industrial
```

and records host CPU/RAM plus component timings.

## Focused General-vs-Authorized benchmark

`scripts/benchmark_authorized_latency.py` now records:

```text
total latency
client TTFT
model TTFT
prompt tokens
output tokens
tokens/s
load duration
queue wait
model duration
model prompt characters
requested num_ctx
requested num_predict
whether the prompt was compacted
```

This is specifically intended to determine whether the Authorized 4.8 tok/s result came mainly from prompt/context pressure, model loading, output length, scheduler contention, or broader machine resource pressure.

---

# 7. What is NOT implemented yet

These gaps are important because they define the real next phase of RAM/local-inference engineering.

## GAP-RAM-001 — No free-RAM-aware admission

Current scheduler behavior is roughly:

```text
one model generation at a time
```

not:

```text
estimate required memory
measure available memory
admit / wait / unload / choose smaller context
```

---

## GAP-RAM-002 — No automatic model eviction policy

There is no SovereignAI policy such as:

```text
RAM > 90% -> unload idle coder/VLM
switching 7B -> 4B -> unload old model first
```

Ollama keep-alive is configured, but active pressure-based eviction is not yet implemented.

---

## GAP-RAM-003 — No adaptive context size from live memory

Authorized context budgets are currently static by request category.

A future version could use something like:

```text
available RAM high   -> 6144 context
available RAM medium -> 4096 context
available RAM low    -> 3072 or stronger evidence compression
```

but only after quality experiments define safe lower bounds.

---

## GAP-RAM-004 — No model-specific measured memory estimates

The registry says `medium` rather than recording experimentally measured resident RAM/VRAM for each local tag/context configuration.

Therefore the scheduler cannot yet predict the cost of:

```text
Qwen 4B @ 4K
Qwen 4B @ 8K
Qwen Coder 7B
Qwen-VL 4B
```

on the actual target machine.

---

## GAP-LAT-001 — Structured JSON generation remains large-budget

The structured `generate_json` path still uses:

```text
num_ctx = 8192
num_predict = 4096
```

This should be audited separately. It may be appropriate for some schemas but is potentially expensive for constrained local hardware.

---

## GAP-LAT-002 — No adaptive generation budget from requested answer shape

Aside from the new Authorized split, generation does not yet derive output budget from whether the user asks for:

```text
one value
three bullets
short explanation
long report
structured object
```

---

## GAP-LAT-003 — No low-level Ollama tuning experiments recorded yet

The project has not yet established controlled experiments for items such as:

- thread count;
- batch size;
- KV-cache quantization where supported;
- Ollama/llama.cpp memory mapping behavior;
- flash-attention support where applicable;
- model quantization variant comparisons;
- CPU affinity / power mode;
- parallel model loading policy.

These should not be changed blindly. They need benchmarked experiments because speed, RAM, and answer quality can move in different directions.

---

# 8. Current architectural assessment

## Strongest work already present

1. **True streaming and TTFT instrumentation** — good user-facing latency foundation.
2. **Single-generation admission** — sensible protection for a 16 GB local machine.
3. **Warm/cold/load/queue/tokens-s observability** — enough instrumentation to perform real experiments.
4. **Versioned preprocessing/retrieval caching** — avoids repeated non-generation cost.
5. **Role-specific local-model routing** — avoids using coder/VLM for every request.
6. **Evidence/context compaction** — directly attacks prompt/KV cost.
7. **Output-token budgeting for Authorized mode** — directly attacks CPU decode time.
8. **Route-specific benchmark harnesses** — optimization can now be evidence-driven.

## Biggest remaining technical weakness

The project is **resource-observant, not yet resource-adaptive**.

It knows:

```text
RAM usage
VRAM where available
model loaded state
queue depth
load time
TTFT
tokens/s
```

but does not yet feed those observations back into a dynamic decision loop for:

```text
model eviction
keep-alive
context budget
output budget
model choice
request admission
```

That feedback loop is the logical next major upgrade.

---

# 9. Recommended next experiment sequence

Do these only after the current Authorized/RAG optimization is benchmarked.

## EXP-RAM-A — Per-model resident-memory profile

Measure the actual target laptop for each model at controlled context sizes.

Record:

```text
cold load RAM
warm idle RAM
4K generation peak RAM
6K/8K peak RAM
TTFT
tok/s
load time
```

This gives the scheduler real capacity data.

## EXP-RAM-B — Adaptive keep-alive / eviction

Compare:

```text
300 s keep-alive
60 s keep-alive
pressure-triggered unload
role-switch unload
```

Measure both repeated-query speed and worst-case RAM.

## EXP-RAM-C — Context vs RAM vs quality curve

For the same Authorized evaluation set:

```text
3072
4096
6144
8192
```

Measure:

```text
RAM
prompt tokens
TTFT
tok/s
total time
citation coverage
answer completeness
unsupported claim rate
```

## EXP-LAT-A — Output budget curve

Test controlled limits such as:

```text
256
384
512
640
```

for fixed workloads and measure truncation/completeness.

## EXP-LAT-B — Structured-generation budget audit

Profile every `generate_json` caller and replace the global 8K/4096 budget with schema/task-appropriate limits only where correctness permits.

## EXP-LAT-C — Low-level runtime tuning

Only after the previous bottlenecks are understood, experiment with quantization/runtime/thread/KV settings one variable at a time.

---

# 10. Bottom line

SovereignAI already has meaningful local-inference systems engineering. The work is more than simply "run Ollama": it has scheduling, lifecycle observation, true streaming, caching, mode selection, model routing, context compilation, output budgeting, and performance benchmarks.

However, the current RAM strategy should not yet be described as fully dynamic memory management.

The accurate description is:

> **SovereignAI currently uses bounded local-model concurrency, configurable model residency, context/output budgets, route-specific caching, and detailed resource/inference telemetry to operate on constrained hardware. The next step is to turn those measurements into adaptive RAM-aware admission, eviction, context, and model-residency decisions.**
