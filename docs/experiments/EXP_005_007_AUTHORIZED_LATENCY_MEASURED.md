# EXP-005 / EXP-006 / EXP-007 — Measured Authorized/RAG Latency Result

**Date:** 2026-09-19  
**Status:** Measured; partial success, current 384-token routine budget not accepted as final  
**Branch:** `optimize-authorized-rag-latency`  
**Related PR:** #6  
**Benchmark:** `scripts/benchmark_authorized_latency.py --runs 3`

---

## 1. Experiment purpose

This run measured the first Authorized/RAG inference optimization after the earlier system benchmark showed approximately:

```text
Authorized total latency: ~184.9-185.1 s
Authorized TTFT:          ~65.6 s
Authorized throughput:    ~4.82 tok/s
Observed output:           ~559-805 tokens
```

The optimization changed only the model-facing Authorized prompt budget while preserving retrieval results, source/provenance records, authorization state, and GraphRAG internals.

Routine Authorized inference was changed from approximately:

```text
num_ctx = 8192
num_predict = 1280
6 full evidence chunks
```

to:

```text
num_ctx = 4096
num_predict = 384
max 4 model-facing evidence chunks
max 1000 characters per model-facing evidence chunk
```

---

# 2. Measured summary

## General route

```text
successes:                  3/3
total mean:                 15.5642 s
client TTFT mean:            8.5738 s
tokens/s mean:               9.262
output tokens mean:         57.67
prompt tokens mean:         85
load duration mean:          4.8022 s
queue wait mean:             0.0 s
model duration mean:        14.8796 s
model prompt characters:    75
num_ctx:                  8192
num_predict:              1280
prompt compacted:          false
```

## Authorized route

```text
successes:                  3/3
total mean:                139.1413 s
client TTFT mean:           63.2587 s
tokens/s mean:               5.1987
output tokens mean:        384
prompt tokens mean:       1286
load duration mean:          4.0651 s
queue wait mean:             0.0 s
model duration mean:       136.8523 s
model prompt characters:  3395
num_ctx:                  4096
num_predict:               384
prompt compacted:           true
```

Every Authorized run reported:

```text
done_reason = length
output_truncated = true
```

---

# 3. Before vs after

Using the prior Authorized system-benchmark mean of approximately `184.9 s` as the comparison baseline:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Authorized total latency | ~184.9 s | 139.14 s | **~24.7% lower** |
| Authorized TTFT | ~65.6 s | 63.26 s | **~3.6% lower** |
| Authorized throughput | ~4.82 tok/s | 5.20 tok/s | **~7.9% higher** |
| Output tokens | ~559-805 observed | 384 exactly | bounded, but **100% truncated** |
| Context budget | 8192 | 4096 | 50% lower |
| Output budget | 1280 | 384 | 70% lower |

The workloads are closely related but the original and focused benchmark harnesses are not identical, so percentages should be treated as engineering comparison evidence rather than a formal controlled-study confidence interval.

---

# 4. Main findings

## Finding A — The first optimization reduced total latency materially

Authorized mean latency fell from roughly 185 s to 139 s.

This is a useful improvement and shows that bounding output/context was directionally correct.

However, the mechanism is not yet accepted as the final design because all three outputs were truncated.

---

## Finding B — Output-token reduction explains a large part of the latency gain

The Authorized run generated exactly 384 tokens in every iteration.

At the measured average `5.1987 tok/s`:

```text
384 / 5.1987 ~= 73.9 s
```

of decode time is expected from output generation alone.

Observed mean elapsed time after client first token was approximately:

```text
139.14 - 63.26 ~= 75.88 s
```

which is very close to the decode-time estimate.

### Interpretation

The post-first-token portion is now well explained by token generation itself.

Therefore reducing unnecessary output tokens was effective, but lowering the limit to 384 went too far for this benchmark request because the model reached the length ceiling every time.

---

## Finding C — The dominant remaining problem is before the first token

Authorized TTFT remains:

```text
63.26 s
```

versus General:

```text
8.57 s
```

Authorized TTFT is therefore about:

```text
7.4x General TTFT
```

This is the most important remaining latency problem.

---

## Finding D — Prompt size is now the strongest measured explanation

General prompt tokens:

```text
85
```

Authorized prompt tokens:

```text
1286
```

So the Authorized model processes roughly:

```text
15.1x as many prompt tokens
```

before answering.

The Authorized prompt contains 3395 model-prompt characters versus 75 for the benchmark General user prompt, plus the Authorized system prompt and tokenizer overhead.

### Interpretation

This strongly supports the hypothesis that prompt evaluation/context work is a major cause of the ~63 s TTFT.

It does **not** yet prove RAM pressure is the only cause. Prompt length, context/KV work, memory bandwidth, and full-stack CPU contention may all contribute.

---

## Finding E — Model loading is not the main bottleneck

Authorized load duration mean:

```text
4.07 s
```

Authorized model duration mean:

```text
136.85 s
```

Load time is only around 3% of model duration in this benchmark.

All Authorized runs were marked warm.

### Decision

Do not prioritize cold-load/keep-alive tuning as the next latency experiment for this route.

---

## Finding F — Scheduler contention is not the cause

Every General and Authorized run reported:

```text
queue_wait_seconds = 0.0
```

### Decision

Do not optimize model scheduler queueing for this single-request workload yet.

The scheduler may still matter under concurrent load, but it does not explain this benchmark.

---

## Finding G — Decode throughput remains substantially lower in Authorized mode

Measured:

```text
General:    9.262 tok/s
Authorized: 5.199 tok/s
```

Authorized decode speed is only about 56% of General, or approximately 44% lower.

This gap remains even though `num_ctx` was reduced from 8192 to 4096.

### Candidate explanations still open

1. larger active prompt/KV state;
2. RAM/memory-bandwidth pressure from the full stack;
3. CPU contention from retrieval/services/processes;
4. model/runtime behavior at larger active sequence length;
5. machine-level thermal/power effects across long sequential runs.

The current focused benchmark did not record host RAM/CPU during each generation, so it cannot yet distinguish these causes.

---

# 5. Quality / correctness warning

The current routine output budget is **not acceptable as final solely from this benchmark** because:

```text
3/3 Authorized runs -> done_reason=length
3/3 Authorized runs -> output_truncated=true
```

A 100% truncation rate violates the earlier acceptance criterion that routine answers should complete naturally.

The benchmark metrics alone also do not contain the generated answer text, so citation correctness, evidence coverage, unsupported-claim rate, sentence completeness, and final-answer usefulness cannot be assessed from this JSON alone.

---

# 6. Decision on EXP-005 / EXP-006 / EXP-007

## EXP-005 — Context reduction

**State: measured, continue.**

Reducing the Authorized inference context from 8192 to 4096 did not break execution and is directionally useful. However, TTFT remains ~63 s, so prompt/context optimization needs another iteration.

## EXP-006 — 384-token output cap

**State: measured, current value rejected as final.**

The cap reduced total generation time but caused 100% output truncation.

Do not return globally to 1280. Instead test an intermediate budget such as 512 while making the prompt/evidence packet materially smaller.

## EXP-007 — Throughput diagnosis

**State: measured; root cause narrowed but not fully isolated.**

The benchmark rules out scheduler queueing and makes model loading a minor contributor. Prompt size/context work and machine resource pressure are now the primary investigation targets.

---

# 7. Next experiment — prompt-first optimization

The next optimization should **not** focus mainly on further reducing output length.

Target the ~63 s prefill/TTFT path.

## Proposed configuration

For the same routine Authorized benchmark, test a more compact evidence representation such as:

```text
num_ctx = 4096
num_predict = 512
model-facing evidence chunks = 3
per-chunk text budget = ~600-800 characters
```

But raw character truncation should gradually be replaced by **query-aware evidence sentence selection** so important evidence is preserved.

### Why increase output 384 -> ~512?

Because all current outputs truncate.

### Why can total latency still improve?

Because the larger output budget should be offset by aggressively reducing prompt processing, which is currently the unresolved ~63 s front-half bottleneck.

---

# 8. Better prompt compaction strategy

Instead of sending long raw chunk text, the next model-facing evidence packet should prefer:

```text
source/citation identity
+ most query-relevant evidence sentence(s)
+ technical values/units/limits
+ revision/page/section
```

rather than:

```text
source metadata
+ 1000 characters of mostly raw chunk text
```

Full evidence must remain retained outside the generation prompt for audit/provenance.

Suggested target for the next experiment:

```text
Authorized prompt tokens: 1286 -> <700 initially
```

This target is experimental and must be evaluated against citation/evidence quality.

---

# 9. Resource instrumentation needed in the next run

The focused benchmark should also record host resources around every General/Authorized request:

```text
RAM used / available before
RAM used / available at first token if practical
RAM used / available after
CPU percent
Ollama resident-model information
```

This is necessary to test whether the remaining 5.2 tok/s throughput is associated with paging or high memory pressure.

---

# 10. Acceptance criteria for next iteration

A subsequent Authorized optimization should aim for all of the following rather than optimizing one metric alone:

```text
success rate               = 100%
output_truncated            = false for normal benchmark request
prompt tokens               materially below 1286
TTFT                        materially below 63 s
Authorized tok/s            improve toward General
citations                   preserved and correct
unsupported claims          no regression
answer completeness         preserved
```

A realistic immediate target is to get routine Authorized latency below 100 s first while eliminating systematic truncation. A more aggressive target can be set only after the next prompt/RAM measurements show what the hardware can sustain.

---

# 11. Raw measured values

### Iteration 1

General:

```text
total 12.1127 s
TTFT 7.6641 s
12.589 tok/s
54 output tokens
85 prompt tokens
cold
```

Authorized:

```text
total 116.1813 s
TTFT 55.3907 s
6.350 tok/s
384 output tokens
1286 prompt tokens
warm
length-truncated
```

### Iteration 2

General:

```text
total 15.9837 s
TTFT 9.2253 s
8.466 tok/s
55 output tokens
85 prompt tokens
warm
```

Authorized:

```text
total 149.1632 s
TTFT 65.1780 s
4.592 tok/s
384 output tokens
1286 prompt tokens
warm
length-truncated
```

### Iteration 3

General:

```text
total 18.5962 s
TTFT 8.8321 s
6.731 tok/s
64 output tokens
85 prompt tokens
warm
```

Authorized:

```text
total 152.0793 s
TTFT 69.2075 s
4.654 tok/s
384 output tokens
1286 prompt tokens
warm
length-truncated
```

The large run-to-run throughput variation itself should be retained as evidence; a later 10-run benchmark should report p50/p95 after the next correctness-preserving optimization is selected.
