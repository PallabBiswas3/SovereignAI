# SovereignAI Engineering Experiment & Optimization Log

This document is the persistent engineering memory for SovereignAI.

It records **why a change was attempted, what was changed, how it was tested, what actually happened, what benefit was obtained, what trade-offs were introduced, and what should happen next**. The purpose is to prevent repeated mistakes, preserve research reasoning, make later ablations possible, and turn project development into reproducible engineering evidence rather than a sequence of undocumented code changes.

> **Rule:** do not claim an optimization worked until a measurement demonstrates it. Expected benefits and measured benefits must remain clearly separated.

> **Scope rule:** GraphRAG internal graph/retrieval architecture is considered complete/frozen. Reliability at its service boundary and integration with the rest of SovereignAI may be changed, but the internal GraphRAG design should not be redesigned unless that scope is explicitly reopened.

---

## 1. How every future experiment must be recorded

Create a new `EXP-XXX` section **before or together with the implementation**. Never delete a failed experiment; mark it rejected or superseded and explain why.

Use this structure:

```text
EXP-XXX — Short title
Date:
Status: proposed | implemented-unverified | measured | accepted | rejected | superseded
Area:
Related branch / PR / commits:

Problem
Baseline
Hypothesis
Constraints / invariants
Experiment / implementation
Metrics to collect
Acceptance criteria
Result
Interpretation
Benefit
Trade-offs / risks
Decision
Next experiment
Reproduction commands
Artifacts
```

### Evidence states

| State | Meaning |
|---|---|
| **Proposed** | Idea only. No implementation yet. |
| **Implemented-unverified** | Code exists, but the relevant local/system benchmark has not yet proved the benefit. |
| **Measured** | Measurements exist, but the design decision is still being evaluated. |
| **Accepted** | Measurement and correctness evidence justify keeping the approach. |
| **Rejected** | Experiment did not provide enough benefit or caused unacceptable regressions. |
| **Superseded** | Replaced by a later experiment; retained for historical reasoning. |

### Measurement rules

1. Keep the **before** measurement.
2. Record machine/runtime conditions when they materially affect the result.
3. Compare the same workload before and after whenever possible.
4. Separate:
   - TTFT,
   - prompt processing,
   - decode/generation,
   - output token count,
   - tokens/s,
   - model load time,
   - scheduler queue time,
   - retrieval/service time,
   - overall user-visible latency.
5. Record correctness/safety regressions as seriously as speed regressions.
6. A faster result that loses required evidence, citations, authorization, verification, or safe failure behavior is **not** an accepted optimization.
7. For local-model experiments, note warm/cold state and RAM pressure.
8. Do not generalize one route's latency to the entire SovereignAI system.

---

# Experiment index

| ID | Area | Goal | Current state |
|---|---|---|---|
| EXP-001 | Integration reliability | Retry transient service failures safely | Implemented; focused tests added |
| EXP-002 | Workflow truthfulness | Prevent unavailable services from being reported as completed | Implemented |
| EXP-003 | System benchmarking | Measure SovereignAI route-by-route instead of guessing | Implemented; local benchmark required |
| EXP-004 | Industrial integration compatibility | Preserve diagnostic request compatibility and exact failure attribution | Implemented; regression tests added |
| EXP-005 | Authorized/RAG latency | Reduce ~185 s authorized generation latency | Implemented-unverified |
| EXP-006 | Output-token control | Stop unnecessarily long authorized answers | Implemented-unverified |
| EXP-007 | Authorized decode throughput | Explain and reduce ~4.8 tok/s vs ~10.6 tok/s general-chat gap | Instrumented; measurement pending |

---

# EXP-001 — Bounded retry for local integration services

**Date:** 2026-09-18  
**Status:** Implemented; focused regression tests added  
**Area:** GraphRAG / Time-Series Diagnostic Agent / ControlPlane service boundaries  
**Related work:** `improve-reliability-runtime-controlplane`, PR #3

## Problem

The shared JSON integration client performed one request only. A temporary timeout, connection reset, HTTP 503, or similar transient local-service problem immediately failed the workflow even when a second attempt could succeed.

This is especially undesirable for a multi-service local industrial workflow because short-lived service startup/resource-contention failures are expected on constrained hardware.

## Baseline

Before this experiment:

```text
request -> one HTTP attempt -> success OR immediate failure
```

No bounded retry/backoff or transient/permanent failure classification existed in the shared integration client.

## Hypothesis

Retrying only clearly transient failures a small bounded number of times should improve day-to-day reliability without hiding permanent errors or producing uncontrolled request storms.

## Constraints / invariants

- Do not change GraphRAG internals.
- Do not retry arbitrary application errors.
- Keep attempts bounded.
- Preserve fail-open/fail-closed orchestration policy.
- Expose whether an error was retryable and how many attempts occurred.

## Experiment / implementation

Added transient-status classification:

```text
408, 425, 429, 500, 502, 503, 504
```

Added bounded handling for:

- `httpx.TimeoutException`
- `httpx.NetworkError`
- `httpx.RemoteProtocolError`

Default policy:

```text
max_retries = 2
initial_backoff = 0.25 s
max_backoff = 2.0 s
```

Permanent HTTP errors, malformed successful responses, and unrelated value errors are not blindly retried.

`IntegrationServiceError` now carries service attribution, status code, attempt count, and retryability.

## Metrics / tests

Focused tests cover:

1. HTTP 503 -> retry -> success.
2. HTTP 400 -> no retry.
3. repeated HTTP 503 -> retry budget exhausted.
4. connection error -> retry -> success.

## Expected benefit

- Better resilience to temporary local-service failures.
- Clearer operator/debugging information.
- No silent infinite retry behavior.
- Same GraphRAG architecture.

## Trade-offs / risks

- A failing transient request can now take longer because retries are deliberate.
- Non-idempotent operations would require stronger semantics; this retry policy is intended for the existing bounded integration calls.

## Decision

Keep bounded retry at integration boundaries. Do **not** automatically apply the same policy to partially streamed LLM generation because retrying after emitted tokens can duplicate or diverge from the original generation.

---

# EXP-002 — Truthful degraded workflow state

**Date:** 2026-09-18  
**Status:** Implemented  
**Area:** Agent workflow / integration state reporting  
**Related work:** `improve-reliability-runtime-controlplane`, PR #3

## Problem

In fail-open conditions, GraphRAG or the diagnostic service could be unavailable while their individual workflow steps were still represented as completed. The overall workflow might legitimately continue, but the step-level state was misleading.

## Hypothesis

Separating **overall workflow continuation** from **individual service success** should preserve fail-open behavior while making the audit trail truthful.

## Implementation

Individual GraphRAG/diagnostic steps are now marked completed only when that service reports available. Otherwise the individual step is failed and the workflow records an explicit warning/error such as service unavailable after bounded retry.

## Benefit

- No false claim that evidence was retrieved when the evidence service was unavailable.
- Better auditability.
- Easier failure diagnosis.
- Fail-open/fail-closed policy remains an explicit orchestration decision rather than being hidden in step status.

## Decision

Accepted architectural principle:

```text
overall workflow success != every dependency succeeded
```

Each dependency must report its own truthful state.

---

# EXP-003 — Overall-system benchmark harness

**Date:** 2026-09-18  
**Status:** Implemented; real local execution required for performance claims  
**Area:** Evaluation / performance / reliability  
**Related work:** `benchmark-overall-system`, PR #4  
**Detailed runbook:** `docs/SYSTEM_BENCHMARK.md`

## Problem

SovereignAI had component tests and isolated model benchmarks, but those did not answer the practical question:

> How fast and reliable is the system a user actually interacts with?

A single model tokens/s number cannot represent retrieval, authorization, agent planning, ControlPlane, GraphRAG, diagnostics, SSE streaming, cold starts, or resource pressure.

## Hypothesis

A route-specific benchmark that records user-visible latency and component metrics will expose the actual bottleneck and prevent optimization by intuition.

## Implementation

Added `scripts/benchmark_system.py` covering:

1. backend/service health,
2. synchronous `/api/chat`,
3. streamed general task TTFT + total latency,
4. streamed Authorized/RAG task,
5. GraphRAG + ControlPlane integration,
6. optional GraphRAG + Time-Series Diagnostic Agent + ControlPlane path,
7. CPU/RAM snapshots,
8. component timing reports.

Generated artifacts:

```text
reports/system_benchmark/system_benchmark.json
reports/system_benchmark/system_benchmark.csv
reports/system_benchmark/system_benchmark.md
```

## Benchmarking principle obtained

Do not quote one overall latency number. Report routes separately:

```text
General chat
Authorized/RAG
GraphRAG + ControlPlane
Full industrial integration
Deep/controlled workflows
```

## Benefit

This harness converts future optimization work into an experimental loop:

```text
baseline -> hypothesis -> code change -> same benchmark -> compare -> keep/reject
```

## Limitation

GitHub CI cannot reproduce the user's local CPU/RAM/Ollama performance. Real latency must be measured on the machine running the stack.

---

# EXP-004 — Diagnostic compatibility and exact failure attribution

**Date:** 2026-09-19  
**Status:** Implemented; regression tests added  
**Area:** Industrial integration  
**Related branch:** `fix-industrial-integration-attribution`

## Problem A — diagnostic run-context compatibility

The synthetic benchmark and the Time-Series Diagnostic Agent did not use exactly the same expected `run_context` structure. Benchmark metadata such as `synthetic=true` needed to be retained without violating the downstream request contract.

## Approach

Normalize legacy/synthetic context into the diagnostic request's metadata while preserving SovereignAI as the calling source.

Expected normalized form conceptually becomes:

```text
run_context.source = sovereign-ai
run_context.metadata.synthetic = true
```

rather than placing arbitrary benchmark fields at the top level.

## Problem B — failure attribution

A failed integrated request must identify which dependency actually failed. A GraphRAG 503 and a diagnostic-agent 422 should not collapse into an ambiguous generic integration failure.

## Tests added

Regression coverage verifies:

- GraphRAG failure retains `service=graph-rag` and HTTP status.
- Diagnostic failure retains `service=time-series-diagnostic-agent`, status, and retryability.
- legacy benchmark context is normalized into diagnostic metadata.

## Benefit

- Better compatibility between the benchmark and real diagnostic API.
- More useful incident/debugging information.
- Correct retry decisions because transient and permanent failures remain distinguishable.
- Better evidence for reliability experiments.

---

# EXP-005 — Authorized/RAG latency reduction

**Date:** 2026-09-19  
**Status:** **Implemented-unverified** — rerun on local machine before accepting  
**Area:** Local inference / Authorized Knowledge mode  
**Related branch:** `optimize-authorized-rag-latency`  
**Related PR:** #6

## Problem

The first local benchmark showed the Authorized/RAG generation path taking approximately:

```text
~185 s total
~4.8 generated tokens/s
559-805 output tokens in observed runs
```

General chat in the same broader benchmark was approximately:

```text
~10.6 generated tokens/s
```

RAM pressure during the broader run was high (approximately 92-96% used in the observed benchmark period).

These are baseline observations, not universal SovereignAI numbers.

## Observed design before optimization

Authorized generation could present:

```text
6 evidence chunks normally
10 evidence chunks for broad briefings
full retrieved chunk text
num_ctx = 8192
num_predict = 1280
```

This created two likely costs:

1. excessive prompt/context work and KV-cache pressure,
2. unnecessarily long generated responses.

## Hypothesis

For ordinary evidence-grounded questions, the model does not need an 8K generation context containing six complete retrieved chunks or an allowance of 1,280 new tokens. Reducing only the **model-facing packet** should lower prompt cost, memory pressure, and decode time while leaving retrieval/audit evidence intact.

## Critical constraint

Do **not** change:

- GraphRAG internal retrieval/graph design,
- persisted retrieved evidence,
- source records,
- audit state,
- authorization decision,
- citation identifiers.

Only the evidence packet sent into local model inference is compacted.

## Implementation

The Ollama boundary detects the existing:

```text
AUTHORIZED_CONTEXT_START
...
AUTHORIZED_CONTEXT_END
```

envelope.

### Routine authorized request budget

```text
model-facing evidence: max 4 chunks
evidence text: max 1,000 characters/chunk
num_ctx: 4096
num_predict: 384
```

### Broad multi-domain / management briefing budget

```text
model-facing evidence: max 6 chunks
evidence text: max 1,600 characters/chunk
num_ctx: 6144
num_predict: 640
```

### General Chat

General Chat remains unchanged:

```text
num_ctx: 8192
num_predict: 1280
```

## Why this is safer than changing retrieval

The retriever can still retrieve and preserve its normal result set. Compaction happens after retrieval and only for the local generation prompt. Therefore the system can retain full source/audit information even while the model receives a smaller inference packet.

## Metrics added

Runtime metadata now includes:

```text
requested_num_ctx
requested_num_predict
prompt_compacted
model_prompt_characters
prompt_token_count
output token_count
time_to_first_token_seconds
tokens_per_second
total_duration_seconds
load_duration_seconds
queue_wait_seconds
warm_status
```

## Acceptance criteria

This experiment should be accepted only if the local rerun shows:

1. a substantial Authorized total-latency decrease,
2. lower prompt token count/model prompt size,
3. output bounded appropriately for routine questions,
4. no material loss of evidence-grounded answer quality,
5. citations/provenance still usable,
6. no increase in unsupported organizational claims,
7. no truncation of answers that genuinely need the broad budget.

## Current result

**Pending local benchmark.**

No latency improvement is claimed yet.

## Expected benefit

- Smaller KV-cache/context workload.
- Lower prompt evaluation cost.
- Shorter response generation.
- Lower RAM pressure during routine Authorized queries.
- Better user-visible latency.

## Risks / trade-offs

- Evidence text truncation could omit a relevant passage if the selected chunk is unusually long.
- 384 output tokens may be too small for some complex questions.
- A fixed character limit is a practical first optimization, not necessarily the final optimal evidence-compression strategy.

## Future direction if quality drops

Prefer smarter context selection/compression rather than simply restoring the entire 8K prompt. Possible experiments:

- query-aware sentence extraction inside retrieved chunks,
- evidence deduplication,
- source-diversity-aware packing,
- token-based rather than character-based budget,
- adaptive prompt budget from execution mode and available RAM.

---

# EXP-006 — Reduce unnecessary Authorized output tokens

**Date:** 2026-09-19  
**Status:** **Implemented-unverified**  
**Area:** Generation budget  
**Related to:** EXP-005

## Problem

Observed Authorized responses generated roughly 559-805 tokens during the slow benchmark. At ~4.8 tok/s, output length alone can account for a very large part of user-visible latency.

Approximate decode-only intuition:

```text
600 tokens / 4.8 tok/s ~= 125 s
800 tokens / 4.8 tok/s ~= 167 s
```

This does not include prompt evaluation, load, routing, or retrieval, so it is not an end-to-end prediction; it demonstrates why output length must be treated as a first-class performance variable.

## Hypothesis

Most routine Authorized questions can be answered with evidence citations in substantially fewer than 500-800 generated tokens.

## Implementation

Routine Authorized:

```text
num_predict = 384
```

Broad briefing:

```text
num_predict = 640
```

The existing prompt already requests compact output and discourages repeated facts/duplicate summary tables.

## Acceptance criteria

Do not evaluate only speed. Check:

- required sections still completed,
- evidence identifiers remain present,
- no sentence is cut off due to `done_reason=length`,
- broad requests are correctly assigned the larger budget.

If `output_truncated=true` becomes common, adjust the classification/budget rather than globally restoring 1,280 tokens.

## Expected benefit

At the same decode speed, avoiding hundreds of unnecessary tokens can save tens of seconds to more than a minute on CPU-only inference.

## Current result

Pending local measurement.

---

# EXP-007 — Explain 4.8 tok/s Authorized vs 10.6 tok/s General Chat

**Date:** 2026-09-19  
**Status:** Instrumented; measurement pending  
**Area:** Local inference performance diagnosis

## Problem

The initial benchmark showed a major decode-throughput difference:

```text
Authorized/RAG: ~4.8 tok/s
General chat:   ~10.6 tok/s
```

Output length explains total latency but does not by itself explain why measured decode throughput was lower.

## Candidate causes

Current hypotheses, not conclusions:

1. much larger Authorized prompt/context,
2. 8K context allocation / larger KV cache,
3. high host RAM pressure and memory bandwidth contention,
4. model cold/warm/load-state differences,
5. scheduler queue/resource contention,
6. different prompt token counts,
7. other concurrent local services competing for CPU/RAM.

## Experiment

Added focused comparator:

```text
scripts/benchmark_authorized_latency.py
```

It compares General and Authorized routes while capturing:

- total latency,
- TTFT,
- prompt tokens,
- output tokens,
- tokens/s,
- model duration,
- load duration,
- queue wait,
- warm/cold state,
- model-prompt characters,
- requested context size,
- requested output limit,
- context-compaction flag.

Output:

```text
reports/system_benchmark/authorized_latency_comparison.json
```

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_authorized_latency.py --runs 3
```

For a more stable comparison after correctness is confirmed, increase repetitions.

## Interpretation plan

### Case A — prompt tokens fall and TTFT falls, tok/s also recovers

Likely context/KV-memory pressure was a major cause.

### Case B — prompt tokens fall and TTFT falls, but tok/s remains ~4.8

Investigate CPU/RAM contention, model placement/load state, and concurrent services.

### Case C — tok/s recovers near General but total latency remains high

Output token count or non-model workflow stages remain the main bottleneck.

### Case D — load duration is large/inconsistent

Model lifecycle/keep-alive/cold-loading requires a separate experiment.

### Case E — queue wait is significant

Resource scheduler contention requires a separate concurrency/admission experiment.

## Decision rule

Do not change quantization, thread counts, model choice, or low-level Ollama settings until this experiment identifies which part of inference is actually responsible.

---

# Next planned experiment slots

These are placeholders, not implemented claims.

## EXP-008 — Evidence-quality regression after Authorized compaction

Compare pre/post optimization answers on a fixed set of Authorized questions for:

- citation correctness,
- unsupported-claim rate,
- evidence coverage,
- answer completeness,
- truncation rate.

Latency optimization is accepted only if evidence quality remains acceptable.

## EXP-009 — Warm vs cold Authorized inference

Measure the same Authorized prompt after:

1. unloaded model,
2. immediate warm repeat,
3. idle interval,
4. rest of SovereignAI services running.

Goal: separate model-load latency from prompt/decode latency.

## EXP-010 — Resource-pressure experiment

Record CPU/RAM and Authorized throughput under:

- normal full stack,
- nonessential local services stopped,
- model warm,
- repeated generation.

Goal: determine whether the ~4.8 tok/s result is caused partly by memory pressure/contention rather than model inference itself.

---

# Decision history / principles learned

These principles should guide future work unless later experiments disprove them.

1. **Measure before optimizing.** A plausible bottleneck is still only a hypothesis.
2. **Route-specific performance matters.** General chat, Authorized/RAG and full industrial workflows have different cost structures.
3. **Keep full evidence for audit even when model context is compressed.** Model prompt budget and evidence retention are different concerns.
4. **Do not optimize away safety.** Authorization, ControlPlane, provenance and truthful failure states are not optional latency costs.
5. **Retry service calls selectively; do not blindly retry partially streamed generation.**
6. **Output length is a systems variable.** On CPU inference, unnecessary tokens can dominate latency.
7. **A smaller context is valuable only if evidence quality survives.** Performance and grounding must be measured together.
8. **Preserve negative results.** Rejected configurations are useful research evidence and prevent repeated work.

---

# Standard update checklist

Whenever an experiment, optimization, model change, retrieval change, reliability fix, security hardening step, benchmark, or architecture upgrade is attempted:

- [ ] Add/update an `EXP-XXX` section here.
- [ ] State the original problem.
- [ ] Preserve the baseline measurement/configuration.
- [ ] State the hypothesis before interpreting the result.
- [ ] Record exact code/config changes.
- [ ] Record invariants that must not regress.
- [ ] Provide reproduction commands.
- [ ] Link branch/PR/commit when available.
- [ ] Save output artifact paths.
- [ ] Record measured result, including failures.
- [ ] Compare result to baseline.
- [ ] Record benefits and trade-offs separately.
- [ ] Mark the experiment accepted/rejected/superseded.
- [ ] State the next experiment.

This file should evolve with the system and serve as the basis for future technical reports, research ablations, portfolio explanations, design reviews, and performance claims.
