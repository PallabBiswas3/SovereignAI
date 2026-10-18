# Past experiments and results

Last consolidated: 1 October 2026

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
  `python -m pytest local_testing/tests/test_inference_tradeoff.py local_testing/tests/test_system_benchmark.py local_testing/tests/test_phase31_retrieval_acl.py local_testing/tests/test_phase38_system_integration.py local_testing/tests/test_phase39_automatic_integration_routing.py local_testing/tests/test_integration_retry.py` (local-only layout).
- Fixtures check randomized balanced repetition blocks, warm-up exclusion, failure
  suppression, invalid configuration before requests, private endpoint restrictions,
  no proxy/redirect use, truncated/empty stream rejection, raw provenance, unknown
  citation penalties, and legacy smoke compatibility. Synthetic timings are test
  inputs, not benchmark results.
- Actual unavailable-service smoke check on loopback ports 8001 and 11434:
  **20 attempted warm-ups, 20 failures, 0 measured successes, exit code 2**.
  Command:
  `python local_testing/benchmarks/inference_tradeoff.py --protocol controlled --vllm-model Qwen/Qwen3-0.6B --ollama-model qwen3:0.6b --vllm-format unavailable --ollama-format unavailable --context-lengths 64 --evidence-budgets 128 --repetitions 5 --timeout 0.2 --output /tmp/sovereign-inference-unavailable`.
  Neither inference runtime was installed/running in this execution environment.
  No live inference latency, throughput, memory, or quality result is claimed.
- Full repository regression suite and networked Pump-102 workflow were not run.
  Verified cold starts, memory instrumentation, exact comparable model formats,
  tokenizer-aware budgets, and manually adjudicated quality remain outstanding.

The harness now reports end-to-end request throughput, unlike the older
post-first-token calculation; do not compare those fields directly with the
September 26 smoke results. Historical observations above remain unchanged.

## Reproducible evidence

- Runtime results: `local_testing/experiments/results/2026-09-26/` (local only)
- Inference harness: `local_testing/benchmarks/inference_tradeoff.py` (local only)
- System harness: `local_testing/scripts/benchmark_system.py` (local only)
- Root regression tests: `local_testing/tests/` (local only)

See [Current features](CURRENT_FEATURES.md) for the implemented system and [Next plan](NEXT_PLAN.md) for the remaining work.

## 1 October 2026: inference engineering controls (no final benchmark)

Base: `26865ff` on `pump102-inference-completion`. Original preliminary evidence
in `local_testing/experiments/results/inference_controlled_live_check/` is byte-for-byte
unchanged; its new README explains scope. It contains 40 successful requests:
20 warm-ups, 20 measurements, zero failures, `experiment_complete=false`.
Ollama used qwen3:0.6b GGUF Q4_K_M with declared digest
`7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435`;
vLLM used Qwen/Qwen3-0.6B BF16 in WSL. These are preliminary v2 observations,
not v3 verified-identity/memory/cold evidence.

Engineering validation in the Linux execution workspace:

- Focused protocol suite: `python -m pytest local_testing/tests/test_inference_controls.py
  local_testing/tests/test_inference_tradeoff.py` — **83 passed**. New tests use explicit
  synthetic HTTP/process doubles, not model results or real human adjudication.
- Complete root backend suite: `python -m pytest` — **241 passed, 5 failed**.
  Unchanged base `26865ff`, same environment/demo preparation, rerun sequentially:
  **192 passed, the same 5 failed**. No new failure was introduced.
- Existing failures: semantic retrieval (`test_phase14_semantic_embeddings`),
  paraphrased support (`test_phase16_structured_grounding`), grounding score
  (`test_phase9_governance::test_claim_grounding_preserves_source`), and two
  missing `workspace/uploads/Pump_Inspection_Report.md` fixture failures
  (`test_phase17_multifile_package`, `test_phase18_approval_execution`).
  PyTorch/Transformers/local MiniLM were unavailable; fallback scores did not meet
  semantic thresholds. The PDF demo generator does not create the missing MD.
- Environment: Python 3.12; pytest 8.4.2, pytest-asyncio 0.26.0, httpx 0.28.1,
  psutil 7.2.2. Dependencies and generated demo assets were staged for tests only.
  The original unprepared run had 186 passed/11 failed; missing PDF assets and
  SOCKS support accounted for additional environment failures. Concurrent baseline
  tests also conflicted over shared test state; the sequential baseline above is
  the comparison used. One existing Starlette/httpx deprecation warning remains.
- Frontend: `npm ci --ignore-scripts --no-audit --no-fund`, `npm run typecheck`,
  `npm run build` — successful; build generated 9/9 static pages. Typecheck also
  passed after restoring the unrelated Next.js-generated declaration change.
- CLI plan-only check: 360 scheduled batches for both providers, budgets 256/512,
  five repetitions; zero runtime requests, expected exit 3, completion false.
- `git diff --check` passes. Existing live-check JSON/CSV hashes match the base.

No local Ollama/vLLM runtime was exercised. Real process inspection, platform
permissions, API compatibility and external restart handling still require
verification on the user's Windows/WSL hosts. Completion remains blocked until
real reviewed fixtures and all measured evidence exist. No architecture,
authorization, retrieval ACL, release behavior or plant-write code changed.

## 4 October 2026: live all-service Pump-102 smoke

The Windows host completed two synthetic live requests through Graph-RAG,
diagnostics, Ollama and ControlPlane. Both returned HTTP 200 and held for human
review; the pinned-offline NLI run took 16.063 seconds and checked seven NLI
pairs. Graph-RAG abstained with zero authorized chunks because all existing
database nodes/chunks lack required scope metadata. The service-path milestone
passed, but a grounded or authenticated-user end-to-end result is still pending.
The detailed report is local-only at `local_testing/reports/PUMP102_LIVE_SERVICE_PATH_2026-10-04.md`.

## 4 October 2026: segregated synthetic seed and grounded rerun

After approval, 15 synthetic, unreviewed Pump-102 documents were inserted as 15
documents, 15 nodes and 15 chunks in the `sai-p102-v1` Supabase namespace.
Preflight found no target IDs; strict postflight verified all 45 IDs and exact
metadata/content. No legacy row, schema or claim table was changed. Live scope
probes passed for an engineer query, a wrong organization and exclusion of a
finance-only source, but used caller-supplied scope rather than a logged-in user.
The full request reached all four services and retrieved six Graph-RAG chunks;
ControlPlane held the answer, diagnostics abstained, and Graph-RAG had no
structured claim verification. See the local-only `local_testing/reports/PUMP102_SUPABASE_SEED_2026-10-04.md`
for the counts, hashes, recovery plan and limits.

## Diagnostic input contracts — 10 October 2026

Seven versioned task forms now share server-side Pydantic input contracts.
Malformed/legacy data is rejected before analysis, and registered models must
match task, ordered channels, units and sampling rate. Backend + diagnostic
verification: **606 passed, 1 skipped**; eight frontend schema-render/decoder
checks passed; TypeScript and production build passed. The active synthetic
Pump-102 probe was migrated. Tests/raw reports remain ignored under
`local_testing/`; details: `experiments/results/diagnostic_input_contract_20261010.md`
inside that directory. No training or fresh all-service/browser run was performed.

## ControlPlane CPU verification study — 17–18 October 2026

This study used an offline cached `cross-encoder/nli-deberta-v3-small` model on
Windows 11, Intel Family 6 Model 154 (10 physical/12 logical processors), 15.7 GiB
RAM, PyTorch 2.1.2 CPU and Transformers 4.41.2. A detached worktree at
`67dd76e` supplied the unchanged baseline. Forty fixed synthetic workloads
crossed 1/3/5/10/20 claims with short/long, shared/independent, and
supporting/conflicting evidence; each ran twice in randomized workload order.
The model was preloaded before timed checker calls. Baseline and candidate
were initially run in separate blocks; the balanced confirmation below is the
stronger timing comparison. The fixture contains no
industrial documents. Its expected labels are diagnostic proxies, not
adjudicated ground truth or production safety evidence.

| Initial 17 October sweep | Baseline | Candidate before final coverage guards |
|---|---:|---:|
| Checker p50 / p95 over 80 runs | 495 / 11,570 ms | 239 / 2,945 ms |
| Five-claim, long independent support p50 | 2,531 ms | 1,837 ms |
| Five-claim, long shared support p50 | 684 ms | 204 ms |
| Requested / actually inferred NLI pairs | 1,176 / 632 | 1,176 / 632 |
| Synthetic conflicting-case releases (40 runs) | 11 | 0 |
| Synthetic supported-case holds (40 runs) | 20 | 16 |
| Claim abstention fraction | 0% | 16.3% |
| Synthetic case outcome proxy | 60% | 60% |

p99 is omitted because there are fewer than 100 runs. Two repeats per case do
not establish stable tail latency; the broad p95 also mixes very different
workloads. Some nominally supported independent-source cases were held because
similar passages for other assets received conflicting NLI verdicts. The five
long independent claims completed under 2 seconds in the final candidate but
were held on factuality. No claim of consistently releasing valid five-claim
answers follows from these timings.

The root cause is visible in per-stage profiles. In two five-claim long
independent-support runs, Phase 5 fell from 459/517 ms to 6/9 ms after removing
unused repeated full-context numeric and date extraction. NLI still took
1,618/1,996 ms in the final runs, of which model forwards took 1,615/1,992 ms;
tokenization and retrieval were small. Five already-supported claims were still
rechecked, with 15 requested and 15 inferred NLI pairs in that case. Across the
full final workload, the model performed 96 forward passes on 72,680 padded
tokens; peak observed process RSS was about 1.32 GiB. These counts are now
recorded per run with model, runtime, device, thread count, batch shape, memory,
claim outcomes and final action. Top-level wall times and nested inclusive
times are labeled separately; they must not be summed together.

Request-scoped retrieval reuse now tokenizes each structured passage once and
reuses a selection only for an identical claim query over the same source bundle.
Legacy deterministic seed evidence stays claim-specific. Evidence-quality
ranking likewise reuses text tokens and computes each rank once. Exact NLI pair
deduplication and length-aware batching were already present. No source text,
authority result, NLI score or citation is cached across requests. Because
Phase 5's numeric/date support is based on value and anchor overlap, it was
not substituted for the semantic recheck of structured evidence.

Changed implementation files under `Controlplane.ai/src/`:

| File | Change |
|---|---|
| `adaptivefact/verification/deterministic.py` | Remove unused whole-context parsing; preserve sentence-level checks. |
| `adaptivefact/verification/nli.py` | Profile existing exact-pair, length-aware model computation; redact local model paths. |
| `adaptivefact/verification/phase6.py` | Reuse structured tokens/selections per record; expose retrieval and coverage counts; hold omitted relevant evidence. |
| `controlplane/verification.py` | Aggregate request-local profiles; preserve semantic coverage, citation, asset and conflict holds across support helpers. |
| `controlplane/evidence_quality.py` | Prepare tokens once and compute each selection rank once. |
| `controlplane/checker.py`, `schema.py`, `detectors/hallucination.py` | Add sanitized response profiling and stage/risk timers. |

The unchanged-model 40-case NLI comparison had maximum probability delta zero.
The eight deterministic baseline-parity cases compare all claim fields except
elapsed time, including wrong asset, range, negation, date and multi-clause
inputs. Parity preserves baseline behavior; it does not make Phase 5 alone a
sufficient factuality gate.

The first timing-only candidate was rejected: unchanged factual states plus
faster execution exposed an additional conflicting-case release previously
held by the deadline. The baseline's top-three structured retrieval could omit a distinct relevant
conflicting passage and still release the claim. The final candidate marks
otherwise supported claims undecidable when relevant distinct passages remain
unexamined, including for optional support scoring. This preserves the
consequential hold path. It increases abstention and still does not solve
semantic retrieval misses below the relevance filter. The benchmark's reduction
from 11 to 0 conflicting-case releases is a synthetic regression result, not a
general unsafe-release estimate. The final implementation also abstains for
structured claims beyond the standard eight/deep twenty claim limit, explicit
citations absent from the supplied source IDs, and support witnessed only by a
different explicitly named asset. Numeric support helpers cannot clear the
semantic coverage hold. These are conservative guards, not new meanings for
the claim states. They can increase unnecessary holds. They do not constitute
a complete semantic asset resolver or prove complete evidence retrieval.

GraphRAG verifies source-linked claims against retrieved chunks before answer
synthesis. ControlPlane verifies atomic claims extracted from the generated
answer afterward; the claims have different obligations. A future boundary
contract may reuse authorized evidence IDs, revision and citation metadata,
while retaining ControlPlane's own answer-claim and policy checks.

This is a systems optimization, not a demonstrated new verification algorithm.
Atomic factual evaluation, claim evidence retrieval, adaptive routing and
alignment scoring are established in [FActScore](https://aclanthology.org/2023.emnlp-main.741/),
[FEVER](https://aclanthology.org/W18-5501/),
[Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/), and
[AlignScore](https://aclanthology.org/2023.acl-long.634/). The proposed
dependency-aware scheduler was not implemented because these measurements do
not establish a safe way to suppress evidence checks. Source-faithful span
shortening also remains experimental: losing negation, revision, units or
operating conditions could change the verdict. ONNX Runtime is installed, but
the ONNX model export package was absent, so ONNX latency is unmeasured.
The scorer retains its existing 512-token truncation; no new premise shortening
or dependency scheduling is enabled. Truncation itself remains a limitation for
long documents with distant qualifications.

The literature also includes independent sampling in
[SelfCheckGPT](https://arxiv.org/abs/2303.08896), search-based atomic checking in
[SAFE](https://arxiv.org/abs/2403.18802), and verification questions in
[Chain-of-Verification](https://arxiv.org/abs/2309.11495). These establish adjacent
verification/reuse strategies, but do not establish an ACL/revision-aware
release guarantee for this pipeline. The request-scoped reuse, batching and
profiling here are established engineering techniques. Latency-bounded
dependency scheduling needs a reviewed benchmark and a coverage argument before
it is a research contribution. Alternative runtime work should follow the
[ONNX Runtime transformer optimization](https://onnxruntime.ai/docs/performance/transformers-optimization.html)
and [quantization accuracy guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

### Balanced confirmation and rejected inference variants — 18 October

Fresh processes ran in baseline/candidate/candidate/baseline order, each with
two warm repetitions of three fixed five-claim workloads. This was repeated
with zero and two local arithmetic worker processes. Each table entry has four
warm samples; p99 is unmeasured. Shared fixtures deliberately repeat one
proposition, exposing exact-pair reuse, rather than five distinct propositions.

| Workload p50, ms | Baseline idle | Candidate idle | Baseline 2 workers | Candidate 2 workers |
|---|---:|---:|---:|---:|
| Long independent support | 3,895 | 2,691 | 5,049 | 2,860 |
| Long shared support | 1,233 | 390 | 2,098 | 494 |
| Long shared conflicts | 2,205 | 523 | 3,588 | 525 |

Per condition and implementation, 180 NLI comparisons were requested and 68
were actually inferred. Candidate forward passes/padded tokens were 12/13,940.
No new conflicting-case release occurred. All independent-support cases still
held; the two-second target is not consistently achieved. Fresh-process first
checker calls (cached weights, OS caches not cleared) took 6.7–12.7 s baseline
and 6.3–11.1 s candidate while idle, and 15.2–15.9/12.8–14.1 s under contention;
all held. Preloading/readiness remains necessary. These are a small local
confirmation with substantial variance, not production p95/p99 estimates.

Six fixed five-claim NLI cases ran five repetitions per setting. Default ten
threads/batch sixteen was faster in this sweep than four threads/batch sixteen
or ten threads/batch four. For long independent support their p50s were
380/651/793 ms respectively. Four-thread scores were identical; batch-four
maximum score difference was 1.2e-7, with no 0.72 threshold crossings. The
configuration runs were sequential rather than balanced, so these are tuning
observations, not a general hardware optimum. Production defaults are unchanged.

Experimental PyTorch dynamic INT8 Linear quantization reduced one measured long
independent-support call from 927 to 585 ms, but over 30 pairs changed nine class
labels and crossed entailment/contradiction thresholds on eleven pairs; maximum
score change was 0.867. It was rejected before release-action validation and is
available only as an offline experiment. It is not a deployable speedup.

Twelve real-model adversarial cases used identical input hashes across baseline
and candidate. Baseline released two invalid cases (wrong asset and fabricated
citation); candidate released neither. Supported/contradicted/missing/conflicting,
superseded revision, wrong range, negation, multi-clause and retrieved instruction
cases otherwise preserved expected states/holds. Unauthorized perfect-match
evidence is covered by the backend retrieval ACL suite, because ControlPlane's
input contract accepts evidence already authorized by its caller. NLI failure,
deadline expiration, source changes and support-helper coverage are focused
regressions. These tests do not prove universal factual correctness.

One accessible fixed five-claim Pump-102 review trace was replayed locally,
without generation or external services: baseline 2,054 ms/review, candidate
1,786 ms/allow, all five claims supported and NLI score delta exactly zero.
The changed action follows the unchanged 2,000-ms deadline. This single replay
has no independent adjudication and is not a live plant or end-to-end result.

After all coverage/citation/asset guards, the full 40-case matrix was rerun twice
on 18 October. Compared with the original 80 baseline runs, pooled p50/p95 were
495/11,570 ms versus 556/5,406 ms. Median latency did not improve in this later
unbalanced comparison; mean fell from 2,300 to 1,187 ms. Environmental variance
is why the balanced confirmation and stage measurements matter more than a
single pooled speedup. Requested/inferred NLI pairs remained 1,176/632, with
96 candidate forwards, 72,680 padded tokens, aggregate inference throughput
7.21 pairs/s and peak sampled RSS 1.19 GiB (baseline equivalents uninstrumented).
The model snapshot was `fa2804872c3b4bd748f38c0185cc85775361e735`.

Final conflicting-case releases were 0/40 versus 11/40 baseline, while nominally
supported-case holds increased to 24/40 versus 20/40. Claim abstention rose to
52.2%; synthetic claim outcome accuracy fell from 52.6% to 33.7%, and case
outcome accuracy from 60% to 50%. These fixture-label proxies count safe
abstentions as incorrect; they are not adjudicated industrial accuracy. The
larger hold count is a material cost of closing the claim/top-k coverage gaps,
particularly for 10/20-claim inputs. There were no new conflicting-case releases.
Zero observed unsafe releases on this sample does not prove universal safety.

Per-run and aggregate JSON are
`local_testing/experiments/results/controlplane_validated_20261018.json` and
`controlplane_final_comparison_20261018.json`; balanced raw runs and their summary
are in `controlplane_paired_20261018/`. The adversarial/Pump replay reports are
`verification_adversarial_{baseline,candidate}_20261018.json`. All are local and
Git-ignored. Baseline forward/token/memory/throughput fields remain null rather
than estimated. Each condition includes fewer than 100 samples, so p99 is null.

Reproduce from the repository root with the cached model and local dependencies:

```powershell
git worktree add ../.baseline_sovereign 67dd76e
python Controlplane.ai/tools/benchmark_controlplane.py --source ../.baseline_sovereign/Controlplane.ai --repetitions 2 --output local_testing/experiments/results/controlplane_baseline.json
python Controlplane.ai/tools/benchmark_controlplane.py --source Controlplane.ai --repetitions 2 --output local_testing/experiments/results/controlplane_candidate.json
python Controlplane.ai/tools/compare_controlplane_runs.py local_testing/experiments/results/controlplane_baseline.json local_testing/experiments/results/controlplane_candidate.json --output local_testing/experiments/results/controlplane_comparison.json
python Controlplane.ai/tools/benchmark_nli_cpu.py --repetitions 2 --output local_testing/experiments/results/nli_cpu.json
python Controlplane.ai/tools/benchmark_paired.py --baseline ../.baseline_sovereign/Controlplane.ai --output-dir local_testing/experiments/results/controlplane_paired
python Controlplane.ai/tools/benchmark_nli_cpu.py --repetitions 5 --cpu-threads 4 --batch-size 16 --case 5_long_independent_supported --output local_testing/experiments/results/nli_threads4.json
python Controlplane.ai/tools/compare_nli_int8.py --cpu-threads 10 --output local_testing/experiments/results/nli_int8.json
python Controlplane.ai/tools/evaluate_verification.py --source Controlplane.ai --output local_testing/experiments/results/adversarial_candidate.json
$env:CONTROLPLANE_BASELINE_SOURCE='../.baseline_sovereign/Controlplane.ai'
$env:PYTHONPATH='Controlplane.ai/src'
python -m pytest Controlplane.ai/tests/test_verification_profiling.py -q
$env:PYTHONPATH='.workspace/test_deps;Controlplane.ai/src;Time-Series-Diagnostic-Agent/src;backend;local_testing'
python -m pytest local_testing/tests -q
```

All raw timing JSON remains Git-ignored in `local_testing/experiments/results/`.
The final focused suite passed 27 regressions with the cached model and unchanged
baseline source. The complete backend suite passed 367 tests with one skip;
the combined run before the final conflict-preservation regression was
393 passed/one skipped. `pint==0.25.3` was installed into `.workspace/test_deps`
(no global installation). The final focused rerun took 11.29 s and the combined
run 135.76 s. An older separate ControlPlane test checkout is not the test
source for this repository and contains absent local data/config fixtures;
its results are not used as a release gate.
