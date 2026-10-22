# Past experiments and results

## Final v1 release verification (18 October 2026)

This section supersedes the initial remediation checkpoint below. Scope is the
existing read-only research prototype; no new agents, frameworks, diagnostic
research, policy relaxation, merge, release or deployment was introduced.
Tested implementation: `c725ce8f94016351607d27e523a109edbcade6bb` on
`verification/v1-final`. The subsequent commit updates maintained documentation
only; obtain the final documentation SHA with `git rev-parse HEAD`.

### Worktree reconciliation and changes

| Worktree / branch | Verified state and disposition |
|---|---|
| `1_SIH`, `research/diagnostic-guards-review` | Clean at `56d742020028657f6c200f24af100ff4e6fa341c`, already pushed; the former eleven files are preserved in three commits, not pending changes |
| `SovereignAI_v1_security`, `security/v1-remediation` | Original security branch retained at `f15e2867c03d322a7049ec5cf05df233612fc2c9` |
| Same isolated worktree, `verification/v1-final` | Two independent corrections based on that security SHA, followed by this documentation update; no research merge |

The research branch contains Process guard/sensitivity/calibration changes
(`b2e3abd`), Wind quality guards (`2cd2b3d`), and research documentation
(`56d7420`) from common base `480ee767933b91c55f01a3f9aa3ed71ce6380fac`.
Its Process changes are not essential to the existing v1 demonstration:
the live path below succeeds with committed Process 3.0. The research evaluation
also reports healthy alert burden 57/80, so it must not be presented as an
accepted accuracy improvement. Wind remains frozen at 1.1 and outside this demo.

Smallest safe integration procedure: review this verification branch against
the security SHA, then obtain explicit approval before any merge or release.
Do not merge the research branch for v1. If Process research is later explicitly
approved, select its Process commit and reconcile documentation separately;
a blanket merge would also bring the frozen Wind changes. Nothing was reset,
stashed, discarded or overwritten in the original worktree.

| Increment | Files and reason | Verification |
|---|---|---|
| `291684a3064346a2a0f1d8dae35cd0e2d6da5638` | Three sanitized `local_testing/Controlplane.ai/tests/` files: adapt historical clock/NLI doubles to the existing profiling interface; preserve assertions | 16 affected tests passed |
| `c725ce8f94016351607d27e523a109edbcade6bb` | `backend/app/tools/file_tools.py`: normalize Windows extended paths before confined I/O; `backend/app/sandbox/executor.py`: use ordinary Windows host syntax only for Docker volume arguments; `backend/tests/security/test_workspace_long_paths.py`: owner/foreign access and mount regression checks | 27 affected tests passed; live upload changed from HTTP 500 to 200; final security 69 passed |
| Documentation | README and the three maintained documents: evidence, current boundaries and final handoff | Final diff reviewed; no runtime reports, database, uploads, credentials or build outputs included |

The original four high findings remain fixed: approval/audit scope, private file
consumers, GraphRAG service authentication and validated scopes, and legacy chat
governance. No new critical finding was observed in this scoped verification.
The extra Windows bug was an availability failure: a 267-character upload path
failed at `Path.open('xb')`. Root and candidate now use the same resolved native
form; the containment check still runs before file access. Foreign absolute
paths, prefixed paths and traversal remain denied. The Docker check uses a
subprocess stub; no live container execution is claimed.

### Every original remaining failure

Paths in the next table are relative to `local_testing/Controlplane.ai/tests/`.
No assertion, release policy or expected outcome was relaxed to obtain a pass.

| Exact test | Original error / cause | Impact and disposition |
|---|---|---|
| `test_latency_budget_boundary.py::test_internal_two_second_budget_boundary[1999-False]` | `StopIteration`: profiling exhausts the finite mocked clock | Test fixture only; fixed with a clock that retains its terminal value; passes |
| `test_latency_budget_boundary.py::test_internal_two_second_budget_boundary[2000-False]` | Same exhausted clock | Fixture only; unchanged exact-boundary allow assertion passes |
| `test_latency_budget_boundary.py::test_internal_two_second_budget_boundary[2001-True]` | Same exhausted clock | Fixture only; unchanged over-budget review assertion passes |
| `test_local_nli_loading.py::test_lazy_nli_forwards_local_only_to_transformers_loader` | Fake NLI lacks `score_profiled` | Fixture only; adapter returns existing scores and empty profile; local-only forwarding assertions pass |
| `test_nli_exact_batching.py::test_identical_pairs_are_inferred_once_but_results_keep_original_positions` | `__new__` fake scorer lacks profiling attributes (`profile_model`, model configuration) | Fixture only; initialize fake profiling/configuration fields; pass |
| `test_nli_exact_batching.py::test_changed_evidence_is_never_reused_across_calls` | Same incomplete fake scorer | No change to cache isolation assertions; pass |
| `test_nli_exact_batching.py::test_small_mixed_length_batch_avoids_padding_short_pairs_to_long_manuals` | Same incomplete fake scorer | No change to batching/parity assertions; pass |
| `test_nli_exact_batching.py::test_request_memo_reuses_exact_pairs_and_does_not_cross_requests` | Same incomplete fake scorer | No change to request isolation assertions; pass |
| `test_nli_exact_batching.py::test_request_memo_resets_on_exception_and_is_context_isolated` | Same incomplete fake scorer | No change to exception/context isolation assertions; pass |
| `test_pump102_factuality_regressions.py::test_pre_authority_ablation_reuses_same_evidence_and_semantic_score`, line 252 | Expected `supported`, actual `undecidable` with `evidence_authority_enabled=False` | Non-blocking historical research ablation; still fails, expectation unchanged |

The ablation no longer isolates its authority switch: Phase 6 filters uncited
superseded evidence before NLI. Inspection showed `nli_pairs=0`,
`no_evidence_claims=1`, and verifier `phase6_no_evidence:undecidable_v3`.
The normal/default authority-enabled path reports **contradicted** with
`evidence_authority:superseded_current`; current production factuality/revision
regressions pass. The disabled experimental flag does not yield an unsafe
release. Restoring obsolete evidence merely to make the experiment pass would
weaken the current policy, so neither source nor expectation was changed.

Two diagnostic failures also remain unchanged, relative to
`local_testing/Time-Series-Diagnostic-Agent/tests/`:

| Exact test | Error / root cause | V1 impact |
|---|---|---|
| `test_plugin_architecture.py::test_structured_wind_request_uses_plugin_policy`, line 87 | Workflow version expected `1.2`, actual frozen `1.1` | Non-blocking research version literal; Wind excluded and unchanged |
| `test_public_pipeline.py::test_process_uses_decomposed_cross_domain_workflow`, line 53 | Workflow version expected `3.3`, actual committed `3.0` | Non-blocking research version literal; actual Process 3.0 all-service behavior verified below |

These local historical tests target the separate research checkout. They do
not justify importing its unaccepted code. They remain visible failures, not
skips, xfails or modified expected outcomes. The full suite is not entirely green.

### Final component and test status

| Check | Actual final result |
|---|---|
| Backend, including security | **436 passed, 1 skipped**, 105.05 s |
| Security separately | **69 passed**, 6.04 s; included in backend total |
| ControlPlane | **221 passed, 1 failed, 7 skipped**, 7.29 s; remaining ablation above |
| Diagnostics | **237 passed, 2 failed**, 66.76 s; version literals above |
| GraphRAG | Server TypeScript build passed; 3 service-auth tests passed; all 8 existing retrieval/verification/adapter test files passed |
| Frontend | Type check passed; 4 stream tests passed; webpack production build passed |
| Local services | Backend 8000, frontend 3000, GraphRAG 3100, ControlPlane 8100, diagnostics 8200 and Ollama 11434 returned HTTP 200 during verification; authenticated integration health was `ok` |

Copied missing ignored local benchmark/config fixtures into the isolated
worktree before the final ControlPlane run; four transient missing-file setup
failures are not included as product failures. The backend skip is unavailable
Windows symlink creation; the seven ControlPlane skips require the undownloaded
RAGTruth corpus. Those cases were not successfully exercised. Dependency
deprecation warnings remain. Temporary verification services were stopped after
testing; the pre-existing Ollama service was left alone.

### Actual authenticated synthetic Pump-102 demonstration

An isolated SQLite database used synthetic engineer, outsider, approver,
foreign-organization engineer/approver and browser identities. The backend used
`SOVEREIGN_AUTH_MODE=local`, CSRF checks and matching randomly generated
server-side GraphRAG bearer credentials. Global GraphRAG operator access was
disabled. A read-only preflight found the existing `sai-p102-v1` namespace and
zero missing documents/nodes/chunks; this session did not seed, clear or rewrite
the shared GraphRAG database. Existing storage uses remote Supabase, so this run
is not an offline/air-gap demonstration.

The released request explicitly asked for a structured cited assessment and
forbade authorizing physical work. Its normalized synthetic pressure/flow/level
signals declare independent generation and explicit linkage to observation log
`LOG-P102-SEP-R1`; they are not physical measurement attestations.

| Measurement | Actual result |
|---|---|
| Run | `ca493758-9241-49dc-a8dc-54529d2ef701`, HTTP 200, `released=true` |
| Required services | GraphRAG, diagnostics, local model and ControlPlane all actually called; precheck/release both `allow` |
| Retrieval | Authorized 5 nodes / 6 raw chunks; 1,975 ms; GraphRAG structured claim verification unavailable (zero claims), dense retrieval disabled in this configuration |
| Diagnostics | Process 3.0, `diagnose`, DPCA/CVA confirmed pressure deviation; asset, specimen, dataset revision, units, sampling and synthetic log linkage retained in metadata |
| Generation | Ollama `qwen3:4b-instruct`, CPU, fallback false; completed source-typed assessment, not truncated; 1,909 prompt tokens / 274 generated tokens |
| Model timing | 144.511 s provider duration, including 8.021 s loading; 4.874 generated tokens/s |
| ControlPlane | 5 supported atomic claims, 9 NLI pairs, no detector errors/findings, no latency-budget violation; release check **1,781 ms** |
| Other timing | Precheck 78 ms, evidence stage 2,078 ms, orchestrator 148,531 ms; actual client request **161,590.62 ms** |
| Artifact | `915b94e4-0ef7-44a1-bfc2-9e524022c174` |
| Evidence Capsule | `da3e5eca-114e-440c-a077-38be4aabd47e`; downloaded capsule integrity **VALID**, unsigned development mode |

The final answer cited the observed **5.1 bar** at
`2026-09-01T08:00:00Z` and maintenance-review request to chunk
`dae6f0f7-8ca9-57e3-b820-562bd7788367`; the **4.8-5.5 bar** normal range to
`8ada71d0-5c35-5079-bcd9-12fffb8da37c`; and the pressure root-cause candidate
to `process-diagnostic-evidence`. Every rendered citation resolved to supplied
evidence. The four rendered statements yielded five checked atomic claims.
This is source correspondence on synthetic evidence, not independent plant
truth, calibrated diagnostic confidence or a new accuracy percentage.

Actual downloaded outputs had matching registered SHA-256, run/answer/citations
and a valid capsule manifest. The owner could list/download both actual IDs;
outsiders listed neither and guessed downloads returned 404; anonymous downloads
returned 401. The restricted synthetic finance canary was absent from the
response and downloaded payloads, including nested ZIP payloads.

### Live negative paths and frontend/backend behavior

| Actual check | Result |
|---|---|
| No login / missing CSRF | Analysis 401 / 403 |
| Anonymous GraphRAG integration retrieval | 401; anonymous global graph/evidence/ingest/clear returned 403 with operator access disabled |
| Outsider without workcell permission | Analysis 403 |
| Foreign organization with workcell permission | Retrieval abstained, zero chunks, held for human review, no artifact/capsule |
| Uploaded file consumer | Owner upload 200 and foreign list empty; foreign engineer OCR 400 before parsing because private path cannot resolve |
| Human approval boundary | Proposal 200; requester self-approval 403; foreign read/decision 404; authorized approver 200 and executed only a 35-byte synthetic private `review-note.txt` write |
| Audit boundary | Foreign audit 404, authorized audit 200 |
| Unsupported claim, no evidence | Injected candidate `Pump-102 discharge pressure is 999 bar.` held for human review, no outputs |
| Unsafe physical-work candidate | Held for human review, no outputs |
| Legacy prompt injection | 403 |
| Actual required-service outage | Stopped this worktree's ControlPlane process; backend returned 503 `INTEGRATION_SERVICE_UNAVAILABLE`, `failed_services=[controlplane]`, three attempts, 6,957.32 ms, no new artifacts; ControlPlane then restarted |

Unsafe/unsupported negative candidates used the existing `candidate_response`
input to exercise the actual live release gate; these were not generated by the
model. The foreign-evidence run and unsupported no-evidence run cover absence
of authorized support without loosening retrieval or release policy.

The real production frontend signed in through its login form and submitted
versioned synthetic diagnostic arrays through `/integrations`. The browser's
actual query remained the default maintenance-action request, unlike the
structured request above. Run `badb30b9-d12e-4d1b-b7a1-2a6b424b046c` returned
HTTP 200 with `held_human_review` and `released=false`. The DOM displayed
**Not released**, `human_review`, and the fixed review notice; there were no
artifact/capsule links, and the busy state reset. Retrieved authorized evidence,
diagnostic state and actual Ollama identity were displayed. This was a real
backend response, not a browser HTTP fixture. Earlier session/stream/error
fixture checks below and four final stream regressions remain separate evidence.

The existing dev launcher encountered Turbopack's external-directory restriction
because dependencies are junctioned to the original checkout. The already
supported webpack production build plus `npm run start` served the actual UI.
Browser-tool file upload/screenshot saving was denied by that tool's configured
roots; diagnostic arrays were entered through normal form controls. No runtime
or security policy was changed for these tooling constraints.

### Exact commands and retained evidence

All commands below ran in `SovereignAI_v1_security` unless a cwd is shown.
Existing dependencies, local model caches, historical ignored fixtures and an
already seeded approved synthetic namespace are prerequisites. The `.workspace`
helpers orchestrate existing scripts/APIs for this local verification and are
ignored; they are not shipped product entry points or a fresh-clone test suite.
They did not print authentication headers or secrets.

```powershell
python .workspace/final_graph_preflight.py
python .workspace/final_runtime.py
python .workspace/final_test_accounts.py
python .workspace/final_live_probe.py health
python .workspace/final_live_probe.py normal
python .workspace/final_resume.py
python .workspace/final_live_probe.py boundaries
python .workspace/final_outage.py
python .workspace/final_resume.py controlplane

$env:PYTHONPATH='backend;Controlplane.ai/src;Time-Series-Diagnostic-Agent/src;local_testing;local_testing/tests;backend/tests/security;../1_SIH/.workspace/test_deps'
$env:SOVEREIGN_GRAPHRAG_SERVICE_TOKEN='synthetic-test-credential-0000000000000000'
$env:CONTROLPLANE_BASELINE_SOURCE='../.baseline_sovereign/Controlplane.ai'
python -m pytest local_testing/tests backend/tests/security -o addopts='' -q --tb=short --junitxml=.workspace/final-backend.xml
python -m pytest backend/tests/security -o addopts='' -q --tb=short --junitxml=.workspace/final-security.xml

# cwd: Controlplane.ai
$env:PYTHONPATH='src;../local_testing'
$env:CONTROLPLANE_BASELINE_SOURCE='../../.baseline_sovereign/Controlplane.ai'
python -m pytest ../local_testing/Controlplane.ai/tests -o addopts='' -q --tb=short --junitxml=../.workspace/final-controlplane.xml

# cwd: Time-Series-Diagnostic-Agent; PYTHONPATH=src;../local_testing
python -m pytest ../local_testing/Time-Series-Diagnostic-Agent/tests -o addopts='' -q --tb=short --junitxml=../.workspace/final-diagnostics.xml

# cwd: Graph-RAG
npm run build --workspace=server
node --test server/security/serviceAuth.test.cjs

# cwd: frontend
npm run typecheck
node --test tests/task-stream.test.cjs
npm run build -- --webpack
npm run start -- --hostname 127.0.0.1 --port 3000
```

The runtime helpers use `scripts/Run-Service.ps1 -Service` with
`sovereign-api`, `graph-rag`, `controlplane`, `diagnostics` and initially
`sovereign-ui`; the production frontend command above replaces its dev launcher.
Backend and GraphRAG inherit matching `SOVEREIGN_GRAPHRAG_SERVICE_TOKEN` /
`GRAPHRAG_SERVICE_TOKEN`, `SOVEREIGN_AUTH_MODE=local`, isolated storage paths,
`SOVEREIGN_LLM_PROVIDER=ollama`, deterministic fallback disabled, and offline
ControlPlane NLI loading. Actual random credentials were never command literals.

The eight existing GraphRAG files were run from repository root as follows,
separately from the live retrieval above:

```powershell
$env:TS_NODE_PROJECT='Graph-RAG/server/tsconfig.json'
$env:TS_NODE_TRANSPILE_ONLY='true'
$env:SUPABASE_URL='http://127.0.0.1:59999'
$env:SUPABASE_SERVICE_ROLE_KEY='synthetic-test-service-role-key'
$taskTests = @(
  'local_testing/Graph-RAG/server/evaluation/agentPolicy.test.ts',
  'local_testing/Graph-RAG/server/evaluation/localModel.test.ts',
  'local_testing/Graph-RAG/server/evaluation/metrics.test.ts',
  'local_testing/Graph-RAG/server/evaluation/retrievalCore.test.ts',
  'local_testing/Graph-RAG/server/retrieval/revisionSelection.test.ts',
  'local_testing/Graph-RAG/server/verification/claimVerifier.test.ts',
  'local_testing/Graph-RAG/server/verification/verificationPolicy.test.ts',
  'local_testing/Graph-RAG/server/verification/verificationScope.test.ts'
)
node --test --require ./Graph-RAG/server/node_modules/ts-node/register @taskTests
```

Raw JUnit files are the four `.workspace/final-*.xml` paths above. Live JSON
evidence is retained locally under `.workspace/final_release_20261018/`:
`health.json`, `normal-request.json`, `normal-response.json`,
`normal-summary.json`, `boundaries.json`, `required-service-outage.json`, and
`browser-summary.json`, plus service logs. The generic summary helper's phrase
"not an authenticated-user validation" refers to lack of enterprise-user
validation: this run **did use actual authentication with synthetic identities**.
Runtime credential material was isolated from Git and removed after services
were stopped. Generated build/runtime artifacts are excluded from commits.

### Remaining limitations and release gate

No observed critical security or essential integration defect remains in the
tested authenticated, supervised synthetic v1 path. The three remaining failures
are non-blocking for that scope for the reasons above. **Verdict: ready for a
controlled research release review with documented limitations; not released.**
User approval is still required before any merge or final release.

This does not approve production deployment: disabled authentication remains
single-user development only; trusted internal bearer transport is not mutual
TLS; remote Supabase is not an air gap; legacy uploads/conversations need
explicit migration; content hashes remain globally unique; private files use
the primary workspace; unsigned capsule integrity is not provenance attestation.
GraphRAG's structured claim verifier was unavailable and this configuration did
not exercise dense retrieval. CPU generation is slow, a default maintenance
request conservatively held, and diagnostic confidence is uncalibrated.
No claim is made about live Docker, vision/OCR model accuracy, enterprise
identity, physical equipment, calibrated system accuracy or full CI portability
of ignored historical fixtures. These limits are disclosed, not new v1 work.

## Initial v1 security and integration remediation — 18 October 2026

### Scope and preserved work

Base: `480ee767933b91c55f01a3f9aa3ed71ce6380fac`.
Tested implementation: `d2c7a05adfa16a2e26592bf36732509ba206aa66`.
Branch/worktree: `security/v1-remediation`, `../SovereignAI_v1_security`.
The final documentation commit is obtained with `git rev-parse HEAD` in that
worktree. No PR merge, remote push, tag or release was created.

SHA-256 comparison confirms all eleven original modified/untracked Time-Series
and documentation files in `1_SIH` are unchanged. Security commits contain none
of those pending edits. No diagnostic or ControlPlane source/policy was changed.

### Verified findings, fixes and changed files

| Severity / finding | Reproduction and impact | Files / minimal correction |
|---|---|---|
| High: approval/audit resource authorization | An org-B approver could read and reject org-A approval and read its audit | `backend/app/api/governance.py`: scope reads/decisions/audit parent before mutation; preserve requester/approver separation and tool-policy revalidation |
| High: shared uploaded-file namespace | Different principals could list/reuse the same filename and consume foreign files | `api/files.py`, `ocr.py`, `vision.py`, `knowledge.py`, `demo.py`, `tasks.py`; `tools/file_tools.py`, `registry.py`; `workflows/inspection.py`: private principal namespace, unique upload directories, scoped resolution in all consumers/tools, authorization-bound cache key |
| High: unauthenticated GraphRAG and forged scopes | Anonymous global graph reads/writes/deletion and wildcard retrieval scope reached privileged service-key access | `Graph-RAG/server/serviceAuth.ts`, `server.ts`, server `.env.example`; backend `integrations/clients.py`, `orchestrator.py`, `core/config.py`, root `.env.example`: backend bearer token, separate operator token, explicit bounded scopes and fingerprint consistency |
| High: legacy chat governance bypass | Scanner-flagged injection reached a provider and raw output returned without ControlPlane | `api/chat.py`, `core/database.py`: existing scanners, precheck/release gate, fixed held response, scoped conversations and idempotent SQLite migration |
| Related file-consumer gaps | Repeated artifact filenames reused bytes; identical ingestion could rewrite ACL metadata | `tools/application_tools.py`, `rag/ingestion.py`: unique artifact directories and exact existing-ACL match for authenticated duplicate ingestion |
| Medium: session/stream/error states | Failed logout appeared successful, EventSource CONNECTING could wait indefinitely, denied asset/metrics responses appeared empty or stuck | `frontend/app/page.tsx`, `task-stream.ts`, `assets/page.tsx`, `metrics/page.tsx`: checked statuses, stream cleanup/revocation/disconnect handling, abort stale requests, suppress candidate text, visible accessible loading/errors |

New sanitized negative/positive regressions are in
`backend/tests/security/test_governance_scope.py`, `test_file_isolation.py`,
`test_graphrag_service_auth.py`, `test_legacy_chat.py`,
`test_output_and_ingest_isolation.py`; GraphRAG
`server/security/serviceAuth.test.cjs`; frontend `tests/task-stream.test.cjs`.
`local_testing/tests/test_phase1_foundation.py` now checks that unavailable
ControlPlane holds chat rather than expecting an ungoverned 200 response.
README and the three maintained documents record configuration and handoff.

### Verification results

| Check | Actual result |
|---|---|
| Approval/audit increment and existing authorization checks | 18 passed |
| File-isolation increment, existing tool/approval checks | 41 passed; separate existing ACL/download checks: 11 passed |
| GraphRAG credential hop | 4 backend tests passed; 3 Node tests passed, including real HTTP denial and scoped retrieval-handler dispatch with stubbed storage/model |
| Legacy chat increment | 16 passed; full suite additionally covers legacy schema migration |
| Artifact collisions and duplicate ingestion scope | 9 passed |
| Full backend/integration suite | **434 passed, 1 skipped**, 110 warnings, 104.95 s |
| Existing GraphRAG retrieval/verification/adapter suites | **8 test files passed**; synthetic Supabase endpoint, no live graph writes |
| Frontend stream regressions | **4 passed** |
| Frontend TypeScript / production build | `npm run typecheck` and `npm run build -- --webpack` passed; 7 application routes plus not-found generated |
| GraphRAG TypeScript build | `npm run build --workspace=server` passed |
| Broader ControlPlane suite, correct service cwd | **212 passed, 10 failed, 7 skipped**, 7.07 s |
| ControlPlane original-checkout comparison (affected four files) | **32 passed, same 10 failed**, 3.32 s |
| Diagnostic suite against original pending working tree | **239 passed**, 30.11 s; this includes excluded uncommitted diagnostic code |
| Diagnostic suite against isolated committed code | **237 passed, 2 failed**, 32.24 s; version assertions expect pending process 3.3 / Wind 1.2 while committed source is 3.0 / 1.1 |

The first broad runs had missing copied local evaluation/upload fixtures and a
wrong ControlPlane working directory. Those setup errors were corrected before
the results above. Local ignored fixtures were copied into the isolated worktree;
the original test/source files were not rewritten. The legacy chat test was
updated for the intentional fail-closed contract. No failing check was marked
passed, skipped or xfailed to obtain these results.

The ten remaining ControlPlane failures are pre-existing: three exhausted mocked
clocks in `test_latency_budget_boundary.py`; one pre-profiling FakeNLI double in
`test_local_nli_loading.py`; five `__new__` scorer fixtures missing profiling
attributes in `test_nli_exact_batching.py`; and the `supported` versus
`undecidable` assertion in the historical-evidence ablation in
`test_pump102_factuality_regressions.py`. They reproduce against original source.
Their resolution is a release gate, not grounds to relax safety policy.

### Frontend/backend and live integration evidence

Chrome used a separate synthetic browser context and an explicitly synthetic
local HTTP fixture with the actual production frontend build. Verified: failed
logout retains the signed-in state and shows HTTP 503; access_revoked returns to
sign-in and clears sensitive state; unreleased token text is absent; evaluation
403 shows an error and resets the run button; an asset subrequest 403 clears the
previous asset rather than presenting empty-success data. The compact browser
view had no horizontal document overflow. The fixture and its frontend process
were stopped after testing. These are UI contract checks, not a live assessment.

Backend/integration tests cover scoped downloads/capsule payloads, approval
boundaries, auth/CSRF, contradictory/unanswerable evidence, required-service
outages, diagnostics contracts and release/hold behavior. Existing GraphRAG
checks confirm retrieval/revision/verification behavior remains intact.

Final TCP availability probe: backend 8000, frontend 3000, GraphRAG 3100,
ControlPlane 8100, diagnostics 8200 and vLLM 8001 were unreachable; only Ollama
11434 was reachable. Therefore no fresh all-service Pump-102 response, citations,
model latency, artifact or capsule identifiers can be reported from this session.
Historical latency/accuracy results elsewhere in this document are not new
measurements or guarantees for this branch.

Raw JUnit evidence remains local in `.workspace/v1-regression.xml`,
`v1-controlplane-correct-cwd.xml`, `v1-controlplane-original-comparison.xml`,
`v1-diagnostics.xml` and `v1-diagnostics-isolated.xml`.

### Reproduction and verdict

From the isolated worktree, with existing dependencies installed:

```powershell
$env:PYTHONPATH = 'backend;Controlplane.ai/src;Time-Series-Diagnostic-Agent/src'
python -m pytest backend/tests/security -q
Set-Location Graph-RAG
npm run build --workspace=server
node --test server/security/serviceAuth.test.cjs
Set-Location ../frontend
npm run typecheck
node --test tests/task-stream.test.cjs
npm run build -- --webpack
```

The broader historical suites require the local ignored test helpers/fixtures.
Run ControlPlane tests from `Controlplane.ai` so `configs/policies` resolves
correctly. For a fresh live demonstration use README's authenticated service-token
configuration and NEXT_PLAN's ordered acceptance steps; no seeded or generated
data should be assumed present on a clean checkout.

**Initial checkpoint (superseded above): BLOCKED.** The four verified high-severity
issues and scoped file-consumer gaps are remediated and focused regressions pass.
The remaining blockers are existing broader-test failures/version alignment and
an unavailable fresh all-service workflow. Default development authentication,
operator credentials, legacy data migration, globally unique document checksums
and internal transport assumptions remain explicit deployment limitations.
This work does not establish production security accreditation or a new accuracy
score. Merge and release require the user's explicit approval after acceptance.

Last consolidated: 1 October 2026

This file keeps the project history and measured outcomes in one place. Results are local engineering evidence, usually from synthetic or small fixtures; they are not production accuracy, safety, or performance claims.

## Review of pending diagnostic changes — 18 October 2026

Retained the eleven pending files on `research/diagnostic-guards-review`: Process
reference/regime and root-identifiability assessment, sensitivity-only review
alerts, optional reference-holdout calibration, paired regime contract fields,
Wind measurement-quality abstention, and the three maintained research documents.
The source changes were reviewed without retuning or altering their behavior.
The untracked helper modules are required by the modified plugins and must be
committed together; discarding only those helpers would break imports.

Verification: **273 passed** (239 existing diagnostic tests plus 34 focused guard
regressions), 45 dependency/numerical warnings, 36.58 seconds. Public diagnostic
input-contract and provenance/linkage checks: **55 passed**. `git diff --check`
passed. JUnit evidence remains ignored in `.workspace/diagnostic-pending-review.xml`
and `.workspace/diagnostic-pending-contracts.xml`; existing local test scripts,
datasets and research archives remain unpublished under the repository convention.

This retains research work rather than accepting an operational alarm policy.
The reported 57/80 healthy-alert burden and weak-fault sensitivity trade-offs
remain limitations. No fresh accuracy experiment, model training, live service
replay, security-branch merge or final release was performed by this review.

## Process sensitivity-first pilot (10 October 2026)

Short-event/variance follow-up: reference-fitted VAR(1) innovation-energy monitoring
plus weighted means, healthy-only calibration and development-only selection.
Three settings lost 10/9/6 current post-onset fault cases on development; none
qualified. Fresh test confirmed further losses despite fewer healthy alerts.
Retained workflow 3.3; 289 tests pass. All evidence and the final selection remain
local: `local_testing/experiments/results/process_short_event_v1/REPORT.md`.

Follow-up weighted candidate: independently frozen 80-family comparison with
healthy-only calibration reduced healthy alerts 51/80 → 13/80 and retained/improved
step/drift post-onset evidence. It lost 19 individual intermittent/variance/oscillatory
cases relative to current sensitivity. Not promoted; production unchanged. 281
tests pass. Report: `local_testing/experiments/results/process_weighted_v1/REPORT.md`.

The reference-holdout calibration candidate reduced healthy alarms but lost weak
fault sensitivity, so it remains experimental. The subsequent additive mean-change
alert recovers all ten development step/drift misses. On 80 fresh synthetic families,
step/drift/intermittent event alerts improve 74/75/42 → 79/79/77 out of 80 each,
without losing old detections. Healthy alerts increase 8/80 → 57/80. Some alerts
precede faults and some faults remain missed; supplemental-only cases require
review, not a confident root diagnosis. 275 tests pass; no live replay/publication.
Detailed local evidence: `local_testing/experiments/results/process_sensitivity_v2/REPORT.md`.

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

## Wind synthetic pilot and quality repair — 11 October 2026

332 frozen records; eight development, twelve healthy-calibration, twenty-four
fresh-test simulation families. Healthy-only event calibration chose defaults;
no threshold candidate promoted. Fresh strong step/drift/intermittent confirmations
24/24 each, weak step 15/24; healthy/transition 0/48 confirmed but prefix reviews
15/24 and 6/24 respectively. Historical CARE results not rerun or repaired.
Dropout/flatline baseline false confirmations 1/24 and 11/24 motivated explicit
quality routing in Wind workflow 1.2. Same-input helper audit holds 48/48 defective
records and affects none of 96 fault records. Real fitted-runner/plugin regressions
test the wiring; no live API run. Conservative sample-based guard, not calibrated.
252 diagnostic/Wind tests passed (39 warnings); no full-repository suite run.
Exact methods, snapshots, hashes and limitations:
`local_testing/research/wind_monitoring/RESEARCH_LOG.md`.

## Paired Process attribution — 11 October 2026

Stationary independent Gaussian AR(1), common random disturbances, phi .3/.95,
innovation SD `sigma*sqrt(1-phi²)` preserving marginal variance. 40 independent
paired families, innovation/marginal fault-sizing arms, 960 stored records;
duplicate healthy/variance controls are not independent. Fixed detector settings.
Current high-phi additive misses 49/160 with innovation-relative amplitude versus
0/160 marginal-relative; low phi 0/160 both. Paired interaction 30.625 percentage
points (bootstrap 95% 19.375–42.5). Healthy high-phi post-onset alerts 21/40;
zero misses is not proof of diagnosis accuracy. Production unchanged.
307 tests passed (93 warnings). Archive retains dirty source snapshots, inputs,
labels, model parameters, masks, commands and hashes. Complete research history:
`local_testing/research/process_monitoring/RESEARCH_LOG.md`. No publication.

## Process distribution/regime generalization — 11 October 2026

Seven conditions, 24 independent families each, six scenarios/family: 1,008 frozen
records with separate labels, input/source/calibration hashes, masks and fitted
reference models. Compared original/current/fixed 3.5/0.4 rescue without retuning.
Across reference-matched conditions: current 37/720 missed fault records versus
rescue 43/720; healthy alerts 109/144 versus 94/144. Strong autocorrelation is the
largest miss concentration: 35/120 versus 33/120. Declared mismatch guard returned
the expected reason in 144/144 direct checks; full plugin/API was not exercised.
Fault amplitudes are innovation-relative, so autocorrelation also changes SNR.
Sensor-spike alerts are not necessarily unreasonable quality alerts. Post-onset
hits can be incidental alarms; retrospective confirmation is not online validation.
303 diagnostic/Process tests passed (93 warnings). Production unchanged; no
publication. Report: `local_testing/experiments/results/process_generalization_v1/REPORT.md`.

## Process trade-off experiment — 11 October 2026

Added local threshold/innovation-energy rescue comparisons with independent
80-family development/calibration/test splits, frozen input hashes and separate
synthetic labels. Development selected current using exploratory cost
`10 * missed fault records + healthy alert records`; this is not a safety utility.
Test current: 390/400 post-onset detections, 40/80 healthy alerts. Illustrative
3.5/0.4 rescue: 384/400 detections, 33/80 healthy alerts; seven paired losses and
one gain. Other settings also trade fewer healthy alerts for more missed faults.
Production unchanged. Same simulator/new seeds, unreviewed synthetic labels,
retrospective baseline confirmation; not proven online or plant performance.
The selected-versus-current bootstrap interval is trivially zero because current
was selected; it is not evidence of uncertainty-free performance.
Validation: 293 diagnostic/Process tests passed (93 warnings); no live replay.
Local audit: `local_testing/experiments/results/process_tradeoff_v1/REPORT.md`.

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
