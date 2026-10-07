# Next implementation plan

Updated: 4 October 2026

The architecture is frozen. The next milestone is a reproducible Pump-102 assurance bundle that demonstrates the project goal: private, authorized, evidence-backed industrial recommendations that remain auditable, advisory, and fail-safe. It is not another agent or a larger feature surface.

The 4 October live smoke reached all services; an approved segregated synthetic Supabase seed then provided scoped chunks. A follow-up fixed the diagnostic policy reference and achieved a live `diagnose` result. The strict release test still held the generated answer for 34 non-supported claims and a consequential latency-budget breach. Isolated local-auth downloads allowed maintenance resources and denied restricted finance resources. Detailed local-only reports and results are in the Git-ignored `local_testing/` folder, not on GitHub. Next prove a genuinely supported live release without weakening that gate, then repeat download checks against its registered artifact and capsule. Do not treat held runs or synthetic fixture downloads as production recommendations.

## 1. Close the three remaining experiments

1. **Controlled local inference comparison**
   - Engineering controls are implemented and fixture-tested: exact runtime/host
     provenance, sampled memory, provider-verified prompt ceilings, randomized
     prefix/concurrency blocks, verified provider-specific resets, versioned
     quality labels and strict completion gates.
   - Stop before the final live run until the existing backend environment/fixture
     failures are resolved on the target host and the runtime APIs/permissions are
     available. Supply actual human-reviewed real evidence; do not relabel the
     shipped synthetic fixture as reviewed or real.
   - Preserve the original live-check as preliminary evidence. No v3 live result
     exists. GGUF Q4_K_M versus BF16 is explicitly a runtime/format/host comparison.
2. **Live authorized GraphRAG**
   - The approved segregated synthetic Supabase namespace is seeded; preserve its ACL metadata and do not reclassify legacy rows. For production-like or sensitive evidence, use approved local/self-hosted storage.
   - Cover direct, numeric, multi-document, conflicting revision, unanswerable, and unauthorized-perfect-match cases.
   - Authenticate the test principal and prove forbidden evidence is absent before scoring and from context, citations, and actual exported/downloaded artifacts.
3. **Networked Pump-102 end to end**
   - The all-service synthetic smoke, scoped Graph-RAG retrieval and a live non-abstaining diagnostic are complete. A policy-released, cited answer with registered artifact and Evidence Capsule is **not** complete. Use `local_testing/scripts/capture_live_pump102.py --require-released-path` on a fresh local-only output directory; held answers fail that structural gate.
   - Preserve latency, citations, release decision, artifact identity, and capsule root hash.
   - Repeat with no evidence, unauthorized evidence, and required-service outage; each must abstain or fail closed.

### Controlled inference commands (run only after prerequisites)

Use the same branch contents and reviewed fixture on Windows and WSL. Run each
provider's harness locally; remote PIDs/RAM are never inferred. The vLLM collector
must run in the same activated Python environment as its server, including
`VIRTUAL_ENV`/`PYTHONPATH`. `psutil` and `httpx` must be installed there. This does
not require installing the application's pinned Transformers into the vLLM env.
Keep both instances dedicated and otherwise idle. Run the providers sequentially
on a shared physical computer to avoid mutual resource interference.

Prepare a **real, independently reviewed** fixture using
`local_testing/benchmarks/fixtures/quality_fixture_v1.schema.json`, saved at
`local_testing/benchmarks/fixtures/pump102_quality_reviewed_v1.json`. Populate actual evidence,
expected facts/citations and real reviewer/date/method metadata. The supplied
`pump102_quality_v1.json` is only a synthetic, unreviewed schema example. Do not
invent labels or reviewer metadata. Output scoring is still an automated lexical
proxy; completion does not certify generated-answer factuality.

From the repository root in PowerShell, first inspect a request-free plan:

```powershell
git switch pump102-inference-completion
git pull --ff-only
python local_testing/benchmarks/inference_tradeoff.py --protocol controlled --plan-only `
  --context-lengths 256 512 --evidence-budgets 256 512 --repetitions 5 --seed 102 `
  --output local_testing/experiments/results/inference_v3_plan
```

Expected exit 3 means incomplete; no model request is made. Never reuse a nonempty
output directory. Larger real fixtures may require raising **both** hosts' budgets
identically; undersized budgets fail instead of truncating facts.

After the reviewed fixture and test prerequisites are ready, run Ollama first from
PowerShell with its dedicated local service already listening on port 11434:

```powershell
python local_testing/benchmarks/inference_tradeoff.py --protocol controlled --providers ollama `
  --ollama-model qwen3:0.6b --ollama-url http://127.0.0.1:11434 `
  --context-lengths 256 512 --evidence-budgets 256 512 --repetitions 5 --seed 102 `
  --quality-fixture local_testing/benchmarks/fixtures/pump102_quality_reviewed_v1.json `
  --timeout 600 --reset-timeout 600 `
  --output local_testing/experiments/results/inference_v3_ollama
```

After the Ollama run, unload its model before measuring vLLM:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:11434/api/generate `
  -ContentType 'application/json' `
  -Body '{"model":"qwen3:0.6b","keep_alive":0,"stream":false}'
```

In WSL terminal A, activate your existing **working vLLM environment**, then resolve
the already-cached snapshot without downloading. Start the dedicated server:

```bash
export HF_HUB_OFFLINE=1
export MODEL_SNAPSHOT="$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True))')"
printf '%s\n' "$MODEL_SNAPSHOT"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_SNAPSHOT" --served-model-name Qwen/Qwen3-0.6B \
  --dtype bfloat16 --host 127.0.0.1 --port 8001 \
  --max-model-len 4096 --enable-prefix-caching
```

The snapshot basename must be the immutable 40-hex revision. Keep the working CPU
vLLM installation and its resource environment settings. Do not start a second
server on an occupied port. When the harness asks for a restart, use **Ctrl+C in
terminal A**, wait for its workers to stop, and repeat that exact launch command.
There is no automatic process termination or arbitrary reset-command hook.
Reset occurs after calibration **before every batch**, including warm batches to
prevent cache contamination. For the commands below, expect 180 batches/restarts
per provider (420 measured requests and 280 excluded warm-up requests, plus untimed
calibration calls). Ollama resets itself through its supported API.

Then, in WSL terminal B, enter the same repository checkout and activate the same
vLLM environment as terminal A. Resolve the same snapshot and run:

```bash
export HF_HUB_OFFLINE=1
export MODEL_SNAPSHOT="$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True))')"
python local_testing/benchmarks/inference_tradeoff.py --protocol controlled --providers vllm \
  --vllm-model Qwen/Qwen3-0.6B --vllm-url http://127.0.0.1:8001/v1 \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --context-lengths 256 512 --evidence-budgets 256 512 --repetitions 5 --seed 102 \
  --quality-fixture local_testing/benchmarks/fixtures/pump102_quality_reviewed_v1.json \
  --timeout 600 --reset-timeout 600 \
  --output local_testing/experiments/results/inference_v3_vllm
```

A successful single-provider collection returns **3**, because the comparison
still lacks the other provider. Exit **2** means a runtime/control failure: inspect
`summary.json`; do not present the partial as complete. Budget disagreement,
missing usage, unsupported residency APIs, inaccessible processes, runtime drift,
or an unverified restart aborts collection and preserves recorded failures.
Cold vLLM request latency excludes server startup; Ollama may include model load.
Neither resets OS page caches; concurrency refers to simultaneous submitted
requests, not guaranteed simultaneous execution by the provider. Warm repeated
prefixes are input conditions, not verified cache-hit counters.

With both output directories visible in the same checkout (copy the whole vLLM
output directory from WSL if you used a separate clone), merge in PowerShell:

```powershell
python local_testing/benchmarks/inference_tradeoff.py --protocol merge `
  --merge-inputs local_testing/experiments/results/inference_v3_ollama local_testing/experiments/results/inference_v3_vllm `
  --output local_testing/experiments/results/inference_v3_final
Get-Content local_testing/experiments/results/inference_v3_final/summary.json
```

Only exit **0** and every completion gate true close this collection protocol.
Merge rejects differing harnesses, fixtures, budgets/seeds/repetitions, duplicate
providers and v2 results. It retains both host identities and source-report hashes.
Memory is sampled RSS/available RAM, not VRAM; brief peaks can be missed and WSL
memory is guest memory. A result still compares runtime + quantization + host.

Exit condition: reproducible commands and machine-readable results for all three experiments.

## 2. Strengthen diagnostic evidence

- Freeze Paderborn specimen-level splits and compare physics-only, 1D-CNN-only, and fusion models.
- Report record-level balanced accuracy and macro-F1 first, plus coverage/abstention, runtime, dataset identity, and split provenance.
- Use Lenze as a held-out operating-condition transfer test for current-only, speed-only, and combined inputs.
- Do not tune against the frozen Wind CARE events or final holdouts.

Exit condition: a leakage-safe diagnostic result that can be inserted into the Pump-102 evidence chain.

## 3. Build decision-grade factuality evaluation

- Create 300-500 manually adjudicated industrial cases.
- Store each claim, exact evidence bundle, provenance, expected policy action, and reviewer metadata.
- Label `SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`, `UNDECIDABLE`, or `CONFLICTING`.
- Compare optional support backends only after the dataset and holdout are frozen.

Exit condition: state-level precision/recall and release/hold error rates with confidence intervals.

## 4. Prepare for an enterprise pilot

- Add document lifecycle controls: deduplication, revision/supersession, quotas, archival/deletion, re-indexing, and orphan cleanup.
- Add service identity and authenticated transport, centralized secrets, managed signing keys, SBOM/build/model provenance, and dependency scanning.
- Replace process-local jobs/events and prototype storage where durability or scale requires it.
- Add append-only/tamper-evident audit, backup/restore tests, retention, monitoring, and incident procedures.
- Produce an intended-use statement, risk register, human-oversight plan, change-control record, and shadow-pilot protocol.

Exit condition: an auditable, read-only, supervised pilot package with explicit residual risks.

## Deferred until evidence justifies it

- Plant write/control capability or autonomous work approval.
- New agents, major architecture rewrites, public Workcell marketplace, or broader graph algorithms.
- Fine-tuning, larger models, or new factuality thresholds without a measured gap.

See [Current features](CURRENT_FEATURES.md) for the implemented baseline and [Past experiments](PAST_EXPERIMENTS.md) for measured evidence.
