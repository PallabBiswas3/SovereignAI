# Authenticated single-host deployment candidate

**Current verdict: BLOCKED.** See the [clean-environment investigation](CLEAN_ENVIRONMENT_REPORT.md)
and [machine-readable execution ledger](remediation-20261023.json). The missing
reranker is now provisioned and hash-verified; the remaining blockers include WSL
transport corruption, Docker engine/storage health, capacity, persistent Ollama
verification and unexecuted live acceptance gates. Earlier results below describe
the initial packaging batch.

Supported target: Linux x86-64 Docker Engine with Compose v2, one internal host,
local Ollama, local evidence storage, scoped local accounts, and human-reviewed
Pump-102 assessment and Word reporting. Windows Docker Desktop is a validation
environment. This is a deployment candidate, not plant acceptance or security
accreditation. The current development tree includes diagnostic research; its
limits still apply. Do not promote that research based on packaging tests.

The profile includes backend, frontend, diagnostics, ControlPlane, GraphRAG,
Ollama, a private PostgreSQL/PostgREST evidence store, and Caddy TLS ingress.
It uses separate volumes and does not reuse development databases or uploads.
PostgreSQL here is GraphRAG storage; the backend continues to use SQLite.
There is no Docker socket mount. Code execution, vision, OCR and plant actions
are disabled in this profile's tool configuration.

## Baseline and checkpoint inventory

The original baseline is commit `5eaf907f9b74e686ec49d1aa61755fd0a5f86b0a` on
`development`. Before changes: 367 maintained Python tests passed; 41 frontend
tests passed; frontend typechecking and GraphRAG build/authentication tests passed.
The broader local ControlPlane suite, run from its service directory, returned
240 passed, 15 skipped and one failed. The remaining failure is
`test_pre_authority_ablation_reuses_same_evidence_and_semantic_score`: it expects
a superseded-only current-limit claim to become supported with the authority
ablation disabled. Current source filtering/support guards leave it undecidable.
This is an ablation-contract mismatch, not evidence of an unsafe release. It
remains an explicit failure; this batch does not change the test or verifier.
Running that suite from the repository root additionally fails policy discovery
because its default policy path is relative. The deployment sets the correct
ControlPlane working directory. Historical ten-failure reports are not the
current baseline.

`readiness-checklist.json` inventories all 73 supplied checkpoints with status,
evidence locations and limits. Existing runtime features are marked implemented
but unverified where this batch does not establish deployed acceptance.

Reproduce the maintained baseline after installing existing project dependencies:

```powershell
python scripts/run_readiness_baseline.py
```

Add `--include-local-controlplane` only when ignored local research tests/data
are available. Each run writes a source-identity manifest and separate logs/XML
under ignored `local_testing/readiness-baseline-*/`. The initial and post-change
results are distinct; a dirty source manifest is not a committed release.

## Provision before starting

From the repository root:

Use Python 3.11+ in an isolated environment. Provisioning helpers require
`PyYAML>=6,<7` and `huggingface-hub==0.36.2`; the backend's documented dependencies
already include these. Host Node.js 22+ and installed frontend/GraphRAG packages
are needed for the source regression runner, not for container runtime startup.

```powershell
python scripts/prepare_production.py
python scripts/stage_production_models.py
python scripts/preflight_production.py --static
docker compose -f deploy/compose.production.yml config --quiet
docker compose -f deploy/compose.production.yml build
```

Preparation creates random credentials in ignored `deploy/secrets/`, never logs
their values, preserves complete existing sets, and refuses partial sets.
Restrict this directory to the operator using filesystem ACLs, particularly on
Windows. Compose mounts each service's required files. These local secret files
are not a vault; include them in the later secure recovery procedure. The graph
service JWT expires after one year; plan credential renewal before expiry. Do
not regenerate database passwords against an initialized volume.

Model staging uses only the commits/files pinned in `deploy/model-lock.json` and
verifies every file's size and SHA-256, including existing directories. It uses
cached snapshots by default and records provenance in `sovereign-snapshot.json`.
To explicitly permit provisioning-time
downloads, use `python scripts/stage_production_models.py --download`. Supply
complete snapshots of the named NLI, embedding and reranker models under
`deploy/models/`. Runtime downloads and hash-embedding fallback are disabled.
Model staging never overwrites an existing directory; an invalid existing copy
fails verification. New copies are promoted atomically only after verification.
Preflight rejects corrupted or mismatched snapshots. Use
`python scripts/verify_production_models.py` for an offline load smoke check;
this is not model-accuracy or workflow acceptance evidence.

Ollama uses a separate volume. `deploy/ollama-model-lock.json` pins official
manifest/blob digests. With a verified existing cache and adequate space, use
`python scripts/provision_ollama_models.py --volume sovereign-production_ollama-models`
to copy only locked model files and verify container recreation. This never
copies the host's Ollama identity keys and refuses conflicting existing tags.
`--storage-only` can verify persisted bytes but cannot establish daemon/inference
readiness. The investigation's provisioning attempt refused insufficient space;
persistent storage is not yet verified on this machine.

Before starting the isolated profile, stage the
exact `qwen3:4b-instruct` and `nomic-embed-text` models in that volume. One online
provisioning approach is a temporary container on Docker's normal bridge:

```powershell
docker volume create sovereign-production_ollama-models
docker run -d --name sovereign-model-provision -v sovereign-production_ollama-models:/root/.ollama ollama/ollama:0.33.2
docker exec sovereign-model-provision ollama pull qwen3:4b-instruct
docker exec sovereign-model-provision ollama pull nomic-embed-text
docker stop sovereign-model-provision
docker rm sovereign-model-provision
```

Use this temporary container only for staging models. The runtime networks are
internal. Export and record model digests for the supported installation; model
tags alone do not establish immutable model identity. GPU passthrough is not
configured in this first CPU profile; benchmark on the intended hardware.

Optionally create ignored `deploy/.env` to set `SOVEREIGN_HOST`, `HTTPS_PORT`
(default 8443), and `BIND_IP` (default 127.0.0.1). Pass
`--env-file deploy/.env` on every Compose command if using it. For LAN use,
configure internal DNS and a specific bind address. Do not expose specialist
service ports. There is one published HTTPS listener.

```powershell
python scripts/preflight_production.py
docker compose -f deploy/compose.production.yml up -d
docker compose -f deploy/compose.production.yml exec backend python -m app.identity.provision --email operator@example.internal --organization company --workspace engineering --department operations --role ENGINEER
```

Use actual scoped identifiers and a unique password entered at the terminal.
The profile never creates demo accounts. Create separate approver/operator
accounts as required; the CLI refuses overwriting accounts or wildcard scopes.
On an empty database `/ready` intentionally returns 503 until an enabled account
is explicitly provisioned. The existing organization-pack importer remains the
path for a full organization, asset and access configuration.

Provision approved, scoped SOP/document evidence into the private graph store
before workflow acceptance. A clean store is empty; healthy storage does not
mean it contains sufficient evidence. Legacy GraphRAG administration APIs remain
disabled, and the service JWT never belongs in the browser. Existing backend
knowledge ingestion and GraphRAG ingestion are separate stores; this batch does
not add automatic synchronization or seed synthetic SOPs into production.

## TLS and readiness

Caddy uses its internal CA, with persistent CA material in the gateway volume.
Trust its public root certificate on authorized clients before using the UI:

```powershell
docker compose -f deploy/compose.production.yml cp gateway:/data/caddy/pki/authorities/local/root.crt deploy/local-root.crt
python scripts/preflight_production.py --url https://localhost:8443 --ca deploy/local-root.crt
```

The internal-CA approach follows [Caddy's TLS documentation](https://caddyserver.com/docs/caddyfile/directives/tls).
Client trust installation is an operator action. Never bypass TLS verification.
The public root certificate is not the CA private key. Do not export the private
CA key or commit secrets, generated certificates or model weights. Organizations
requiring managed certificates should replace `tls internal` with their approved
certificate provisioning before rollout.

`/health` is backend process liveness. `/ready` returns 200 only when its checks
pass, otherwise 503, including configuration, writable storage, database,
provisioned identity, required service health and matching installed generation
model. Probes are bounded and do not generate model answers. GraphRAG health
checks its credential configuration, required table access and model presence;
ControlPlane health stays unavailable until its local NLI model preloads.
Responses omit credentials and raw exceptions. Readiness does not establish
cross-service credential correctness, corpus coverage, role enforcement,
successful inference or authenticated workflow acceptance.

PostgREST reads its URI/signing key from mounted files using its documented
[`@filename` configuration](https://docs.postgrest.org/en/latest/references/configuration.html).
There is no anonymous database role. Its REST prefix is routed on a private
gateway listener, never through the public HTTPS route.

## Validation boundaries and remaining work

This batch establishes the baseline, profile and preflight behavior. Keep build
results, static validation and live acceptance separate. Backups/restores,
rollback drills, CI release gates, operational alerting, authenticated browser
acceptance and representative operator-value evaluation remain later checkpoints.
No service was published remotely and no production-readiness claim is made.

Validation on this machine: the maintained post-change suite passed 389 tests
(367 existing plus 22 initial deployment tests); 41 frontend tests, typechecking,
GraphRAG TypeScript build and three GraphRAG authentication tests passed. Three
additional account-provisioning tests and two stricter origin-validation cases
were added afterwards and checked separately.
Final focused verification passed 97 cases: 70 existing security tests and all
27 deployment/provisioning cases after the final origin-validation changes.
The non-root gateway image built and its actual Caddy configuration validated.
An isolated PostgreSQL 17.6/PostgREST 12.2.3 stack initialized all eight evidence
tables. Ten read-only checks passed, including denial of anonymous/invalid tokens
and access to all eight tables with the generated service JWT. These checks do
not establish the complete authenticated application workflow.

The subsequent remediation run passed 403 maintained Python tests, 41 frontend
tests, typechecking, GraphRAG build and three auth tests. The broader ControlPlane
suite retained one failure. The [investigation report](CLEAN_ENVIRONMENT_REPORT.md)
supersedes the initial blocker inventory; it explicitly records the unexecuted
authenticated deployment gates and current Docker engine failure.

Recorded container-build attempts on 23 October encountered Debian checksum/
transport failures and npm TLS cipher errors. Integrity/TLS verification was not
disabled. Consult `docs/CURRENT_FEATURES.md` for the final validation result of
this batch rather than treating a profile file as proof of a successful build.
