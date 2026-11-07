# Docker implementation and validation — 5 November 2026

## Resumed validation — current run

**Final result: the complete application is running at
https://localhost:8443.** Live smoke/inference, authorization, restart/persistence
and simulated-offline checks pass. The host was not physically disconnected;
industrial workflow/model accuracy and cold-backup restoration are not certified
by these tests. The previous disk/Engine blocker is resolved for this run.

Start/reuse the provisioned installation from `1_SIH`:

```powershell
docker compose --env-file deploy/.env -f compose.yaml up -d --no-build --pull never
```

Login email: `operator@sovereign.local`. Its generated random password is stored
only in ignored `deploy/secrets/local_operator.json`. Trust the deployment public
CA `deploy/local-root.crt` on your browser/Windows account before using HTTPS;
the test clients explicitly verified this CA. No OS certificate-trust settings
were changed automatically. The account has ENGINEER scope, not ADMIN.

The host initially had 43.1 GB free. Docker Desktop was still hung after the
earlier disk-full incident. A controlled restart timed out; stopping the stuck
Desktop processes and restarting only `docker-desktop` exposed a stale WSL
guest-socket mount error. A full, approved WSL restart recovered Engine 29.4.3
(Linux/overlayfs). No reset, unregister, prune or data/volume deletion was used.
Ubuntu was stopped as part of that approved WSL restart. Docker reports about
7.6 GiB RAM and 12 CPUs. The Docker backing filesystem reports adequate capacity;
host C: capacity is monitored separately during builds/provisioning.

The failed shared Python image was rebuilt; successful frontend, GraphRAG and
gateway images were reused. Live startup then exposed a packaging error:
`.dockerignore`'s broad `**/data` exclusion removed `adaptivefact.data` from the
ControlPlane wheel. The exclusion is narrowed to the research data directory,
and the Python build now checks imports for ControlPlane, adaptivefact data and
diagnostics plus `pip check`. This correction requires one further shared-image
build; no unrelated application images are rebuilt.

The absolute-path smoke probe also needed an explicit backend import path.
A regression test now covers launching it outside the project working directory.
HF cache writes are redirected to temporary storage while model weights remain
read-only, with offline flags preserved.

An all-internal gateway had the requested HostConfig port but no actual host
binding under this Engine. Only the gateway now additionally joins a normal
`edge` bridge; six application networks remain internal. The actual binding is
`127.0.0.1:8443`, and the frontend returned HTTP 200 over TLS verified against
the exported public CA. This gateway can initiate outbound traffic when allowed
by the host; application/inference/database containers do not join `edge`.
Preflight and configuration tests explicitly enforce this boundary.

Current completed source checks: **159 deployment/security/checkpoint/resume
tests passed**, **11 targeted ControlPlane tests passed**, and **11 focused
configuration/operation/preflight tests passed** after the runtime corrections.
All three staged HF models also loaded and produced finite outputs inside a
read-only container with `--network none`. Existing model files/secrets were
verified and preserved. Evidence is under `local_testing/docker-resume-20261105/`.

As explicitly approved, a scoped ENGINEER account was created with random
credentials at ignored `deploy/secrets/local_operator.json`. The file is readable
only by the operator Windows account, SYSTEM and Administrators; its password is
not included in logs or this report. No existing account was overwritten.

The Ollama runtime image was absent and was pulled at the existing pinned version.
The local cache passed its lock for two models/nine blobs and was copied once
into the new production volume. The provisioner's verifier initially checked the
volume root instead of its nested `models/` directory; that path is corrected and
regression-tested. The copied files were verified in place without copying again.
All hashes and API digests also passed after Ollama container recreation.

GraphRAG's successful compile/build did not establish a runnable image. Live
startup exposed missing server-workspace `node_modules` in its runtime stage,
then a secret-loading wrapper which required the module without opening its
guarded listener. Both packaging issues are fixed. Final-image dependency imports
and a wrapper-listener regression test now cover these cases. Only this failed
runtime image was rebuilt; frontend/gateway images remain unchanged.

**The complete nine-container stack is running.** Backend `/health` and `/ready`
return 200; ControlPlane, diagnostics, GraphRAG, Ollama, frontend, gateway and
PostgreSQL health checks pass. PostgREST runs and its table access is verified by
the authenticated GraphRAG health check and the separate ten-case graph probe.
Only `127.0.0.1:8443` is published. There are no public database/model/governance
ports or Docker socket mounts.

| Resumed live check | Result |
|---|---|
| Final deployment regression suite | **52 passed** |
| Deployment/security/checkpoint/resume rerun | **159 passed** |
| Targeted ControlPlane routing/readiness/latency tests | **11 passed** |
| Full Docker smoke with CPU inference | **19 checks passed**, including generation, embeddings, policy denial, audit, SQLite integrity, service health and HTTPS frontend/API |
| Verified HTTPS identity, role denial, CSRF, logout and local assets | **14 checks passed**, eight assets fetched locally |
| Graph PostgreSQL/PostgREST access | **10 checks passed**: anonymous/invalid JWT denied, eight tables accessible with service credential |
| Backend/ControlPlane recreation | **18 smoke checks passed**, including data/evidence markers and actual denial audit retained |
| Ollama recreation | **2 models / 9 blobs**, all hashes and daemon digests match |
| HF model loads with network disabled | All three models produced finite tensors |
| Live ControlPlane prechecks | Benign prompt allowed and synthetic secret blocked; both produced audit IDs |
| Simulated offline run | **12 checks passed**: gateway external bridge detached; internal checks and CPU generation/embeddings passed; external TCP blocked in all nine namespaces; HTTPS restored |
| PostgreSQL/PostgREST restart | Database identity, eight-table schema and existing row counts retained; full backend readiness recovered |
| HTTPS acceptance after restart/offline restoration | All **14 checks passed** again, including the persisted ENGINEER login |
| Final full preflight over HTTPS | Passed, including all model hashes, credentials and deployed readiness |

Test counts overlap; they should not be added into a purported total. An initial
smoke call was made before Compose finished starting the backend and correctly
failed. Its final rerun passed. A post-restart HTTPS check overlapped the planned
offline bridge detachment and failed because ingress was disconnected; its
post-restoration rerun passed all 14 checks. During simulated offline acceptance,
the sole external gateway endpoint was disconnected, all internal smoke checks
and real generation/embeddings succeeded, and TCP connection attempts to
`1.1.1.1:443` failed in each service's network namespace. The gateway endpoint
and verified localhost HTTPS were then restored. This is stronger evidence than
a successful build, but remains a simulated disconnection with one external TCP
destination; physical host/network isolation and customer corpus workflows are
not established by it.

Ollama now explicitly sets `OLLAMA_NO_CLOUD=true` and `OLLAMA_NOPRUNE=true`,
disabling optional cloud behavior and automatic model pruning. The same volume
and model selection are retained. CPU cold loading is slow on this laptop;
passing synthetic inference is not a latency SLA or model-accuracy evaluation.

No data, model weights, project files or Docker volumes were deleted, and no
pruning/cleanup was attempted in the resumed work. More than 15 GiB remained
free after images/model provisioning; the final handover snapshot reports
**20.38 GiB free**, nine running services and all configured health checks healthy.
Existing policy/checker/model source,
integration orchestrator and diagnostic pipeline hashes still match the prior
source-identity manifest. The sections below are historical results from the
earlier disk-full attempt and are superseded by this resumed-run evidence.

Additional files changed during resumed validation (beyond the original inventory
below): `scripts/preflight_production.py`, `scripts/provision_ollama_models.py`,
`deploy/graphrag-entrypoint.cjs`, and `tests/deployment/test_ollama_provision.py`.
Existing task files were further corrected as described above. Ignored evidence
scripts and logs reside under `local_testing/docker-resume-20261105/`:
`backend-packaging-build.log`, `graphrag-runtime-build.log`,
`graphrag-entrypoint-build.log`, `smoke-inference-final.log`, `smoke-restart.log`,
`offline-acceptance.log`, `graph-store-auth.log`, `graph-restart.log`,
`https-after-restart-final.log`, `ollama-recreated-verified.log`,
`preflight-final.log`, `deployment-final.xml`, `security-recovery.xml`,
`controlplane.xml` and a non-secret `runtime-manifest.json` with image identities.

The fresh graph/corpus was not seeded with invented plant evidence. Existing
organization packs/documents require deliberate onboarding for useful scoped
industrial answers. No consequential plant operation was executed. The prior
broader ControlPlane authority-ablation failure remains a documented limitation;
that unrelated research expectation was not changed to pass deployment checks.
Live cold-backup/restore, offline image-transfer to another machine, physical
disconnection and representative authenticated plant workflows remain future
acceptance gates. Internal specialist APIs still trust network membership, and
the gateway retains an external route during normal host access.

## Historical outcome before disk recovery

Deployment files, development/host-Ollama overrides, smoke/backup helpers and
operating documentation are implemented in `1_SIH`. Full container acceptance
is **blocked by disk capacity and Docker storage/engine failure**. Offline
operation, authenticated deployed workflows and recovery are not established.
Use [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for architecture, configuration,
exact provisioning/start/stop commands, offline transfer and recovery procedures.

## Historical tests before disk recovery

| Check | Actual result |
|---|---|
| Production Compose configuration | Passed |
| Development Compose configuration | Passed, including final duplicate-key correction |
| Development + host-Ollama Compose configuration | Passed |
| Static production preflight | Passed all seven checks |
| Existing maintained Python regression runner | **459 passed**, 1,130 warnings; 364.24 seconds |
| Existing frontend tests | **41 passed** |
| Frontend TypeScript check | Passed |
| GraphRAG TypeScript build | Passed |
| GraphRAG service authentication tests | **3 passed** |
| Broader local ControlPlane suite | **247 passed, 15 skipped, 1 failed** |
| Initial focused config/preflight/readiness checks | **29 passed** |
| Final new deployment/operation safeguard tests | **7 passed** |
| All three pre-existing Hugging Face snapshots | Hash/size/revision verification passed; existing copies preserved |
| Gateway image | Built |
| Frontend image | Built, including production Next.js compilation |
| GraphRAG image | Built |
| Shared backend/ControlPlane/diagnostics image | Dependency resolution/install and Dockerfile steps completed; final image unpack failed with storage I/O error |
| Locked Ollama model-volume provisioning | Refused: insufficient free staging space; no model volume populated |
| Smoke helper with unavailable engine | Correctly returned failed JSON and exit status 1 |
| Follow-up combined deployment/security/checkpoint/resume rerun | Invalidated by disk exhaustion: filesystem errors and pytest internal error; **not a passing run** |
| Helper syntax and Git whitespace check | Passed |

The existing ControlPlane failure is
`test_pre_authority_ablation_reuses_same_evidence_and_semantic_score`. With
authority gating disabled, it expects superseded-only evidence to become
supported; the existing verifier returns undecidable. It was already documented
in `deploy/README.md`. Policies, verifier code and that test were not changed.

The seven new tests cover Ollama registry overrides, service-network separation,
durable mounts/fail-closed settings, refusal to back up active services, refusal
to restore over an existing volume, checksum failure before restoration, and
explicit failure when Docker is unavailable. These are source/mocked tests,
not live recovery claims.

The maintained runner's local evidence is under
`local_testing/readiness-baseline-20261104T200514Z/` (the directory name uses host
UTC; this report uses the client's date). Its logs/XML/manifest are retained.
The disposable `baseline.db` created by that run was removed during approved
disk recovery; it was not an application database. The later
`local_testing/docker-focused.xml` is incomplete and must not be treated as a
valid report. The initial 29-case run and final seven-case run printed their
successful results directly.

## Capacity incident and blocked live checks

Docker was initially stopped. Desktop started and the Linux engine responded.
The host had about 12.5 GiB free before building. Parallel dependency installation,
image exports and extraction exhausted it. Backend extraction failed at a Torch
header with `input/output error` in containerd's overlay snapshot storage.
An image tag appearing in `docker image ls` is not proof that this failed image
is runnable; rebuild/verify it after storage recovery.

Ollama provisioning refused at 973,701,120 bytes free, below its required
3,845,338,077 bytes staging gate. Approved unused-build-cache cleanup attempts
timed out because the engine stopped responding. The failed build processes
were stopped. The subsequent Desktop stop attempt also timed out, reporting
remaining Desktop/backend/WSL processes; a clean engine shutdown is not confirmed.
No application/model volumes,
existing images or repository user data were intentionally deleted. Do not
resume builds until the host and Docker backing disk have adequate free space
and engine/storage health have been checked.

No new complete stack was started. These checks remain unexecuted:

- Container startup, deployed backend health/readiness/API acceptance.
- Browser → TLS gateway → frontend/backend connectivity.
- Actual local generation and GraphRAG embedding/inference.
- Deployed ControlPlane allow/deny behavior and retained authorization audit.
- Real evidence/model-volume persistence across recreation.
- Container restart/recovery and cold-backup/restore drills.
- Physically disconnected, fully provisioned offline workflow acceptance.

Source inspection verifies same-origin API configuration, internal service URLs,
local-only model loading and absence of external frontend CDN/font imports.
This evidence does not replace the blocked live checks. The smoke script and
offline guide provide repeatable commands for them.

## Complete task-specific file inventory

Added:

- `compose.yaml`
- `compose.dev.yaml`
- `compose.host-ollama.yaml`
- `backend/constraints-docker.txt`
- `backend/scripts/docker_probe.py`
- `deploy/.env.example`
- `scripts/docker_smoke.py`
- `scripts/docker_backup.py`
- `tests/deployment/test_docker_configuration.py`
- `tests/deployment/test_docker_operations.py`
- `docs/DOCKER_DEPLOYMENT.md`
- `docs/DOCKER_VALIDATION.md`

Modified:

- `.dockerignore`
- `.env.example`
- `README.md`
- `backend/Dockerfile`
- `backend/app/router/model_registry.py`
- `frontend/Dockerfile`
- `deploy/compose.production.yml`
- `deploy/Caddyfile`
- `deploy/Dockerfile.gateway`
- `deploy/Dockerfile.graphrag`
- `deploy/README.md`

Local operator configuration `deploy/.env` was copied from its example only
because it was absent; it is ignored. No existing secrets/model snapshots were
overwritten. The unrelated working-tree edits present before this task remain.
Hashes of the pre-existing ControlPlane policy/checker/model edits, diagnostic
pipeline, integration orchestrator, frontend integration page and maintained
feature/plan documents were checked against the run's source manifest and matched.

## Remaining limitations and next gates

This remains a single-host, single-backend-worker candidate using SQLite.
Internal specialist APIs trust their network peers; host/backend compromise is
not contained by Compose alone. Plain Compose secret files are not an encrypted
vault. Ollama retains a root-owned model volume for image compatibility;
PostgreSQL initializes as root then drops privilege. Existing production tool
restrictions and unsigned research capsule/workcell behavior are retained.

Direct Python constraints, npm locks and digest-pinned application bases improve
repeatability, but a complete transitive Linux hash lock and Debian snapshot
are still needed for source-build reproducibility. Offline promotion should use
verified saved images and artifact inventories. Backup helpers' refusal guards
are tested; their live extraction/ownership and full restoration remain untested.

Next gates: recover disk/engine health; rebuild the failed shared image; provision
locked models; execute smoke/inference/recreation/backup/restore and disconnected
checks; validate actual authenticated corpus/evidence workflows. Then address
service authentication, SBOM/vulnerability review, credential rotation, immutable
audit retention, measured CPU sizing, monitoring and explicit migration/release
gates before industrial operation.
