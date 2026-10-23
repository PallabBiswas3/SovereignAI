# Clean-environment investigation — 23 October 2026

**Release verdict: BLOCKED.** Complete application images and the authenticated
deployment acceptance suite have not passed. Passing source tests is not a
deployment success. The branch remains `development`, with uncommitted changes;
no release, merge, remote rollout, factory reset or data prune was performed.

## Findings and causal boundaries

| Finding | Evidence | Conclusion |
| --- | --- | --- |
| Windows host completed all nine HTTPS probes: npm ping, 15,840,331-byte npm metadata, and 8,790,396-byte Debian index with default/ChaCha20/TLS 1.2 modes | `host-tls-unrestricted.log` | TLS certificates were authorized; repeated downloads had matching hashes. The destinations and host TLS path work. |
| Untouched `node:22-alpine` failed large downloads with `ERR_SSL_CIPHER_OPERATION_FAILED` / bad-record-MAC; small ping succeeded | `container-tls-default.log` | SovereignAI source, npm configuration, model code and credentials are not required to reproduce the transport failure. |
| Software OpenSSL, alternative authenticated ciphers, MTU 1280 and Docker host networking also failed | Corresponding `container-tls-*` logs | These mitigations did not resolve the fault; no insecure cipher, certificate bypass or checksum relaxation was used. |
| Pristine Python/Debian image failed normal HTTPS APT index acquisition with I/O errors and exit 100 | `container-apt-verified.log` | The Debian transport problem is reproducible outside the repository build. APT signatures/checksums remain enabled. |
| Ubuntu WSL, independently of Docker, failed a Debian HTTPS transfer with bad-record-MAC | `ubuntu-independent-tls.log` | Fault scope includes the shared Windows/WSL transport path. Ubuntu's npm transfer succeeded; this is not a universal inability to reach HTTPS. |
| Reversible virtual-adapter RSC before/disabled/restored probes all failed | `rsc-result.json`, `rsc-admin.log`, three probe logs | Disabling this virtual adapter's receive coalescing did not resolve the failure. Original IPv4/IPv6 flags were restored. |
| Ollama image pull failed writing a containerd ingest file; kernel log shows offline virtual-device writes, aborted ext4 journal and I/O errors | `pull-ollama-runtime.log`, `docker-vm-kernel-errors.log` | A separate Docker storage/recovery blocker is present. The write error occurred during the recovery/restart window. It is not proof that storage caused the earlier TLS errors. |
| Docker restart timed out; bounded stop/start recovery failed to restore a usable engine; the final complete-build attempt got daemon HTTP 500 | Restart/recovery logs and `complete-deployment-build-attempt.log` | The engine is currently unhealthy. No forced disk repair, distribution deletion, Docker reset or user-data deletion was attempted. |
| Final read-only `docker version` timed out after 30 seconds | `final-engine-status.log` | Engine recovery still cannot be established at closeout. |
| Host volume had about 2.1 GiB free; model-volume provisioning required at least 3,845,338,077 bytes including reserve | `ollama-volume-provision-attempt.log` | Provisioning refused before allocating the model volume. There is inadequate build/extraction headroom. |

**Exact upstream Windows/WSL component remains unproven.** The demonstrated fault
is corrupted/incomplete large TLS transfers in the WSL path, independent of the
repository. Enabled Fortinet/Realtek network filters were observed; they were
neither disabled nor established as causal. Do not report a vendor driver,
hardware acceleration, certificate trust, mirror rotation or disk capacity as
the proven cause of the historical npm/Debian corruption.

The earlier checksum failure is preserved in
`local_testing/production-readiness-20261023/container-build-2.log`. Current
HTTPS probes reproduce transport/authentication failures rather than fabricate
a fresh checksum mismatch. The original failure, the current reproductions and
the later Docker I/O failure are separate observations.

## Changes made

| Files | Change and purpose |
| --- | --- |
| `deploy/model-lock.json` | Pin official Hub commits, sizes, Git blob identities and SHA-256 for every required model file. |
| `backend/app/monitoring/model_manifest.py` | Reject wrong revisions, missing/traversing files, external references and byte/hash mismatches. |
| `scripts/stage_production_models.py` | Resolve only pinned commits; copy only locked files to a temporary directory; verify before atomic promotion; reject corrupt existing directories without overwrite. |
| `scripts/preflight_production.py` | Verify locked model bytes and add an explicit 10 GiB provisioning-space lower bound on the checkout filesystem. This does not prove sufficient capacity on a different Docker backing disk. |
| `scripts/verify_production_models.py` | Load all three local models with remote code/downloads disabled, use safetensors, and check finite CPU outputs. This is a load smoke test, not a detector/verification accuracy benchmark. |
| `deploy/ollama-model-lock.json` | Pin manifests checked against the official TLS registry and their nine content-addressed blobs. |
| `scripts/provision_ollama_models.py`, `tests/deployment/ollama_volume_verify.py` | Verify cached blobs, refuse conflicting existing tags and Docker read errors, copy only model files, and support verification across container recreation. Synchronous directory creation removes an entrypoint/copy race. Host identity keys are never copied. Capacity failure prevents writes. `--storage-only` cannot claim daemon/inference success. |
| `backend/app/monitoring/readiness.py` | Malformed model-list entries now report unready instead of raising an unhandled exception. Valid application behavior is unchanged. |
| `backend/Dockerfile` | APT explicitly fails on any repository-update error. Signature, TLS and package-integrity verification remain active. |
| `scripts/deployment_evidence.py` | Capture bounded commands, arguments, exit status, duration and output hashes; refuse overwritten evidence labels. Preserve Unicode output without masking the command's real exit code. |
| `scripts/Test-WslTransportRsc.ps1` | Administrator diagnostic for the virtual adapter only; records and restores original settings. Refuses to interrupt running Docker workloads. |
| `tests/deployment/*transport*`, `*model_lock*`, `test_ollama_lock.py`, `test_readiness.py` | Reproducers and regression checks for locked files, same-size corruption, forged provenance, Ollama blob/manifest tampering and malformed health payloads. |
| `scripts/summarize_deployment_investigation.py`, deployment/current-plan documentation | Publish command/test evidence, preserved-source hashes, remaining gates and the explicit BLOCKED verdict. |

Authorization, access/tool policies, governance/release thresholds, diagnostic
logic and the historical failing test were not changed. The machine-readable
report compares SHA-256 of ten protected files against the preceding batch's
recorded source identity.

## Model and storage results

All required model bytes verified against locked metadata and all three models
loaded offline and produced finite outputs:

| Model | Immutable revision | Smoke output shape |
| --- | --- | --- |
| NLI DeBERTa | `fa2804872c3b4bd748f38c0185cc85775361e735` | `[1,3]` |
| MiniLM embedding | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` | `[1,10,384]` |
| MiniLM reranker | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | `[1,1]` |

The host Ollama cache contains two verified models and nine verified blobs.
Official and cached manifest digests matched:

- `qwen3:4b-instruct`: `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`
- `nomic-embed-text:latest`: `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`

**Container persistence and Ollama inference remain BLOCKED**, not passed. The
runtime image manifest exists, but its pull failed. The persistent-volume
provisioner exited 2 on its capacity check; no successful daemon recreation is
claimed and the host cache was not modified.

## Tests and acceptance

Fresh maintained source regression: **403 passed**. All 41 frontend tests,
frontend typechecking, GraphRAG TypeScript build and three GraphRAG auth tests
passed. Broader local ControlPlane: **240 passed, 15 skipped, one failed**.
The failure is the same historical ablation expectation:
`test_pre_authority_ablation_reuses_same_evidence_and_semantic_score`.
No assertion was removed, marked xfail, skipped or weakened; the aggregate
baseline command correctly returned nonzero.

After the final provisioning-helper correction, **12 targeted cases passed**:
five new cases cover matching/absent/conflicting tags and read/daemon errors;
seven rerun model-lock cases cover integrity. This supplements the 403-test
baseline; the complete baseline was not rerun after this isolated helper change.
These mock-based helper checks do not establish successful volume provisioning.

Final preflight returned nonzero: model integrity and credential checks passed,
while provisioning capacity and Docker engine health failed. Static Compose
configuration validity does not establish a functioning deployment.

| Authenticated deployment gate | Result |
| --- | --- |
| Trusted TLS, real browser login and scoped sessions | NOT RUN — complete runtime unavailable |
| Role/workspace isolation and denied downloads | NOT RUN against deployed runtime |
| Browser upload, continuation, diagnosis and authorized retrieval | NOT RUN |
| Governed release, approval/rejection and durable receipts | NOT RUN |
| Report delivery and capsule verification | NOT RUN |
| Dependency outages, readiness and restart/replay behavior | Source regressions passed; live deployment gate NOT RUN |
| Persistent Ollama volume, daemon recreation and digest checks | BLOCKED by capacity/image/engine |

## Commands and evidence

Exact probe/build/provision/test invocations, working directories, exit codes,
durations and output SHA-256 are in
`local_testing/deployment-remediation-20261023/commands.jsonl`. Each label has a
matching log. `deploy/remediation-20261023.json` publishes the same ledger with
repository paths normalized and verifies the log hashes. Scripted child actions
are reviewable in the named script; the elevated RSC diagnostic additionally
has a transcript and before/result JSON. Initial transport collection used a
collector that exited zero even when individual transfers failed; the report
uses the per-transfer error records, not that exit code, as transport evidence.
Subsequent probe versions return nonzero for failed transfers.

Full source regression logs, JUnit XML and exact source identity are in
`local_testing/readiness-baseline-20261023T150845Z/`. No database files, secret
values, identity keys or model weights belong in an evidence export.
The final uncommitted source hashes, including the supplemental helper correction
and this report, are recorded in
`local_testing/deployment-remediation-20261023/final-source.json`.

Read-only inspections also included Git status/source files; Docker version,
architecture, image/container/volume inventory; CPU/NIC/RSC/LSO/filter state;
WSL versions/distributions; host free space and Docker VM kernel errors. No
driver was disabled, no WSL-wide shutdown was performed, and the already-running
Ubuntu distribution was preserved. Temporary Docker network removal was attempted
but could not be verified while the engine was unavailable.

## Remaining release gates and safe continuation

1. Restore adequate disk headroom and Docker VM/engine health without deleting
   unreviewed user data. The provisioner reports its exact payload requirement;
   preflight's 10 GiB floor is a lower bound, not a guarantee for all image builds.
2. Resolve or conclusively identify the Windows/WSL transfer fault with the host
   administrator/vendor. Re-run the same verifying probes; do not disable network
   security filters, TLS authentication or package checks to force a pass.
3. Rebuild all images using the complete profile and retain successful build logs.
4. Provision and verify the locked Ollama volume across container recreation,
   then run actual generation/embedding checks.
5. Run every authenticated deployment gate above with approved, explicitly
   scoped test identities/evidence and the trusted deployment CA.
6. Resolve the recorded legacy test contract explicitly; retain its failure until
   then. Release remains BLOCKED while live gates are unexecuted.

Diagnostic options were checked against the official
[OpenSSL capability documentation](https://docs.openssl.org/4.0/man3/OPENSSL_ia32cap/)
and [Microsoft RSC documentation](https://github.com/MicrosoftDocs/windows-powershell-docs/blob/main/docset/winserver2022-ps/NetAdapter/Set-NetAdapterRsc.md).
Those documents explain the diagnostic mechanisms; they do not establish this
machine's root cause.
