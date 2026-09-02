# SovereignAI 2.0 Batch 3 — Workcell Platform and Verifiable Evidence Capsules

Batch 3 is implemented and stops before Batch 4. The pre-change baseline is recorded in `SOVEREIGNAI_2_BATCH3_BASELINE.md`.

## Delivered architecture

The existing FastAPI modular monolith remains the orchestrator. `app.api.tasks` selects a Workcell explicitly or from a task class, validates it through the local registry, runs its declarative DAG through `WorkcellExecutor`, and delegates executable behavior to the trusted handler registry. Pump Inspection's validated Batch 2 engineering implementation is reused by a registered adapter; its extraction, hybrid retrieval, ContextCompiler, unit conversion, deterministic calculations, verification, conflict handling, artifacts, governance, audit, and SSE behavior were not rewritten.

The persisted `AgentRunState` now adds backward-compatible optional Workcell ID/version/hash/workflow fields and a detailed step state. Workcell version participates in the Pump deterministic cache identity. Generated artifacts persist SHA-256, Workcell identity, artifact type, and claim lineage.

## Workcell platform

- Strict Pydantic domain models cover manifests, versions, inputs/outputs, steps, tools, evidence, policy, artifacts, evaluations, validation, trust, catalog, and execution state.
- `WorkcellLoader` uses safe YAML, strict JSON/YAML parsing, allow-listed directories/extensions, file-size limits, deterministic ordering, traversal containment, and symlink rejection. No pack code is imported.
- `WorkcellValidator` checks semantic/platform versions, schemas, DAG topology, dependencies, terminal step, handlers, artifact handlers/paths, evidence references, tools, policy, and Ed25519 trust.
- Tool authority is an intersection with `config/tools.yaml`; a Workcell can reduce but cannot increase permission.
- `WorkcellRegistry` discovers local directories, detects duplicate ID/version, exposes typed status, resolves explicit versions, and resolves ready task-class candidates without silently selecting invalid packs.
- `WorkcellHandlerRegistry` is the sole declarative-name-to-application-capability boundary.
- `WorkcellExecutor` deterministically orders a validated DAG, validates bounded input schemas, records step inputs/outputs/completion/failure, supports bounded conditions/failure behavior, and emits existing SSE events.

The official `workcells/pump_inspection` pack is version 1.0.0. It declares schemas, policy, evidence requirements, workflow, prompt identities, artifact definitions, rules, and an evaluation case. It has two trusted steps: input validation and delegation to the existing Pump Inspection service. This deliberately preserves deterministic engineering behavior instead of duplicating it in a second orchestration stack.

A second demo pack was not added because generic execution is already proven with isolated handler-registry/DAG tests, and Batch 3 made it optional. Keeping a single real pack avoids shipping a shallow workflow merely for catalog size.

## Content identity and capsule verification

`ContentIdentityService` provides streamed file hashing, byte hashing, stable canonical JSON hashing, and deterministic directory-manifest identity. Workcell prompts record template path and SHA-256 separately from dynamic model input.

`EvidenceCapsuleBuilder` constructs in a temporary local directory and renames only after success. It exports the final answer, input manifest, Batch 2 structured evidence, Workcell/workflow/prompt/model identity, policy decisions, tool calls, human decisions, audit JSONL, and registered artifacts. Ollama weight digests remain `null` when unavailable rather than being fabricated.

`hashes.sha256` is sorted by normalized relative path. `capsule_root_hash` is SHA-256 of canonical JSON over the sorted list of `{path, sha256}` payload identities. This is not called a Merkle tree. `EvidenceCapsuleVerifier` independently rejects schema errors, missing/extra files, byte/size changes, hash-manifest changes, root inconsistency, and invalid signatures with typed path-specific failures.

SQLite stores capsule metadata and `BUILDING`, `COMPLETE`, `VERIFIED`, `INVALID`, or `FAILED` state. The API exposes build, metadata, verify, and safe ZIP download. Audit and task events record Workcell selection/validation/steps plus capsule build/create/hash/sign/verify/failure.

## Signing and trust

`CapsuleSigner` is an abstract sign/verify boundary. `Ed25519CapsuleSigner` uses the local `cryptography` package. `WorkcellTrustStore` stores only approved public keys by key ID. The repository contains no private key and creates only ephemeral test keys.

The same content-signing abstraction validates optional Workcell `signature.json` without putting that signature into the signed content hash. Development allows unsigned Workcells/capsules by configuration. Strict policy rejects unsigned content. Unknown keys are `SIGNED_UNVERIFIED`; mismatches are invalid.

Hash integrity and signature authenticity are intentionally reported separately. Neither asserts factual correctness.

## APIs and frontend

Added endpoints:

- `GET /api/workcells`
- `GET /api/workcells/{id}`
- `POST /api/workcells/{id}/validate`
- `POST /api/tasks/{task_id}/capsule`
- `GET /api/tasks/{task_id}/capsule`
- `GET /api/capsules/{id}`
- `POST /api/capsules/{id}/verify`
- `GET /api/capsules/{id}/download`

`POST /api/tasks` and `/start` accept optional `workcell_id` without breaking existing requests. The frontend now loads the local READY catalog, supports automatic/manual workflow selection, displays Workcell/version/hash/trust context, integrates capsule inclusion into “Why This Answer?”, and exposes create, verify, integrity/signature/artifact status, exact failures, root hash, and ZIP download.

## Security controls

- Declarative definitions cannot load arbitrary code.
- Disabled/unknown tools invalidate a pack; approval cannot override global disablement.
- Pack and artifact paths are bounded and normalized.
- No network registry, cloud signature service, timestamp authority, external model API, or telemetry was added.
- Capsules exclude hidden chain-of-thought and retain concise auditable observations only.
- Runtime outputs remain under local filesystem/SQLite storage.

## Tests and validation

Focused Batch 3 coverage includes valid/invalid manifests, versions, duplicate IDs, handlers, tools, disabled-tool escalation, traversal, symlink escape where supported, missing dependencies, cycles, DAG order, failure behavior, input/state output, canonical hashing, Workcell trust/signature modification, capsule creation/schema/files/root, tampering, missing/extra files, invalid manifests, Ed25519 valid/wrong-key/tamper cases, unsigned development/strict policy, Workcell API, Pump state persistence, capsule API verification, and ZIP download.

Final exact suite/build results are recorded after the final clean-room regression at the end of this document.

## Honest limitations

- Ed25519 local trust is not enterprise PKI, formal certification, HSM key custody, or certificate lifecycle management.
- SQLite remains a single-node store.
- Full authentication/RBAC and production document ACL authorization are absent.
- Model digest availability depends on local Ollama metadata.
- Deterministic findings are reproducible for identical inputs/rules/version; LLM presentation prose is not bit-for-bit deterministic.
- Capsule verification proves stored-content integrity and optional key-holder signature, not factual correctness or completeness.
- Unsigned development capsules can be rebuilt by a filesystem holder; strict signed deployment needs external enterprise key governance.
- SSE channels remain process-local even though events are persisted.

## Exit boundary and Batch 4 recommendation

Batch 4 was not started. The recommended next batch is identity and authorization hardening: authentication, OIDC/LDAP integration, full RBAC, production document ACL enforcement, PostgreSQL/durable job infrastructure, and enterprise key management. GraphRAG, asset graphs, OPC-UA/CMMS connectors, Kubernetes, LoRA, and larger models should remain separate decisions backed by evaluation need.

## Final verification record

- Focused Batch 3 suite: **19 passed, 1 skipped**. The skip is the symlink-escape test on a Windows environment that did not grant symlink creation; traversal and allow-list tests still ran.
- Complete backend suite: **87 passed, 1 skipped, 2 warnings** in 49.49 seconds with the live-model generation timeout bounded to 2 seconds for tests.
- Python compile validation: **passed** (`python -m compileall -q backend tests`).
- Frontend TypeScript validation: **passed** (`npm run typecheck`).
- Frontend production build: **passed** (`npm run build`), producing `/`, `/_not-found`, `/metrics`, and `/sovereignty`.
- Pre-existing warnings retained: Starlette `TestClient`/`httpx` deprecation and the installed Torch/NumPy ABI warning.
