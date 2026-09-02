# SovereignAI 2.0 Batch 3 Baseline

Recorded before Batch 3 implementation on 2026-09-02.

## Validation results

- Backend regression: **68 passed** (`SOVEREIGN_MODEL_GENERATION_TIMEOUT_SECONDS=2`, `python -m pytest -q`).
- Python compile validation: **passed** (`python -m compileall -q backend tests`).
- Frontend TypeScript validation: **passed** (`npm run typecheck`).
- Frontend production build: **passed** (`npm run build`).
- Production routes generated: `/`, `/_not-found`, `/metrics`, and `/sovereignty`.

The backend test run retained two pre-existing warnings: the Starlette `TestClient`/`httpx` deprecation warning and a Torch warning caused by the installed NumPy ABI. Neither warning failed the suite.

## Inspected architecture

- `app.evidence.models` contains the Batch 2 typed source, fragment, measurement, rule, calculation, claim, finding, recommendation, conflict, requirement, and bundle models.
- `EvidenceFirstExecutor` performs deterministic unit conversion, rule calculation, evidence requirement checks, and claim generation before optional prose.
- `VerificationEngine` independently applies schema, evidence, numerical, unit, rule, and lineage checks.
- `InspectionWorkflow` owns the validated Pump Inspection extraction, retrieval, rule parsing, deterministic findings, conflicts, recommendation, and evidence bundle construction.
- `CodingWorkflow` owns the bounded local coding flow and Docker-only execution/repair loop.
- `ToolRegistry` exposes trusted in-process tools. `ActionGuard` enforces the global `config/tools.yaml` policy and correctly prevents approval from enabling disabled tools.
- `ArtifactService` constrains artifacts to the configured artifact root and stores SQLite metadata.
- `AuditLogger`, `TaskEventRecord`, and `TaskEventBroker` provide persisted audit data and live SSE events.
- `Settings` resolves all configuration from project-relative local paths and environment overrides.
- `app.api.tasks` is the existing integration seam for routing, workflows, governance, persistence, artifacts, audit, background execution, cancellation, and SSE.
- The frontend already displays live execution, evidence lineage, calculations, conflicts, sources, and artifact downloads in `frontend/app/page.tsx`.

No `workcells/` directory or Workcell runtime existed at this baseline.

## Batch 3 implementation plan

1. Add strict typed Workcell models, safe loader, deterministic content identity, registry, validator, DAG validation, trusted handler registry, and focused security tests.
2. Add a Workcell executor that writes compatible state and emits events while delegating capabilities to registered application handlers.
3. install Pump Inspection as a declarative `pump-inspection` Workcell and route its validated engineering implementation through a trusted handler; add a small `document-summary` pack as a genericity proof if it remains bounded.
4. Add capsule metadata persistence, atomic local storage, canonical SHA-256 manifests/root identity, structured evidence export, independent verification, Ed25519 signing abstraction/test support, and local trust-store policy.
5. Add Workcell/capsule APIs and integrate selection, state, audit, and SSE into the existing task lifecycle without breaking current endpoints.
6. Extend the existing frontend with local Workcell selection/catalog information plus capsule build, verify, and download controls.
7. Run focused tests at each foundation/integration boundary, then the complete regression, compile, typecheck, and production build gates; document the architecture and stop before Batch 4.

## Compatibility constraints

- Workcells are declarative data, never imported executable modules.
- A Workcell can only reduce global/workspace permissions.
- Existing Pump deterministic logic is reused, not rewritten.
- Workcell routing remains separate from model routing.
- New persisted fields are optional/defaulted for old run records.
- SQLite and local filesystem storage remain in place.
- No cloud runtime, remote registry, GraphRAG, production PKI, or Batch 4 feature is introduced.
