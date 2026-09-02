# SovereignAI 2.0 Batch 5 baseline

Recorded before Batch 5 implementation on 2026-09-02.

## Exact validation

- Complete backend regression: **103 passed, 1 skipped** (104 tests collected).
- Python compilation: **passed** (`python -m compileall -q backend tests scripts`).
- Frontend TypeScript: **passed** (`npm run typecheck`).
- Next.js production build: **passed** (`npm run build`).
- Existing routes: `/`, `/_not-found`, `/metrics`, `/organization`, `/sovereignty`.

The existing skip is the Windows symlink-capability test. The two existing warnings are the Starlette TestClient/httpx deprecation and the installed Torch/NumPy ABI warning.

## Repository inspection

- The APEL generator has one canonical `assets.yaml`, creates 55 deterministic files, attaches Batch 4 document ACLs during ingestion, and safely resets only APEL-scoped data. Its current asset entries are document-generation metadata rather than persistent Asset Passports.
- `EvidenceSource`, `Measurement`, `Rule`, `Finding`, `Recommendation`, `Calculation`, `Claim`, and `EvidenceConflict` already provide the correct typed lineage foundation. `UnitService` is the single Pint-based unit normalization seam and will be reused.
- `ContextCompiler` already bounds evidence by model window, execution mode, chunk count, token count, revision, relevance, and deduplication. It has no structured asset-context field yet.
- Dense and BM25 retrieval exclude unauthorized documents before scoring. Hybrid fusion, reranking, context compilation, and caches operate only on that authorized set. Asset-aware preference must be added after this ACL gate.
- The Pump Inspection Workcell is a validated declarative DAG resolved only to trusted handlers. The underlying inspection workflow currently accepts uploaded inspection/SOP evidence and must retain that mode when optional asset context is added.
- Approval persistence already records the requester, canonical exact-action hash, separate authenticated approver, requester/tool reauthorization, schema validation, and current ActionGuard policy.
- Evidence Capsules already create deterministic, independently verifiable payload manifests. Exact telemetry snapshots and trend objects can be added as bounded structured payloads without querying live telemetry during later verification.
- Artifact registration already persists task lineage and Batch 4 ACL scope.
- Model-selected tools are registry-only and intersect principal permission with global ActionGuard policy. No plant connector or control method exists.

## Batch 5 architecture plan

1. Add generic typed Plant/Area/Unit/Asset Passport, asset-reference, link, inspection, telemetry, trend, finding, recommendation, maintenance-history, and maintenance-draft models.
2. Add relational SQLite records plus `AssetRepository`, exact-alias `AssetResolver`, and resource-scope authorization. No graph database is introduced.
3. Extend the existing APEL source of truth with stable passports, aliases, evidence links, maintenance history, and deterministic scenario-based telemetry.
4. Implement a read-only `TelemetryProvider` abstraction and APEL simulator. The interface will contain latest/history reads only—no write, command, setpoint, start/stop, alarm, PLC, DCS, or SCADA methods.
5. Add configurable quality/freshness evaluation, reuse `UnitService`, and implement deterministic trend/threshold arithmetic without LLM calculations or predictive-maintenance claims.
6. Assemble bounded authorized `AssetContext`, extend `ContextCompiler`, and add asset-linked preference to hybrid retrieval only after Batch 4 ACL exclusion.
7. Add optional asset-aware Pump Workcell input while retaining uploaded-only behavior, structured discrepancy reporting, audit/SSE lineage, and exact telemetry snapshots in Evidence Capsules.
8. Add a local CMMS connector for governed draft actions only, secured APIs/read tools, focused deterministic tests, then the minimal asset frontend.

## Safety boundary

Batch 5 is read-only with respect to plant data. The local CMMS simulator may persist a draft record, but it cannot control equipment. APEL telemetry is synthetic, SQLite is not a historian, and no OPC-UA network service or new runtime dependency is required.
