# SovereignAI 2.0 Batch 5 — Asset intelligence and safe plant integration

## Outcome

Batch 5 is implemented and verified. SovereignAI now supports asset-aware industrial workflows using structured Asset Passports, authorized simulated plant telemetry, historical condition data, deterministic trend and rule analysis, explicit source discrepancies, and human-governed local maintenance drafts.

The implementation extends the existing Batch 1–4 architecture. It does not add a cloud runtime, larger model, GraphRAG, production historian, live plant connector, real CMMS, or autonomous control.

## Delivered architecture

- Generic Plant, Area, Unit, AssetPassport, alias, evidence-link, inspection, telemetry, trend, finding, recommendation, maintenance-history, and maintenance-draft models.
- SQLite records and an authorization-aware `AssetRepository`.
- Exact-only `AssetResolver`, including not-found, ambiguity, and access-denied states.
- Deterministic enrichment of the existing 20-asset/55-file APEL seed; identifiers remain stable across runs.
- Read-only `TelemetryProvider`, APEL scenario provider, quality warnings, configurable freshness, timezone preservation, UnitService normalization, bounded history, and deterministic `TrendAnalyzer`.
- Evidence-backed condition comparison and an explicit `MEASUREMENT_SOURCE_CONFLICT` between the 7.4 mm/s inspection and 8.2 mm/s later telemetry snapshot.
- Bounded `AssetContextService`, structured ContextCompiler input, and post-ACL asset-aware hybrid retrieval.
- Optional asset context for the Pump Inspection Workcell while uploaded-only behavior remains supported.
- Local draft-only `CMMSConnector`, supporting-claim requirement, hashed approval, separate approver, and no plant execution.
- Asset APIs, read-only agent tools, audit/SSE events, task state lineage, and exact operational snapshots in Evidence Capsules.
- `/assets` frontend with passport, quality/freshness, trend, timeline, maintenance history, and simulated/read-only labels.

The deterministic telemetry seed contains 20 measurement rows across three assets. Pump-102 has four metrics (`vibration`, `bearing_temperature`, `discharge_pressure`, and `speed`) and four controlled scenarios; Pump-101 and Compressor-201 provide comparison vibration readings.

## Files created

- `backend/app/assets/`: domain models, repository, resolver, telemetry providers, trend and condition engines, context service, and local CMMS abstraction.
- `backend/app/api/assets.py`: authenticated asset, telemetry, history, timeline, context, and draft endpoints.
- `frontend/app/assets/page.tsx`: asset-focused frontend.
- `tests/test_phase35_asset_intelligence.py`: focused functional and security coverage.
- `docs/SOVEREIGNAI_2_BATCH5_BASELINE.md`, this report, `docs/ASSET_INTELLIGENCE.md`, `docs/PLANT_DATA_CONNECTORS.md`, and `docs/APEL_ASSET_DEMO.md`.

## Files modified

- Core/data: `backend/app/core/config.py`, `backend/app/core/database.py`, `.env.example`.
- Identity/governance: `backend/app/identity/models.py`, `authorization.py`, `backend/app/api/governance.py`, `config/access.yaml`, and `config/tools.yaml`.
- APEL: `backend/app/demo/apel.py` and `demo/apel/assets.yaml`.
- Evidence/retrieval: `backend/app/evidence/context.py`, `backend/app/rag/ingestion.py`, `retrieval.py`, and `hybrid.py`.
- Tasks/Workcells/tools/capsules: `backend/app/agent/state.py`, `backend/app/api/tasks.py`, `backend/app/workflows/inspection.py`, `backend/app/tools/application_tools.py`, `registry.py`, `backend/app/capsules/builder.py`, and `backend/app/main.py`.
- Frontend/docs/tests: `frontend/app/page.tsx`, `frontend/app/globals.css`, `README.md`, and `tests/test_phase28_capsules.py`.

## Verification

Baseline before Batch 5: 103 passed, 1 skipped; Python compilation, frontend typecheck, and production build passed.

Final result:

- Backend: 115 tests collected; 114 passed and the existing Windows symlink-capability test skipped.
- Python: `python -m compileall -q backend/app` passed.
- Frontend: `npm run typecheck` passed.
- Production build: `npm run build` passed, including `/assets`.
- Focused tests cover asset models/hierarchy/aliases/ACL/links; latest/history/units/timestamps/quality/freshness; deterministic trends; conflicts; bounded context; asset-aware retrieval; tool security; maintenance draft approval; API security; and Capsule telemetry tampering.
- Existing Batch 4 identity, retrieval ACL, approval, artifact, task/SSE, and capsule security tests remain green.

Two non-failing environment warnings remain: the existing Starlette TestClient/httpx deprecation and installed Torch/NumPy ABI warning.

## Evidence Capsule behavior

Capsules store the exact passport/context, telemetry measurement IDs and values, timestamps, quality/freshness, provider, trends, maintenance references, and draft used by the completed task. Verification hashes those stored files. It does not query fresh telemetry or assert that current operational values remain unchanged.

## Exact limitations

- APEL telemetry is synthetic/simulated; no live industrial plant is connected.
- Connector layers expose no PLC/DCS/SCADA control, tag write, setpoint, start/stop, alarm acknowledgement, or interlock operation.
- Trend analysis is deterministic condition analysis, not predictive maintenance, failure forecasting, or remaining-useful-life estimation.
- SQLite is not a production historian.
- Asset relationships are relational local records, not GraphRAG.
- CMMS is a local draft-action simulator; no SAP/Maximo transaction is sent.
- Operational safety certification has not been performed.
- Evidence-backed recommendations require qualified engineering review.
- Capsule integrity proves stored-byte integrity, not engineering correctness.
- LLM prose remains nondeterministic and local model availability remains a runtime dependency.

## Recommendation for Batch 6

Do not start Batch 6 as part of this change. The next design phase should first define production-readiness requirements: historian volume/retention, site network and identity zones, connector allowlists, operational safety review, observability/SLOs, and formal evaluation criteria. Only after those are agreed should the project consider production time-series storage or real read-only connectors; plant writes should remain a separately governed and safety-certified program.
