# Asset intelligence architecture

Batch 5 adds a generic, local asset domain under `backend/app/assets`. It does not turn the language model into a source of plant truth.

## Domain and hierarchy

The hierarchy is `Organization -> Plant -> Area -> Unit -> AssetPassport`. A passport records the canonical asset ID and name, type, hierarchy IDs, workspace/department/classification ACL, criticality, lifecycle status, manufacturer/model, commissioning time, structured design parameters, aliases, and tags. The APEL hierarchy uses Plant A, Utilities/Process Unit 1/Tank Farm, six units, and 20 stable assets.

`AssetResolver` performs exact normalized canonical/alias matching only. It does not fuzzy-match industrial tags. Results are `RESOLVED`, `ASSET_NOT_FOUND`, `AMBIGUOUS_ASSET`, or `ASSET_ACCESS_DENIED`. The deliberately shared `P102` alias demonstrates safe ambiguity; `P-102` resolves to `Pump-102`.

`AssetRepository` is the SQLite persistence boundary for passports, aliases, measurements, inspections, maintenance records, drafts, and evidence links. Every read receives the authenticated `Principal`; resource authorization is checked before asset or telemetry content is returned.

## Evidence relationships

Relational `AssetEvidenceLink` rows preserve `HAS_DOCUMENT`, `HAS_INSPECTION`, `HAS_MEASUREMENT`, `HAS_FINDING`, `SUPPORTED_BY`, `EVALUATED_USING`, and `DERIVED_FROM` relationships. APEL assigns explicit deterministic metadata at generation/ingestion time. It does not use LLM entity extraction or GraphRAG.

## Operational evidence

`OperationalMeasurement` preserves the measurement ID, asset and metric, normalized and original values/units, timezone-aware timestamp, quality, provider, source tag, age, freshness, and typed warnings. `UnitService` remains the single Pint-backed conversion engine.

`TrendAnalyzer` is pure deterministic arithmetic over usable stored measurements. It calculates latest, mean, min/max, change, percentage change, least-squares slope per day, rolling mean, threshold crossings, time above threshold, abnormal count, and increasing/decreasing/flat direction. It is condition analysis—not predictive maintenance, remaining useful life, or failure forecasting.

`ConditionAssessmentEngine` consumes an explicit measurement and evidence-backed `Rule`. Pump-102's threshold is sourced from SOP-MNT-017 Rev 4 and retained with source document, revision, and section. When inspection and telemetry values differ materially, `MEASUREMENT_SOURCE_CONFLICT` preserves both values, timestamps, qualities, and sources instead of silently overwriting either.

## Context and retrieval

`AssetContextService` assembles a bounded authorized passport, at most 12 latest measurements, eight trends, five inspections, 20 document references, ten maintenance records/drafts, applicable rules, calculations, findings, recommendations, conflicts, and warnings. `ContextCompiler` accepts this structured context and still enforces the existing input-token budget.

Hybrid retrieval remains dense + BM25 + RRF + optional offline reranking. Document ACL filtering occurs inside both retrievers before fusion. A resolved `asset_id` then adds a deterministic preference for already-authorized chunks explicitly linked to that asset. Asset affinity is never an ACL bypass.

## Interfaces

Read APIs:

- `GET /api/assets`
- `GET /api/assets/{asset_id}`
- `GET /api/assets/{asset_id}/measurements/latest`
- `GET /api/assets/{asset_id}/measurements/history`
- `GET /api/assets/{asset_id}/inspections`
- `GET /api/assets/{asset_id}/documents`
- `GET /api/assets/{asset_id}/maintenance`
- `GET /api/assets/{asset_id}/context`
- `GET /api/assets/{asset_id}/timeline`

The only mutation is `POST /api/assets/{asset_id}/maintenance/drafts`, which stores a local draft and a hashed human-approval request. It does not send a work order or plant command.

Agent-facing tools are bounded reads: `get_asset_telemetry` and `get_asset_history`. No tag-write, setpoint, start/stop, PLC, DCS, or SCADA tool is registered.

## Measured development performance

On the development Windows machine with an in-memory SQLite APEL seed, 20 warm iterations averaged: asset resolution 1.534 ms, latest telemetry 1.708 ms, history 2.207 ms, trend calculation 1.904 ms, and full asset-context assembly 403.285 ms. These are local observations, not production capacity guarantees; no generative inference was used.

