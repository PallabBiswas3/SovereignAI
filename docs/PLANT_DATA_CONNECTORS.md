# Plant data connectors and safety boundary

## Telemetry

`TelemetryProvider` is intentionally read-only and exposes only `get_latest` and `get_history`. Batch 5 implements `APELSimulatorTelemetryProvider`. `OPCUAConnector` is an unconfigured future read-only contract; no OPC-UA package, listener, or network dependency was introduced.

Each read is authorized against the asset scope before records are queried. The API never accepts a client-supplied role. Unauthorized responses contain a typed error and do not expose values or source-tag names.

Quality is explicit: `GOOD`, `UNCERTAIN`, `BAD`, or `UNKNOWN`. BAD and UNKNOWN values carry `BAD_TELEMETRY_QUALITY` or `UNKNOWN_TELEMETRY_QUALITY`; they are not silently treated as reliable facts. The default freshness policy is five minutes fresh and 24 hours before expiration, with a replaceable metric-override map. Measurements are labelled `FRESH`, `STALE`, `EXPIRED`, or `UNKNOWN` and retain age and timestamp.

## Maintenance / CMMS

`CMMSConnector` supports reading asset history, reading a local draft, and creating a local work-order draft. `APELLocalCMMSConnector` is the only Batch 5 implementation. A draft must cite at least one supporting claim, stays `DRAFT`, and creates an exact SHA-256-bound approval request.

A different authorized approver may accept or reject the draft under Batch 4 separation-of-duties rules. Approval changes only the local record to `APPROVED` or `REJECTED`; the response explicitly reports `plant_action_executed: false`. There is no SAP PM, Maximo, or external CMMS connection.

## Prohibited operations

No provider, API, or registered tool exposes `write_tag`, `set_value`, `execute_command`, `start_asset`, `stop_asset`, PLC writes, DCS writes, SCADA writes, setpoint changes, alarm acknowledgement, or interlock bypass. Human approval cannot create a missing plant-control capability.

## Production replacement seams

A future deployment could implement a read-only historian/OPC-UA provider and a governed enterprise CMMS adapter behind the same interfaces. That work requires site-specific network zones, credentials, tag allowlists, safety review, audit retention, time-series storage, availability engineering, and operational certification. SQLite is suitable for this deterministic demo, not a production historian.

