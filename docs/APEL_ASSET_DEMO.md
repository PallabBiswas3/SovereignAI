# APEL asset-intelligence demo

APEL is a completely fictional organization. Its plant telemetry, inspection evidence, maintenance history, identities, and approvals are synthetic.

## Prepare

```powershell
$env:SOVEREIGN_DEMO_ORG_ENABLED = "true"
$env:SOVEREIGN_AUTH_MODE = "local"
.\.venv\Scripts\python.exe scripts\seed_apel_demo.py
```

Sign in as the Maintenance Engineer with `arjun.rao@apel.local` / `ApelDemo!2026`. Open `/assets` and select Pump-102. The page must display `SIMULATED PLANT DATA` and `READ ONLY`.

## Pump-102 scenario

Pump-102 is Cooling Water Pump B in Plant A / Utilities / Cooling Water. It links a passport, aliases, datasheet, current and superseded SOP revisions, inspection, sensor evidence, six-month vibration history, five maintenance events, current telemetry, findings, recommendation, local maintenance draft, approval, and Evidence Capsule lineage.

Controlled scenarios:

- `NORMAL`: four GOOD measurements within the synthetic normal state.
- `PUMP_102_DEGRADING`: vibration rises deterministically from 4.1 to 8.2 mm/s RMS over six periods; temperature is 86 degC, pressure 4.4 bar, and speed 1475 rpm as original values.
- `PUMP_102_STALE_DATA`: the latest values are eight hours old and must be labelled STALE rather than presented as current.
- `PUMP_102_BAD_QUALITY`: the vibration reading is BAD and carries a typed warning.

Suggested demonstrations:

1. `What is the current condition of Pump-102?`
2. `How has Pump-102 vibration changed over the last six months?`
3. Attach the inspection report and run `Assess Pump-102 using all available evidence` with the Pump Inspection Workcell.
4. Create a local maintenance draft citing the supported vibration claim; approve it as the separate Plant Manager account.
5. Create and verify the Evidence Capsule. Tampering with `asset/asset_context_snapshot.json` must invalidate it.
6. Switch the telemetry query parameter to `PUMP_102_STALE_DATA` or `PUMP_102_BAD_QUALITY` to demonstrate warnings.
7. Attempt telemetry access as a principal without `telemetry.read` or Plant-A scope; the response is denied without values or tag leakage.

The inspection Workcell still supports its original uploaded-report/SOP-only mode when no authorized asset is resolved.

