# APEL synthetic demonstration organization

Apex Petrochemical & Energy Ltd. (APEL) is completely fictional. It exists only to demonstrate a coherent organization-wide SovereignAI deployment without proprietary documents or real-person data.

## Plant A

Plant A contains Utilities, Process Unit 1, and Tank Farm. The canonical asset registry in `demo/apel/assets.yaml` defines 20 assets, including Pump-101 through Pump-104, Compressor-201/202, Motor-201/202, HeatExchanger-301/302, Tank-401/402, Valve-XV-101/102, Fan-501, Boiler-601, Filter-701, PSV-801, Analyzer-901, and Transformer-1001.

Pump-102 is the deepest vertical slice:

- Utilities cooling-water service;
- normal discharge pressure 4.8–5.5 bar;
- maximum normal vibration 6.0 mm/s RMS;
- controlled shutdown threshold 9.0 mm/s RMS;
- latest inspection 5.1 bar and 7.4 mm/s RMS;
- bearing replacement on 2026-08-18;
- current SOP-MNT-017 Rev 4 and an explicitly superseded Rev 3.

The deterministic generator creates 55 files spanning engineering datasheets, maintenance histories, SOPs, sensor CSVs, inspection/incident/permit records, vendor proposals, shift logs, quality evidence, management inputs, and one restricted Finance canary. Difficult fixtures cover revision conflict, missing unit, incorrect unit, low-confidence OCR, embedded prompt injection, stale readings, unsupported vendor claims, missing evidence, and restricted evidence.

## Departments and access

APEL includes Operations, Maintenance, Engineering, HSE, Quality, Procurement, Management, Finance, and IT. The persisted matrix is in `demo/apel/access_matrix.yaml`.

- Maintenance Engineer: Maintenance plus shared Engineering/Operations; Pump Inspection Workcell; no Finance.
- HSE Officer: HSE and relevant Operations evidence; restricted incident/permit scope; no Finance.
- Procurement Engineer: Procurement and shared Engineering requirements; no Finance.
- Plant Manager: broad non-Finance Plant A operational view plus business approval.
- Auditor: audit and authorized capsule verification; no tools or approval.
- Administrator: platform administration; no automatic business approval.

## Demo accounts

All accounts use the development-only password `ApelDemo!2026`.

| Account | Role context |
|---|---|
| `arjun.rao@apel.local` | Maintenance Engineer |
| `meera.sen@apel.local` | HSE Officer |
| `vikram.shah@apel.local` | Procurement Engineer |
| `ananya.iyer@apel.local` | Plant Manager + Approver |
| `auditor@apel.local` | Auditor |
| `admin@apel.local` | Administrator |
| `operations@apel.local` | Operations Engineer |

These credentials are isolated in `demo/apel/users.yaml` and are never silently seeded.

## Seed and reset

From the repository root in PowerShell:

```powershell
$env:SOVEREIGN_DEMO_ORG_ENABLED = "true"
.\.venv\Scripts\python.exe scripts\seed_apel_demo.py
```

To remove only APEL-scoped database rows and `demo/apel/generated`:

```powershell
$env:SOVEREIGN_DEMO_ORG_ENABLED = "true"
.\.venv\Scripts\python.exe scripts\reset_apel_demo.py
```

The reset validates the exact generated-directory shape before recursive removal and never deletes unrelated organizations. Rerunning the seed regenerates byte-identical corpus content; password salts and session tokens intentionally remain random.

For the authenticated demo, also set:

```powershell
$env:SOVEREIGN_AUTH_MODE = "local"
```

Then start the backend from the repository root so configuration paths resolve consistently.
