# Time-Series Diagnostic Agent

Adaptive, evidence-backed industrial time-series diagnostics.

Project priorities, retained benchmark baselines, scientific rules, and the
improvement log are maintained in
[`docs/PROJECT_DIRECTION.md`](docs/PROJECT_DIRECTION.md). Update that document
instead of creating additional planning or status notes.

## Integration API

Install the optional API dependencies and expose the existing strict diagnostic tool contract:

```powershell
pip install -e ".[api]"
uvicorn tsdiag.api:app --host 127.0.0.1 --port 8200
```

The service provides `GET /health` and `POST /v1/diagnose`. Authorization remains the responsibility of the hosting platform; the API validates the envelope and delegates directly to `diagnose_tool`.

## Stable pipeline API

Pipeline version `1.1.0` exposes one entry point for all six domains and always returns a validated `DiagnosticResult`:

```python
from tsdiag import diagnose

result = diagnose(
    "process",
    signal_matrix=current,
    normal_reference=healthy_reference,
    channel_names=["pressure", "flow", "level"],
    sampling_rate_hz=1.0,
)
print(result.decision, result.detection, result.localization)
```

| Domain | Executable baseline | Optional capability |
| --- | --- | --- |
| `bearing` | spectral-kurtosis/envelope physics plus Paderborn CNN–FFT feature fusion | Lenze current/encoder validation and persisted PyTorch models |
| `process` | hybrid DPCA/CVA monitoring, CVA-SVM/FDA diagnosis, contribution, onset, Granger and root-cause ranking | topology and fault catalog |
| `wind_scada` | regime-aware normal behavior residuals, CUSUM corroboration, persistence and physics checks | `train_matrix`, `prediction_matrix`, channel names and optional timestamps |
| `battery` | cell-to-pack voltage/temperature deviation, localization and risk | trained prognostic model |
| `turbofan` | decomposed sensor screening, regime handling, health index, RUL, uncertainty and explanation workflow | trained RUL model |
| `transformer` | wavelet denoising, correlation weighting and multisensor anomaly evidence; AD-TFM-AT is the default classifier for compatible 100x6 SGAH electrical waveforms when its versioned checkpoint is available | deterministic electrical rules and explicit model overrides |

The pipeline returns `abstain` with a reason when metadata is missing, a registered step fails, or evidence cannot support the requested label. Every executed step records its status and duration. Results support JSON-safe `to_dict()` and `to_json()` serialization, while `get_input_schema(domain)` describes public inputs. See [`examples/run_full_pipeline.py`](examples/run_full_pipeline.py) for normal and fault scenarios across all six domains.

For Wind SCADA, `train_matrix` and `prediction_matrix` are the canonical input
names. The older `normal_reference` and `signal_matrix` names remain temporary
compatibility aliases and execute through the same plugin workflow.

> **Deprecated v0 API:** `IndustrialDiagnosticOrchestrator` and the classes in
> `tsdiag.agents` remain available for reproducibility, but new integrations
> should use `diagnose()` and `DiagnosticResult`. They will not be removed before
> a major-version release.

### External tool boundary

Local LLM or service hosts can call the same pipeline through a strict JSON-safe
adapter. The host remains responsible for authentication and authorization.

```python
from tsdiag.integrations import DIAGNOSE_TOOL_MANIFEST, diagnose_tool

response = diagnose_tool({
    "domain": "turbofan",
    "task": "remaining_useful_life",
    "inputs": {
        "signal_matrix": sensor_rows,
        "channel_names": sensor_names,
        "cycle_index": cycles,
    },
    "policy_ref": "turbofan-policy-v2",
})
```

`scripts/run_diagnostic_tool.py` exposes the adapter over stdin/file JSON without
adding a network server or weakening the request contract.

### Cross-domain provenance report

Combine serialized `DiagnosticResult` files into versioned JSON and Markdown:

```bash
python scripts/generate_cross_domain_report.py result1.json result2.json --output-dir outputs/cross_domain
```

The report records pipeline, workflow, policy and model versions together with
dataset/protocol identifiers, input hashes, artifact checksums and Git SHAs.

### CARE v6 on Kaggle

Build the code-only Kaggle bundle locally:

```bash
python scripts/build_kaggle_care_bundle.py
```

Upload `dist/tsdiag-care-kaggle.zip` and the complete official
`CARE_To_Compare.zip` as separate private Kaggle Datasets. The bundle includes
`kaggle/care_benchmark.ipynb`, an auto-discovering runner, optional checksum
validation, a three-event smoke mode, and the full canonical benchmark mode.
The runner accepts both the original ZIP and the directory tree produced when
Kaggle automatically expands uploaded archives.

The benchmark fits the normal-behavior model on each event's declared healthy
training partition and scores its prediction partition. It is not supervised
deep-model training, and event labels do not enter the diagnostic request.

The validated 95-event canonical run completed with zero failures and produced
recall `0.9111`, precision `0.4659`, F1 `0.6165`, abstention rate `0.0316`, and
normal-event false-alarm rate `0.9400`. This is the current reproducibility
baseline, not an operational performance claim: false-alarm control remains the
principal Wind research problem, and CARE must not be tuned retrospectively.

Wind threshold calibration now has a healthy-only target-event-FAR workflow.
It uses a contiguous tail holdout from each development event, searches the
residual/CUSUM/event-burden configuration, reports a bootstrap interval, and
requires a checksummed split manifest that explicitly declares an event-,
asset-, or temporal-leakage boundary. Asset mode fails when asset identity is
missing or shared; temporal mode additionally proves development intervals
precede evaluation intervals on shared assets. No calibrated configuration is
shipped yet: the existing CARE run used all 95 events, so an independent
healthy development pool is required before the configuration can be frozen.
Run `python scripts/calibrate_wind_scada.py --help` once that pool is available.

New Wind reports call the historical `event_end - first_detection` quantity
`time_to_event_end`. Genuine signed `early_warning_lead_time` is reported
separately against event onset (positive before onset, negative after onset).
The frozen CARE JSON retains its historical field names and is not rewritten.

This project is designed as a **specialist-agent platform**, not a single bearing classifier. The long-term goal is to combine deterministic signal analysis, statistical monitoring, causal/root-cause reasoning, learned models, multimodal evidence, physics/manual verification, prognostics, and later sovereign industrial orchestration.

## Current specialist agents

### 1. SignalProcessingAgent
Generic first-pass signal diagnostics:
- RMS / peak / crest factor
- Pearson kurtosis
- Welch PSD
- STFT-based temporal spectral variability
- generic anomaly/impulsiveness score

### 2. BearingDiagnosticAgent
Rotating-bearing diagnostics using:
- band-pass resonance isolation
- Hilbert envelope
- envelope spectrum
- BPFO / BPFI / BSF / FTF harmonic matching
- diagnosis confidence and explicit evidence

### 3. StatisticalMonitoringAgent
Reference-based process monitoring:
- PCA
- reconstruction/SPE monitoring
- Hotelling-style T2 monitoring
- Local Outlier Factor
- 99% reference limits

### 4. ProbabilisticDiagnosticAgent
Uncertainty-aware reference comparison:
- multivariate Gaussian baseline
- Mahalanobis-distance scoring
- reference-derived anomaly threshold

### 5. CausalRootCauseAgent
For multivariate industrial processes:
- pairwise Granger-causality screening
- directed dependency graph
- candidate upstream/root-cause channel ranking

> Granger relationships are treated as predictive evidence, not proof of physical causality.

### 6. TransferRobustnessAgent
Checks whether deployment data differs strongly from the source/training domain:
- normalized mean shift
- scale shift
- operating-domain shift score

This is intended to flag when domain adaptation or recalibration may be needed.

### 7. LearnedModelAgent
Adapter for trained models supplied by the application:
- CNN
- LSTM / GRU / BiLSTM
- Transformer
- autoencoder
- ensemble classifier
- custom anomaly/fault model

Heavy learned models are **not falsely bundled as pretrained models**. A trained predictor must be registered explicitly.

### 8. MultimodalFusionAgent
Fuses evidence from combinations of:
- raw/time-series analysis
- learned models
- text/manual/KG evidence
- images/time-frequency artifacts
- external diagnostic tools

### 9. EvidenceVerificationAgent
Verification layer for candidate diagnostic claims:
- physics checks
- domain constraints
- maintenance/manual evidence
- custom verifier hooks
- SUPPORTED / CONTRADICTED / INSUFFICIENT-style aggregation

### 10. PrognosticsAgent
Initial prognostics baseline:
- health-index trend analysis
- threshold-crossing estimate
- simple RUL estimate

This is intentionally a baseline; validated industrial RUL models should replace it for real maintenance use.

## Orchestration

The deprecated `IndustrialDiagnosticOrchestrator` routes a record to the specialist agents that are applicable to the available data/context. It is retained as the transparent v0 comparison baseline.

```text
SignalRecord
   |
   +--> SignalProcessingAgent
   |
   +--> BearingDiagnosticAgent            if BPFO/BPFI/BSF/FTF available
   |
   +--> StatisticalMonitoringAgent        if normal-reference data available
   +--> ProbabilisticDiagnosticAgent
   +--> TransferRobustnessAgent
   |
   +--> CausalRootCauseAgent              if multichannel data available
   |
   +--> LearnedModelAgent                 if a trained predictor is registered
   |
   +--> MultimodalFusionAgent             if external/modal evidence is supplied
   |
   +--> EvidenceVerificationAgent         if candidate claims/verifiers are supplied
   |
   +--> PrognosticsAgent                  if health-history data is supplied
   |
   v
DiagnosticReport
```

The current router is deliberately deterministic and inspectable. Later we can benchmark an LLM/learned/RL router against this baseline instead of assuming agentic routing is automatically better.

## Install

```bash
pip install -e ".[dev]"
pytest
```

## Example

```python
import numpy as np
from tsdiag import SignalRecord, IndustrialDiagnosticOrchestrator

fs = 12000.0
x = np.random.randn(24000)

record = SignalRecord(
    signal=x,
    fs=fs,
    fault_frequencies={
        "BPFO": 85.0,
        "BPFI": 120.0,
        "BSF": 50.0,
        "FTF": 11.0,
    },
)

report = IndustrialDiagnosticOrchestrator().run(record)
print(report.decision, report.label, report.confidence)
```

## Research-method coverage

The repository now has concrete homes/interfaces for the major method families we identified from the literature:

- signal processing
- statistical process monitoring
- probabilistic diagnosis
- deep/learned time-series models
- graph/causal root-cause analysis
- transfer/domain-shift robustness
- multimodal diagnosis
- agentic/tool-based diagnosis
- evidence and physics verification
- prognostics / predictive maintenance

Not every research method is implemented as a full production model yet. For example, domain-specific CNN/Transformer models, digital twins, full Bayesian networks, advanced wavelet/framelet models, and validated RUL networks require separate datasets/model artifacts and will be added as specialized implementations rather than placeholder claims.

## Benchmark-improvement phase

The six deterministic domain paths are complete at pipeline version `1.1.0`.
The immediate domain-research priorities are leakage-safe Wind calibration
and Paderborn/Lenze bearing validation. In parallel, the integration track is
building one grounded Turbofan/Process user story. See the [project direction and improvement
log](docs/PROJECT_DIRECTION.md) for the maintained sequence and scientific
guardrails.

The frozen 30-epoch SGAH AD-TFM-AT run achieved 0.9569 multiclass accuracy,
0.7971 balanced accuracy and 0.7847 macro-F1. Main-transformer-fault detection
was 64/64 with no false positives, while all 14 inter-phase short-circuit
events were misclassified. The public Transformer path automatically selects
the versioned AD-TFM-AT checkpoint only for the compatible `Ua, Ub, Uc, Ia,
Ib, Ic` 100-sample input contract and records its version and checksum.
