# Time-Series Diagnostic Agent

Adaptive, evidence-backed industrial time-series diagnostics.

Project priorities, retained benchmark baselines, scientific rules, and the
improvement log are maintained in
[`docs/PROJECT_DIRECTION.md`](docs/PROJECT_DIRECTION.md). Update that document
instead of creating additional planning or status notes.

## Latest diagnostic changes - 3 November 2026

Source baseline: `development`, commit `d157ed6`. The additions below are
available as Python research utilities; they do not automatically run inside
`diagnose()` or `POST /v1/diagnose`. Package `0.4.0`, pipeline `1.1.0`, existing
domain models and public input contract `industrial-diagnostic-v1` are retained.
No new Process detector was introduced in this batch. Model promotion, real-data
evaluation and production acceptance are separate from these source additions.

| Module under `src/tsdiag/` | Main inputs | Main outputs |
| --- | --- | --- |
| [`uncertainty/conformal.py`](src/tsdiag/uncertainty/conformal.py) | Calibration probabilities `(N, K)` and integer class-column labels, or aligned targets/predictions and optional positive scales | Boolean prediction sets `(N, K)`, selective decisions, or lower/upper interval arrays |
| [`uncertainty/selective.py`](src/tsdiag/uncertainty/selective.py) | Logits/probabilities and labels; confidence plus correctness; answer/abstain masks | Temperature, ECE, risk/coverage arrays, AURC, accuracy-at-coverage and selective summaries |
| [`calibration/event_conformal.py`](src/tsdiag/calibration/event_conformal.py) | Named healthy events containing train/prediction residual matrices, a declared split manifest and optional validation IDs | Frozen event threshold/configuration, calibration/validation FAR, intervals, exclusions and per-event decisions |
| [`calibration/split_guard.py`](src/tsdiag/calibration/split_guard.py) | Event IDs, optional asset groups and temporal intervals | Compatibility adapter to the existing manifest and group-preserving development splits |
| [`challengers/transformer_sequence.py`](src/tsdiag/challengers/transformer_sequence.py) | Six-channel phase waveforms `(N, T, 6)` and training labels, optionally independent model probabilities | Named sequence/harmonic features, class probabilities/predictions and per-class recall intervals |
| [`challengers/turbofan_rul.py`](src/tsdiag/challengers/turbofan_rul.py) | Operating settings/sensors, engine/cycle identities, windows and exact RUL targets | Normalized sensors, causal windows, ensemble mean/spread, conformal intervals and evaluation metrics |
| [`challengers/battery_prognostics.py`](src/tsdiag/challengers/battery_prognostics.py) | Identified capacity histories and EOL threshold; optionally discharge time/voltage/current | Weighted RUL samples/quantiles, per-cell evaluation and censoring reports, discharge features |
| [`challengers/bearing_cyclostationary.py`](src/tsdiag/challengers/bearing_cyclostationary.py) | Vibration, sampling rate, fault frequencies and optional band/tachometer trace | Modulation/order spectra, fault-frequency features and a healthy-score threshold |

### What the uncertainty results mean

`ConformalClassifier.decide()` returns `label`, `abstain`, `set_size` and `sets`.
It answers only for singleton sets; both empty and multiple-class sets abstain.
Its labels refer to probability-column indices. With a learned classifier whose
`classes_` are not `0..K-1`, map calibration labels into those columns and map
answered indices back to `classes_`. A class with insufficient calibration data
gets an infinite class-conditional threshold and remains included conservatively.

`SelectiveDecision.coverage` is the fraction answered; `set_coverage()` is the
fraction whose true class is inside the set. They measure different properties.
Report answer coverage together with selective accuracy/risk. Marginal set
coverage does not promise accuracy conditional on answering. Temperature fitting
requires its own tuning data, separate from final conformal calibration.

The following standalone example demonstrates result shapes and abstention;
the repeated synthetic probabilities are not independent validation evidence:

```python
import numpy as np
from tsdiag.uncertainty import ConformalClassifier, selective_summary

calibration_probabilities = np.tile([0.4, 0.6], (20, 1))
calibration_labels = np.zeros(20, dtype=int)
calibrator = ConformalClassifier(alpha=0.1, method="lac").fit(
    calibration_probabilities, calibration_labels,
)
decision = calibrator.decide([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
print(decision.label.tolist())    # [0, -1, 1]; -1 means abstain
print(decision.set_size.tolist()) # [1, 2, 1]
print(selective_summary(decision.label, [0, 1, 1], decision.abstain))
```

For regression, `SplitConformalRegressor.fit(y, y_pred, scale=...)` calibrates
absolute residuals; `predict_interval()` returns `(lo, hi)`. CQR accepts the
model's lower/upper quantile predictions at fitting and prediction. Insufficient
calibration can produce unbounded intervals; callers must hold or explicitly
represent that state, not silently publish a finite interval. `clip_min=0`
requires a known nonnegative target. General utilities validate arrays but cannot
infer which physical assets produced them: callers own train/tune/calibration/
test identity separation. The event and RUL APIs add their explicit ID checks.

### Wind calibration lifecycle and frozen artifacts

1. Fit each event's normal-behavior model using healthy training observations
   only, then supply its train and prediction residuals as `EventResiduals`.
   The calibrator does not train that upstream model or certify healthy labels.
2. Fix an `EventScorer` or call `select_statistic()` on separate healthy/anomalous
   development tuning events. Its manifest must declare exactly their union.
   The returned scorer carries the selection manifest into final fitting.
3. Fit `EventFARCalibrator` on a fresh healthy pool exactly matching the final
   manifest's development IDs. Optional `validation_ids` stay out of threshold
   fitting and must obey the declared unit/group/temporal boundary. With 10%
   target FAR, at least nine scorable fitting events are required for a finite
   threshold; this minimum is not a recommended validation sample size.
4. Freeze `EventAlarmCalibration.to_json()` and its `config_sha256()`. The artifact
   includes scorer/window/fusion/drift, missing-data policy, target/threshold,
   calibration and validation IDs/counts/exclusions, observed FAR, optional Wilson
   and bootstrap intervals, manifest boundary/checksum, selection checksum and
   code version. `n_boot=0` omits the bootstrap interval.
5. Reload using `EventAlarmCalibration.from_json()` and
   `EventFARCalibrator.from_calibration()`. `detect()` returns an `EventDecision`
   with `alarm`, `status`, `score`, `threshold` and optional `first_alarm_index`.
   `alarm=None` is an unscorable abstention. Alarm requires `score > threshold`.

The checksum identifies artifact contents; it is not a signature or proof of data
authenticity. The manifest prevents declared identity/time overlap, not hidden
physical dependence. FAR control assumes fixed scoring and exchangeable scorable
healthy events; comparable prediction-window lengths matter for maximum-based
statistics. Validation intervals assume independent events. Missing rows cannot
complete persistence windows, and exclusions must be reported with FAR.

`early_warning_lead_time(first_alarm, onset)` is signed: positive before onset,
negative after it. `time_to_event_end(first_alarm, event_end)` is a separate
quantity. These helpers return sample-index differences, not seconds. No Wind
artifact has been accepted on independent CARE data, and this calibrator has not
been wired into `WindScadaPlugin`.

### Verification and remaining integration work

Recorded verification for this source batch: **227 passed**, combining the new
uncertainty/calibration/challenger tests with selected existing battery, bearing
and public-pipeline regression checks. This includes a small optional CNN CPU
forward/backward test; it is not GPU training validation. GP kernel-bound
convergence and existing statistical-test warnings remain visible. The separate
32-test application benchmark/timing check is not part of the diagnostic total.

Run the six tracked test files from this project directory with the application
virtual environment, or an environment installed with this project's dependencies:

```powershell
$env:PYTHONPATH = "src"
..\.venv\Scripts\python.exe -m pytest -q tests
```

The recorded 227-case run additionally used these local-only regression modules,
which are not available in a fresh GitHub clone:

```powershell
..\.venv\Scripts\python.exe -m pytest -o "addopts=" -q tests `
  ../local_testing/Time-Series-Diagnostic-Agent/tests/test_battery_incomplete_history.py `
  ../local_testing/Time-Series-Diagnostic-Agent/tests/test_battery_prognosis.py `
  ../local_testing/Time-Series-Diagnostic-Agent/tests/test_bearing_evidence_regressions.py `
  ../local_testing/Time-Series-Diagnostic-Agent/tests/test_bearing_resonance_selection.py `
  ../local_testing/Time-Series-Diagnostic-Agent/tests/test_public_pipeline.py
```

No new CARE/SGAH/C-MAPSS/NASA-battery/CWRU/Paderborn/Lenze evaluation was run.
Existing historical results remain in [project direction](docs/PROJECT_DIRECTION.md#retained-real-data-baselines).
Before runtime promotion, predeclare independent physical-unit splits, compare
against existing baselines, freeze models and calibration separately, and add an
explicit domain adapter with provenance, abstention and contract checks. The
public JSON/HTTP boundary continues to reject executable model objects and
user-supplied decision-threshold overrides. Model-based NumPy infinities/NaNs
must be converted into explicit unavailable/censored output by an adapter before
crossing that JSON boundary.

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

### Shared uncertainty utilities

`tsdiag.uncertainty` provides `ConformalClassifier` (LAC/APS, optional
class-conditional thresholds), singleton-or-abstain decisions,
`SplitConformalRegressor`, `ConformalizedQuantileRegressor`, and selective
metrics: risk/coverage, accuracy at coverage, AURC, ECE and temperature scaling.
These are NumPy utilities for explicitly supplied model predictions; the domain
plugins do not automatically apply them. Fix the model, temperature and scoring
rule before fitting conformal thresholds on independent calibration units.
Prediction-set coverage is marginal under exchangeability and does not imply
accuracy conditional on answering. Regression `clip_min=0` is appropriate only
when the target is known to be nonnegative.

Run the supplied and extended synthetic tests from this project directory:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

### Opt-in domain challengers

`tsdiag.challengers` contains the supplied transformer sequence classifier,
turbofan RUL ensemble, battery fade models, and bearing modulation/order features.
They are research APIs; importing them does not replace domain models or policies.
No new real-data improvement or trained challenger checkpoint is claimed.

`SequenceFeatureExtractor` expects finite waveforms ordered
`Ua, Ub, Uc, Ia, Ib, Ic`. Use verified `cycles_in_window` metadata where available;
otherwise the dominant voltage bin is estimated separately for each record.
Feature columns remain fixed, with observability flags for unavailable harmonics.
Non-integer cycle metadata is supported without rounding. Circular augmentation
requires verified whole-cycle windows. Verify channel grounding/zero-sequence
observability before interpreting an imbalance as a physical fault. V/I coupling
features remain dependent on sensor units and load.

`SequenceFeatureClassifier` trains only on supplied training records, optionally
oversamples minority classes, and supports `extra` probability columns when
augmentation is disabled. Stacking inputs for training must be out-of-fold or
come from an independently frozen base model. Fit conformal thresholds on
independent calibration records and align label indices with `classes_`.

`RULChallenger` supports HGB and an optional lazily imported PyTorch CNN (CPU by
default). Fit `RegimeNormalizer` on training engines only. Point-model targets
are capped, but interval calibration defaults to uncapped RUL. Engine identities
are required for calibration and interval evaluation; use one prefix per engine,
with no train/calibration/test overlap. Refit invalidates calibration.

```python
from tsdiag.challengers import RULChallenger, evaluate_rul

model = RULChallenger(backend="hgb", window=30).fit(
    training_windows, training_rul, training_engine_ids,
)
model.calibrate(calibration_windows, calibration_rul, units=calibration_engine_ids)
metrics = evaluate_rul(model, test_windows, uncapped_test_rul, units=test_engine_ids)
```

The caller supplies those disjoint datasets and predeclares truncation rules.
Use `target="capped"` explicitly for capped-target intervals; reports identify
the target and include both capped and uncapped interval coverage. Random
synthetic truncations do not establish exchangeability with official C-MAPSS
test prefixes. RMSE/MAE/NASA score use uncapped truth, and `n_engines` counts
independent test engines rather than repeated windows. These APIs require an
explicit adapter and real-data validation before use through `diagnose()`.

`CellData`, `FadeParticleFilter`, `GPFadeModel` and `leave_one_cell_out` provide
capacity-fade research tools. Histories must have strictly increasing cycles and
finite positive capacities; missing/latest-stale observations are not silently
filled. Population priors from `prior_from_cells()` retain training cell IDs;
pass `cell_id=` to filter fitting so overlap is rejected. Leave-one-cell-out
evaluation excludes the held cell from its prior, rejects duplicate cell IDs,
and keeps right-censored cells out of point-error metrics. Fraction-based cutoffs
are retrospective evaluation protocols, not prospective operating schedules.

Both battery models report **uncalibrated model-based uncertainty**. For the GP,
infinite samples/quantiles mean no crossing within `censoring_horizon_`, not a
known infinite lifetime. Supply that horizon to `survival_consistency()`;
leave-one-cell-out reports mark checks beyond it unscorable. Reports include
finite-point forecast coverage, unbounded-interval fraction and per-cell errors;
interval coverage excludes true lifetimes beyond the GP horizon and reports
the scorable/unscorable counts instead of treating its unknown tail as covered.
Repeated cutoffs are not independent cells. Incremental-capacity features require
a cleaned monotone discharge segment, increasing timestamps and a consistent
current sign. These tools have not been validated on NASA battery records here.

`bearing_cyclo_features()` computes STFT-power modulation features and
energy-weighted spectral-kurtosis descriptors; `order_envelope_spectrum()` adds
Hilbert demodulation and tachometer-based order tracking. This is a modulation
proxy, not a full cyclic-coherence estimator. Use finite signals, positive
sampling/speed metadata, an even STFT window with at least four complete frames,
and frequency bands inside Nyquist. Angular downsampling applies a conservative
anti-alias low-pass based on minimum shaft speed; high orders can be attenuated.
Silent/unresolvable recordings produce unscorable scores or explicit errors.

Freeze score settings before fitting `healthy_threshold()` on independent
healthy recordings. Its optional `recording_ids=` and `split_manifest=` arguments
reuse the existing manifest validator and reject calibration records sharing a
specimen/asset. Without those arguments, split provenance remains the caller's
responsibility. Marginal FAR requires exchangeable calibration/test units with
comparable acquisition/window protocols; it is not a per-record fault probability
or a fault-class diagnosis. Alarm only when `score > threshold`, with equality
treated as no alarm. No Paderborn/Lenze/CWRU benchmark was rerun here.

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
python ../local_testing/Time-Series-Diagnostic-Agent/scripts/generate_cross_domain_report.py result1.json result2.json --output-dir ../local_testing/Time-Series-Diagnostic-Agent/outputs/cross_domain
```

The report records pipeline, workflow, policy and model versions together with
dataset/protocol identifiers, input hashes, artifact checksums and Git SHAs.

### CARE v6 on Kaggle

Build the code-only Kaggle bundle locally:

```bash
python ../local_testing/Time-Series-Diagnostic-Agent/scripts/build_kaggle_care_bundle.py
```

Upload `../local_testing/Time-Series-Diagnostic-Agent/dist/tsdiag-care-kaggle.zip` and the complete official
`CARE_To_Compare.zip` as separate private Kaggle Datasets. The bundle includes
`../local_testing/Time-Series-Diagnostic-Agent/kaggle/care_benchmark.ipynb`, an auto-discovering runner, optional checksum
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
Run `python ../local_testing/Time-Series-Diagnostic-Agent/scripts/calibrate_wind_scada.py --help` once that pool is available. These benchmark and calibration tools are local-only and are not included in GitHub clones.

New Wind reports call the historical `event_end - first_detection` quantity
`time_to_event_end`. Genuine signed `early_warning_lead_time` is reported
separately against event onset (positive before onset, negative after onset).
The frozen CARE JSON retains its historical field names and is not rewritten.

`tsdiag.calibration.EventFARCalibrator` additionally provides event-statistic
split-conformal thresholds from healthy residual events, with `persistence`,
`max_smoothed` and `cusum` scorers. It accepts the existing
`DevelopmentSplitManifest` from `tsdiag.evaluation.split_manifest`; the compact
`SplitManifest` API is a compatibility adapter to that same schema and validator.
Calibration inputs must exactly match its development units. Validation IDs are
held out from threshold fitting and must respect the declared leakage boundary.
If selecting a statistic, declare the healthy and anomalous tuning IDs together;
the returned scorer requires fresh, leakage-safe units for final calibration.
Unscorable events abstain, and missing rows cannot complete a persistence window.

`EventAlarmCalibration.to_json()` freezes the complete scorer, threshold,
calibration/validation IDs, exclusion counts, FAR estimates/intervals and manifest
checksum. Reload with `EventAlarmCalibration.from_json()` and
`EventFARCalibrator.from_calibration()`. Setting `n_boot=0` omits bootstrap
intervals. Wilson and bootstrap intervals assume independent validation events;
the event-FAR claim also requires exchangeable healthy event scores and
comparable prediction-window lengths. Synthetic checks do not establish CARE
performance. This new calibrator is not wired into `WindScadaPlugin` and no
production calibration artifact is frozen: the independent healthy pool is
still required. The existing Wind search/configuration path remains separate.

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
