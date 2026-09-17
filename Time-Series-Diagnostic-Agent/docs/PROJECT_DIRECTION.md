# Project Direction and Improvement Log

Last reviewed: 2026-09-17

This is the single maintained record of the project's technical direction,
scientific baselines, and improvement decisions. Update it when an approach,
benchmark protocol, or priority changes. The README remains the user-facing
entry point; generated benchmark reports belong in `outputs/` and are not
source documentation.

## Direction

The project is moving from a collection of specialist experiments to one
auditable industrial time-series diagnostic platform. Its stable boundary is:

```text
DiagnosticRequest -> diagnose() -> DomainPlugin -> Workflow -> DecisionPolicy
                                                        |
                                                        v
                                                DiagnosticResult
```

The platform supports six domain packs: bearing, process, wind SCADA, battery,
turbofan, and transformer. Deterministic workflows are the baseline. Learned
models, physics models, manuals, and other evidence are explicit adapters; the
system must abstain when its inputs or evidence are insufficient.

Pipeline version `1.1.0` is the current public
`diagnose(domain, **inputs) -> DiagnosticResult` contract. Detection,
localization, diagnosis, verification, prognosis, uncertainty, and abstention
remain separate result fields. Each run records an ordered tool trace and
provenance. New work should strengthen algorithms and evidence without creating
a second public runtime.

## Current priority: Wind false-alarm calibration

The canonical CARE v6 run remains a frozen regression result, but its 0.94
normal-event FAR excludes Wind from platform demonstrations. A healthy-only
target-event-FAR calibration workflow now exists in
`src/tsdiag/evaluation/wind_calibration.py`. It:

- consumes a checksummed, reusable development/evaluation split manifest with
  an explicit unit-, group-, or ordered-temporal leakage boundary;
- fails loudly when an asset boundary has missing identities, shared assets,
  or a temporal boundary has overlapping/reversed intervals;
- uses contiguous fit/head and holdout/tail partitions to respect SCADA
  autocorrelation;
- searches residual threshold, persistence, CUSUM drift/threshold, and event
  residual burden against the actual event decision rather than a surrogate
  sample metric;
- chooses the loosest candidate with the highest event FAR still at or below
  the target;
- reports event FAR and sample FAR with a bootstrap event-FAR interval;
- verifies that calibration inputs exactly match the declared development
  units before fitting or scoring; and
- freezes a configuration only when the target is met.

No calibrated configuration has been accepted. All 95 CARE events already
participated in the frozen run, so they cannot now serve as a disjoint
development pool for a new headline evaluation. Obtain independent healthy
SCADA events, or predeclare a new development/evaluation protocol before
examining its evaluation results.

The historical `event_end - first_detection` metric is now named
`time_to_event_end`. A separate signed `early_warning_lead_time` is computed
against event onset; positive values precede onset and negative values are
detection delays. Frozen CARE artifacts remain immutable and retain their
historical keys.

## Parallel priority: bearing hybrid validation

The active development track is the hybrid bearing benchmark in
`src/tsdiag/benchmarks/bearing_hybrid.py`:

1. Use Paderborn vibration and motor-current waveforms with rotational speed as
   supporting physics metadata.
2. Compare three predeclared ablations: physics features, 1D CNN features, and
   CNN-plus-physics fusion.
3. Hold out complete bearing specimens so windows from one physical bearing
   never appear in both training and evaluation.
4. Evaluate Lenze separately as an operating-condition transfer test. Compare
   current-only, speed-only, and current-plus-speed inputs while holding out the
   maximum-RPM condition.
5. Report window and record metrics, but use record-level balanced accuracy and
   macro-F1 for the principal comparison.

The physics branch uses shaft orders, bearing-frequency families, envelope
harmonics, current sidebands, kurtosis, crest factor, RMS, and spectral entropy.
The learned branch uses a reproducible 1D representation rather than an
arbitrary 2D signal reshape. Paderborn combined-damage labels remain excluded by
default unless their taxonomy is explicitly reported.

Commands:

```powershell
python scripts/run_bearing_hybrid_benchmark.py --dataset paderborn --epochs 15
python scripts/run_bearing_hybrid_benchmark.py --dataset lenze --download --epochs 15
```

Completion criteria for this track:

- datasets and split identities are recorded;
- all ablations run with the same windows, seed, and evaluation units;
- trained models and JSON reports are reproducible in `outputs/`;
- results include failure/abstention behavior and runtime;
- conclusions distinguish benchmark evidence from production claims.

## Architecture status

| Domain | Current path | Status and next concern |
| --- | --- | --- |
| Bearing | Named workflow plus CWRU and hybrid benchmarks | Active priority: complete leakage-safe Paderborn/Lenze validation and improve coverage on independent development data. |
| Process / TEP | DPCA/CVA monitoring, optional SVM/FDA diagnosis, temporal/causal ranking | Deployable path is decomposed; the research calibration/ablation evaluator remains separate by design. |
| Wind SCADA | One plugin workflow plus healthy-only target-event-FAR calibration | Excluded from demos. Obtain a disjoint healthy development pool, freeze a configuration that meets its target, then evaluate once on untouched events. |
| Battery | Pack diagnosis plus an atomic capacity-prognosis task | Real-data evaluated; a materially different model and independent validation are needed. |
| Turbofan | Ten-step shared-executor workflow with `turbofan-policy-v2` | Preserve the locked C-MAPSS method and metrics while adding calibrated uncertainty/abstention on independent data. |
| Transformer | Decomposed workflow; AD-TFM-AT is the default learned classifier for compatible SGAH-shaped electrical inputs when the versioned checkpoint is present | Preserve the frozen test partition. Resolve the complete inter-phase-short-circuit failure using train/development data, then validate on independent protection data. |

The remaining migration boundary is the deprecated v0 agent API. Domain-owned
code now lives under `domains/`, and the old compatibility plugin module has
been removed. Retire legacy APIs only after searches and tests prove no public
caller or benchmark depends on them.

## Sovereign system integration boundary

The broader system should remain a composition of independently auditable
layers rather than merging GraphRAG, authorization, LLM generation, and signal
diagnosis into one policy engine:

```text
Authenticated user -> Sovereign orchestrator -> GraphRAG retrieval
                              |                        |
                              +-> tsdiag diagnose() <--+
                                           |
                                           v
                         Control-plane evidence gate -> answer or abstention
```

The host platform owns identity, role/asset authorization, secrets, rate
limits, and audit retention. GraphRAG supplies versioned manual/log excerpts.
tsdiag supplies sensor evidence, verification findings, provenance, and an
explicit abstention. The control plane may release a diagnostic claim only when
it cites permitted document chunks and matching tsdiag evidence IDs. The
diagnostic `PolicyRegistry` remains separate from RBAC; only its versioned
resolve-and-validate design pattern should be reused.

The first host-neutral integration slice now exists under
`src/tsdiag/integrations/`: a sentence-by-sentence claim release gate requires
the generating LLM to provide JSON-pointer references into the exact
`DiagnosticResult`, rejects unannotated sentences and unsupported numbers, and
fails closed on mismatched values or evidence IDs. Versioned `operator-v1` and
`engineer-v1` presentation policies provide two output shapes without taking
over host authorization. A real GraphRAG corpus and platform-LLM invocation
remain outside this repository and are still required for the complete demo.

## Retained real-data baselines

These are regression references, not claims that every domain is
research-ready.

| Domain / dataset | Retained result | Interpretation |
| --- | --- | --- |
| Bearing / CWRU | accuracy 0.3125; macro-F1 0.35; coverage 0.3125; covered accuracy 1.0; normal FPR 0 | Correct when covered but abstains too often. |
| Process / Braatz TEP | Fixed calibration and root-cause regression gates | Continue verification work without post-hoc tuning on the frozen fault cases. |
| Wind / CARE v6 | 95/95 successful; recall 0.9111; precision 0.4659; F1 0.6165; abstention 0.0316; normal-event FAR 0.94 | Valid canonical `1.1.0` Kaggle run with official archive MD5 and zero failures. Reproducible, but not operationally acceptable because 47/50 normal events were flagged. |
| Battery / NASA B0005/6/7/18 | exact-event coverage 0.90; MAE 14.89 cycles; RMSE 17.92; interval coverage 0.7778 | Evaluated but not research-ready; B0007 exposes censor-interval weakness. |
| Turbofan / C-MAPSS | 707 engines; coverage 1.0; MAE 24.08 cycles; RMSE 33.76 | Validated train-only baseline; add uncertainty and abstention. |
| Transformer / SGAH RF baseline | recall 0.2031; specificity 0.9683; precision 0.5909; F1 0.3023; balanced accuracy 0.5857 | Retained conservative binary baseline; do not use it to characterize the later AD-TFM-AT model. |
| Transformer / SGAH AD-TFM-AT | multiclass accuracy 0.9569; balanced accuracy 0.7971; macro-F1 0.7847; main-transformer precision/recall/F1 1.0 | Strong frozen-test main-transformer result, but all 14 inter-phase short-circuit events were misclassified, principally as single-phase ground faults. |

## Scientific rules

- Never tune a method on the same frozen benchmark used for the final claim.
- Split by physical unit, machine, event, or operating condition—not random
  overlapping windows—whenever leakage is possible.
- Fit preprocessing, thresholds, feature selection, and models using training or
  declared development data only. Load test labels only for final scoring.
- Keep fault detection, localization, classification, prognosis, coverage, and
  abstention metrics distinct.
- Do not convert warnings, domain shift, or weak harmonic rankings into fault
  labels without the evidence required by policy.
- Preserve benchmark configuration, per-unit outputs, aggregate metrics,
  failures, runtime, commit SHA, workflow/policy/model versions, and artifact
  checksums.
- Treat architecture-only changes as parity regressions, not new performance
  experiments.

## Improvement sequence

1. Treat the completed CARE v6 run as a frozen regression baseline. Calibrate
   Wind event FAR on disjoint healthy development data and refuse any overlap
   with the 95 previously evaluated events.
2. Finish the bearing hybrid benchmark and record a clean, leakage-safe result.
3. Add calibrated uncertainty and abstention only where an independent
   calibration protocol exists.
4. Strengthen domain models on independent development data: bearing coverage,
   wind false alarms, battery censor-aware prognosis, turbofan uncertainty, and
   transformer sensitivity.
5. Integrate the host-neutral JSON diagnostic tool with an authenticated
   platform service; keep authorization outside the diagnostic policy registry.
6. Evaluate LLM/RL planning only after deterministic workflows and mandatory
   safety/verification policies are stable. Adaptive routing must be compared
   against the deterministic baseline, not assumed to be better.

## Development and repository hygiene

Reference environment: Python 3.11.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
python examples/final_demo.py
```

Repository policy:

- source, tests, benchmark runners, configuration, README, and this document are
  tracked;
- downloaded datasets stay under ignored `data/`;
- generated models and reports stay under ignored `outputs/`;
- Python and test caches are disposable and must not be committed;
- do not add a new planning/status Markdown file—update this one;
- use `main` as the canonical development line.

## Improvement log

| Date | Decision |
| --- | --- |
| 2026-09-17 | Added healthy-only Wind target-event-FAR calibration with contiguous holdouts, disjoint-event enforcement, loosest-passing selection, bootstrap confidence intervals, and frozen-config loading. No config was frozen because no valid disjoint development pool is currently available. |
| 2026-09-17 | Added reusable checksummed development/evaluation split manifests with explicit unit, group, and ordered-temporal leakage boundaries; Wind calibration is the first consumer. |
| 2026-09-17 | Renamed Wind's historical event-end interval to `time_to_event_end` and added a separate signed onset-relative `early_warning_lead_time`. |
| 2026-09-17 | Added a fail-closed DiagnosticResult claim-grounding gate plus versioned operator and engineer presentation policies for the first platform-integration slice. |
| 2026-09-17 | Made the versioned AD-TFM-AT checkpoint the default Transformer classifier for compatible 100x6 SGAH electrical inputs; retained the general evidence workflow for incompatible sensor layouts and recorded model checksum/version provenance. |
| 2026-09-17 | Reproduced the 30-epoch AD-TFM-AT result: 0.9569 multiclass accuracy, 0.7971 balanced accuracy, 0.7847 macro-F1, perfect main-transformer binary metrics, and zero recall for the 14-event inter-phase-short-circuit class. |
| 2026-09-17 | Closed the Kaggle provenance gap found in the canonical run: future manifests and per-event provenance embed the source-tree SHA-256 and verified CARE archive MD5. |
| 2026-09-17 | Accepted the canonical CARE v6 Kaggle run: official archive MD5 matched, 95/95 events succeeded, recall 0.9111, precision 0.4659, F1 0.6165, abstention 0.0316, and normal-event FAR 0.94. CARE is now frozen against retrospective tuning. |
| 2026-09-17 | CARE error analysis found weak class separation: event-evidence ROC-AUC 0.558 and fused-alarm ROC-AUC 0.639; residual/CUSUM corroboration fired on 48/50 normal events. Independent healthy data is required for false-alarm development. |
| 2026-09-17 | Added a code-only Kaggle CARE bundle with direct-ZIP execution, smoke/full modes, checksum validation, run integrity checks, and provenance artifacts. |
| 2026-09-17 | Defined the Sovereign/GraphRAG/tsdiag/control-plane trust boundary; authorization remains outside diagnostic policy resolution. |
| 2026-09-17 | Decomposed turbofan into ten named workflow steps with `turbofan-policy-v2`; removed `compat_plugins.py`. |
| 2026-09-17 | Added a versioned cross-domain provenance report and strict JSON diagnostic tool boundary. |
| 2026-09-17 | Consolidated wind physics from `domain/` into `domains/`. |
| 2026-09-17 | Marked historical CARE metrics pending regeneration after the Wind single-path migration. |
| 2026-09-17 | Unified both Wind SCADA call styles behind the plugin workflow and made train/prediction matrices canonical. |
| 2026-09-17 | Standardized Wind CUSUM defaults at drift 0.5, threshold 10.0, and hold 6 across code and benchmark entry points. |
| 2026-09-17 | Made unspecified pipeline provenance report `unknown`, removed superseded compatibility classes, and deprecated the v0 agent API. |
| 2026-09-17 | Made the hybrid Paderborn/Lenze bearing evaluation the immediate benchmark priority. |
| 2026-09-17 | Returned active development to the Paderborn/Lenze bearing benchmark after completing the architecture cleanup above. |
| 2026-09-17 | Consolidated migration notes, benchmark status, protocols, and future approaches into this document. |
| 2026-09-17 | Removed generated caches/reports from the working tree while retaining ignored benchmark datasets needed for current work. |
