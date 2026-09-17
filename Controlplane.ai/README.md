---
title: ControlPlane AI
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
suggested_hardware: cpu-upgrade
models:
  - cross-encoder/nli-deberta-v3-small
---

# ControlPlane.ai

ControlPlane.ai is a policy-driven safety gateway for enterprise generative AI. It checks prompts, model responses, grounding context, and conversation state for overlapping privacy, bias, hallucination, and governance risks before a response reaches a user or downstream system.

The original `adaptivefact` research package remains in this repository as the hallucination-risk and factual-verification subsystem.

## What works now

- Parallel privacy, bias, hallucination, and conversation-risk checks
- Multi-label findings: one passage can carry several risk categories
- Three versioned policy profiles with different latency and risk tolerances
- Independent verification depth (`quick`, `standard`, `deep`) and enforcement action
- Explainable `allow`, `allow_with_warning`, `redact`, `human_review`, and `block` decisions
- Span-preserving redaction for PII, payment cards, credentials, and secrets
- Conservative hallucination handling using `supported`, `suspected`, `contradicted`, and `unknown`
- Human feedback and JSONL audit events containing the policy snapshot
- CLI, FastAPI service, Streamlit demonstration, and controlled cross-risk evaluation
- Existing RAGTruth/HaluEval research pipeline through NLI threshold tuning (Phases 0–6.1)
- Integrated Phase-7 route execution and bounded Phase-8 evidence-seeking verification

## Architecture

```text
Prompt + response + evidence + conversation
                    │
                    ▼
          Versioned policy profile
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
    Privacy        Bias     Hallucination    Conversation
       └────────────┼────────────┘
                    ▼
          Normalized multi-label findings
                    │
                    ▼
             Policy decision engine
                    │
       allow / warn / redact / review / block
                    │
                    ▼
          Safe response + audit + feedback
```

See [`docs/architecture/CONTROLPLANE.md`](docs/architecture/CONTROLPLANE.md)
for the detailed design and current limitations. The complete project-level
explanation remains easy to find at
[`OVERALL_ARCHITECTURE_AND_REVIEW.md`](OVERALL_ARCHITECTURE_AND_REVIEW.md).

## Setup

Use a fresh virtual environment. NumPy is intentionally constrained below 2.0 because older Torch wheels in common enterprise environments are not NumPy-2-compatible.

```bash
pip install -e ".[dev,api,demo]"
```

Install the heavy research dependencies only when running model training or NLI:

```bash
pip install -e ".[ml]"
```

Clone and install from GitHub after the repository is published:

```bash
git clone https://github.com/YOUR_USERNAME/controlplane-ai.git
cd controlplane-ai
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -e ".[dev,api,demo]"
```

On macOS or Linux:

```bash
source .venv/bin/activate
pip install -e ".[dev,api,demo]"
```

## Quick start

Run one check:

```bash
python scripts/run_controlplane_demo.py \
  --profile customer_support \
  --prompt "How can I contact the customer?" \
  --response "Email the customer at priya@example.com." \
  --context "The customer requested a status update."
```

Start the API:

```bash
uvicorn controlplane.api:app --reload
```

Enable the integrated depth-aware AdaptiveFact verifier when the heavy ML
dependencies and NLI model are available:

```powershell
$env:CONTROLPLANE_ADAPTIVE_VERIFICATION = "1"
$env:CONTROLPLANE_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
uvicorn controlplane.api:app --reload
```

Verification depth now controls execution: `quick` runs deterministic checks,
`standard` adds retrieval/NLI, and `deep` adds the bounded evidence agent.
AdaptiveFact results are normalized before the final policy decision. A
configured backend failure becomes a detector failure so consequential policies
can fail closed to human review.

Useful endpoints:

- `GET /health`
- `GET /v1/policies`
- `POST /v1/precheck`
- `POST /v1/check`
- `POST /v1/feedback`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

Start the visual demonstration:

```bash
streamlit run demo/app.py
```

On Windows PowerShell, start the complete UI with AdaptiveFact and the default
DeBERTa-v3-small model using one command:

```powershell
.\scripts\start_controlplane_ui.ps1
```

AdaptiveFact/NLI is enabled by default. The launcher explicitly sets
`CONTROLPLANE_ADAPTIVE_VERIFICATION=1`, selects
`cross-encoder/nli-deberta-v3-small`, disables Streamlit's optional-module file
watcher, and starts the application. A different local or Hugging Face model
can be selected with `-Model`, and `-Device cpu` or `-Device cuda` can override
automatic device selection.

The interface explicitly shows whether AdaptiveFact/NLI is configured, keeps
the overall policy risk index separate from factuality status, and displays
claim-level entailment, contradiction, and neutral scores when NLI runs.
Privacy values are masked before factuality feature extraction so emails,
phones, identifiers, credentials, and card numbers do not create misleading
entity findings. Confirmed contradictions are never released: customer-facing
outputs are blocked, while internal and regulated outputs require review.

Run the controlled cross-risk evaluation:

```bash
python scripts/evaluate_controlplane.py
```

Build the larger reviewer-candidate benchmark and compare enabled verification
models:

```bash
python scripts/build_benchmark_splits.py
python scripts/compare_verification_models.py
```

The generated 200-development, 100-validation, and 400-blinded-test items are
synthetic reviewer candidates, not gold labels. Independent review and
adjudication are required before reporting final benchmark accuracy. API judges
are disabled in `configs/model_comparison.yaml` by default and require the
optional `judge` dependency, credentials, and `--include-api-models`.

Run all tests:

```bash
pytest -q
```

## Deploy to a Hugging Face Docker Space

The Docker deployment keeps AdaptiveFact enabled and caches
`cross-encoder/nli-deberta-v3-small` in the image during the build. The default
device is CPU.

1. Create a new Space at <https://huggingface.co/new-space>.
2. Choose **Docker** as the Space SDK and select the desired visibility.
3. Clone the new Space repository.
4. Copy this repository's tracked files into the Space repository, then commit
   and push them.

Alternatively, add the Space as a second remote from this repository:

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push --force space HEAD:main
```

Use `--force` only for a newly created, empty Space because it replaces the
Space repository's generated starter commit. For later deployments, use:

```bash
git push space HEAD:main
```

The container listens on port `7860`. These runtime defaults are already set in
the `Dockerfile`:

```text
CONTROLPLANE_ADAPTIVE_VERIFICATION=1
CONTROLPLANE_NLI_MODEL=cross-encoder/nli-deberta-v3-small
CONTROLPLANE_NLI_DEVICE=cpu
```

The first build downloads and embeds the model weights, so it will take longer
than later application-only rebuilds. CPU Basic can run the application but NLI
inference will be slower; CPU Upgrade is the recommended hardware profile.

## Publishing this project to GitHub

The repository includes a `.gitignore` that prevents downloaded datasets, generated results, audit logs, virtual environments, caches, secrets, and binary model artifacts from being uploaded.

From the project root, create the repository history:

```bash
git init
git add .
git status
git commit -m "Build ControlPlane.ai responsible AI gateway"
git branch -M main
```

Create an empty repository on GitHub named `controlplane-ai`, then connect and push it:

```bash
git remote add origin https://github.com/YOUR_USERNAME/controlplane-ai.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. If an `origin` remote already exists, inspect it with `git remote -v` and update it with `git remote set-url origin <repository-url>` instead of adding it again.

Before every push, verify the staged files:

```bash
git status
git diff --cached --stat
```

Do not force-add ignored datasets, audit logs, `.env` files, Streamlit secrets, or model binaries. They may contain third-party data, sensitive interactions, or environment-specific serialized objects.

### Files intentionally committed

- Application and research source code under `src/`
- Policy profiles under `configs/policies/`
- Controlled synthetic scenarios under `data/controlplane/`
- Tests, scripts, documentation, and dependency definitions
- Lightweight JSON model metadata and selected threshold descriptions

### Files intentionally excluded

- `data/raw/`: downloaded RAGTruth and HaluEval files
- `results/`: generated experiments, local audit events, and feedback
- Binary model files such as `.joblib`, `.pt`, `.safetensors`, and `.onnx`
- Virtual environments, Python caches, IDE settings, credentials, and local secrets

Users can recreate the research datasets locally:

```bash
python scripts/download_ragtruth.py
python scripts/download_halueval.py
```

They can regenerate model artifacts with the relevant `scripts/run_phase*.py` commands. ControlPlane.ai runs with its deterministic prototype controls without committed model binaries; trained hallucination risk is disabled in the default policies until compatible Phase 2 artifacts are regenerated.

## Policy profiles

Policies live under `configs/policies/`:

- `customer_support`: strict output privacy, CPU-friendly verification latency, and warnings for material factuality/bias risk
- `internal_assistant`: balanced verification and auditability, review for high bias or compounding risk
- `regulated_decision_support`: safety-first, review on evidence gaps, bias, sensitive output, or failed controls

Policies contain detector settings, a latency budget, explicit rule conditions, action mappings, and user-facing safe responses. Geography and industry are policy metadata rather than hard-coded legal conclusions.

## Repository layout

```text
README.md                              quick start and project navigation
OVERALL_ARCHITECTURE_AND_REVIEW.md     complete problem, design, and review
docs/                                  detailed architecture, evaluation, research, training
src/controlplane/                      product-level gateway and policy enforcement
src/adaptivefact/                      factuality research and verification subsystem
configs/                               policy and experiment configuration
data/controlplane/                     benchmarks, generated packs, and NLI datasets
models/                                lightweight metadata and local trained-model location
scripts/                               evaluation, training, and research entry points
results/                               generated logs and reports, separated by subsystem/phase
demo/                                  Streamlit interface
tests/                                 component and end-to-end tests
```

Use [`docs/README.md`](docs/README.md) as the documentation index. Generated
outputs under `results/` and binary checkpoints under `models/` remain excluded
from Git; their README files explain where each local artifact belongs.

## Existing factuality research

The `adaptivefact` package implements RAGTruth/HaluEval loading, always-on baselines, risk estimation, calibration, route assignment, claim extraction, deterministic verification, retrieval/NLI, source-safe NLI threshold tuning, an integrated adaptive dispatcher, a bounded agentic evidence loop, and evidence-grounded automatic remediation.

The trained risk estimator is disabled in the new product policies by default because persisted scikit-learn estimators must be retrained or loaded with the same library version that created them. Enable `use_trained_risk` after regenerating Phase 2 artifacts in the active environment.

Research commands remain available under `scripts/run_phase*.py` and are
described in [`docs/research/PHASES.md`](docs/research/PHASES.md).

Run the integrated Phase 7/8 research pipeline after generating compatible model artifacts:

```bash
python scripts/run_phase7_adaptive.py --config configs/phase7_adaptive.yaml
```

Start with a small real-model smoke run:

```bash
python scripts/run_phase7_adaptive.py --max-records 5
```

The configured experiment draws a deterministic, label-stratified sample instead of taking the first ordered dataset rows. Its report separates confirmed factuality decisions from abstentions:

- `metrics.selective`: coverage, abstention/review burden, confirmed detection, and unsafe releases
- `metrics.safety_triage`: conservative interception of every non-supported response
- `metrics.automatic_remediation`: automation, actual human review, useful answers, containment, and supported-answer retention

Tune the risk model, calibrator, and routing thresholds on training/validation data with the Phase 2/3 scripts. Do not manually choose thresholds from Phase 7 test metrics.

The pre-evidence-guard 100-response prototype baseline achieved 100% automated handling, 0% human review, 100% hallucination containment, and 0% unsafe original releases for non-consequential benchmark traffic. It produced a useful unchanged or extractively corrected answer for 52% of responses, retained useful answers for 60% of supported inputs, and achieved 74% safe handling overall. That inspected sample is now a historical baseline, not an unseen evaluation of the current evidence-aware configuration. See [`docs/architecture/AUTOMATIC_REMEDIATION.md`](docs/architecture/AUTOMATIC_REMEDIATION.md) for the full interpretation.

The agent uses only approved search tools, enforces search/NLI/latency/cost budgets, preserves conflicting or insufficient evidence as `unknown`, and records its complete step trace. Phase 9 then releases supported responses, retains only verified claims in extractive corrections, blocks contradictions, safely abstains for unresolved low-impact cases, and reserves humans for consequential uncertainty. See [`docs/architecture/AGENTIC_VERIFICATION.md`](docs/architecture/AGENTIC_VERIFICATION.md) and [`docs/architecture/AUTOMATIC_REMEDIATION.md`](docs/architecture/AUTOMATIC_REMEDIATION.md) for the design and safety boundary.

See [`docs/architecture/ARCHITECTURE_REVIEW.md`](docs/architecture/ARCHITECTURE_REVIEW.md) for an honest assessment of what is structurally strong, the evidence-quality safeguards added after evaluation, and the limitations that remain before production use.

The latest source-safe validation rerun improved predicted-contradiction span alignment from 7.76% to 42.86% while reducing NLI evidence pairs by 35.8%. Coverage decreased, and no threshold candidate satisfied every support-safety constraint, so these are validation diagnostics rather than a claim of final model accuracy.

Run the controlled agent example against the included synthetic enterprise corpus:

```bash
python scripts/run_phase8_agent_demo.py
```

Run the controlled automatic-remediation examples:

```bash
python scripts/run_phase9_remediation_demo.py
```

## Safety boundary

This is a working prototype, not a compliance certification. Bias heuristics are screening controls, NLI scores are not ground truth, and `unknown` deliberately triggers policy rather than being treated as correctness. Regulated profiles hold consequential uncertain outputs for an accountable human reviewer.
