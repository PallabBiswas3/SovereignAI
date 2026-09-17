# Implementation roadmap

## ControlPlane.ai product track

| Milestone | Scope | Status |
|---|---|---|
| CP0 | Common interaction, finding, decision, audit, and feedback schemas | **Implemented** |
| CP1 | Versioned customer-support, internal-assistant, and regulated policies | **Implemented** |
| CP2 | Privacy, bias, hallucination, and conversation-risk detectors | **Implemented (prototype scope)** |
| CP3 | Parallel orchestration and allow/warn/redact/review/block enforcement | **Implemented** |
| CP4 | CLI, API, Streamlit demo, audit trail, and feedback capture | **Implemented** |
| CP5 | Controlled cross-risk evaluation harness | **Implemented; expand dataset next** |
| CP6 | Production stores, authentication, policy administration, monitoring | Not started |
| CP7 | External model connectors and asynchronous post-audit queue | Not started |
| CP8 | Validated sector/geography policy packs | Not started |

## Adaptive factuality research track

| Phase | Focus | Dataset | Status |
|---|---|---|---|
| 0 | Schema, loader, benchmark harness | RAGTruth | **Implemented** |
| 1 | Always-on verification baselines | RAGTruth | **Implemented** |
| 2 | Risk Estimator V1 | RAGTruth | **Implemented** |
| 3 | Calibration and routing thresholds | RAGTruth | **Implemented** |
| 4 | Adaptive route evaluation | RAGTruth + HaluEval | **Implemented** |
| 5 | Claim extraction and deterministic verification | RAGTruth | **Implemented** |
| 6 | Retrieval and NLI for unresolved claims | RAGTruth | **Implemented** |
| 6.1 | Source-safe NLI threshold tuning | RAGTruth | **Implemented; constraints not fully met** |
| 7 | Full response-level selective dispatcher | RAGTruth | **Implemented; 100-record representative benchmark completed** |
| 8 | Bounded agentic escalation with approved tools | Controlled corpus | **Implemented; representative evaluation runner added** |
| 9 | Evidence-grounded correction, blocking, safe abstention, and selective human review | RAGTruth + controlled scenarios | **Implemented** |
| 9 | Open-domain web verification | FActScore/HaluEval-Wild | Not started |
| 10 | Optimization and ablations | Held-out sets | Not started |

## Important current conclusions

- Risk ranking is useful, but the safe RAGTruth fast path is effectively empty under the configured constraints.
- Calibration does not transfer reliably from RAGTruth to HaluEval.
- NLI resolves roughly 30% of Phase-5 unknown claims in the recorded runs and dominates latency.
- No Phase-6.1 threshold candidate met every safety constraint; the saved threshold is a documented fallback.
- The Phase-7 dispatcher and bounded Phase-8 agent now exist. The next milestone is a held-out comparison proving selective verification against an always-on baseline.
