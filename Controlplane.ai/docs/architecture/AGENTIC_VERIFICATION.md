# Phase 7/8 adaptive and agentic verification

## Purpose

Phases 0–6 implement the individual factuality components. Phase 7 connects them into one route-aware runtime. Phase 8 adds bounded evidence-seeking verification for important claims that remain `UNKNOWN` after deterministic and NLI checks. Phase 9 applies evidence-grounded automatic remediation after the final claim decision.

The agent is not allowed to treat model memory or generated reasoning as evidence. A resolved decision requires evidence returned by an approved search tool and an auditable citation.

## Execution flow

```text
ResponseRecord
    │
    ▼
Risk model → calibrator → route
    │
    ├── FAST_ACCEPT
    │      no claim verification
    │
    ├── LIGHTWEIGHT
    │      Phase 5 deterministic verification
    │      NLI for a limited number of priority claims
    │
    └── AGENTIC
           Phase 5 deterministic verification
           broader Phase 6 retrieval/NLI
           bounded search for important UNKNOWN claims
                    │
                    ▼
           response-level aggregation
                    │
                    ▼
           extractive correction / block / safe abstain / review
```

## Implemented modules

- `adaptivefact.pipeline.adaptive_pipeline`: complete route dispatcher and benchmark component
- `adaptivefact.pipeline.prioritization`: claim-type, response-risk, conflict, and consequentiality priority
- `adaptivefact.pipeline.aggregation`: claim-to-response decision rule
- `adaptivefact.agents.verifier`: bounded iterative evidence search and NLI verification
- `adaptivefact.agents.tools.context`: search within the supplied grounding context
- `adaptivefact.agents.tools.corpus`: offline approved-enterprise-corpus search
- `adaptivefact.agents.schema`: citations, step trace, result, and budget configuration

## Agent constraints

The agent enforces independent limits for:

- iterations
- search calls
- NLI pairs
- evidence sources
- total latency
- monetary cost
- claims per response
- total agent latency shared across all claims in a response
- minimum source authority and retrieval relevance

Possible stop reasons include `resolved_supported`, `resolved_contradicted`, `conflicting_evidence`, `no_reliable_evidence`, `search_budget_exhausted`, `nli_budget_exhausted`, `latency_budget_exhausted`, `cost_budget_exhausted`, and `max_iterations_reached`.

Failure to resolve is represented as `UNKNOWN`; it is never silently converted to support.

## Approved sources

The default experiment uses the provided context as its approved source. `CorpusSearchTool` demonstrates an enterprise search connector using an offline controlled corpus. A future internal database or web connector must implement the same `SearchTool` interface and return source identity, text, authority, retrieval score, and optional URL.

Unrestricted web search is deliberately not enabled by default.

## Running Phase 7/8

Regenerate Phase 2 and Phase 3 artifacts in the active environment before running serialized models:

```bash
python scripts/run_phase2_risk.py --config configs/phase2_ragtruth.yaml
python scripts/run_phase3_calibration.py --config configs/phase3_ragtruth.yaml
```

Then run the integrated experiment:

```bash
python scripts/run_phase7_adaptive.py --config configs/phase7_adaptive.yaml
```

Use `--max-records 5` for the first real-model smoke run.

To inspect one complete agent trace against the included controlled enterprise corpus:

```bash
python scripts/run_phase8_agent_demo.py
```

The default run is limited to 100 RAGTruth records because transformer NLI is CPU-intensive. The runner uses a reproducible, label-stratified random sample (`sampling.seed: 42`) rather than the first 100 records, which are ordered by source/generator. Increase `experiment.max_records` only after validating the smoke run.

## Reading the Phase 7 metrics

The summary deliberately reports two different views:

- `metrics.selective` treats `SUPPORTED` and `HALLUCINATED` as confirmed decisions, while `UNKNOWN` and `MIXED` are pre-remediation abstentions requiring intervention. It reports coverage, abstention rate, confirmed-detection precision/recall, intervention burden, and unsafe releases.
- `metrics.safety_triage` measures the conservative operational action in which every non-`SUPPORTED` response is intercepted. This can have perfect recall and a high false-positive rate without implying that every abstention was classified as a hallucination.
- `metrics.automatic_remediation` measures human-free handling separately from useful verified answers, containment, and supported-answer retention.

For a safety gateway, prioritize a low `unsafe_release_rate` while reducing human review through validation-set calibration and bounded remediation. Do not select routing, NLI, or aggregation thresholds using the test summary.

See [`AUTOMATIC_REMEDIATION.md`](AUTOMATIC_REMEDIATION.md) for Phase 9 actions and the safety boundary.

## Evaluation warning

RAGTruth provides response spans, not gold atomic claims. Response-level selective and safety-triage metrics are valid, but agent claim-resolution accuracy also requires a controlled claim/evidence set with explicit `SUPPORTED`, `CONTRADICTED`, and `UNKNOWN` labels. The included Phase 8 demo supplies one such auditable scenario; it is a mechanism demonstration rather than a statistical test set.
