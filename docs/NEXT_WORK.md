# SovereignAI — Next Work

Updated: 2026-09-21

This note records the next engineering/research priorities after integrating the validated ControlPlane factuality v3 work. It is intentionally short and should be updated when a priority is completed or materially changes.

## Current frozen result

ControlPlane factuality v3 is the current accepted architecture for the prototype. The focused regression gate passed locally, and the DeBERTa-only 1,000-case validation reached approximately:

- action accuracy: 76.9%
- over-intervention: 0.0% on the legacy validation labels
- contradiction precision: 66.3%
- contradiction recall: 97.6%
- contradiction F1: 78.9%
- p50 factuality latency: ~12 ms
- p95 factuality latency: ~452 ms

These numbers are engineering validation results on the existing synthetic corpus. The corpus predates the v3 state taxonomy, so it cannot establish independent semantic precision for UNSUPPORTED, UNDECIDABLE, and CONFLICTING.

Do not redesign factuality v3 or tune its thresholds without a new measured experiment.

## 1. Authorized/RAG latency — next systems priority

The first optimization reduced Authorized/RAG total latency from roughly 185 s to ~139 s, but the tested routine configuration capped output at 384 tokens and all 3/3 measured responses ended by length truncation. The ~63 s TTFT also remained the dominant bottleneck.

Next experiment:

- keep full evidence and provenance outside the generation prompt;
- replace raw character truncation with query-aware evidence sentence selection;
- send source/citation identity + the most relevant sentence(s) + technical values/units + revision/page/section;
- initially target Authorized prompt size below ~700 tokens rather than ~1286;
- raise the routine output budget from 384 toward ~512 so normal answers complete naturally;
- record RAM/CPU around each run;
- accept a configuration only if citations, evidence coverage, unsupported-claim behavior, and answer completeness do not regress.

Do not merge the current 384-token configuration as the final runtime policy.

## 2. Industrial Time-Series Diagnostic Agent — next research priority

Finish the leakage-safe bearing hybrid benchmark:

### Paderborn

Compare on identical specimen-level splits:

1. physics features only;
2. 1D CNN features only;
3. CNN + physics fusion.

Use record-level balanced accuracy and macro-F1 as primary comparison metrics, while retaining window metrics, abstention/coverage, runtime, dataset identities, and split provenance.

### Lenze

Use as an operating-condition transfer test. Compare:

- current-only;
- speed-only;
- current + speed;

while holding out the predeclared maximum-RPM condition.

Wind CARE remains a frozen regression baseline; do not tune on the 95 already evaluated events. False-alarm calibration requires an independent healthy development pool or a predeclared new protocol.

## 3. Pump-102 flagship end-to-end workflow

After the diagnostic improvement, demonstrate the full evidence chain:

sensor/history -> Time-Series Diagnostic Agent -> DiagnosticResult/evidence IDs -> GraphRAG manuals/SOP/history -> structured evidence -> local LLM candidate -> ControlPlane factuality v3 -> approval when required -> maintenance recommendation/artifacts -> Evidence Capsule.

The goal is to show that the three specialist systems work together without collapsing their responsibilities:

- GraphRAG retrieves documentary evidence;
- the diagnostic agent produces sensor/engineering evidence;
- ControlPlane verifies/release-controls claims;
- SovereignAI owns identity, authorization, orchestration, audit, and artifacts.

## 4. Later ControlPlane evaluation

Build a manually adjudicated industrial factuality set (roughly 300–500 cases) containing:

- claim;
- exact evidence bundle available to the system;
- provenance;
- gold state: SUPPORTED / CONTRADICTED / UNSUPPORTED / UNDECIDABLE / CONFLICTING;
- expected policy action;
- reviewer/adjudication metadata.

Only after this should MiniCheck/Align-style backends or broader bias/conversation detector redesign be compared as separate experiments.

## 5. Storage / deployment hardening later

Before a real enterprise pilot, add explicit document lifecycle controls (deduplication, quotas, revision handling, archival/deletion/re-indexing, orphan-chunk cleanup, storage metrics) and use an approved local/self-hosted database/vector stack for strict offline GraphRAG deployments.
