# Next implementation plan

Updated: 30 September 2026

The architecture is frozen. The next milestone is a reproducible Pump-102 assurance bundle, not another agent or a larger feature surface.

## 1. Close the three remaining experiments

1. **Controlled local inference comparison**
   - Run Ollama and vLLM with comparable Qwen3 0.6B formats.
   - Separate cold and warm runs, randomize condition order, and collect at least five warm repetitions.
   - Report TTFT, total latency, throughput, memory, evidence budget, groundedness, and citation recall.
2. **Live authorized GraphRAG**
   - Move the evaluation corpus to approved local/self-hosted storage.
   - Cover direct, numeric, multi-document, conflicting revision, unanswerable, and unauthorized-perfect-match cases.
   - Prove forbidden evidence is absent before scoring and from context/citations.
3. **Networked Pump-102 end to end**
   - Run SovereignAI, GraphRAG, diagnostics, local inference, and ControlPlane together.
   - Preserve latency, citations, release decision, artifact identity, and capsule root hash.
   - Repeat with no evidence, unauthorized evidence, and required-service outage; each must abstain or fail closed.

Exit condition: reproducible commands and machine-readable results for all three experiments.

## 2. Strengthen diagnostic evidence

- Freeze Paderborn specimen-level splits and compare physics-only, 1D-CNN-only, and fusion models.
- Report record-level balanced accuracy and macro-F1 first, plus coverage/abstention, runtime, dataset identity, and split provenance.
- Use Lenze as a held-out operating-condition transfer test for current-only, speed-only, and combined inputs.
- Do not tune against the frozen Wind CARE events or final holdouts.

Exit condition: a leakage-safe diagnostic result that can be inserted into the Pump-102 evidence chain.

## 3. Build decision-grade factuality evaluation

- Create 300-500 manually adjudicated industrial cases.
- Store each claim, exact evidence bundle, provenance, expected policy action, and reviewer metadata.
- Label `SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`, `UNDECIDABLE`, or `CONFLICTING`.
- Compare optional support backends only after the dataset and holdout are frozen.

Exit condition: state-level precision/recall and release/hold error rates with confidence intervals.

## 4. Prepare for an enterprise pilot

- Add document lifecycle controls: deduplication, revision/supersession, quotas, archival/deletion, re-indexing, and orphan cleanup.
- Add service identity and authenticated transport, centralized secrets, managed signing keys, SBOM/build/model provenance, and dependency scanning.
- Replace process-local jobs/events and prototype storage where durability or scale requires it.
- Add append-only/tamper-evident audit, backup/restore tests, retention, monitoring, and incident procedures.
- Produce an intended-use statement, risk register, human-oversight plan, change-control record, and shadow-pilot protocol.

Exit condition: an auditable, read-only, supervised pilot package with explicit residual risks.

## Deferred until evidence justifies it

- Plant write/control capability or autonomous work approval.
- New agents, major architecture rewrites, public Workcell marketplace, or broader graph algorithms.
- Fine-tuning, larger models, or new factuality thresholds without a measured gap.

See [Current features](CURRENT_FEATURES.md) for the implemented baseline and [Past experiments](PAST_EXPERIMENTS.md) for measured evidence.
