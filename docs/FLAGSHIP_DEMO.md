# Pump-102 flagship demo

Open `/integrations` after signing in. The page is preloaded with the canonical
request and a deterministic sensor sequence. Start GraphRAG, the diagnostic
service, ControlPlane, and the configured local model provider, then run once.

The screen shows authorized evidence with source/revision/page/ACL, sensor
diagnosis, local runtime TTFT/tok/s/prompt tokens, governance findings, the final
cited recommendation, a maintenance artifact, and an Evidence Capsule root hash.

Artifacts and capsules are created only for a released Pump‑102 run with a
diagnostic input. They remain advisory; physical maintenance requires separate
human authorization.

Failure demonstrations:

- stop vLLM/Ollama: `model_unavailable`, unreleased, no artifact/capsule;
- omit GraphRAG authorization scope: HTTP 400;
- query a finance-only document as maintenance: absent before ranking;
- provide no supported evidence: abstention;
- introduce contradictory SOP revisions: conflict surfaced;
- stop Docker during coding: no host execution;
- make ControlPlane block/hold: candidate never released.

Do not describe local endpoints as an air gap. For offline proof, deploy every
dependency internally, block egress, run the sovereignty verifier, and preserve
its output with the demo evidence.
