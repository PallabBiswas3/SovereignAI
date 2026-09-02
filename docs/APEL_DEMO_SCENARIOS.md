# APEL demonstration scenarios

Seed APEL, start with `SOVEREIGN_AUTH_MODE=local`, and use the browser at `http://127.0.0.1:3000`.

## A — Maintenance evidence workflow

Sign in as `arjun.rao@apel.local` and ask:

> Assess Pump-102 using the latest available evidence.

Authorized evidence establishes 5.1 bar pressure, 7.4 mm/s RMS vibration, the current Rev 4 limits, and recent maintenance. The expected disposition is elevated vibration requiring controlled maintenance review, below the 9.0 mm/s RMS shutdown point. For the installed inspection Workcell, attach the generated inspection document or existing demo inspection upload, execute, then create and verify an Evidence Capsule.

## B — Restricted Finance denial

As the same Maintenance Engineer ask:

> Show me executive compensation for this year.

The restricted Finance document is excluded before dense/BM25 scoring, fusion, reranking, context compilation, and model input. The user receives an access-safe no-authorized-evidence response. The `FIN-XC-926` canary is used by automated leakage tests and must never appear.

## C — HSE incident

Sign in as `meera.sen@apel.local` and ask:

> Analyze Incident-2026-014 and identify applicable safety requirements.

The authorized set includes the restricted incident, draft permit, and line-breaking SOP. The result should identify verified isolation, gas test, PPE, and standby-observer requirements while clearly stating that the draft permit is not work authorization.

## D — Procurement comparison

Sign in as `vikram.shah@apel.local` and ask:

> Compare the available vendor proposals against Compressor-201 technical requirements.

Atlas supplies all listed technical fields, Boreal omits specific power and has an unsupported certificate statement, and Cascade falls below minimum capacity and omits a noise guarantee. Batch 4 uses the generic evidence workflow; a dedicated vendor-comparison Workcell is future work.

## E — Management briefing

Sign in as `ananya.iyer@apel.local` and ask:

> Prepare today's Plant A management briefing.

The manager can compile authorized Maintenance, Operations, HSE, Quality, Procurement, and Management evidence. The restricted Finance canary remains outside their scope.

## Role distinction checks

- Auditor can inspect authorized audits and verify explicitly shared Evidence Capsules, but cannot execute tools or approve actions.
- Administrator can inspect demo users and validate managed Workcells, but cannot approve business actions.
- Even a user holding approver authority cannot approve their own proposal.

The versioned benchmark `demo/apel/evaluation/v1.json` contains 50 exact, paraphrased, numeric, cross-document, missing-evidence, conflict, unit, role, and unauthorized questions. It is a test target, not a fabricated performance claim. If local reranker weights are absent, the honest benchmark status is `RERANKER_UNAVAILABLE` and hybrid RRF remains operational.
