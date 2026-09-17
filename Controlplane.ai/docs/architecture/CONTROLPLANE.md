# ControlPlane.ai design

## Design principles

1. Risk categories overlap. Findings are independent records, not one exclusive label.
2. Evidence absence is `unknown`, never automatically `false` or `safe`.
3. Verification depth is separate from enforcement action.
4. Policy is versioned configuration with an explicit rule trace.
5. Enabled detector failures become policy findings instead of silent passes.
6. Consequential uncertain outputs can fail closed to human review.
7. Every completed check records the exact policy snapshot used.

## Request model

An interaction contains a policy profile, prompt, response, optional grounding context, conversation turns, geography/industry hints, a consequential-use flag, and provider-specific metadata. It does not depend on model logits or a particular foundation-model API.

## Detector contract

Each detector receives the same interaction and profile-specific settings. It returns normalized findings, latency, metadata, and an isolated error field. The orchestrator executes enabled detectors concurrently.

Privacy detection currently handles common direct identifiers, payment cards with Luhn validation, API keys, bearer tokens, and private-key material. Prompt findings are distinguished from actionable response spans. The `/v1/precheck` endpoint applies the same policy engine before a prompt is sent to a model provider.

Bias detection currently provides high-precision screening for explicit generalizations, protected-attribute adverse actions, and caller-supplied counterfactual response inconsistency. It is not a general fairness certification.

Hallucination detection adapts the existing `adaptivefact` claim extractor and deterministic verifier. The research subsystem now includes an integrated Phase-7 dispatcher and bounded Phase-8 agentic verifier. The default ControlPlane request path remains low-latency and does not invoke transformer NLI unless that backend is explicitly integrated and enabled.

Conversation detection flags current outputs that explicitly rely on an earlier turn carrying elevated risk.

## Adaptive verification integration

The optional `AdaptiveFactVerificationService` connects configured factuality
depth to execution:

```text
quick    -> Phase 5 deterministic verification
standard -> Phase 5 + Phase 6 retrieval/NLI
deep     -> Phase 5 + Phase 6 + bounded Phase 8 evidence agent
```

The service returns `claim_contradicted` and `claim_unknown` findings using the
common ControlPlane schema. Supported claims emit no risk finding, while mixed
responses retain their separate claim-level findings. ControlPlane applies the
final policy only after these results are appended, allowing deep evidence to
change the enforcement action.

Heavy verification is enabled by default and can be disabled with
`CONTROLPLANE_ADAPTIVE_VERIFICATION=0` for lightweight installations that
should never download model weights. Once the
service is enabled, unavailable required NLI or agent backends become
policy-visible detector failures rather than silent fallbacks.

## Decision behavior

All matching policy rules are recorded. When several rules match, the strongest action wins:

```text
allow < allow_with_warning < redact < human_review < block
```

Redaction is span-preserving and applied from the end of the response toward the beginning. Block and review actions replace the response with profile-controlled text.

## Audit and feedback

The prototype stores append-only JSONL events under `results/controlplane/`. Audit events include the report and complete policy snapshot. Sensitive response spans are redacted before persistence by default; raw storage requires an explicit store configuration. Feedback events record reviewer outcomes, corrected actions, notes, actor, and timestamps.

Production deployment should replace JSONL with an authenticated append-only event store, encrypt sensitive fields, add retention policies, and separate reviewer permissions from policy-administrator permissions.

## Known limitations

- Heuristic privacy/bias detectors prioritize demonstrability and precision over exhaustive coverage.
- Regex entity extraction can treat sentence-initial capitalized words as entities.
- The policy engine does not claim that configured geography/industry metadata constitutes legal compliance.
- Thread-based parallelism records a latency-budget breach but does not forcibly cancel a running ML call.
- Phase-6 NLI and the bounded agent are integrated but opt-in and synchronous;
  a durable asynchronous human-review system is still required for production.
- JSONL is appropriate for a local prototype, not a multi-instance production deployment.
- Controlled scenarios demonstrate mechanics; a larger independently labeled dataset is required for trustworthy accuracy claims.
