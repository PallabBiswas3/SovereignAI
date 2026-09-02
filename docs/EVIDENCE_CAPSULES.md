# Sovereign Evidence Capsules

An Evidence Capsule is a portable local package that preserves what a completed Workcell used and produced. It supports independent stored-content integrity checks after execution without storing hidden chain-of-thought.

## Contents

A Pump Inspection capsule can contain:

```text
capsule_manifest.json
hashes.sha256
signature.json                         optional
final_answer.md
inputs/manifest.json
evidence/sources.json
evidence/fragments.json
evidence/measurements.json
evidence/rules.json
evidence/calculations.json
evidence/claims.json
evidence/conflicts.json
execution/workcell_manifest.json
execution/workflow_definition.json
execution/prompt_manifest.json
execution/model_manifest.json
execution/policy_decisions.json
execution/tool_calls.json
execution/human_decisions.json
execution/audit.jsonl
artifacts/*
```

Optional empty evidence categories are omitted. Audits contain concise events, tool observations, verification results, and decisions—not private reasoning tokens.

## Identity algorithm

Every payload file receives SHA-256 over its exact bytes. Paths are normalized to forward-slash relative paths and sorted lexically. `hashes.sha256` is rendered deterministically.

The capsule root identity is:

```text
SHA256(canonical JSON([
  {"path": normalized_relative_path, "sha256": file_sha256},
  ... sorted by path
]))
```

Canonical JSON is UTF-8, key-sorted, compact (`separators=(",", ":")`), non-ASCII preserving, and rejects NaN. Timestamps are not inputs to the root identity except where a timestamp is intentionally stored inside a payload file. This is called `capsule_root_hash`; it is not described as a Merkle tree.

`capsule_manifest.json`, `hashes.sha256`, and `signature.json` are metadata and excluded from the payload root to avoid circular identities. Their consistency is checked independently.

## Verification

`EvidenceCapsuleVerifier` checks:

- strict versioned manifest schema;
- declared required payload presence;
- unexpected payload files;
- every byte hash and file size;
- deterministic root identity;
- exact `hashes.sha256` rendering;
- Workcell/artifact identity recorded by the manifest;
- Ed25519 signature when a trusted public key is available.

Failures are typed and identify the exact path, for example `CAPSULE_HASH_MISMATCH` on `artifacts/approval_note.docx`. A mandatory tamper test generates a capsule, verifies it, modifies the artifact, and confirms rejection with that exact path.

## Signing and trust

`CapsuleSigner` separates signing from verification. The Batch 3 implementation uses Ed25519 through `cryptography`. Signature metadata records algorithm, key ID, signed root hash, and Base64 signature; no private key is written into the capsule or repository.

Hash integrity means the stored bytes match the manifest. Signature authenticity means a holder of the corresponding private key signed the root hash. Neither proves that the engineering conclusion is factually correct.

Test keys are created ephemerally in tests. Production private-key generation/storage is deliberately absent. Enterprise deployments should later integrate approved PKI, HSM, smart-card, or offline signing services.

Unsigned capsules are valid under the default development policy and report `UNSIGNED`. Set `SOVEREIGN_UNSIGNED_CAPSULES_ALLOWED=false` to reject them. A signature with an unknown local key reports `SIGNED_UNVERIFIED`; a wrong key or invalid signature is rejected.

## Storage and API

Capsules are built in a temporary directory under `workspace/evidence_capsules/` and renamed to their final UUID directory only after successful construction. SQLite tracks `BUILDING`, `COMPLETE`, `VERIFIED`, `INVALID`, and `FAILED` states. Runtime capsule content is git-ignored.

APIs:

- `POST /api/tasks/{task_id}/capsule`
- `GET /api/tasks/{task_id}/capsule`
- `GET /api/capsules/{id}`
- `POST /api/capsules/{id}/verify`
- `GET /api/capsules/{id}/download`

Downloads are ZIP archives built only from the resolved registered capsule directory. No API accepts arbitrary filesystem paths.

## Honest limitations

Verification proves stored-content integrity relative to the manifest, not factual correctness, completeness of the original evidence, or certification of a decision. Unsigned manifests can be rebuilt by someone with filesystem access. Model digest availability depends on Ollama metadata; missing digests remain `null` and are never fabricated. LLM prose is not deterministic. Local Ed25519 trust is not enterprise PKI, and production key management/HSM integration remains future work.
