# Sovereign Workcell Packs

Workcell Packs are versioned, declarative local workflow definitions. They are not Python plugins and are never imported as executable code. A pack describes what should happen; every executable capability resolves through SovereignAI's trusted `WorkcellHandlerRegistry`.

## Runtime architecture

```text
existing task API / classifier
        -> WorkcellRegistry
        -> WorkcellLoader + WorkcellValidator
        -> WorkcellExecutor
        -> trusted registered handlers
        -> existing inspection, evidence, tool, artifact, audit, and SSE services
```

Workcell routing chooses the workflow. Model routing independently chooses a local model role/instance. Selecting a Workcell never overrides model or tool governance.

## Pack format

Installed packs live under `workcells/`. The official `pump_inspection` pack contains:

```text
manifest.yaml
workflow.yaml
schemas/input.schema.json
schemas/output.schema.json
evidence.yaml
rules.yaml
policy.yaml
prompts/*.txt
artifacts/*.yaml
evaluations/*.yaml
signature.json                 optional
```

Only YAML, JSON, text, and Markdown definition files are accepted in allow-listed locations. Hidden files, arbitrary Python, unknown root files/directories, oversized files, traversal, absolute paths, and symlinks are rejected. YAML uses `safe_load`.

The strict manifest declares ID, semantic version, platform constraint, task classes, execution modes, required/optional tools, risk class, schemas, and entry workflow. Unknown fields are rejected. A deterministic SHA-256 identity covers canonical relative paths and each validated file hash; `signature.json` is excluded from this identity to avoid circular signing.

## Workflow DSL

Batch 3 intentionally implements a small DAG, not a programming language. A step supports:

- a bounded identifier and registered handler name;
- dependencies;
- named input references and declared outputs;
- `always` or `input_present` conditions;
- `stop` or `continue` failure behavior;
- evidence requirement references;
- an approval-required marker.

Validation rejects duplicate step IDs, missing dependencies, cycles, missing terminal steps, unsupported conditions, unknown handlers, missing evidence requirements, and invalid artifact handlers/paths. Execution order is deterministic among simultaneously ready steps.

## Security and effective policy

The authority order is:

```text
system policy > workspace policy > Workcell policy > model proposal
```

The effective permission set is an intersection. A pack may disable a globally enabled tool, but it cannot enable a disabled or unknown tool. `delete_file` and `execute_shell` therefore remain blocked even if declared by a pack. Workcells cannot import code, access network resources, choose arbitrary filesystem roots, or bypass `ActionGuard`, approvals, sandbox controls, classifications, or registered services.

## Trust and signing

Workcell content can be signed with Ed25519. The local `WorkcellTrustStore` maps a `key_id` to an approved public key; private keys are never stored by the application. States are `UNSIGNED`, `SIGNED_UNVERIFIED`, `TRUSTED`, and `INVALID_SIGNATURE` (with `MODIFIED` reserved in the model for future lifecycle reporting).

Development may set `SOVEREIGN_UNSIGNED_WORKCELLS_ALLOWED=true`. A strict deployment can set it to `false`. A signed pack whose key is absent is untrusted; a signature that no longer matches the pack identity is invalid. This local mechanism is not enterprise PKI, an HSM, certificate-chain validation, or formal software certification.

## Adding a Workcell

1. Create only declarative files in a new `workcells/<directory>`.
2. Reference existing trusted handler names. Adding a new executable capability requires an explicit reviewed application-code change and registry entry.
3. Declare the smallest required tool set and reduce policy where possible.
4. Add input/output schemas, evidence requirements, prompts with embedded ID/version, and evaluation cases.
5. Call `POST /api/workcells/{id}/validate` and run focused tests before enabling the pack.
6. Optionally sign the deterministic content hash with an approved offline development/enterprise process.

There is no remote registry or marketplace. Discovery is deterministic and local-only.
