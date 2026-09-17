# SovereignAI 2.0 Batch 6 — Reusable organization onboarding

Last verified: 3 September 2026

## Outcome

SovereignAI can now onboard a new organization from a versioned Organization Pack without changing Python source code. The pack describes the organization hierarchy, departments, workspaces, users, access rules, assets, approved documents, and optional organization-scoped Workcells.

This capability is an onboarding control plane, not a claim that an unknown company's identity provider or plant systems are already connected. Those integrations require site-approved endpoints, certificates, mappings, credentials, and acceptance tests.

## Implemented controls

- Strict Pydantic schemas reject unknown fields and plaintext passwords.
- The loader rejects hidden files, symbolic links, path traversal, oversized content, ambiguous aliases, broken references, and unsupported schema versions.
- Dry-run validation reports the pack fingerprint, object counts, required secret names, warnings, errors, and database collisions without mutating organization data.
- Actual import is transactional, additive, idempotent, organization-bound, and audited.
- Passwords are supplied through environment variables; existing hashes are preserved unless rotation is explicitly requested.
- Document access rules are applied before ingestion and retained on documents and chunks.
- Asset-to-evidence relationships are explicit rather than inferred from filenames.
- Organization-scoped Workcells pass the existing trust, schema, handler, tool-policy, and DAG validation gates.
- Authenticated Workcell discovery and task resolution filter organization-specific packs by tenant.
- The admin onboarding API permits same-organization validation and update only. Bootstrapping a new organization remains a local CLI administrator operation.
- Inspection retrieval can be restricted to the exact supplied source documents, preventing unrelated tenant/demo evidence from entering a workflow.
- Organization import never enables plant commands.

## Operator surfaces

- Example pack: `organizations/example-industrial`
- CLI: `scripts/import_organization.py`
- Admin UI: `/admin/onboarding`
- Operator guide: [ORGANIZATION_PACKS.md](ORGANIZATION_PACKS.md)
- API: `/api/admin/organization-packs`

## Verification

The complete backend suite collected 135 tests and passed with exit code 0 (134 passed and one expected Windows symbolic-link skip). Frontend TypeScript checking and the optimized Next.js production build also passed. Focused tests cover schema validation, secret handling, traversal and alias rejection, dry-run immutability, idempotency, ACL persistence, asset links, audit attribution, cross-organization collision rejection, Workcell tenant visibility, and admin API authorization.

## Deliberate production boundaries

- AD, LDAP, and OIDC accounts may be declared as externally managed, but a real identity adapter must be commissioned for the target company.
- OPC-UA, historian, SAP PM, Maximo, and document-management connectors remain site-specific, read-only integration projects.
- One deployment and database per company is recommended for the strongest sovereign boundary.
- Production deployment still requires approved TLS, encryption and key management, backup and recovery, signed packages/capsules, audit retention, malware scanning, and site security accreditation.
- Any future plant-write capability is a separate safety-certified program and is not enabled by Organization Packs.
