# Organization Packs

Organization Packs provide a versioned, fail-closed way to onboard a company without changing Python source code. Each company should use its own on-premise SovereignAI deployment and database. The importer does not merge or export data between deployments.

## Layout

```text
organizations/company-name/
|-- organization.yaml
|-- assets.yaml
|-- access_matrix.yaml
|-- users.yaml                 # optional for external identity
|-- documents/
|   |-- manifest.yaml
|   `-- ... approved source files
`-- workcells/                 # optional, organization-scoped packs
    `-- company-workcell/
```

All IDs are stable and globally unique inside one deployment. Prefix plant, area, unit, department, workspace, user, asset, and Workcell IDs with the organization ID. Cross-organization ID, email, and document-checksum collisions are rejected.

Every asset references an `asset_rules` entry. Every document references a `document_rules` entry and declares an approval status. Document ACLs are applied before ingestion and retained on both document and chunk records. Asset links are explicit; the importer does not infer them from filenames.

## Validate first

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\import_organization.py --pack organizations\example-industrial --dry-run
```

Dry-run validates schema version, references, paths, file bounds, Workcells, database collisions, and required secret names. It writes a JSON report under `workspace/onboarding_reports` but does not alter organization data.

## Import

Local development identities reference environment variables; plaintext passwords are rejected from pack YAML:

```powershell
$env:EXAMPLE_INDUSTRIAL_ENGINEER_PASSWORD = "replace-with-a-strong-temporary-password"
$env:EXAMPLE_INDUSTRIAL_ADMIN_PASSWORD = "replace-with-a-different-strong-password"
python scripts\import_organization.py --pack organizations\example-industrial
```

The importer is additive and idempotent. Re-import updates records owned by the same organization and never deletes records omitted from a later pack. Existing local password hashes are preserved unless `--rotate-local-passwords` is explicitly supplied.

## Workcells

Workcells under the pack must declare `organization_id` in `manifest.yaml`, matching the Organization Pack. They pass the normal Workcell loader, handler, tool-policy, schema, DAG, version, and trust validation before installation. API discovery filters organization-scoped Workcells by the authenticated principal's organization.

## External identity and plant connectors

Set `identity_provider: external` and omit `users.yaml` when AD/LDAP/OIDC is authoritative. The current repository defines the provider interface but does not fabricate an external connection; real endpoints, certificates, group mappings, and service accounts require a site-approved adapter. Likewise, historian, OPC-UA, SAP PM, Maximo, and document-management connections remain site-specific read-only adapters. Organization Pack import never enables plant commands.

## Production controls

- Run one deployment per company for strongest sovereign isolation.
- Back up the database and artifact roots before an approved update.
- Review the dry-run report under change control.
- Use production identity, database encryption, TLS, signed Workcells and signed Evidence Capsules.
- Treat the included `example-industrial` pack as fictional scaffolding only.
