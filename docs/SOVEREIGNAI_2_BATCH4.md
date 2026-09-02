# SovereignAI 2.0 Batch 4 — Organizational intelligence, identity, and access control

Batch 4 is implemented as an authorization boundary around the existing local inference, retrieval, Workcell, artifact, capsule, approval, audit, and SSE systems. It does not start Batch 5 and does not add LDAP/AD, cloud identity, GraphRAG, PostgreSQL, Kubernetes, or public services.

## Delivered

- Generic organization, department, workspace, user, principal, role, permission, clearance, ACL, and access-decision models.
- Provider-neutral identity seam with a local PBKDF2 password/session provider.
- HttpOnly/SameSite session cookies, server-side expiration/revocation, logout, disabled-user handling, and CSRF validation.
- Central role permissions plus organization/workspace/department/classification/user/role/owner checks.
- Production fail-closed configuration when authentication is disabled.
- Server-derived identity on task, knowledge, tool, Workcell, artifact, capsule, approval, audit, and SSE paths.
- Typed document ACL enforcement before all ranking/model-context stages and permission-safe cache fingerprints.
- Task ownership for synchronous runs and background tracking streams, including access revalidation during SSE delivery.
- Artifact/capsule enumeration and download filtering with generic denial responses.
- Requester/approver separation, canonical exact-action hashes, requester/tool/policy/schema revalidation, and immutable approval execution.
- A deterministic fictional APEL seed/reset system with 20 assets, 55 files, seven users, five scenarios, difficult evidence/security fixtures, and a versioned 50-question evaluation set.
- Local login/logout UI, visible principal context, credentialed API/SSE requests, minimal organization page, and existing Workcell/capsule UI preservation.

## Security invariants verified by tests

1. Disabled and unknown accounts do not reveal account state during login.
2. Raw passwords and session tokens are not stored.
3. Session expiration/revocation and mid-stream invalidation stop authenticated use.
4. Browser-supplied role/user/workspace/department fields cannot enter strict task or tool schemas.
5. Unauthorized Finance canary content never reaches dense candidates, BM25 candidates, fusion, reranker input, compiled context, or a narrow-scope cache hit.
6. Auditor, administrator, approver, engineer, and manager permissions remain distinct.
7. Artifact/capsule/task identifiers do not bypass resource authorization.
8. Approval cannot change exact arguments or override global disabled-tool policy.
9. Requester and approver must be different authenticated principals.
10. APEL generation is deterministic and reset is organization-scoped.

See [IDENTITY_AND_ACCESS.md](IDENTITY_AND_ACCESS.md), [APEL_DEMO_ORGANIZATION.md](APEL_DEMO_ORGANIZATION.md), and [APEL_DEMO_SCENARIOS.md](APEL_DEMO_SCENARIOS.md).

## Limitations

- SQLite and process-local SSE remain single-node prototype choices.
- Uploaded-file ACL persistence is workspace-level rather than a full records-management system.
- Local authentication has no password-reset, MFA, lockout/rate-limit service, SCIM, LDAP, or Active Directory integration.
- Audit records are structured and scoped but are not cryptographically signed or shipped to a SIEM.
- APEL is synthetic and the 50-question set is not a production accuracy certification.
- Reranking uses staged local weights only; no runtime download occurs.
- Role/ACL controls reduce application-layer exposure but do not replace host hardening, encrypted disks, network segmentation, backups, or formal accreditation.

## Validation

The pre-change baseline was 87 passed, 1 skipped. After Batch 4, 104 tests are collected and the complete suite passes with 103 passed and the existing Windows symlink-capability test skipped. Python compilation, frontend TypeScript validation, and the Next.js production build also pass. The two existing warnings are the Starlette TestClient/httpx deprecation and the installed Torch/NumPy ABI warning.
