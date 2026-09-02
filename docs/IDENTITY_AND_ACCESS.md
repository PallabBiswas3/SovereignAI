# Identity and access architecture

SovereignAI Batch 4 adds a local, organization-aware security boundary. The domain model is generic; APEL is only one synthetic seed.

## Identity flow

`IdentityProvider` is the provider seam. The current `LocalIdentityProvider` verifies versioned salted PBKDF2-HMAC-SHA256 password hashes and creates opaque, random server-side sessions. Only a SHA-256 hash of the session token is stored. Sessions expire, may be revoked, and stop resolving when a user is disabled.

In `local` mode the browser receives:

- an HttpOnly, SameSite=Strict session cookie;
- a separate SameSite=Strict CSRF cookie;
- no role, clearance, user, department, or workspace authority in request payloads.

Every unsafe cookie-authenticated request must echo the CSRF token in `X-CSRF-Token`. Login is the only unsafe pre-session exception. Logout revokes the server-side record and clears both cookies. Invalid email, password, unknown account, and disabled-account login attempts return the same public failure.

`SOVEREIGN_AUTH_MODE=disabled` is an explicit compatibility mode for development and tests. It resolves one bounded development principal. Configuration fails closed if disabled authentication is selected with `SOVEREIGN_ENVIRONMENT=production`.

## Principal and authorization

The server constructs `Principal` from the authenticated user record plus the role-permission registry in `config/access.yaml`. A principal contains organization, departments, workspaces, roles, clearance, effective permissions, and the server-side session identifier.

`AuthorizationService` centralizes stable permissions and contextual resource checks:

- organization must match;
- clearance must meet `PUBLIC < INTERNAL < CONFIDENTIAL < RESTRICTED`;
- workspace must match;
- department must match unless an explicit cross-department permission or ACL grant applies;
- explicit users, roles, and ownership can grant the bounded resource scope;
- the required action permission must still be present.

The six roles are `USER`, `ENGINEER`, `APPROVER`, `MANAGER`, `AUDITOR`, and `ADMIN`. Roles are permission bundles, not complete access decisions. In particular:

- `AUDITOR` can read audit/capsule evidence within scope but cannot execute tools, administer users, or approve business actions.
- `ADMIN` can manage platform users/workcells but is not automatically a business `APPROVER`.
- an approver cannot approve an action they requested, even if they possess both roles.

## Resource security

Tasks, task-stream tracking IDs, knowledge documents, artifacts, Evidence Capsules, approvals, and audit events persist the relevant principal/organization/workspace/department/classification fields. Artifact and capsule list/download endpoints filter before returning metadata. Unauthorized resource reads use generic not-found responses to avoid confirming hidden resource existence.

Tool execution is the intersection of:

1. user `tool.execute` authorization;
2. Workcell-declared tool capability where applicable;
3. the global `ActionGuard` configuration;
4. registered-tool argument validation.

A Workcell cannot grant a permission absent from the user or global policy.

## Retrieval security invariant

`KnowledgeIngestionService` requires a typed `DocumentACL` in secure flows. New secure documents cannot silently fall back to broad visibility.

For authenticated retrieval, ACL exclusion happens before:

1. vector parsing and dense scoring;
2. BM25 tokenization and sparse scoring;
3. reciprocal-rank fusion;
4. reranking;
5. context compilation;
6. model/tool observation.

The hybrid cache key uses a deterministic effective-access fingerprint containing organization, user, clearance, cross-department state, departments, workspaces, and roles. It never contains session tokens or passwords. Tests use a restricted Finance canary and assert it is absent from candidates, reranker input, final context, and narrower-user cache reads.

## Approval integrity

Approval creation stores the authenticated requester and a canonical SHA-256 hash over the exact tool name and arguments. Approval decision derives the approver from the session; browser `decided_by` is ignored in local mode. Immediately before execution, SovereignAI rechecks:

- approver authority and requester/approver separation;
- requester identity is still enabled and authorized for the tool;
- exact action hash is unchanged;
- registered-tool schema;
- current global action policy.

Approval never overrides a disabled tool. Mutated arguments produce `APPROVAL_ARGUMENTS_CHANGED` and are not executed.

## Expected denial codes

Stable codes include `AUTHENTICATION_REQUIRED`, `SESSION_EXPIRED`, `ACCESS_DENIED`, `INSUFFICIENT_CLEARANCE`, `DEPARTMENT_SCOPE_MISMATCH`, `WORKSPACE_SCOPE_MISMATCH`, `DOCUMENT_ACCESS_DENIED`, `ARTIFACT_ACCESS_DENIED`, `CAPSULE_ACCESS_DENIED`, `WORKCELL_ACCESS_DENIED`, `TOOL_ACCESS_DENIED`, `APPROVER_SEPARATION_REQUIRED`, and `ACCESS_SCOPE_REQUIRED`.

## Deployment notes

For HTTPS deployments set `SOVEREIGN_AUTH_COOKIE_SECURE=true`. SQLite remains appropriate for this single-node prototype, but production identity operations should additionally use TLS termination, encrypted storage, key management, rate limiting, operational session cleanup, and formal security review.
