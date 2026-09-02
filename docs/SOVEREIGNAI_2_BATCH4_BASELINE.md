# SovereignAI 2.0 Batch 4 Baseline

Recorded before Batch 4 implementation on 2026-09-02.

## Exact validation results

- Complete backend regression: **87 passed, 1 skipped, 2 warnings** in 35.84 seconds.
- Python compile validation: **passed** (`python -m compileall -q backend tests`).
- Frontend TypeScript validation: **passed** (`npm run typecheck`).
- Next.js production build: **passed** (`npm run build`).
- Static routes: `/`, `/_not-found`, `/metrics`, and `/sovereignty`.

The skip is the Windows symlink-capability test from Batch 3. Existing warnings are the Starlette `TestClient`/`httpx` deprecation and the installed Torch/NumPy ABI warning.

## Security inspection

- Dense and BM25 retrieval already filter candidates before RRF/reranking, but the filter is a prototype intersection of lowercase metadata tokens such as `internal` or `maintenance`. It is not derived from an authenticated Principal and cannot express organization, workspace, clearance, users, roles, or ownership coherently.
- Retrieval and hybrid caches already include normalized scope. Batch 4 must replace that scope with a stable authorization fingerprint; no session secret or password may enter a cache key.
- Knowledge ingestion accepts caller-supplied department/classification without authority validation. Missing scope can become broadly visible through the current `internal` fallback.
- `BoundedToolAgent` and approval execution enforce global `ActionGuard` policy and registered-tool arguments, but do not receive a Principal or user permission decision.
- Workcell validation correctly prevents pack privilege escalation, but task execution does not yet intersect pack permissions with user permissions/workspace scope.
- Approval decisions trust a browser-supplied `decided_by` string. Requester/approver separation and action hashes are absent.
- Task state and event streams have no persisted owner/scope. Any caller knowing an ID can read/cancel/subscribe.
- Artifacts and capsules have integrity metadata but no organization/workspace/classification/owner ACL. List/download/verify endpoints are globally enumerable.
- Audit records are structured and avoid hidden reasoning, but the audit endpoint has no role/scope authorization.
- Files/uploads are shared globally and have no owner/scope metadata.
- No authentication, server-side session, Principal, role assignment, clearance hierarchy, or centralized authorization service exists.

## Implementation plan

1. Add generic typed identity/access models, local identity-provider abstraction, PBKDF2 password hashing, opaque expiring server-side sessions, secure cookie handling, CSRF protection for cookie-authenticated state changes, and centralized FastAPI dependencies. Keep `auth_mode=disabled` explicit for backwards-compatible development tests, and reject disabled auth in production configuration.
2. Add organization/user/session/resource-scope persistence and a centralized `AuthorizationService` that evaluates permission, organization, workspace, department, classification, role/user ACL, ownership, and separation of duties.
3. Persist scope on tasks, documents, artifacts, capsules, approvals, and task-stream tracking. Enforce authorization on route enumeration and access without leaking hidden metadata.
4. Replace retrieval token filtering with typed `DocumentACL` checks before dense/BM25 scoring, RRF, reranking, ContextCompiler, and model input. Key caches by a deterministic effective-access fingerprint and add mixed-corpus/cache-isolation tests.
5. Pass Principal/task/Workcell/tool/resource context through tool and Workcell authorization. Preserve global disabled-tool and Workcell intersection rules.
6. Build a deterministic APEL seed/reset system with fictional users, departments, Plant A, 15–25 assets, 30–60 coherent files, difficult security/evidence cases, scenarios, and a versioned 40–60 question evaluation set.
7. Add the minimum login/user/organization/security UI after backend enforcement is proven.
8. Run focused authentication/RBAC/ACL/cache/tool/Workcell/artifact/capsule/approval/SSE/APEL tests, then the complete regression, compile, typecheck, and production build gates. Stop before Batch 5.

## Compatibility policy

`SOVEREIGN_AUTH_MODE=disabled` is a deliberate development/test compatibility mode and will map to one bounded local development Principal. Local multi-user demonstrations use `SOVEREIGN_AUTH_MODE=local` with the explicit APEL demo seed. Production configuration must not start with authentication disabled. Browser role, department, workspace, user ID, clearance, and permissions are never accepted as authority.
