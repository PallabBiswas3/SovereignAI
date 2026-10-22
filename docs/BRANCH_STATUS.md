# Branch status — 22 October 2026

The repository uses two active branches:

| Branch | Purpose |
| --- | --- |
| `main` | Published application baseline at `ee004d5`: workflow selection, structured plans, saved tasks, safe resume, clarification inputs, CPU voice, and maintenance reports. |
| `development` | Application baseline plus retained Process/Wind diagnostic research and the separate v1 security/verification work. Integration and research acceptance remain pending before promotion to `main`. |

Before consolidation there were 15 local branches and 13 remote branches
(18 distinct branch names). A commit saves work on its branch; it does not merge
or delete the branch. Several older branches were integrated through squash or
replacement commits, so Git ancestry alone did not mark them as merged.

## Integrated development work

- `checkpoint/research-and-application-20261022` / `research/diagnostic-guards-review`:
  original application work and Process/Wind research, including its limitations.
- `security/v1-remediation` / `verification/v1-final`: scoped raw-file access,
  approval/audit ownership, authenticated GraphRAG hops, legacy-chat release
  checks, session revocation handling, Windows path handling, and regression tests.
- `main`: portable fixtures, fresh-database test isolation, and published features.

Conflict resolution retains the current workflow selector and numeric tools,
principal-scoped workspaces, receipt-aware approvals, and the authenticated stream
helper used by both new and resumed tasks. A circular import introduced by
combining capsule path handling and workflow routing is resolved with a deferred
helper import.

## Retired experiment branches

Older benchmark, factuality v2/v3, integration-attribution, RAG optimization,
and runtime-latency branches are historical experiment snapshots. Their commit
histories are retained by a history-only merge into `development`; their old trees
are not overlaid on the newer application. This avoids restoring superseded
runtime implementations, deleted datasets, or old generated benchmark outputs.
Archive tags under `archive/20261022/` preserve the remaining divergent snapshot
tips for inspection and future porting. These tags are not active branches.

The original branch names and commit IDs are recorded in the consolidation commit
message. A complete pre-consolidation Git bundle and ref manifest are also kept
locally in `local_testing/branch-consolidation-20261022/`.

## Remaining development work

Consolidation checks: the broad Python run passed 366 cases and exposed one old
artifact-path fixture assumption. After correcting that fixture, all 117 cases
in the affected structured-plan, workflow-selector, file-isolation and approval
suites passed, covering all 367 collected cases across the two runs. All 41
frontend cases and three GraphRAG authentication cases passed. The frontend
production build with TypeScript checking and the GraphRAG TypeScript build
passed. Test runs used hash embeddings and disabled model caching; these are
development regression checks, not a new live-model performance measurement.

1. Re-run the authenticated industrial workflow against the combined runtime.
   The former application's and security branch's live results are historical
   evidence for their separate configurations, not acceptance of this combination.
2. Verify principal-scoped uploads, task-input continuations, file receipts,
   approval execution, and resume together. Their cross-feature behavior needs
   live authenticated validation before promotion.
3. Complete the documented Process/Wind research evaluations; synthetic detector
   sensitivity and healthy false-alert rates remain release limitations.
4. Review archived optimization experiments before porting any additional behavior;
   their historical measurements do not establish the current runtime's latency.

Keep experimental work on `development`. Promote only reviewed, validated changes
to `main`. No plant deployment or new production-readiness claim is implied by
this branch cleanup.
