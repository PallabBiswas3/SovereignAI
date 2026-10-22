export type TaskLifecycleInfo = {
  schema_version: number;
  status: string;
  terminal: boolean;
  execution_complete: boolean;
  completed_step_ids: string[];
  remaining_step_ids: string[];
  current_step_id: string | null;
  failure_reason: string | null;
  waiting_reason: string | null;
  required_inputs: { key: string; prompt: string; kind: string }[];
  approval_ids: string[];
  response_release: string;
  completion_checks: { code: string; passed: boolean; detail: string }[];
};

export type ExecutionCheckpointInfo = {
  schema_version: number; revision: number; boundary: string; phase: string;
  saved_at: string | null; requires_reconciliation: boolean; interruption_reason: string | null;
};

export const taskOutcomeEvents = ["task_completed", "task_paused", "task_blocked", "task_failed", "task_cancelled", "task_interrupted", "task_snapshot"];

export default function TaskLifecycle({ lifecycle, checkpoint }: { lifecycle?: TaskLifecycleInfo | null; checkpoint?: ExecutionCheckpointInfo | null }) {
  if (!lifecycle || lifecycle.schema_version !== 1) return null;
  return <section className="agentTrace" aria-label="Task outcome">
    <h3>Task: {lifecycle.status.replaceAll("_", " ")}</h3>
    <p>{lifecycle.completed_step_ids.length} steps completed; {lifecycle.remaining_step_ids.length} remaining.</p>
    <p>Response release: {lifecycle.response_release.replaceAll("_", " ")}. Execution completion and response release are checked separately.</p>
    {lifecycle.waiting_reason && <p>{lifecycle.waiting_reason}</p>}
    {lifecycle.failure_reason && <p>{lifecycle.failure_reason}</p>}
    {lifecycle.approval_ids.length > 0 && <p>Approval references: {lifecycle.approval_ids.join(", ")}</p>}
    {checkpoint?.schema_version === 1 && <div>
      <p>Saved checkpoint {checkpoint.revision}: {checkpoint.boundary.replaceAll("_", " ")} ({checkpoint.phase}).</p>
      {checkpoint.saved_at && <p>Saved at: {checkpoint.saved_at}</p>}
      {checkpoint.interruption_reason && <p>{checkpoint.interruption_reason}</p>}
      {checkpoint.requires_reconciliation && <p>An operation has no confirmed durable result. Review its outcome before resuming.</p>}
    </div>}
    <details><summary>Execution completion checks</summary>
      <ul>{lifecycle.completion_checks.map(check => <li key={check.code}>
        {check.passed ? "Passed" : "Unmet"}: {check.detail}
      </li>)}</ul>
    </details>
  </section>;
}
