type TraceEvent = { type: string; timestamp: string; payload: Record<string, unknown> };
export type ToolRecord = {
  tool: string; arguments?: Record<string, unknown>; reason_summary?: string;
  success?: boolean; error?: string | null; output_summary?: string; duration_ms?: number;
  validation?: string; waiting_for_approval?: boolean; governance_decision?: string;
  execution_status?: string;
  reconciliation?: { type: string; id: string };
};

export default function AgentTrace({ events, records, status }: { events: TraceEvent[]; records: ToolRecord[]; status?: string }) {
  const live = events.filter((event) => ["tool_proposed", "tool_started", "tool_completed", "tool_approval_required", "agent_verified", "agent_decision_rejected"].includes(event.type));
  if (!live.length && !records.length) return null;
  const verification = live.findLast((event) => event.type === "agent_verified");
  return <section className="agentTrace" aria-label="Agent tool execution trace">
    <h3>Agent workflow</h3>
    <p>Plan → permission check → tool → observation → completion check</p>
    {verification && <p className={verification.payload.completed ? "traceSuccess" : "traceWarning"}>
      {verification.payload.completed ? "Requested tool steps completed" : "Tool workflow incomplete or awaiting approval"}
      {status ? ` · Run: ${status}` : " · Final governance checks pending"}. This checks tool execution; the answer still uses the existing grounding and policy checks.
    </p>}
    {records.length ? records.map((record, index) => <details key={`${record.tool}-${index}`} open>
      <summary>{index + 1}. {record.tool} · {record.waiting_for_approval ? "Approval required" : record.execution_status === "unknown" ? "Outcome unknown; review required" : record.execution_status === "in_flight" ? "Running; result not yet confirmed" : record.success ? "Succeeded" : "Failed or denied"}{record.duration_ms != null ? ` · ${record.duration_ms} ms` : ""}</summary>
      <p>{record.reason_summary}</p><b>Input</b><pre>{JSON.stringify(record.arguments, null, 2)}</pre>
      {record.reconciliation && <p>Previous execution verified from its saved result; this action was not repeated on resume.</p>}
      {record.output_summary && <><b>Observed output</b><pre>{record.output_summary}</pre></>}
      {record.error && <p className="error">{record.error}</p>}
      <p>{record.validation || record.governance_decision}</p>
    </details>) : <ol>{live.filter((event) => event.type !== "agent_verified").map((event, index) => <li key={`${event.timestamp}-${index}`}>
      <strong>{String(event.payload.tool ?? "Model decision")} · {event.type === "tool_proposed" ? "Planned / policy checked" : event.type === "tool_started" ? "Running" : event.type === "tool_approval_required" ? "Awaiting approval" : event.type === "agent_decision_rejected" ? "Rejected; model must retry" : event.payload.success ? "Observed success" : "Failed or denied"}</strong>
      {event.type === "tool_proposed" && event.payload.reason_summary != null && <p>{String(event.payload.reason_summary)}</p>}
      {event.payload.arguments != null && <pre>{JSON.stringify(event.payload.arguments, null, 2)}</pre>}
      {event.payload.output_summary != null && <pre>{String(event.payload.output_summary)}</pre>}
      {event.payload.error != null && <p className="error">{String(event.payload.error)}</p>}
    </li>)}</ol>}
  </section>;
}
