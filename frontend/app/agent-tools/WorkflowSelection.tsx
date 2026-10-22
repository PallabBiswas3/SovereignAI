export type WorkflowSelectionInfo = {
  strategy: string; handler: string; rule: string; reason: string; domain: string;
  capabilities: string[]; services: string[]; requires_validated_input: boolean;
  planning_candidate: boolean; fallback_reason?: string | null; selection_seconds: number;
};

const labels: Record<string, string> = {
  DIRECT: "Direct answer", RETRIEVAL: "Authorized evidence retrieval", WORKFLOW: "Established workflow",
  PLANNED: "Structured plan", ADAPTIVE: "Adaptive tool agent", BLOCKED: "Execution blocked",
};

export default function WorkflowSelection({ selection }: { selection?: WorkflowSelectionInfo | null }) {
  if (!selection) return null;
  return <section className="agentTrace" aria-label="Selected execution workflow">
    <h3>Selected workflow: {labels[selection.strategy] ?? selection.strategy}</h3>
    <p>{selection.reason}</p>
    {selection.services.length > 0 && <p>Selected services: {selection.services.join(", ")}</p>}
    {selection.requires_validated_input && <p className="traceWarning">Validated diagnostic input is required before diagnosis can run.</p>}
    {selection.fallback_reason && <p>{selection.fallback_reason}</p>}
    <details><summary>Selection details</summary>
      <p>Domain: {selection.domain} · Handler: {selection.handler} · Rule: {selection.rule}</p>
      <p>Capabilities: {selection.capabilities.join(", ")}</p>
      <p>Selection took {(selection.selection_seconds * 1000).toFixed(3)} ms without an LLM call.</p>
    </details>
  </section>;
}
