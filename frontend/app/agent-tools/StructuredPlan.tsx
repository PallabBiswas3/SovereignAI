export type StructuredPlanInfo = {
  status?: string; revisions?: number;
  steps?: { id: string; tool: string; arguments: Record<string, unknown>; depends_on: string[]; status: string }[];
};

export default function StructuredPlan({ plan }: { plan?: StructuredPlanInfo | null }) {
  if (!plan?.status) return null;
  return <section className="agentTrace" aria-label="Structured execution plan">
    <h3>Structured plan: {plan.status.replaceAll("_", " ")}</h3>
    <p>Revisions: {plan.revisions ?? 0}. Tool results and permissions are shown in the execution trace.</p>
    <ol>{plan.steps?.map((step) => <li key={step.id}>
      <strong>{step.id}: {step.tool}</strong> — {step.status.replaceAll("_", " ")}
      <p>Depends on: {step.depends_on.join(", ") || "Request inputs"}</p>
      <details><summary>Planned arguments and output references</summary><pre>{JSON.stringify(step.arguments, null, 2)}</pre></details>
    </li>)}</ol>
  </section>;
}
