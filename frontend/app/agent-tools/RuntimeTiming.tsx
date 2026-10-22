type Call = {
  call: number; model: string; provider: string; purpose: string; outcome: string;
  wall_seconds?: number; prompt_token_count?: number; token_count?: number;
  prompt_processing_seconds?: number; generation_seconds?: number;
  load_duration_seconds?: number; queue_wait_seconds?: number;
  time_to_first_token_seconds?: number; tokens_per_second?: number; warm_status: string;
};
type Timing = {
  schema_version: number; task_handler_seconds: number; llm_call_count: number;
  llm_wall_seconds: number; tool_seconds: number; stages_seconds: Record<string, number>;
  llm_calls: Call[]; scope: string; coverage: string; partial?: boolean;
};
const seconds = (value: unknown) => typeof value === "number" && Number.isFinite(value) && value >= 0 ? `${value.toFixed(3)} s` : "Unavailable";
const count = (value: unknown) => typeof value === "number" && Number.isFinite(value) && value >= 0 ? String(value) : "Unavailable";

export default function RuntimeTiming({ metrics }: { metrics?: Record<string, unknown> }) {
  const timing = metrics?.timing as Timing | undefined;
  const response = metrics?.response_generation as { mode?: string } | undefined;
  if (!timing || timing.schema_version !== 1 || !Array.isArray(timing.llm_calls)) return null;
  const stages = [
    ["Input checks", "input_checks"], ["Execution", "execution"],
    ["Grounding and governance", "grounding_and_governance"], ["Persistence and audit", "persistence_and_audit"],
  ];
  return <section className="agentTrace runtimeTiming" aria-label="Task timing breakdown">
    <h3>Where the time went</h3>
    {response?.mode === "verified_formatter" && <p>Completed tool results were formatted directly, without an answer-generation model call.</p>}
    {timing.partial && <p>Partial timing report: the task stopped before final checks completed.</p>}
    <p>Measured handler: <strong>{seconds(timing.task_handler_seconds)}</strong> · {count(timing.llm_call_count)} model calls · model calls {seconds(timing.llm_wall_seconds)} · recorded tools {seconds(timing.tool_seconds)}</p>
    <table><caption>Task stages</caption><tbody>{stages.map(([label, key]) => <tr key={key}><th scope="row">{label}</th><td>{seconds(timing.stages_seconds?.[key])}</td></tr>)}</tbody></table>
    <p>Model and tool times are included within execution; these figures must not be added to the stage totals. Concurrent call durations can overlap.</p>
    {timing.llm_calls.map((call) => <details key={call.call}>
      <summary>Call {call.call}: {call.model} · {call.purpose} · {seconds(call.wall_seconds)} · {call.outcome}</summary>
      <table><caption>Call {call.call} measurements ({call.provider}, {call.warm_status})</caption><tbody>
        <tr><th scope="row">Prompt tokens</th><td>{count(call.prompt_token_count)}</td></tr>
        <tr><th scope="row">Generated tokens</th><td>{count(call.token_count)}</td></tr>
        <tr><th scope="row">Prompt processing</th><td>{seconds(call.prompt_processing_seconds)}</td></tr>
        <tr><th scope="row">Token generation</th><td>{seconds(call.generation_seconds)}</td></tr>
        <tr><th scope="row">Generation rate (tokens/s)</th><td>{count(call.tokens_per_second)}</td></tr>
        <tr><th scope="row">Model loading</th><td>{seconds(call.load_duration_seconds)}</td></tr>
        <tr><th scope="row">Queue wait</th><td>{seconds(call.queue_wait_seconds)}</td></tr>
        <tr><th scope="row">Time to first token</th><td>{seconds(call.time_to_first_token_seconds)}</td></tr>
      </tbody></table>
    </details>)}
    <p>{timing.scope}</p><p>{timing.coverage}</p>
    <p>Unavailable means the provider did not expose that measurement. Structured responses do not measure time to first token.</p>
  </section>;
}
