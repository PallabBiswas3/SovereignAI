"use client";

export const agentExamples = [
  { title: "Calculator", expected: "Expected result: 1050", prompt: "Use calculate to compute (125 * 8) + 50. Report the tool result." },
  { title: "Unit conversion", expected: "Expected result: 2.5 bar", prompt: "Use convert_units to convert 250 kPa to bar. Report the tool result." },
  { title: "Three-step workflow", expected: "Expected: mean 20, scaled 160, then 1.6 bar", prompt: "Use summarize_numbers on [10, 20, 30]. Then use calculate to multiply the observed mean by 8. Then use convert_units to convert that calculated value from kPa to bar. Report the actual results from all three tools. These are example numbers, not plant measurements." },
];

export default function AgentExamples({ disabled, onChoose }: { disabled: boolean; onChoose: (prompt: string) => void }) {
  return <section className="agentExamples" aria-label="Simple agent workflow examples">
    <h3>Try a simple agent workflow</h3>
    <p>Choose an example, review it, then send. Follow actual tool calls below the conversation. Requires a local model and tool permission.</p>
    <div>{agentExamples.map((example) => <button type="button" key={example.title} disabled={disabled} onClick={() => onChoose(example.prompt)}>
      <strong>{example.title}</strong><small>{example.expected}</small>
    </button>)}</div>
  </section>;
}
