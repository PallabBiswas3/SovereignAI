const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup: render } = require('react-dom/server');

function load(name) {
  const filename = path.resolve(__dirname, `../app/agent-tools/${name}.tsx`);
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true,
  } }).outputText;
  const component = new Module(filename);
  component.filename = filename;
  component.paths = Module._nodeModulePaths(path.dirname(filename));
  component._compile(compiled, filename);
  return component.exports;
}
const { default: Examples, agentExamples } = load('AgentExamples');
const { default: Trace } = load('AgentTrace');
const { default: RuntimeTiming } = load('RuntimeTiming');
const { default: WorkflowSelection } = load('WorkflowSelection');
const { default: StructuredPlan } = load('StructuredPlan');
const { default: TaskLifecycle, taskOutcomeEvents } = load('TaskLifecycle');
const { default: TaskInputReply } = load('TaskInputReply');

test('task outcome distinguishes a paused plan from completion and shows its approval reference', () => {
  const html = render(React.createElement(TaskLifecycle, { lifecycle: {
    schema_version: 1, status: 'waiting_for_approval', terminal: false, execution_complete: false,
    completed_step_ids: ['read'], remaining_step_ids: ['write', 'report'], current_step_id: 'write',
    failure_reason: null, waiting_reason: 'Approve the proposed write action.', required_inputs: [],
    approval_ids: ['approval-1'], response_release: 'waiting_for_approval',
    completion_checks: [{ code: 'all_steps_completed', passed: false, detail: 'Every planned step must complete.' }],
  } }));
  for (const value of ['waiting for approval', '1 steps completed; 2 remaining', 'approval-1', 'Unmet:', 'Approve the proposed write']) assert.ok(html.includes(value));
  assert.ok(!html.includes('Task: completed'));
  for (const value of ['task_completed', 'task_paused', 'task_blocked', 'task_failed', 'task_cancelled']) assert.ok(taskOutcomeEvents.includes(value));
});

test('task outcome shows response review separately and escapes failure details', () => {
  const html = render(React.createElement(TaskLifecycle, { lifecycle: {
    schema_version: 1, status: 'failed', terminal: true, execution_complete: false,
    completed_step_ids: [], remaining_step_ids: ['read'], current_step_id: 'read',
    failure_reason: '<script>failure</script>', waiting_reason: null, required_inputs: [],
    approval_ids: [], response_release: 'needs_revision', completion_checks: [],
  } }));
  assert.match(html, /needs revision/); assert.match(html, /checked separately/);
  assert.ok(!html.includes('<script>')); assert.match(html, /&lt;script&gt;/);
  assert.equal(render(React.createElement(TaskLifecycle, {})), '');
});

test('interrupted checkpoints retain progress without claiming safe resume', () => {
  const html = render(React.createElement(TaskLifecycle, { lifecycle: {
    schema_version: 1, status: 'interrupted', terminal: false, execution_complete: false,
    completed_step_ids: ['calc'], remaining_step_ids: ['report'], current_step_id: 'report',
    failure_reason: null, waiting_reason: 'Worker stopped.', required_inputs: [],
    approval_ids: [], response_release: 'not_evaluated', completion_checks: [],
  }, checkpoint: { schema_version: 1, revision: 7, boundary: 'worker_interrupted', phase: 'interrupted',
    saved_at: '2026-10-21T12:00:00Z', requires_reconciliation: true, interruption_reason: '<script>stopped</script>' } }));
  for (const value of ['Task: interrupted', 'Saved checkpoint 7', '1 steps completed; 1 remaining', 'Review its outcome before resuming']) assert.ok(html.includes(value));
  assert.ok(!html.includes('<script>'));
  assert.ok(taskOutcomeEvents.includes('task_interrupted') && taskOutcomeEvents.includes('task_snapshot'));
});

test('uncertain tool outcomes are displayed as unknown rather than confirmed success or failure', () => {
  const html = render(React.createElement(Trace, { events: [], records: [
    { tool: 'generate_docx', success: null, execution_status: 'unknown' },
    { tool: 'calculate', execution_status: 'in_flight' },
  ] }));
  assert.match(html, /Outcome unknown; review required/);
  assert.match(html, /Running; result not yet confirmed/);
  assert.ok(!html.includes('Succeeded') && !html.includes('Failed or denied'));
});

test('examples expose expected results and buttons never submit the form', () => {
  const html = render(React.createElement(Examples, { disabled: false, onChoose: () => {} }));
  assert.equal((html.match(/type="button"/g) ?? []).length, 3);
  assert.match(html, /1050/); assert.match(html, /2.5 bar/); assert.match(html, /1.6 bar/);
  assert.match(html, /review it, then send/);
  assert.equal(agentExamples.length, 3);
  for (const tool of ['summarize_numbers', 'calculate', 'convert_units']) assert.ok(agentExamples[2].prompt.includes(tool));
});

test('choosing an example provides the editable prompt only', () => {
  const chosen = [];
  const element = Examples({ disabled: false, onChoose: prompt => chosen.push(prompt) });
  const buttons = element.props.children[2].props.children;
  buttons[2].props.onClick();
  assert.deepEqual(chosen, [agentExamples[2].prompt]);
});

test('examples can be disabled during recording, work, or insufficient permission', () => {
  const html = render(React.createElement(Examples, { disabled: true, onChoose: () => {} }));
  assert.equal((html.match(/disabled=""/g) ?? []).length, 3);
});

test('trace renders actual inputs, outputs, timing and validation after completion', () => {
  const html = render(React.createElement(Trace, {
    events: [{ type: 'agent_verified', timestamp: 'now', payload: { completed: true } }], status: 'completed',
    records: [{ tool: 'calculate', arguments: { expression: '20*8' }, success: true, duration_ms: 0.5,
      output_summary: 'result: 160', validation: 'Finite arithmetic result' }],
  }));
  for (const value of ['calculate', '20*8', 'result: 160', '0.5 ms', 'Finite arithmetic result', 'Succeeded']) assert.ok(html.includes(value));
  assert.match(html, /grounding and policy checks/);
});

test('failed and held calls are displayed honestly and output text is escaped', () => {
  const html = render(React.createElement(Trace, { events: [], records: [
    { tool: 'calculate', success: false, error: '<img src=x onerror=alert(1)>' },
    { tool: 'write_file', waiting_for_approval: true },
  ] }));
  assert.match(html, /Failed or denied/); assert.match(html, /Approval required/);
  assert.ok(!html.includes('<img')); assert.ok(!html.includes('Succeeded'));
});

test('live trace shows a running tool before any completed result exists', () => {
  const html = render(React.createElement(Trace, { records: [], events: [
    { type: 'tool_started', timestamp: 'now', payload: { tool: 'calculate', arguments: { expression: '1+2' } } },
  ] }));
  assert.match(html, /Running/); assert.match(html, /1\+2/); assert.ok(!html.includes('Succeeded'));
});

test('incomplete verification never claims completion and empty trace is hidden', () => {
  const html = render(React.createElement(Trace, { records: [], events: [
    { type: 'agent_verified', timestamp: 'now', payload: { completed: false } },
  ] }));
  assert.match(html, /incomplete or awaiting approval/);
  assert.equal(render(React.createElement(Trace, { records: [], events: [] })), '');
});

test('timing report exposes stages and every call without adding overlapping totals', () => {
  const html = render(React.createElement(RuntimeTiming, { metrics: { timing: {
    schema_version: 1, task_handler_seconds: 89.8, llm_call_count: 2,
    llm_wall_seconds: 79, tool_seconds: .47, stages_seconds: { execution: 79.5, grounding_and_governance: 10 },
    llm_calls: [1, 2].map(call => ({ call, model: 'test', purpose: `tool_decision_${call}`, provider: 'OllamaProvider',
      outcome: 'completed', wall_seconds: 39.5, prompt_token_count: 120, token_count: 30,
      prompt_processing_seconds: .25, generation_seconds: 1.5, tokens_per_second: 20, warm_status: 'warm' })),
    scope: 'Handler only', coverage: 'Local calls',
  } } }));
  assert.match(html, /89.800 s/);
  assert.match(html, /Call 1: test/); assert.match(html, /Call 2: test/);
  assert.match(html, /Prompt tokens/); assert.match(html, /Token generation/);
  assert.match(html, /0.250 s/); assert.match(html, /1.500 s/);
  assert.match(html, /must not be added/); assert.match(html, /Unavailable/);
});

test('old runs without timing remain readable', () => {
  for (const metrics of [undefined, {}, { timing: { schema_version: 2, llm_calls: [] } }]) {
    assert.equal(render(React.createElement(RuntimeTiming, { metrics })), '');
  }
});

test('timing identifies direct formatting without claiming that all model calls disappeared', () => {
  const html = render(React.createElement(RuntimeTiming, { metrics: {
    response_generation: { mode: 'verified_formatter' }, timing: { schema_version: 1, llm_call_count: 1, llm_calls: [] },
  } }));
  assert.match(html, /without an answer-generation model call/);
  assert.match(html, /1 model calls/);
});

test('timing report escapes model names and distinguishes missing values from measured zero', () => {
  const html = render(React.createElement(RuntimeTiming, { metrics: { timing: {
    schema_version: 1, llm_calls: [{ call: 1, model: '<script>bad()</script>', outcome: 'fallback', queue_wait_seconds: 0 }],
  } } }));
  assert.ok(!html.includes('<script>'));
  assert.match(html, /fallback/); assert.match(html, /0.000 s/); assert.match(html, /Unavailable/);
});

test('stopped tasks identify partial timing reports', () => {
  const html = render(React.createElement(RuntimeTiming, { metrics: { timing: {
    schema_version: 1, partial: true, llm_calls: [],
  } } }));
  assert.match(html, /Partial timing report/);
  assert.match(html, /stopped before final checks/);
});

const selectedWorkflow = {
  strategy: 'DIRECT', handler: 'direct_answer', rule: 'conversational_response',
  reason: 'Answer the question directly.', domain: 'general', capabilities: ['answer_generation'],
  services: [], requires_validated_input: false, planning_candidate: false, selection_seconds: .000012,
};

test('workflow choice is hidden for old runs and explains current execution', () => {
  assert.equal(render(React.createElement(WorkflowSelection)), '');
  for (const [strategy, label] of [['DIRECT', 'Direct answer'], ['RETRIEVAL', 'Authorized evidence retrieval'],
    ['WORKFLOW', 'Established workflow'], ['BLOCKED', 'Execution blocked']]) {
    const html = render(React.createElement(WorkflowSelection, { selection: { ...selectedWorkflow, strategy } }));
    assert.ok(html.includes(label)); assert.match(html, /0.012 ms without an LLM call/);
  }
});

test('planning candidates show the actual adaptive executor and missing inputs', () => {
  const html = render(React.createElement(WorkflowSelection, { selection: {
    ...selectedWorkflow, strategy: 'ADAPTIVE', handler: 'tool_agent', planning_candidate: true,
    fallback_reason: 'Structured-plan execution is not available yet; the bounded adaptive agent executes this task.',
    requires_validated_input: true, services: ['diagnostic-service'],
  } }));
  assert.match(html, /Selected workflow: Adaptive tool agent/);
  assert.match(html, /Structured-plan execution is not available yet/);
  assert.match(html, /Validated diagnostic input is required/);
  assert.match(html, /Selected services: diagnostic-service/);
  assert.ok(!html.includes('Selected workflow: Structured plan'));
});

test('workflow metadata is escaped and selection does not claim successful execution', () => {
  const html = render(React.createElement(WorkflowSelection, { selection: {
    ...selectedWorkflow, reason: '<script>bad()</script>', services: ['<img src=x>'],
  } }));
  assert.ok(!html.includes('<script>')); assert.ok(!html.includes('<img'));
  assert.ok(!html.includes('Succeeded')); assert.ok(!html.includes('Approved'));
});

test('structured plans display dependencies and actual completion statuses', () => {
  const html = render(React.createElement(StructuredPlan, { plan: { status: 'completed', revisions: 1, steps: [
    { id: 'read', tool: 'read_file', arguments: { path: 'input.txt' }, depends_on: [], status: 'completed' },
    { id: 'report', tool: 'generate_docx', arguments: { content: { $ref: 'read.text' } }, depends_on: ['read'], status: 'completed' },
  ] } }));
  assert.match(html, /Structured plan: completed/); assert.match(html, /Revisions: 1/);
  assert.match(html, /Depends on: read/); assert.match(html, /read.text/);
  assert.match(html, /Request inputs/); assert.match(html, /generate_docx/);
});

test('pending and stopped plans never claim that steps succeeded; old runs hide the panel', () => {
  assert.equal(render(React.createElement(StructuredPlan)), '');
  assert.equal(render(React.createElement(StructuredPlan, { plan: {} })), '');
  const html = render(React.createElement(StructuredPlan, { plan: { status: 'stopped', steps: [
    { id: 'report', tool: '<script>bad()</script>', arguments: { text: '<img src=x>' }, depends_on: ['read'], status: 'waiting_for_approval' },
  ] } }));
  assert.match(html, /stopped/); assert.match(html, /waiting for approval/);
  assert.ok(!html.includes('<script>')); assert.ok(!html.includes('<img'));
  assert.ok(!html.includes('completed')); assert.ok(!html.includes('Succeeded'));
});


test('reconciled actions explain that the previous result was verified without replay', () => {
  const html = render(React.createElement(Trace, { events: [], records: [
    { tool: 'write_file', success: true, execution_status: 'succeeded', reconciliation: { type: 'approval', id: 'saved-approval' } },
  ] }));
  assert.match(html, /Previous execution verified from its saved result/);
  assert.match(html, /not repeated on resume/);
  assert.match(html, /Succeeded/);
  assert.ok(!html.includes('Outcome unknown'));
});


test('missing-input form renders only declared questions with accessible labels and bounds', () => {
  const html = render(React.createElement(TaskInputReply, { taskId: 'saved-task', revision: 7, disabled: false,
    questions: [{ key: 'asset_id', prompt: 'Which asset should I inspect?', kind: 'text' },
      { key: 'diagnostic_input', prompt: 'Attach diagnostic JSON', kind: 'attachment' }],
    apiFetch: () => {}, onBusy: () => {}, onContinue: () => {} }));
  assert.match(html, /Continue this task/);
  assert.match(html, /Which asset should I inspect/);
  assert.match(html, /for="task-input-asset_id"/);
  assert.match(html, /maxLength="4000"/i);
  assert.match(html, /type="file"/);
  assert.match(html, /accept=".json"/);
  assert.match(html, /Submit inputs and continue/);
  assert.ok(!html.includes('Run task'));
});

test('missing-input form is hidden without questions and disabled while other work runs', () => {
  const props = { taskId: 'saved-task', revision: 1, apiFetch: () => {}, onBusy: () => {}, onContinue: () => {} };
  assert.equal(render(React.createElement(TaskInputReply, { ...props, questions: [], disabled: false })), '');
  const html = render(React.createElement(TaskInputReply, { ...props, disabled: true,
    questions: [{ key: 'answer', prompt: '<script>unsafe</script>', kind: 'text' }] }));
  assert.match(html, /fieldset disabled/);
  assert.ok(!html.includes('<script>'));
  assert.match(html, /&lt;script&gt;/);
});
