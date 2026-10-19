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
