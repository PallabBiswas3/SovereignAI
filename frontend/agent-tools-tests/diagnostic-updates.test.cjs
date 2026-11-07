const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');

const filename = path.resolve(__dirname, '../app/integrations/DiagnosticUpdates.tsx');
const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const component = new Module(filename);
component.filename = filename;
component.paths = Module._nodeModulePaths(path.dirname(filename));
component._compile(compiled, filename);
const View = component.exports.default;
const render = (updates) => renderToStaticMarkup(React.createElement(View, { updates }));

test('no selected update has no supplementary result panel', () => {
  assert.equal(render(undefined), '');
  assert.equal(render([]), '');
});

test('ambiguous classification shows hold, prediction set and synthetic provenance', () => {
  const html = render([{ model_ref: 'sequence-test', version: 'v1', method: 'sequence', status: 'abstained',
    summary: '<script>ambiguous</script>', calibrated: true, prediction_set: ['normal', 'fault'], synthetic_artifact: true }]);
  for (const text of ['withheld', 'normal, fault', 'Synthetic demonstration model', 'not an individual correctness probability', 'sequence-test']) assert.ok(html.includes(text));
  assert.ok(!html.includes('<script>'));
});

test('zero-cycle forecast and uncalibrated interval remain visible without invented certainty', () => {
  const html = render([{ method: 'battery-pf', summary: 'Capacity forecast', status: 'completed',
    point_estimate: 0, interval: [0, 3], unit: 'cycles', calibrated: false, details: { policy_hold_remains: true } }]);
  assert.ok(html.includes('0.00'));
  assert.ok(html.includes('3.00'));
  assert.ok(html.includes('model-based or not calibrated'));
  assert.ok(html.includes('policy still requires review'));
});

test('unscorable event never renders a negative alarm as healthy', () => {
  const html = render([{ method: 'wind-event', summary: 'Unscorable event', status: 'abstained', details: { alarm: null } }]);
  assert.ok(html.includes('withheld'));
  assert.ok(!html.includes('Additional alarm:'));
});
