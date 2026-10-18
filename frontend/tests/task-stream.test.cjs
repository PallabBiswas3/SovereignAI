const { test } = require('node:test');
const assert = require('node:assert/strict');
const ts = require('typescript');
const fs = require('node:fs');
const vm = require('node:vm');
const compiled = ts.transpileModule(fs.readFileSync('app/task-stream.ts', 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
}).outputText;
const context = { exports: {}, Set, Error, JSON };
vm.runInNewContext(compiled, context);
const { openTaskStream } = context.exports;
class Source {
  constructor(url, options) { this.url = url; this.options = options; this.listeners = {}; Source.last = this; }
  addEventListener(type, receive) { this.listeners[type] = receive; }
  close() { this.closed = true; }
  emit(type, payload = {}) { this.listeners[type]({ data: JSON.stringify({ type, payload }) }); }
}
function connect(receive = () => {}) {
  const failures = [];
  const close = openTaskStream('synthetic', ['task_completed', 'model_token'], receive, m => failures.push(m), Source);
  return { source: Source.last, close, failures };
}
test('CONNECTING errors stop retrying and report incomplete task', () => {
  const { source, failures } = connect();
  source.readyState = 0;
  source.onerror(); source.onerror();
  assert.equal(source.closed, true);
  assert.equal(failures.length, 1);
});
test('revoked session is terminal, credentials included, late events ignored', () => {
  const events = [];
  const { source } = connect(e => events.push(e.type));
  assert.equal(source.options.withCredentials, true);
  source.emit('access_revoked'); source.emit('model_token');
  assert.equal(events.join(','), 'access_revoked');
  assert.equal(source.closed, true);
});
test('malformed and mismatched events fail safely', () => {
  for (const data of ['{', '{"type":"other","payload":{}}']) {
    const { source, failures } = connect();
    source.listeners.model_token({ data });
    assert.equal(source.closed, true);
    assert.equal(failures.length, 1);
  }
});
test('completion and explicit cleanup close streams', () => {
  let received = false;
  const { source, close } = connect(() => { received = true; });
  source.emit('task_completed');
  assert.equal(received, true); assert.equal(source.closed, true);
  close(); source.onerror();
});
