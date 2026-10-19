const { test, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const filename = path.resolve(__dirname, '../app/voice/recorder.ts');
const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const moduleUnderTest = new Module(filename);
moduleUnderTest._compile(compiled, filename);
const { VoiceRecorder } = moduleUnderTest.exports;
const flush = () => new Promise(resolve => setImmediate(resolve));
const limits = { max_duration_seconds: 30, max_upload_bytes: 1024, timeout_seconds: 90,
  mime_types: ['audio/webm', 'audio/ogg', 'audio/mp4'] };
const controllers = [];
afterEach(() => controllers.splice(0).forEach(c => c.cancel(false)));

function setup({ permission, fetch, supported = true } = {}) {
  let stopped = 0, recordings = 0;
  const track = { stop() { stopped++; }, onended: null };
  const stream = { getTracks: () => [track] };
  class Recorder {
    static isTypeSupported() { return supported; }
    constructor(stream, options) { this.state = 'inactive'; this.mimeType = options.mimeType; recordings++; }
    start() { this.state = 'recording'; }
    stop() {
      this.state = 'inactive';
      queueMicrotask(() => {
        this.ondataavailable?.({ data: new Blob(['audio']) });
        this.onstop?.();
      });
    }
  }
  Object.defineProperty(globalThis, 'isSecureContext', { value: true, configurable: true });
  Object.defineProperty(globalThis, 'navigator', { configurable: true,
    value: { mediaDevices: { getUserMedia: permission ?? (async () => stream) } } });
  globalThis.MediaRecorder = Recorder;
  const states = [], transcripts = [], requests = [];
  const c = new VoiceRecorder({
    state: (state, detail, elapsed) => states.push({ state, detail, elapsed }),
    transcript: text => transcripts.push(text),
    fetch: async (url, init) => {
      requests.push({ url, init });
      if (fetch) return fetch(url, init);
      return Response.json(url.endsWith('/config') ? limits : { text: 'What is the condition of Pump-102?' });
    },
  });
  controllers.push(c);
  return { c, states, transcripts, requests, stream, track,
    get stopped() { return stopped; }, get recordings() { return recordings; } };
}

test('start, elapsed state, stop, local transcription, and explicit review', async () => {
  const x = setup();
  await x.c.start();
  assert.equal(x.states.at(-1).state, 'recording');
  assert.equal(x.states.at(-1).elapsed, 0);
  x.c.stop();
  await flush();
  assert.equal(x.stopped, 1);
  assert.equal(x.states.at(-1).state, 'ready');
  assert.match(x.states.at(-1).detail, /Review or edit/);
  assert.deepEqual(x.transcripts, ['What is the condition of Pump-102?']);
  assert.deepEqual(x.requests.map(r => r.url), ['/api/voice/config', '/api/voice/transcribe']);
  assert.equal(x.requests[1].init.body.type, 'audio/webm;codecs=opus');
});

test('cancel recording releases microphone without uploading', async () => {
  const x = setup();
  await x.c.start(); x.c.cancel(); await flush();
  assert.equal(x.stopped, 1);
  assert.equal(x.requests.length, 1);
  assert.deepEqual(x.transcripts, []);
  assert.equal(x.states.at(-1).state, 'idle');
});

test('repeated start and stop clicks produce a single recording and upload', async () => {
  const x = setup();
  await Promise.all([x.c.start(), x.c.start(), x.c.start()]);
  x.c.stop(); x.c.stop(); await x.c.start(); await flush();
  assert.equal(x.recordings, 1);
  assert.equal(x.requests.length, 2);
});

test('permission denial, no microphone, and microphone in use are clear errors', async () => {
  for (const name of ['NotAllowedError', 'NotFoundError', 'NotReadableError']) {
    const x = setup({ permission: async () => { throw Object.assign(new Error(), { name }); } });
    await x.c.start();
    assert.equal(x.states.at(-1).state, 'error');
    assert.ok(x.states.at(-1).detail.length > 30);
    assert.equal(x.requests.length, 1);
  }
});

test('cancel while permission is pending stops any later stream and never starts recording', async () => {
  let grant;
  const x = setup({ permission: () => new Promise(resolve => { grant = resolve; }) });
  const pending = x.c.start(); await flush(); x.c.cancel(); grant(x.stream); await pending;
  assert.equal(x.stopped, 1); assert.equal(x.recordings, 0);
});

test('cancel during inference aborts upload and ignores a late transcript', async () => {
  let complete, signal;
  const x = setup({ fetch: async (url, init) => {
    if (url.endsWith('/config')) return Response.json(limits);
    signal = init.signal;
    return new Promise(resolve => { complete = resolve; });
  } });
  await x.c.start(); x.c.stop(); await flush(); x.c.cancel();
  assert.equal(signal.aborted, true);
  complete(Response.json({ text: 'ignored' })); await flush();
  assert.deepEqual(x.transcripts, []); assert.equal(x.states.at(-1).state, 'idle');
});

test('unsupported recording format never requests microphone permission', async () => {
  let permissions = 0;
  const x = setup({ supported: false, permission: async () => { permissions++; } });
  await x.c.start(); assert.match(x.states.at(-1).detail, /no supported/); assert.equal(permissions, 0);
});

test('network errors and unavailable model leave draft untouched', async () => {
  for (const code of ['network', 'ASR_MODEL_UNAVAILABLE', 'VOICE_BUSY', 'TRANSCRIPTION_TIMEOUT']) {
    const x = setup({ fetch: async url => {
      if (url.endsWith('/config')) return Response.json(limits);
      if (code === 'network') throw new TypeError('failed fetch');
      return Response.json({ detail: { code } }, { status: 503 });
    } });
    await x.c.start(); x.c.stop(); await flush();
    assert.equal(x.states.at(-1).state, 'error'); assert.deepEqual(x.transcripts, []);
  }
});

test('microphone disconnect cancels safely without upload', async () => {
  const x = setup(); await x.c.start(); x.track.onended(); await flush();
  assert.match(x.states.at(-1).detail, /disconnected/); assert.equal(x.requests.length, 1);
});

test('empty recordings and browser byte limit do not upload', async () => {
  for (const payload of ['', 'x'.repeat(1025)]) {
    const x = setup();
    globalThis.MediaRecorder.prototype.stop = function () {
      this.state = 'inactive';
      queueMicrotask(() => { this.ondataavailable?.({ data: new Blob([payload]) }); this.onstop?.(); });
    };
    await x.c.start(); x.c.stop(); await flush();
    assert.equal(x.states.at(-1).state, 'error'); assert.equal(x.requests.length, 1);
  }
});

test('session rejection and empty transcription are handled without modifying draft', async () => {
  for (const status of [401, 403, 200]) {
    const x = setup({ fetch: async url => Response.json(url.endsWith('/config') ? limits : { text: '' },
      { status: url.endsWith('/config') ? 200 : status }) });
    await x.c.start(); x.c.stop(); await flush();
    assert.equal(x.states.at(-1).state, 'error'); assert.deepEqual(x.transcripts, []);
  }
});

test('elapsed feedback and automatic stop stay below the hard recording limit', async context => {
  let now = 1000, tick;
  context.mock.method(Date, 'now', () => now);
  context.mock.method(globalThis, 'setInterval', callback => { tick = callback; return undefined; });
  const x = setup(); await x.c.start();
  now = 6500; tick();
  assert.equal(x.states.at(-1).elapsed, 5);
  now = 30500; tick(); await flush();
  assert.equal(x.states.at(-1).state, 'ready');
  assert.equal(x.requests.length, 2);
});

test('HTTP timeout aborts transcription and allows retry', async context => {
  const callbacks = [];
  context.mock.method(globalThis, 'setTimeout', callback => { callbacks.push(callback); return undefined; });
  const x = setup({ fetch: async (url, init) => {
    if (url.endsWith('/config')) return Response.json(limits);
    return new Promise((resolve, reject) => init.signal.addEventListener('abort', () =>
      reject(Object.assign(new Error(), { name: 'AbortError' }))));
  } });
  await x.c.start(); x.c.stop(); await flush();
  callbacks.at(-1)(); await flush();
  assert.equal(x.states.at(-1).state, 'error');
  assert.match(x.states.at(-1).detail, /timed out/);
  await x.c.start();
  assert.equal(x.states.at(-1).state, 'recording');
});
