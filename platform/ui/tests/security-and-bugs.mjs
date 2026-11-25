/**
 * Regression suite for the 13 verified security & correctness fixes in platform/ui.
 *
 *     npm run test:security-and-bugs
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import * as esbuild from 'esbuild';
import { UI, SHARED, suite } from './support/loadTs.mjs';

const { check, finish } = suite('security & bug regressions');

// Basic browser-like globals for bundled TS modules.
const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};
globalThis.window = {
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  addEventListener() {},
  removeEventListener() {},
  dispatchEvent() { return true; },
  innerWidth: 1200,
  innerHeight: 800,
  location: { protocol: 'https:', origin: 'https://flows.example.org', href: 'https://flows.example.org/app.html', hostname: 'flows.example.org', port: '' },
};
globalThis.location = globalThis.window.location;

const uiStub = `import { createElement } from 'react';
export function invalidateDynamicOptions() {}
export function subscribeToDynamicOptionsInvalidation() { return () => {}; }
export function definitionOf() { return { name: 'File Input' }; }
export function CollapsibleNodeWrapper({ children }) { return createElement('div', null, children); }
`;

const dialogStub = `export async function open() { return null; }`;
const xyflowStub = `export const Position = { Left: 'left', Right: 'right', Top: 'top', Bottom: 'bottom' };`;
const sandboxRuntimeStub = `export default "/*sandbox*/";`;

const dialogUiStub = `import { createElement } from 'react';
export default function Dialog({ children }) { return createElement('div', { 'data-dialog': '' }, children); }
`;

const aliases = {
  name: 'oaiy-test-aliases',
  setup(build) {
    build.onResolve({ filter: /^\.\/Dialog$/ }, (a) => (
      a.importer.endsWith('RunWorkflowModal.tsx') ? { path: 'dialog-ui', namespace: 'stub' } : null
    ));
    build.onLoad({ filter: /^dialog-ui$/, namespace: 'stub' }, () => ({ contents: dialogUiStub, loader: 'js', resolveDir: UI }));
    build.onResolve({ filter: /^oaiy-ui-components$/ }, () => ({ path: 'oaiy-ui-components', namespace: 'stub' }));
    build.onLoad({ filter: /^oaiy-ui-components$/, namespace: 'stub' }, () => ({ contents: uiStub, loader: 'js', resolveDir: UI }));
    build.onResolve({ filter: /^@tauri-apps\/plugin-dialog$/ }, () => ({ path: 'dialog', namespace: 'stub' }));
    build.onLoad({ filter: /^dialog$/, namespace: 'stub' }, () => ({ contents: dialogStub, loader: 'js' }));
    build.onResolve({ filter: /^@xyflow\/react$/ }, () => ({ path: 'xyflow', namespace: 'stub' }));
    build.onLoad({ filter: /^xyflow$/, namespace: 'stub' }, () => ({ contents: xyflowStub, loader: 'js' }));
    build.onResolve({ filter: /^virtual:sandbox-runtime$/ }, () => ({ path: 'sandbox-rt', namespace: 'stub' }));
    build.onLoad({ filter: /^sandbox-rt$/, namespace: 'stub' }, () => ({ contents: sandboxRuntimeStub, loader: 'js' }));
    build.onResolve({ filter: /^oaiy-core\/modules\// }, (a) => ({
      path: path.join(UI, 'src/bundled-modules', `${a.path.slice('oaiy-core/modules/'.length)}.ts`),
    }));
    build.onResolve({ filter: /^oaiy-core$/ }, () => ({ path: path.join(UI, 'vendor/oaiy-core/src/index.ts') }));
    build.onResolve({ filter: /^oaiy-core\// }, (a) => {
      const rest = a.path.slice('oaiy-core/'.length);
      const p = path.join(UI, 'vendor/oaiy-core', rest);
      return { path: fs.existsSync(p) ? p : `${p}.ts` };
    });
    build.onResolve({ filter: /^@oaiy\/shared\// }, (a) => {
      const p = path.join(SHARED, a.path.slice('@oaiy/shared/'.length));
      return { path: fs.existsSync(p) ? p : `${p}.ts` };
    });
    build.onResolve({ filter: /^\.\/ffmpeg$/ }, () => ({ path: './ffmpeg', external: true }));
  },
};

const bundlePath = path.join(os.tmpdir(), `oaiy-sec-bugs-${process.pid}.mjs`);
await esbuild.build({
  stdin: {
    contents: `
export { default as BrowserRuntime } from './src/bundled-modules/core-browser/runtime.ts';
export { default as ServiceRuntime } from './src/bundled-modules/core-service/runtime.ts';
export { default as AIRuntime } from './src/bundled-modules/core-ai/runtime.ts';
export { default as AudioRuntime } from './src/bundled-modules/core-audio/runtime.ts';
export { default as ImageRuntime } from './src/bundled-modules/core-image/runtime.ts';
export { default as VideoRuntime } from './src/bundled-modules/core-video/runtime.ts';
export { default as SubflowRuntime } from './src/bundled-modules/core-flow-control/runtime.ts';
export { default as Model3DRuntime } from './src/bundled-modules/core-3d/runtime.ts';
export { default as AgentRuntime } from './src/bundled-modules/core-agent/runtime.ts';
export { OAIYRuntime, getModuleLoader } from 'oaiy-core';
export { parseImportedProject, stripUnsafeImportedSettings, sanitizeProjectRecord, looksLikeSecret, UNSAFE_IMPORTED_SETTING_KEYS } from './src/utils/ProjectIO.ts';
export { filterNodeData, prepareWorkflowForExport } from './src/utils/WorkflowIO.ts';
export { isReadOnlySql } from './src/lib/sqlReadOnly.ts';
export { normalizeSql } from './src/tauri-shim/sql.ts';
export { PermissionRequestQueue } from './src/lib/permissionQueue.ts';
export { default as RunWorkflowModal } from './src/components/ui/RunWorkflowModal.tsx';
export { QuickConnectPopup } from './src/components/OAIYBuilder/QuickConnectPopup.tsx';
export { forgetPageHost } from '@oaiy/shared/capabilities/host';
export { createElement } from 'react';
export { renderToStaticMarkup } from 'react-dom/server';
`,
    resolveDir: UI,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: bundlePath,
  logLevel: 'silent',
  jsx: 'automatic',
  define: { 'import.meta.env': '{"DEV":false,"PROD":true,"MODE":"test"}' },
  banner: { js: "import { createRequire as __oaiyRequire } from 'node:module'; const require = __oaiyRequire(import.meta.url);" },
  plugins: [aliases],
});
const M = await import(pathToFileURL(bundlePath).href);
fs.rmSync(bundlePath, { force: true });

const mockResp = (status, body, headersObj = {}) => {
  const h = new Headers(headersObj);
  const text = typeof body === 'string' ? body : JSON.stringify(body);
  return {
    status,
    ok: status >= 200 && status < 300,
    statusText: '',
    headers: h,
    url: '',
    async text() { return text; },
    async json() { return JSON.parse(text); },
    async arrayBuffer() { return new TextEncoder().encode(text).buffer; },
    async blob() { return new Blob([text], { type: h.get('content-type') || 'application/octet-stream' }); },
  };
};

// ---------------------------------------------------------------------------
// 1. Local-network permission / SSRF bypass via direct http_request
// ---------------------------------------------------------------------------
await check('1a. Browser.request routes through ctx.secureFetch even with ctx.tauri and allowLocalNetwork=true', async () => {
  const secureCalls = [];
  const tauriCalls = [];
  const ctx = {
    log() {},
    onNodeStatus() {},
    abortSignal: undefined,
    tauri: { invoke: async (cmd, args) => { tauriCalls.push({ cmd, args }); return { status: 200, headers: {}, body: 'bypassed' }; } },
    secureFetch: async (url, init) => {
      secureCalls.push({ url, init });
      return mockResp(200, 'ok-via-secure', { 'content-type': 'text/plain' });
    },
    fetch: async () => { throw new Error('raw ctx.fetch should not be used when secureFetch exists'); },
  };
  const res = await M.BrowserRuntime.createMethods(ctx).request(
    'http://127.0.0.1:8188/api', 'POST', '{"X-Test":"1"}', '{"hi":1}', undefined, 'n1', true,
  );
  assert.equal(tauriCalls.length, 0, 'must not bypass via tauri.invoke("http_request")');
  assert.equal(secureCalls.length, 1);
  assert.equal(secureCalls[0].url, 'http://127.0.0.1:8188/api');
  assert.equal(secureCalls[0].init.method, 'POST');
  assert.equal(secureCalls[0].init.headers['X-Test'], '1');
  assert.equal(res.body, 'ok-via-secure');
});

await check('1b. core-service httpCall routes through ctx.secureFetch instead of raw tauri.invoke("http_request")', async () => {
  const secureCalls = [];
  const tauriHttpCalls = [];
  const ctx = {
    log() {},
    onNodeStatus() {},
    abortSignal: undefined,
    tauri: {
      invoke: async (cmd, args) => {
        if (cmd === 'http_request') tauriHttpCalls.push(args);
        return { success: false };
      },
    },
    secureFetch: async (url, init) => {
      secureCalls.push({ url, init });
      return mockResp(200, { answer: 'safe' }, { 'content-type': 'application/json' });
    },
  };
  const out = await M.ServiceRuntime.createMethods(ctx).call(
    { input: 'hello' }, 'http://127.0.0.1:9123/run', 'POST', '{}', '{"q": {{input}}}', 'json', 'answer', '', 'n1',
  );
  assert.equal(tauriHttpCalls.length, 0, 'must not call raw tauri.invoke("http_request")');
  assert.equal(secureCalls.length, 1);
  assert.equal(out, 'safe');
});

await check('1c. OAIYRuntime buildModuleInvokeBroker enforces local-network permission on raw tauri.invoke("http_request")', async () => {
  const nativeCalls = [];
  const runtime = new M.OAIYRuntime({ useWorkerForUntrusted: false });
  runtime.setTauriInvoke(async (cmd, args) => {
    nativeCalls.push({ cmd, args });
    return { status: 200, headers: {}, body: 'secret', url: 'http://127.0.0.1:9999/secret' };
  });
  let capturedCtx = null;
  await runtime.registerDynamicModule({
    manifest: { id: 'probe-mod', name: 'Probe', version: '1.0.0', nodes: [] },
    runtime: {
      name: 'Probe',
      createMethods(ctx) {
        capturedCtx = ctx;
        return { ping: () => 'pong' };
      },
      methods: {},
    },
  });
  // No onLocalNetworkPermission callback and not whitelisted -> must reject even with allow_private_networks: true!
  await assert.rejects(
    capturedCtx.tauri.invoke('http_request', {
      request: { url: 'http://127.0.0.1:9999/secret', method: 'GET', allow_private_networks: true },
    }),
    /Local network access to 127\.0\.0\.1:9999 is not allowed/,
  );
  assert.equal(nativeCalls.length, 0, 'native invoke must not be called when local network access is denied');

  // Now with explicit permission granted:
  const prompts = [];
  const runtimeAllowed = new M.OAIYRuntime({
    useWorkerForUntrusted: false,
    callbacks: {
      onLocalNetworkPermission: async (req) => {
        prompts.push(req);
        return { allowed: true, remember: false };
      },
    },
  });
  runtimeAllowed.setTauriInvoke(async (cmd, args) => {
    nativeCalls.push({ cmd, args });
    return { status: 200, headers: {}, body: 'allowed-body', url: 'http://127.0.0.1:9999/ok' };
  });
  let ctx2 = null;
  await runtimeAllowed.registerDynamicModule({
    manifest: { id: 'probe-mod-2', name: 'Probe2', version: '1.0.0', nodes: [] },
    runtime: { name: 'Probe2', createMethods(ctx) { ctx2 = ctx; return {}; }, methods: {} },
  });
  const res = await ctx2.tauri.invoke('http_request', {
    request: { url: 'http://127.0.0.1:9999/ok', method: 'GET', allow_private_networks: true },
  });
  assert.equal(prompts.length, 1);
  assert.equal(res.body, 'allowed-body');
  assert.equal(nativeCalls.length, 1);
});

// ---------------------------------------------------------------------------
// 2. core-ai: OAIY_SERVER_TOKEN allowlist, AI.vision apiKey, eager unload, 503 abort
// ---------------------------------------------------------------------------
await check('2a. core-ai only attaches OAIY_SERVER_TOKEN on explicit AI/gateway paths in Node mode', async () => {
  const seenHeaders = [];
  const ctx = {
    log() {},
    onNodeStatus() {},
    getModuleSetting: () => undefined,
    getConstant: () => undefined,
    secureFetch: async (url, init) => {
      seenHeaders.push({ url, auth: init.headers?.Authorization ?? init.headers?.authorization ?? null });
      return mockResp(200, { choices: [{ message: { content: 'ok' } }] });
    },
  };
  const ai = M.AIRuntime.createMethods(ctx);
  const base = 'http://127.0.0.1:17972';
  process.env.OAIY_SERVER_URL = base;
  process.env.OAIY_SERVER_TOKEN = 'admin-secret-token';
  try {
    const allowedPaths = [
      '/api/ai/v1/chat/completions',
      '/api/ai/providers/oaiy-engine/v1/chat/completions',
      '/api/ai/engine/gateway/v1/chat/completions',
      '/api/network/nodes/worker01/v1/chat/completions',
    ];
    for (const p of allowedPaths) {
      seenHeaders.length = 0;
      await ai.chat('', 'hi', null, `${base}${p}`, 'm', '', false, 64, 0, 'text', false, 'auto', 'n1');
      assert.equal(seenHeaders[0].auth, 'Bearer admin-secret-token', `expected token on ${p}`);
    }
    const forbiddenPaths = [
      '/api/settings',
      '/api/services',
      '/api/secrets',
      '/api/network/settings',
      '/api/network/nodes/worker01/v1/chat/completions/extra',
      '/api/ai/v1/%2e%2e/settings',
    ];
    for (const p of forbiddenPaths) {
      seenHeaders.length = 0;
      await ai.chat('', 'hi', null, `${base}${p}`, 'm', '', false, 64, 0, 'text', false, 'auto', 'n1');
      assert.equal(seenHeaders[0].auth, null, `must NOT send OAIY_SERVER_TOKEN to ${p}`);
    }
  } finally {
    delete process.env.OAIY_SERVER_URL;
    delete process.env.OAIY_SERVER_TOKEN;
  }
});

await check('2b. AI.vision() passes apiKey (both constant name and direct literal key) into chat()', async () => {
  const seen = [];
  const ctx = {
    log() {},
    onNodeStatus() {},
    getModuleSetting: () => undefined,
    getConstant: (name) => (name === 'MY_VISION_KEY' ? 'resolved-from-const' : undefined),
    secureFetch: async (_url, init) => {
      seen.push(init.headers?.Authorization ?? null);
      return mockResp(200, { choices: [{ message: { content: 'a cat' } }] });
    },
  };
  const ai = M.AIRuntime.createMethods(ctx);
  await ai.vision('describe', '', '', 'gpt-4o', 'n1', 'openai', 'https://api.example.com/v1/chat/completions', 'MY_VISION_KEY', 'none', 4096, 100);
  await ai.vision('describe', '', '', 'gpt-4o', 'n1', 'openai', 'https://api.example.com/v1/chat/completions', 'direct-short-key', 'none', 4096, 100);
  await ai.vision('describe', '', '', 'gpt-4o', 'n1', 'openai', 'https://api.example.com/v1/chat/completions', 'UNSET_API_KEY', 'none', 4096, 100);
  assert.deepEqual(seen, ['Bearer resolved-from-const', 'Bearer direct-short-key', null]);
});

await check('2c. OAIY_LLM_EAGER_UNLOAD=1 runs in finally after oaiy-local chat completes', async () => {
  const invoked = [];
  const ctx = {
    log() {},
    onNodeStatus() {},
    getModuleSetting: () => undefined,
    getConstant: () => undefined,
    tauri: {
      invoke: async (cmd, args) => {
        invoked.push({ cmd, args });
        if (cmd === 'plugin:oaiy-llm|llm_chat') {
          return { text: 'hello', modelId: 'm1', backend: 'cpu', promptTokens: 2, generatedTokens: 1 };
        }
        return true;
      },
    },
  };
  process.env.OAIY_LLM_EAGER_UNLOAD = '1';
  try {
    const out = await M.AIRuntime.createMethods(ctx).chat('', 'hi', null, 'oaiy-llm://local', 'm1', '', false, 64, 0, 'text', false, 'auto', 'n1');
    assert.equal(out, 'hello');
    assert.ok(invoked.some((c) => c.cmd === 'plugin:oaiy-llm|llm_unload_model' && c.args?.modelId === 'm1'), 'expected llm_unload_model to run');
  } finally {
    delete process.env.OAIY_LLM_EAGER_UNLOAD;
  }
});

await check('2d. core-ai 503 retry sleep aborts immediately on ctx.abortSignal', async () => {
  const ac = new AbortController();
  const ctx = {
    log() {},
    onNodeStatus() {},
    abortSignal: ac.signal,
    getModuleSetting: () => undefined,
    getConstant: () => undefined,
    secureFetch: async () => {
      setTimeout(() => ac.abort(), 25);
      return mockResp(503, 'loading model');
    },
  };
  const t0 = Date.now();
  const res = await M.AIRuntime.createMethods(ctx).chat('', 'hi', null, 'https://api.example.com/v1/chat/completions', 'm', '', false, 64, 0, 'text', false, 'auto', 'n1');
  const elapsed = Date.now() - t0;
  assert.equal(res, '__ABORT__');
  assert.ok(elapsed < 1000, `expected immediate abort (<1000ms), took ${elapsed}ms`);
});

// ---------------------------------------------------------------------------
// 3. compiler-sandbox-worker.ts blocklist (WebTransport + navigator.storage)
// ---------------------------------------------------------------------------
await check('3. compiler-sandbox-worker neutralizes WebTransport and navigator.storage across prototypes', async () => {
  const workerBundle = path.join(os.tmpdir(), `oaiy-worker-test-${process.pid}.mjs`);
  await esbuild.build({
    entryPoints: [path.join(UI, 'src/sandbox/compiler-sandbox-worker.ts')],
    bundle: true,
    format: 'esm',
    platform: 'node',
    outfile: workerBundle,
    logLevel: 'silent',
  });
  const navProto = { storage: { getDirectory: () => 'opfs' } };
  const fakeNav = Object.create(navProto);
  const workerProto = {
    WebTransport: function WebTransport() {},
    fetch: () => {},
    postMessage() {},
    addEventListener() {},
  };
  const fakeSelf = Object.create(workerProto);
  fakeSelf.navigator = fakeNav;
  const prevSelf = globalThis.self;
  globalThis.self = fakeSelf;
  try {
    await import(pathToFileURL(workerBundle).href);
    assert.equal(fakeSelf.WebTransport, undefined);
    assert.equal(workerProto.WebTransport, undefined);
    assert.equal(fakeSelf.navigator.storage, undefined);
    assert.equal(navProto.storage, undefined);
  } finally {
    globalThis.self = prevSelf;
    fs.rmSync(workerBundle, { force: true });
  }
});

// ---------------------------------------------------------------------------
// 4. Untrusted project & settings JSON import sanitization
// ---------------------------------------------------------------------------
await check('4. parseImportedProject strips unsafe endpoint & local-network override settings', () => {
  const malicious = {
    name: 'Shared Project',
    flows: [],
    settings: {
      defaultAIModel: 'safe-model',
      showDemoFlows: false,
      defaultAIEndpoint: 'https://attacker.example/v1/chat',
      ollamaEndpoint: 'https://attacker.example/ollama',
      lmstudioEndpoint: 'https://attacker.example/lmstudio',
      defaultImageEndpoint: 'https://attacker.example/image',
      defaultVideoEndpoint: 'https://attacker.example/video',
      allowAllLocalNetwork: true,
      localNetworkWhitelist: ['169.254.169.254:80'],
    },
  };
  const parsed = M.parseImportedProject(JSON.stringify(malicious));
  assert.equal(parsed.settings.defaultAIModel, 'safe-model');
  assert.equal(parsed.settings.showDemoFlows, false);
  assert.equal(parsed.settings.defaultAIEndpoint, 'http://localhost:11434/v1/chat/completions');
  assert.equal(parsed.settings.defaultImageEndpoint, 'http://localhost:8188');
  assert.equal(parsed.settings.defaultVideoEndpoint, 'http://localhost:8188');
  assert.equal(parsed.settings.ollamaEndpoint, undefined);
  assert.equal(parsed.settings.lmstudioEndpoint, undefined);
  assert.equal(parsed.settings.allowAllLocalNetwork, undefined);
  assert.equal(parsed.settings.localNetworkWhitelist, undefined);

  const { cleaned, stripped } = M.stripUnsafeImportedSettings(malicious.settings);
  assert.deepEqual(new Set(stripped), new Set(M.UNSAFE_IMPORTED_SETTING_KEYS));
  assert.deepEqual(cleaned, { defaultAIModel: 'safe-model', showDemoFlows: false });
});

// ---------------------------------------------------------------------------
// 5. Secret leakage across workflow export & runtime logs/errors
// ---------------------------------------------------------------------------
await check('5a. WorkflowIO.filterNodeData/prepareWorkflowForExport strip nested/header/URL/literal secrets and keep persistent _ fields', () => {
  const sk = `sk-${'a'.repeat(30)}`;
  const wf = M.prepareWorkflowForExport(
    [
      {
        id: 'n1',
        type: 'ai_llm',
        position: { x: 0, y: 0 },
        data: {
          _collapsed: true,
          _status: 'running',
          maxTokens: 512,
          apiKeyConstant: 'OPENAI_API_KEY',
          accessToken: 'plain-access-token',
          Password: 'secret-password',
          endpoint: 'https://user:pass@api.example.com/v1?api_key=mysecret&page=2',
          customHeaders: `{"Authorization": "Bearer ${sk}", "X-Api-Key": "plain-hdr", "Accept": "application/json"}`,
          customBodyTemplate: '{"secret_key": "plain-body-secret", "model": "{{model}}"}',
          headers: { Authorization: 'Basic dXNlcjpwYXNz', Accept: 'application/json' },
          nested: { clientSecret: 'nested-secret', ok: true },
        },
      },
      {
        id: 'n2',
        type: 'ai_llm',
        position: { x: 100, y: 0 },
        data: { apiKeyConstant: 'sk-pasted-literal', _macroWorkflowId: 'm-1' },
      },
    ],
    [],
    'Exported',
  );
  const text = JSON.stringify(wf);
  for (const leaked of ['plain-access-token', 'secret-password', 'user:pass', 'mysecret', sk, 'plain-hdr', 'plain-body-secret', 'dXNlcjpwYXNz', 'nested-secret', 'sk-pasted-literal']) {
    assert.equal(text.includes(leaked), false, `workflow export leaked ${leaked}`);
  }
  const d1 = wf.nodes[0].data;
  assert.equal(d1._collapsed, true);
  assert.equal('_status' in d1, false);
  assert.equal(d1.maxTokens, 512);
  assert.equal(d1.apiKeyConstant, 'OPENAI_API_KEY');
  assert.deepEqual(d1.headers, { Accept: 'application/json' });
  assert.deepEqual(d1.nested, { ok: true });
  assert.equal(wf.nodes[1].data._macroWorkflowId, 'm-1');
  assert.equal('apiKeyConstant' in wf.nodes[1].data, false);
});

await check('5b. core-service & core-ai malformed template errors and core-audio TTS log never leak interpolated apiKey', async () => {
  const secretKey = 'sk-supersecretkeyvalue1234567890';
  // core-service malformed headers JSON
  const svcCtx = {
    log() {},
    onNodeStatus() {},
    getConstant: (k) => (k === 'SVC_KEY' ? secretKey : undefined),
    secureFetch: async () => mockResp(200, '{}'),
  };
  await assert.rejects(
    M.ServiceRuntime.createMethods(svcCtx).call(
      { input: 'x' }, 'https://api.example.com/v1', 'POST', '{"Authorization": "Bearer {{apiKey}}" broken}', '{}', 'json', '', 'SVC_KEY', 'n1',
    ),
    (err) => {
      assert.equal(err.message.includes(secretKey), false, `Service error leaked secret: ${err.message}`);
      assert.match(err.message, /did not render to valid JSON/);
      return true;
    },
  );

  // core-ai malformed customBodyTemplate JSON
  const aiCtx = {
    log() {},
    onNodeStatus() {},
    getModuleSetting: () => undefined,
    getConstant: (k) => (k === 'AI_KEY' ? secretKey : undefined),
    secureFetch: async () => mockResp(200, '{}'),
  };
  await assert.rejects(
    M.AIRuntime.createMethods(aiCtx).chat(
      '', 'hi', null, 'https://api.example.com/v1', 'm', 'AI_KEY', false, 64, 0, 'text', false, 'auto', 'n1',
      [], '', 0, 0, false, '', null, 8, '{"key": {{apiKey}} broken}', '', '',
    ),
    (err) => {
      assert.equal(err.message.includes(secretKey), false, `AI error leaked secret: ${err.message}`);
      assert.match(err.message, /Custom body template did not render to valid JSON/);
      return true;
    },
  );

  // core-audio external TTS log does not include interpolated apiKey
  const audioLogs = [];
  const audioCtx = {
    log: (_lvl, msg) => audioLogs.push(msg),
    onNodeStatus() {},
    getConstant: (k) => (k === 'TTS_KEY' ? secretKey : undefined),
    tauri: {
      invoke: async (cmd) => {
        if (cmd === 'plugin:oaiy-filesystem|get_app_data_dir') return '/tmp/app';
        if (cmd === 'get_media_url') return 'blob:audio';
        return null;
      },
    },
    fetch: async () => mockResp(200, { path: '/tmp/out.wav' }, { 'content-type': 'application/json' }),
  };
  await M.AudioRuntime.createMethods(audioCtx).textToSpeechV2({
    nodeId: 'tts1',
    text: 'hello world',
    provider: 'external',
    externalUrl: 'https://tts.example.com/speak?key={{apiKey}}',
    externalMethod: 'POST',
    externalHeaders: '{}',
    externalBodyTemplate: '{"text": {{text}}}',
    externalResponseType: 'file_path',
    externalResponsePath: 'path',
    externalApiKeyConstant: 'TTS_KEY',
  });
  assert.ok(audioLogs.length > 0);
  assert.equal(audioLogs.join('\n').includes(secretKey), false, `TTS logs leaked secret: ${audioLogs.join('\n')}`);
});

// ---------------------------------------------------------------------------
// 6. DataViewer isReadOnlySql & tauri-shim/sql normalizeSql
// ---------------------------------------------------------------------------
await check('6a. isReadOnlySql blocks multi-statement, writable CTEs, mutating PRAGMAs, and EXPLAIN writes', () => {
  const allowed = [
    'SELECT * FROM docs',
    'SELECT * FROM docs;',
    "SELECT ';' FROM docs WHERE op = 'DELETE'",
    '/* comment */ SELECT 1',
    '-- lead\nWITH cte AS (SELECT 1 AS a) SELECT * FROM cte;',
    'PRAGMA table_list',
    'PRAGMA main.user_version;',
    'EXPLAIN SELECT * FROM docs',
    'EXPLAIN QUERY PLAN SELECT * FROM docs',
  ];
  for (const q of allowed) {
    assert.equal(M.isReadOnlySql(q), true, `expected read-only: ${q}`);
  }

  const rejected = [
    '',
    '   -- comment only',
    'SELECT 1; DROP TABLE docs',
    'SELECT 1; DELETE FROM docs;',
    'SELECT 1;;',
    'WITH cte AS (SELECT 1) DELETE FROM docs',
    'WITH cte AS (SELECT 1) INSERT INTO docs VALUES (1)',
    'WITH cte AS (SELECT 1) UPDATE docs SET x = 1',
    'WITH cte AS (SELECT 1) REPLACE INTO docs VALUES (1)',
    'WITH cte AS (SELECT 1) DROP TABLE docs',
    'WITH cte AS (SELECT 1) ALTER TABLE docs ADD COLUMN y',
    "WITH cte AS (SELECT 1) ATTACH DATABASE 'x.db' AS x",
    'WITH cte AS (SELECT 1) VACUUM',
    'PRAGMA writable_schema = ON',
    'PRAGMA user_version(42)',
    'PRAGMA wal_checkpoint(TRUNCATE)',
    'EXPLAIN DELETE FROM docs',
    "SELECT 'unterminated",
    '/* unterminated comment SELECT 1',
  ];
  for (const q of rejected) {
    assert.equal(M.isReadOnlySql(q), false, `expected rejected: ${q}`);
  }
});

await check('6b. normalizeSql rewrites $N placeholders without corrupting string literals, identifiers, or comments', () => {
  assert.equal(
    M.normalizeSql("SELECT 'Price: $100', 'it''s $2', \"col$3\", `b$4`, [br$5], json_extract(data, '$.a') -- $6\n/* $7 */ FROM t WHERE id = $1 AND x = $2"),
    "SELECT 'Price: $100', 'it''s $2', \"col$3\", `b$4`, [br$5], json_extract(data, '$.a') -- $6\n/* $7 */ FROM t WHERE id = ?1 AND x = ?2",
  );
  assert.equal(M.normalizeSql('INSERT INTO docs (id, val) VALUES ($1, $2)'), 'INSERT INTO docs (id, val) VALUES (?1, ?2)');
});

// ---------------------------------------------------------------------------
// 7. Fail-closed package permission declarations
// ---------------------------------------------------------------------------
await check('7. Package permission broker classifies Subflow.*, Browser V2, Service.*, Model3D.*, and AgentTask.ask', async () => {
  const runPkg = async (script, perms) => {
    const rt = new M.OAIYRuntime({ useWorkerForUntrusted: false });
    rt.setFlowContext('f1', 'pkg-untrusted');
    rt.setPackagePermissions(perms);
    await rt.registerDynamicModule({
      manifest: { id: 'core-flow-control', name: 'Subflow', version: '1.0.0', nodes: [] },
      runtime: M.SubflowRuntime,
    });
    await rt.registerDynamicModule({
      manifest: { id: 'core-browser', name: 'Browser', version: '1.0.0', nodes: [] },
      runtime: {
        name: 'Browser',
        createMethods: () => Object.fromEntries(
          ['createSessionV2', 'goto', 'evaluateJson', 'getText', 'getTitle', 'getUrl', 'waitForSelector', 'getCookies', 'setCookies']
            .map((m) => [m, async () => `ok:${m}`]),
        ),
        methods: {},
      },
    });
    await rt.registerDynamicModule({
      manifest: { id: 'core-service', name: 'Service', version: '1.0.0', nodes: [] },
      runtime: { name: 'Service', createMethods: () => ({ call: async () => 'ok:call', callContract: async () => 'ok:contract' }), methods: {} },
    });
    await rt.registerDynamicModule({
      manifest: { id: 'core-3d', name: 'Model3D', version: '1.0.0', nodes: [] },
      runtime: { name: 'Model3D', createMethods: () => ({ make: async () => 'ok:make', generate: async () => 'ok:gen' }), methods: {} },
    });
    await rt.registerDynamicModule({
      manifest: { id: 'core-agent', name: 'AgentTask', version: '1.0.0', nodes: [] },
      runtime: { name: 'AgentTask', createMethods: () => ({ ask: async () => 'ok:ask' }), methods: {} },
    });
    return rt.executeScript(script);
  };

  // Subflow.checkAborted needs NO permission
  assert.equal(await runPkg('return await Subflow.checkAborted();', []), false);

  // All Browser V2 methods require 'browser' (never 'not permission-declared')
  for (const m of ['createSessionV2', 'goto', 'evaluateJson', 'getText', 'getTitle', 'getUrl', 'waitForSelector', 'getCookies', 'setCookies']) {
    await assert.rejects(
      runPkg(`return await Browser.${m}();`, []),
      /does not have the 'browser' permission/,
      `Browser.${m} without browser permission`,
    );
    assert.equal(await runPkg(`return await Browser.${m}();`, ['browser']), `ok:${m}`);
  }

  // Service.call, Service.callContract, Model3D.make, Model3D.generate require 'network'
  for (const call of ['Service.call', 'Service.callContract', 'Model3D.make', 'Model3D.generate']) {
    await assert.rejects(runPkg(`return await ${call}();`, []), /does not have the 'network' permission/, call);
    assert.match(await runPkg(`return await ${call}();`, ['network']), /^ok:/);
  }

  // AgentTask.ask requires high-risk 'process:execute', NOT low-risk 'network'
  await assert.rejects(runPkg('return await AgentTask.ask();', ['network']), /does not have the 'process:execute' permission/);
  assert.equal(await runPkg('return await AgentTask.ask();', ['process:execute']), 'ok:ask');
});

// ---------------------------------------------------------------------------
// 9. customNodeCompiler direct entry preserves export default & TSX
// ---------------------------------------------------------------------------
await check('9. customNodeCompiler direct virtual entry preserves export default and named exports (and TSX)', async () => {
  const compilerSrc = fs.readFileSync(path.join(UI, 'src/services/customNodeCompiler.ts'), 'utf8');
  const codeOnly = compilerSrc.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
  assert.equal(codeOnly.includes("export * from './source'"), false, 'must not use export * wrapper that drops default export');
  // Verify with real esbuild that direct entry preserves default export in IIFE global
  const compileVirtual = async (entryPath, code, globalName) => {
    const sources = new Map([[entryPath, code]]);
    const escaped = entryPath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const res = await esbuild.build({
      entryPoints: [entryPath],
      bundle: true,
      write: false,
      format: 'iife',
      globalName,
      platform: 'browser',
      target: 'es2020',
      jsx: 'automatic',
      plugins: [{
        name: 'v',
        setup(build) {
          build.onResolve({ filter: new RegExp(`^${escaped}$`) }, () => ({ path: entryPath, namespace: 'virtual' }));
          build.onResolve({ filter: /^react\/jsx-runtime$/ }, () => ({ path: 'jsx', namespace: 'ext' }));
          build.onLoad({ filter: /.*/, namespace: 'ext' }, () => ({ contents: 'module.exports = { jsx: (t, p) => ({ t, p }), jsxs: (t, p) => ({ t, p }) };', loader: 'js' }));
          build.onLoad({ filter: /.*/, namespace: 'virtual' }, (a) => ({ contents: sources.get(a.path), loader: a.path.endsWith('.tsx') ? 'tsx' : 'ts' }));
        },
      }],
    });
    return new Function(`${res.outputFiles[0].text}\nreturn ${globalName};`)();
  };

  const uiMod = await compileVirtual('source.tsx', 'export default function MyUI() { return <div>ok</div>; }', '__UI_TEST__');
  assert.equal(typeof uiMod.default, 'function');
  assert.deepEqual(uiMod.default(), { t: 'div', p: { children: 'ok' } });

  const namedMod = await compileVirtual('source.ts', 'export function compile() { return "x"; }', '__NAMED_TEST__');
  assert.equal(typeof namedMod.compile, 'function');
  assert.equal(namedMod.compile(), 'x');
});

// ---------------------------------------------------------------------------
// 10. ComfyUI multi-image input index shift on bypass
// ---------------------------------------------------------------------------
await check('10a. core-image ComfyUI bypass pairs imageInputs before filtering bypassed slots', async () => {
  const uploaded = [];
  let submittedWorkflow = null;
  const ctx = {
    log() {},
    onNodeStatus() {},
    onImage() {},
    getModuleSetting: () => undefined,
    getConstant: () => undefined,
    secureFetch: async (url, init) => {
      if (String(url).includes('/upload/image')) {
        uploaded.push({ url, body: init.body });
        return mockResp(200, { name: `uploaded_${uploaded.length}.png` }, { 'content-type': 'application/json' });
      }
      if (String(url).includes('/prompt')) {
        submittedWorkflow = JSON.parse(init.body).prompt;
        return mockResp(200, { prompt_id: 'p1' }, { 'content-type': 'application/json' });
      }
      if (String(url).includes('/history/p1')) {
        return mockResp(200, {
          p1: { status: { completed: true }, outputs: { '9': { images: [{ filename: 'out.png', subfolder: '', type: 'output' }] } } },
        }, { 'content-type': 'application/json' });
      }
      return mockResp(200, '{}');
    },
  };
  const workflowJson = JSON.stringify({
    '1': { class_type: 'LoadImage', inputs: { image: 'default_A.png' } },
    '2': { class_type: 'LoadImage', inputs: { image: 'default_B.png' } },
    '9': { class_type: 'SaveImage', inputs: { images: ['2', 0] } },
  });
  const configs = [
    { nodeId: '1', title: 'Slot A', allowBypass: true },
    { nodeId: '2', title: 'Slot B', allowBypass: false },
  ];
  const imgB = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
  // Slot 0 has no input (bypassed!), Slot 1 has imgB. Before fix, slot 1 got imageInputs[0] (undefined)!
  await M.ImageRuntime.createMethods(ctx).generate(
    'a fox', null, 'http://127.0.0.1:8188', '', '', 512, 512, 1, 'comfyui', 'n1',
    workflowJson, null, ['1', '2'], [undefined, imgB], configs, 'fixed', 42, ['1', '2'],
  );
  assert.equal(uploaded.length, 1, 'slot 1 image must be uploaded');
  assert.equal('1' in submittedWorkflow, false, 'bypassed slot 0 must be removed');
  assert.equal(submittedWorkflow['2'].inputs.image, 'uploaded_1.png', 'slot 1 node must receive slot 1 uploaded image');
});

await check('10b. core-video ComfyUI bypass pairs imageInputs before filtering bypassed slots', async () => {
  const uploaded = [];
  let submittedWorkflow = null;
  const ctx = {
    log() {},
    onNodeStatus() {},
    getModuleSetting: () => undefined,
    getConstant: () => undefined,
    secureFetch: async (url, init) => {
      if (String(url).includes('/upload/image')) {
        uploaded.push({ url });
        return mockResp(200, { name: `vid_up_${uploaded.length}.png` }, { 'content-type': 'application/json' });
      }
      if (String(url).includes('/prompt')) {
        submittedWorkflow = JSON.parse(init.body).prompt;
        return mockResp(200, { prompt_id: 'vp1' }, { 'content-type': 'application/json' });
      }
      if (String(url).includes('/history/vp1')) {
        return mockResp(200, {
          vp1: { status: { completed: true }, outputs: { '9': { gifs: [{ filename: 'out.mp4', subfolder: '', type: 'output' }] } } },
        }, { 'content-type': 'application/json' });
      }
      return mockResp(200, '{}');
    },
  };
  const workflowJson = JSON.stringify({
    '1': { class_type: 'LoadImage', inputs: { image: 'default_A.png' } },
    '2': { class_type: 'LoadImage', inputs: { image: 'default_B.png' } },
    '9': { class_type: 'VHS_VideoCombine', inputs: { images: ['2', 0] } },
  });
  const configs = [
    { nodeId: '1', title: 'Slot A', allowBypass: true },
    { nodeId: '2', title: 'Slot B', allowBypass: false },
  ];
  const imgB = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
  await M.VideoRuntime.createMethods(ctx).generate(
    'http://127.0.0.1:8188', 'n1', 'waves', workflowJson, null,
    ['1', '2'], [undefined, imgB], configs, 'fixed', 42, ['1', '2'],
  );
  assert.equal(uploaded.length, 1, 'slot 1 image must be uploaded');
  assert.equal(submittedWorkflow['1'].class_type, 'EmptyImage', 'bypassed slot 0 replaced with EmptyImage');
  assert.equal(submittedWorkflow['2'].inputs.image, 'vid_up_1.png', 'slot 1 node must receive slot 1 uploaded image');
});

// ---------------------------------------------------------------------------
// 11. Concurrent local-network permission FIFO queue
// ---------------------------------------------------------------------------
await check('11. PermissionRequestQueue serializes concurrent requests without dropping resolvers', async () => {
  const q = new M.PermissionRequestQueue();
  const r1 = q.enqueue({ url: 'http://127.0.0.1:11434/a', hostPort: '127.0.0.1:11434', purpose: 'A' });
  const r2 = q.enqueue({ url: 'http://127.0.0.1:8188/b', hostPort: '127.0.0.1:8188', purpose: 'B' });
  const r3 = q.enqueue({ url: 'http://127.0.0.1:8188/c', hostPort: '127.0.0.1:8188', purpose: 'C' });
  const r4 = q.enqueue({ url: 'http://127.0.0.1:9000/d', hostPort: '127.0.0.1:9000', purpose: 'D' });

  assert.equal(r1.becameActive, true);
  assert.equal(r2.becameActive, false);
  assert.equal(q.current.hostPort, '127.0.0.1:11434');

  // Respond once (no remember) -> resolves r1 and advances to r2
  const nextAfter1 = q.respond(true, false);
  assert.deepEqual(await r1.promise, { allowed: true, remember: false });
  assert.equal(nextAfter1.hostPort, '127.0.0.1:8188');

  // Respond with remember=true -> resolves r2 AND queued duplicate r3, advances to r4
  const nextAfter2 = q.respond(true, true);
  assert.deepEqual(await r2.promise, { allowed: true, remember: true });
  assert.deepEqual(await r3.promise, { allowed: true, remember: true });
  assert.equal(nextAfter2.hostPort, '127.0.0.1:9000');

  // drainDeny settles remaining r4 cleanly
  q.drainDeny();
  assert.deepEqual(await r4.promise, { allowed: false, remember: false });
  assert.equal(q.current, null);
});

// ---------------------------------------------------------------------------
// 12. RunWorkflowModal mediaUrlBlocked guard
// ---------------------------------------------------------------------------
await check('12. RunWorkflowModal blocks local-network image preview URLs on an unlinked tab', () => {
  M.forgetPageHost();
  store.clear();
  const htmlBlocked = M.renderToStaticMarkup(
    M.createElement(M.RunWorkflowModal, {
      isOpen: true,
      nodes: [{
        id: 'f1',
        type: 'input_file',
        position: { x: 0, y: 0 },
        data: { label: 'Photo', filePath: 'http://127.0.0.1:8188/view?filename=secret.png' },
      }],
      onRun() {},
      onCancel() {},
    }),
  );
  assert.equal(htmlBlocked.includes('<img'), false, 'must NOT render <img> for blocked local-network preview URL');

  const htmlAllowed = M.renderToStaticMarkup(
    M.createElement(M.RunWorkflowModal, {
      isOpen: true,
      nodes: [{
        id: 'f2',
        type: 'input_file',
        position: { x: 0, y: 0 },
        data: { label: 'Photo', fileContent: 'data:image/png;base64,AAAA', fileName: 'ok.png' },
      }],
      onRun() {},
      onCancel() {},
    }),
  );
  assert.equal(htmlAllowed.includes('<img'), true, 'data:image preview must still render');
});

// ---------------------------------------------------------------------------
// 13. QuickConnectPopup clamping & SandboxedCustomNodeUI theme sync
// ---------------------------------------------------------------------------
await check('13. QuickConnectPopup clamps coordinates >= 8px on narrow viewports & SandboxedCustomNodeUI removes dark-only #e5e7eb', () => {
  globalThis.window.innerWidth = 200;
  globalThis.window.innerHeight = 200;
  const html = M.renderToStaticMarkup(
    M.createElement(M.QuickConnectPopup, {
      state: { isVisible: true, position: { x: 50, y: 50 }, searchQuery: '' },
      handlePosition: { x: 10, y: 10 },
      compatibleNodes: [],
      onNodeSelect() {},
      onClose() {},
      onSearchChange() {},
    }),
  );
  assert.match(html, /left:8px/, `expected clamped left:8px in ${html}`);
  assert.match(html, /top:8px/, `expected clamped top:8px in ${html}`);

  const sandboxSrc = fs.readFileSync(path.join(UI, 'src/components/SandboxedCustomNodeUI.tsx'), 'utf8');
  assert.equal(sandboxSrc.includes('#e5e7eb'), false, 'SandboxedCustomNodeUI must not hardcode dark-only #e5e7eb');
  assert.ok(sandboxSrc.includes('--color-text-primary'), 'SandboxedCustomNodeUI must sync host theme tokens');
});

finish();
