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
export { default as AudioRuntime, sanitizeAudioFilename, sanitizeAudioFormat, resolveLocalMediaUrl, redactMediaToken } from './src/bundled-modules/core-audio/runtime.ts';
export { default as ImageRuntime } from './src/bundled-modules/core-image/runtime.ts';
export { default as VideoRuntime, sanitizeVideoFilename, sanitizeVideoFormat, sanitizeFfmpegColor } from './src/bundled-modules/core-video/runtime.ts';
export { default as SubflowRuntime } from './src/bundled-modules/core-flow-control/runtime.ts';
export { default as Model3DRuntime } from './src/bundled-modules/core-3d/runtime.ts';
export { default as AgentRuntime, taskText } from './src/bundled-modules/core-agent/runtime.ts';
export { default as FilesystemRuntime, validatePath } from './src/bundled-modules/core-filesystem/runtime.ts';
export { default as FilesystemCompiler } from './src/bundled-modules/core-filesystem/compiler.ts';
export { default as DatabaseRuntime } from './src/bundled-modules/core-database/runtime.ts';
export { default as DatabaseCompiler } from './src/bundled-modules/core-database/compiler.ts';
export { default as UtilityRuntime } from './src/bundled-modules/core-utility/runtime.ts';
export { default as UtilityCompiler } from './src/bundled-modules/core-utility/compiler.ts';
export { default as Core3DCompiler } from './src/bundled-modules/core-3d/compiler.ts';
export { desktopCredential, keepFile } from './src/bundled-modules/core-service/contractRuntime.ts';
export { OAIYRuntime, OAIYCompiler, JobManager, getModuleLoader } from 'oaiy-core';
export { isLocalNetworkUrl, matchesWhitelistEntry, getHostPort } from 'oaiy-core/src/runtime/NetworkUtils.ts';
export { convertSelectionToMacro, analyzeSelection } from 'oaiy-core/src/macro/selection-analyzer.ts';
export { parseImportedProject, sanitizeProjectForExport, stripUnsafeImportedSettings, sanitizeProjectRecord, looksLikeSecret, isSecretConstant, UNSAFE_IMPORTED_SETTING_KEYS } from './src/utils/ProjectIO.ts';
export { filterNodeData, prepareWorkflowForExport } from './src/utils/WorkflowIO.ts';
export { sanitizeDataForSandbox, sanitizeSandboxUpdates } from './src/lib/sandboxData.ts';
export { withdrawTool, withdrawHook } from './src/lib/oaiyAgentTools.ts';
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

// ===========================================================================
// Wave-2 Security & Bug Regressions (14..24)
// ===========================================================================

await check('14. JobManager enforces hardened sandbox on untrusted flows and transitive subflows/macros', async () => {
  store.clear();
  const seenLogs = [];
  const jm = new M.JobManager({
    config: { maxConcurrency: 1 },
    databaseHandler: async () => ({ success: true, data: [] }),
    moduleRegistry: {
      nodes: new Map([['subflow', { id: 'subflow', name: 'Subflow', inputs: [], outputs: [], compiler: {} }]]),
      nodeToModule: new Map([['subflow', 'core-flow-control']]),
      getAllModules: () => [],
      getAllNodeDefinitions: () => [],
      getNodeDefinition: (t) => (t === 'subflow' ? { id: 'subflow', name: 'Subflow', inputs: [], outputs: [], compiler: {} } : undefined),
      getModuleForNode: (type) => (
        type === 'subflow'
          ? { manifest: { id: 'core-flow-control', name: 'FlowControl', version: '1.0.0', nodes: ['subflow'] }, nodes: new Map(), compiler: { name: 'FlowControl', compileNode: () => 'let node_s1_out = null;' } }
          : undefined
      ),
      getCompiler: (type) => (type === 'subflow' ? { name: 'FlowControl', compileNode: () => 'let node_s1_out = null;' } : undefined),
    },
  });
  jm.setAvailableFlows([
    {
      id: 'untrusted-child',
      name: 'Untrusted Child',
      untrusted: true,
      createdAt: '',
      updatedAt: '',
      graph: { nodes: [], edges: [] },
    },
    {
      id: 'caller-flow',
      name: 'Caller',
      createdAt: '',
      updatedAt: '',
      graph: {
        nodes: [{ id: 's1', type: 'subflow', position: { x: 0, y: 0 }, data: { flowId: 'untrusted-child' } }],
        edges: [],
      },
    },
  ]);
  jm.onLog((_jobId, entry) => seenLogs.push(entry));
  const jobId = jm.submit('caller-flow', 'Caller', {
    nodes: [{ id: 's1', type: 'subflow', position: { x: 0, y: 0 }, data: { flowId: 'untrusted-child' } }],
    edges: [],
  });
  // Wait for job to settle
  const finalJob = await new Promise((resolve) => {
    const unsub = jm.onStateChange((jobs) => {
      const j = jobs.find((x) => x.id === jobId);
      if (j && (j.status === 'completed' || j.status === 'failed' || j.status === 'aborted')) {
        unsub();
        resolve(j);
      }
    });
  });
  assert.equal(finalJob.status, 'failed');
  assert.match(String(finalJob.error || ''), /Refusing to run untrusted package code in-thread/, `JobManager must engage hardened mode for flow reaching untrusted subflow (got: ${finalJob.error})`);
  assert.ok(seenLogs.some((l) => l.source === 'Security' && /sandboxed/i.test(l.message)), 'must log Security sandbox notice');

  // Verify App.tsx sanitizes flowJson.settings & CliWorkflowRunner passes packageContext for .oaiy
  const appSrc = fs.readFileSync(path.join(UI, 'src/App.tsx'), 'utf8');
  assert.ok(appSrc.includes('stripUnsafeImportedSettings('), 'App.tsx must sanitize shared flowJson.settings');
  const cliSrc = fs.readFileSync(path.join(UI, 'src/components/CliWorkflowRunner.tsx'), 'utf8');
  assert.ok(cliSrc.includes('packageTrustLevel') && cliSrc.includes('cli-package:'), 'CliWorkflowRunner must pass hardened packageContext for .oaiy packages');
});

await check('15. untrusted-executor.ts neutralizes WebTransport, caches, indexedDB, LockManager, locks, storage & navigator.storage', () => {
  const src = fs.readFileSync(path.join(UI, 'vendor/oaiy-core/src/untrusted-executor.ts'), 'utf8');
  for (const name of ['WebTransport', 'caches', 'indexedDB', 'LockManager', 'locks', 'storage']) {
    assert.ok(src.includes(`'${name}'`), `untrusted-executor egress must block ${name}`);
  }
  assert.ok(src.includes("var nav = self.navigator"), 'untrusted-executor must neutralize navigator properties across prototypes');
});

await check('16. buildModuleInvokeBroker enforces per-command permissions, drops 3 admin commands, allows calculate_file_chunks, and gates browser URLs', async () => {
  const invoked = [];
  const prompted = [];
  const rt = new M.OAIYRuntime({
    useWorkerForUntrusted: false,
    callbacks: {
      onLocalNetworkPermission: async (req) => {
        prompted.push(req);
        return { allowed: false, remember: false };
      },
    },
  });
  rt.setTauriInvoke(async (cmd, args) => {
    invoked.push({ cmd, args });
    return { ok: true };
  });
  let broker;
  await rt.registerDynamicModule({
    manifest: { id: 'probe', name: 'Probe', version: '1.0.0', nodes: [] },
    runtime: {
      name: 'Probe',
      createMethods(ctx) {
        broker = ctx.tauri;
        return {};
      },
      methods: {},
    },
  });

  // 3 admin commands removed from allowlist even in trusted mode
  for (const removed of [
    'plugin:oaiy-browser|browser_engine_set_custom_path',
    'plugin:oaiy-agent|agent_call_tool',
    'plugin:oaiy-agent|agent_set_api_base',
  ]) {
    await assert.rejects(() => broker.invoke(removed, {}), /not on the allowlist/, `must reject removed command ${removed}`);
  }

  // calculate_file_chunks allowed in trusted mode
  await broker.invoke('plugin:oaiy-filesystem|calculate_file_chunks', { path: '/tmp/a', chunkSize: 100, overlap: 0 });
  assert.equal(invoked.at(-1).cmd, 'plugin:oaiy-filesystem|calculate_file_chunks');

  // Per-command package permission enforcement
  rt.setFlowContext('f1', 'pkg-test');
  rt.setPackagePermissions(['network']);
  await assert.rejects(() => broker.invoke('plugin:oaiy-filesystem|read_file', { path: '/tmp/a' }), /filesystem:read/);
  await assert.rejects(() => broker.invoke('plugin:oaiy-filesystem|calculate_file_chunks', { path: '/tmp/a' }), /filesystem:read/);
  await assert.rejects(() => broker.invoke('plugin:oaiy-filesystem|write_file', { path: '/tmp/a' }), /filesystem/);
  await assert.rejects(() => broker.invoke('plugin:oaiy-filesystem|run_command', { command: 'ffmpeg' }), /process:execute/);
  await assert.rejects(() => broker.invoke('ensure_service_ready_by_port', { port: 8188 }), /service:start/);

  // Browser local-network URL interception (both browser_v2_http_request and browser_v2_goto)
  rt.setPackagePermissions(['browser', 'network']);
  await assert.rejects(
    () => broker.invoke('plugin:oaiy-browser|browser_v2_http_request', { request: { url: 'http://127.0.0.1:8188/api' } }),
    /Local network access denied/,
  );
  await assert.rejects(
    () => broker.invoke('plugin:oaiy-browser|browser_v2_goto', { sessionId: 's1', url: 'http://localhost:11434/' }),
    /Local network access denied/,
  );
  assert.equal(prompted.length, 2, 'must prompt local-network consent for browser_v2_http_request & browser_v2_goto');

  // Package with browser but WITHOUT network/network:local is rejected outright for local URLs
  rt.setPackagePermissions(['browser']);
  await assert.rejects(
    () => broker.invoke('plugin:oaiy-browser|browser_v2_goto', { sessionId: 's1', url: 'http://127.0.0.1:8188/' }),
    /network:local/,
  );
});

await check('17. NetworkUtils detects trailing dots, *.localhost, 127/8, 0/8, 100.64/10, IPv4-mapped IPv6, and parses [::1]:8080 whitelist', () => {
  for (const local of [
    'http://localhost./',
    'http://api.localhost:3000/v1',
    'http://sub.localhost./',
    'http://127.0.0.2:8080/',
    'http://127.255.255.254/',
    'http://0.0.0.1/',
    'http://100.64.0.1:8080/',
    'http://100.127.255.254/',
    'http://[::ffff:127.0.0.1]:8080/',
    'http://[::ffff:7f00:1]/',
    'http://[::ffff:10.0.0.5]/',
    'http://[::ffff:169.254.169.254]/latest/meta-data/',
    'http://[fe80::1]/',
    'http://[fd12:3456::1]/',
  ]) {
    assert.equal(M.isLocalNetworkUrl(local), true, `expected local/private: ${local}`);
  }
  for (const pub of [
    'https://example.com/',
    'http://100.63.255.255/',
    'http://100.128.0.1/',
    'http://[::ffff:8.8.8.8]/',
    'http://[2606:4700:4700::1111]/',
  ]) {
    assert.equal(M.isLocalNetworkUrl(pub), false, `expected public: ${pub}`);
  }
  assert.equal(M.matchesWhitelistEntry('[::1]:8080', '[::1]'), true);
  assert.equal(M.matchesWhitelistEntry('[::1]:8080', '[::1]:*'), true);
  assert.equal(M.matchesWhitelistEntry('[::1]:8080', '::1'), true);
  assert.equal(M.matchesWhitelistEntry('[::1]:8080', '[::1]:8080'), true);
  assert.equal(M.matchesWhitelistEntry('[::1]:8080', '[::1]:9000'), false);
  assert.equal(M.matchesWhitelistEntry('[::1]:8080', ''), false, 'empty whitelist entry must not match IPv6');
});

await check('18. desktopCredential blocks encoded/backslash traversal and anchors /api/voice/transcribe; keepFile sanitizes ext', async () => {
  const desktop = { origin: 'http://127.0.0.1:17972', token: 'secret-tok' };
  assert.equal(M.desktopCredential('http://127.0.0.1:17972/api/voice/transcribe', desktop), 'secret-tok');
  assert.equal(M.desktopCredential('http://127.0.0.1:17972/api/voice/transcribe/v1', desktop), 'secret-tok');
  assert.equal(M.desktopCredential('http://127.0.0.1:17972/api/voice/transcribe-admin', desktop), undefined);
  assert.equal(M.desktopCredential('http://127.0.0.1:17972/api/ai/engine/%2e%2e/secrets', desktop), undefined);
  assert.equal(M.desktopCredential('http://127.0.0.1:17972/api/ai/engine/a%2fb', desktop), undefined);
  assert.equal(M.desktopCredential('http://127.0.0.1:17972/api/ai/engine/a%5cb', desktop), undefined);
  assert.equal(M.desktopCredential('http://127.0.0.1:17972/api/ai/engine/a\\..\\secrets', desktop), undefined);

  let writtenPath = '';
  const kept = await M.keepFile(
    {
      log() {},
      tauri: {
        invoke: async (cmd, args) => {
          if (cmd === 'plugin:oaiy-filesystem|get_temp_dir') return '/tmp';
          if (cmd === 'plugin:oaiy-filesystem|write_file') { writtenPath = args.path; return ''; }
          if (cmd === 'get_media_url') return args.filePath;
          return '';
        },
      },
    },
    new Uint8Array([1, 2, 3]),
    'wav/../../evil.sh',
    '../clip',
  );
  assert.equal(writtenPath.includes('..'), false, `keepFile must sanitize ext & base: ${writtenPath}`);
  assert.match(kept.path, /^\/tmp\/oaiy-media\/_clip_\d+\.[A-Za-z0-9]{1,12}$/);
});

await check('19. FileSystem mints unforgeable file_ref tokens, validates Windows device paths, and clamps text_chunker overlap', async () => {
  const statuses = [];
  const tauriCalls = [];
  const ac = new AbortController();
  const fsMod = M.FilesystemRuntime.createMethods({
    log() {},
    abortSignal: ac.signal,
    onNodeStatus: (id, s) => statuses.push({ id, s }),
    tauri: {
      invoke: async (cmd, args) => {
        tauriCalls.push({ cmd, args });
        if (cmd === 'plugin:oaiy-filesystem|read_file') {
          return { content: '', size: 50_000_000, path: '/tmp/big.txt', name: 'big.txt', nameWithoutExt: 'big', ext: 'txt', isLargeFile: true };
        }
        if (cmd === 'plugin:oaiy-filesystem|calculate_file_chunks') {
          return [{ path: args.path, start: 0, length: 10, index: 0, total: 1 }];
        }
        if (cmd === 'plugin:oaiy-filesystem|read_chunk_content') {
          return 'chunk-body';
        }
        return null;
      },
    },
  });

  // Forged raw path or bogus token rejected
  await assert.rejects(() => fsMod.calculateFileChunks('/etc/passwd', 100, 0, 'n1'), /Invalid or unissued file reference token/);
  await assert.rejects(() => fsMod.readChunkContent('/etc/passwd', 0, 10, 'text'), /Invalid or unissued file reference token/);

  // Real readFile issues opaque token & emits completed status
  const refJson = await fsMod.readFile('/tmp/big.txt', 'text', 'read1');
  const parsedRef = JSON.parse(refJson);
  assert.equal(parsedRef.__type, 'file_ref');
  assert.equal(typeof parsedRef.token, 'string');
  assert.equal('path' in parsedRef, false, 'file_ref must not expose raw path');
  assert.ok(statuses.some((e) => e.id === 'read1' && e.s === 'completed'), 'large-file readFile must emit completed status');

  const chunks = await fsMod.calculateFileChunks(parsedRef.token, 100, 10, 'chunk1');
  assert.equal(chunks.length, 1);
  assert.equal(chunks[0].token, parsedRef.token);
  assert.equal('path' in chunks[0], false);
  const text = await fsMod.readChunkContent(chunks[0].token, 0, 10, 'text');
  assert.equal(text, 'chunk-body');

  // Abort check
  ac.abort();
  await assert.rejects(() => fsMod.calculateFileChunks(parsedRef.token, 100, 10, 'chunk2'), /aborted/i);
  await assert.rejects(() => fsMod.readChunkContent(parsedRef.token, 0, 10, 'text'), /aborted/i);

  // Windows path validation (unconditional)
  for (const bad of ['//./COM1', '//?/C:/secret', '\\\\.\\PhysicalDrive0', '/tmp/CON.txt', '/tmp/nul/file.txt', 'C:relative.txt']) {
    assert.throws(() => M.validatePath(bad, 'test'), `validatePath must reject ${bad}`);
  }
  assert.doesNotThrow(() => M.validatePath('/tmp/report..final.txt', 'test'));
  assert.doesNotThrow(() => M.validatePath('C:\\Users\\demo\\file.txt', 'test'));

  // text_chunker compiler clamps negative & NaN overlap
  const compiled = M.FilesystemCompiler.compileNode('text_chunker', {
    node: { id: 'tc1', type: 'text_chunker', data: { chunkSize: 100, overlap: -50 } },
    inputs: new Map([['default', 'node_a_out']]),
    outputVar: 'node_tc1_out',
    sanitizedId: 'tc1',
    skipVarDeclaration: false,
    escapeString: (s) => s,
    sanitizeId: (s) => s,
    isInLoop: false,
  });
  assert.ok(compiled.includes('_ref_tc1.token'), 'text_chunker compiler must use file_ref.token');
  assert.equal(compiled.includes('-50'), false, 'negative overlap must be clamped to >= 0');
});

await check('20. OAIYCompiler validates edge handles, logic_block validates handleId, and loops hoist multi-output vars', () => {
  const compiler = new M.OAIYCompiler();
  assert.throws(
    () => compiler.compile({
      nodes: [
        { id: 'a', type: 'input_text', position: { x: 0, y: 0 }, data: {} },
        { id: 'b', type: 'logic_block', position: { x: 100, y: 0 }, data: { code: 'input' } },
      ],
      edges: [{ id: 'e1', source: 'a', sourceHandle: 'text', target: 'b', targetHandle: 'x=1;evil()' }],
    }),
    /Invalid edge targetHandle/,
  );
  assert.throws(
    () => compiler.compile({
      nodes: [
        { id: 'a', type: 'input_text', position: { x: 0, y: 0 }, data: {} },
        { id: 'b', type: 'output', position: { x: 100, y: 0 }, data: {} },
      ],
      edges: [{ id: 'e1', source: 'a', sourceHandle: 'bad handle', target: 'b', targetHandle: 'result' }],
    }),
    /Invalid edge sourceHandle/,
  );

  // logic_block skips JS reserved words / non-identifier handleIds
  const lbCode = M.UtilityCompiler.compileNode('logic_block', {
    node: { id: 'lb1', type: 'logic_block', data: { code: 'input' } },
    inputs: new Map([['input', 'v0'], ['class', 'v1'], ['bad-name', 'v2'], ['goodVar', 'v3']]),
    outputVar: 'node_lb1_out',
    sanitizedId: 'lb1',
    skipVarDeclaration: false,
    escapeString: (s) => s,
    sanitizeId: (s) => s,
    isInLoop: false,
  });
  assert.equal(lbCode.includes('let class ='), false, 'must not emit let for JS reserved word handle');
  assert.equal(lbCode.includes('let bad-name ='), false, 'must not emit let for invalid identifier handle');
  assert.ok(lbCode.includes('let goodVar = v3;'), 'valid handle identifier must still be bound');

  // Multi-output node inside a loop pre-declares suffixed vars outside the loop and assigns inside
  const fakeRegistry = {
    getNodeDefinition: (type) => {
      if (type === 'model_3d') {
        return {
          id: 'model_3d',
          name: '3D',
          inputs: [{ id: 'prompt', name: 'prompt', type: 'string' }],
          outputs: [{ id: 'model', name: 'model', type: 'any' }, { id: 'path', name: 'path', type: 'string' }, { id: 'picture', name: 'picture', type: 'any' }],
          compiler: {},
        };
      }
      return undefined;
    },
    getModuleForNode: (type) => (type === 'model_3d' ? { compiler: M.Core3DCompiler, manifest: { id: 'core-3d' } } : undefined),
    getCompiler: (type) => (type === 'model_3d' ? M.Core3DCompiler : undefined),
    getAllNodeDefinitions: () => [],
  };
  const loopCompiler = new M.OAIYCompiler({ moduleRegistry: fakeRegistry });
  const script = loopCompiler.compile({
    nodes: [
      { id: 'ls', type: 'loop_start', position: { x: 0, y: 0 }, data: { iterations: 2, loopMode: 'count' } },
      { id: 'm1', type: 'model_3d', position: { x: 100, y: 0 }, data: { prompt: 'chair' } },
      { id: 'le', type: 'loop_end', position: { x: 200, y: 0 }, data: {} },
    ],
    edges: [
      { id: 'e1', source: 'ls', sourceHandle: 'item', target: 'm1', targetHandle: 'prompt' },
      { id: 'e2', source: 'm1', sourceHandle: 'path', target: 'le', targetHandle: 'input' },
      { id: 'e3', source: 'ls', sourceHandle: 'loop', target: 'le', targetHandle: 'loop' },
    ],
  });
  assert.ok(script.includes('let node_m1_out_path = null;'), 'loop must pre-declare suffixed output at outer scope');
  assert.equal(/\n  let node_m1_out_path = node_m1_out\.path;/.test(script), false, 'loop body must assign without re-declaring let');
  assert.ok(/\n  node_m1_out_path = node_m1_out\.path;/.test(script), 'loop body must assign pre-declared suffixed var');
});

await check('21. SandboxedCustomNodeUI sanitization & ProjectIO macro preservation + secret constant redaction', () => {
  const safeData = M.sanitizeDataForSandbox({
    label: 'My Node',
    count: 3,
    nested: { ok: true, fn: () => 1 },
    onChange: () => {},
    onDataChange: () => {},
    _status: 'running',
    projectConstants: [{ key: 'OPENAI_API_KEY', value: 'sk-secret', isSecret: true }],
    projectSettings: { apiKey: 'secret' },
    llmEndpoints: [{ id: 'x' }],
    availableFlows: [{ id: 'f1' }],
  });
  assert.deepEqual(safeData, { label: 'My Node', count: 3, nested: { ok: true } });
  // Must be structured-cloneable (no DataCloneError)
  assert.deepEqual(structuredClone(safeData), safeData);

  const cleanUpdates = M.sanitizeSandboxUpdates(
    JSON.parse('{"text":"hi","__proto__":{"polluted":true},"constructor":"bad","_status":"error","onChange":"x","projectConstants":[]}'),
  );
  assert.deepEqual(cleanUpdates, { text: 'hi' });
  assert.equal(M.sanitizeSandboxUpdates(['not-an-object']), null);
  assert.equal(M.sanitizeSandboxUpdates(new Date()), null);

  // ProjectIO preserves _macro* & _collapsed and redacts api_key / looksLikeSecret constants even if isSecret:false
  const fakeKey = `sk-${'z'.repeat(30)}`;
  const proj = {
    version: '1.0',
    name: 'Test',
    createdAt: '',
    updatedAt: '',
    flows: [{
      id: 'f1',
      name: 'F1',
      createdAt: '',
      updatedAt: '',
      graph: {
        nodes: [{
          id: 'm1',
          type: 'macro',
          position: { x: 0, y: 0 },
          data: {
            _macroWorkflowId: 'wf-1',
            _macroName: 'MyMacro',
            _macroInputs: [{ id: 'in1', name: 'x', type: 'text' }],
            _macroOutputs: [{ id: 'out1', name: 'y', type: 'text' }],
            _collapsed: true,
            _status: 'running',
            _temp: 'drop-me',
          },
        }],
        edges: [],
      },
    }],
    constants: [
      { id: 'c1', key: 'API_KEY_UNFLAGGED', value: 'plain-api-key', category: 'api_key', isSecret: false },
      { id: 'c2', key: 'LOOKS_SECRET', value: fakeKey, category: 'custom', isSecret: false },
      { id: 'c3', key: 'NORMAL', value: 'hello', category: 'custom', isSecret: false },
    ],
  };
  const exported = M.sanitizeProjectForExport(proj);
  const macroData = exported.flows[0].graph.nodes[0].data;
  assert.equal(macroData._macroWorkflowId, 'wf-1');
  assert.equal(macroData._macroName, 'MyMacro');
  assert.deepEqual(macroData._macroInputs, [{ id: 'in1', name: 'x', type: 'text' }]);
  assert.deepEqual(macroData._macroOutputs, [{ id: 'out1', name: 'y', type: 'text' }]);
  assert.equal(macroData._collapsed, true);
  assert.equal('_status' in macroData, false);
  assert.equal('_temp' in macroData, false);
  assert.equal(exported.constants[0].value, '', 'category=api_key constant must be cleared on export');
  assert.equal(exported.constants[1].value, '', 'looksLikeSecret constant must be cleared on export');
  assert.equal(exported.constants[2].value, 'hello');

  const imported = M.parseImportedProject(JSON.stringify(proj));
  assert.equal(imported.flows[0].untrusted, true, 'imported project flows must be marked untrusted');
  const c1 = imported.constants.find((c) => c.key === 'API_KEY_UNFLAGGED');
  const c2 = imported.constants.find((c) => c.key === 'LOOKS_SECRET');
  assert.equal(c1.isSecret, true);
  assert.equal(c2.isSecret, true);
});

await check('22. core-audio & core-video sanitize paths/colors, parse tokenized media URLs, and drop network protocols from FFmpeg appendVideos', async () => {
  assert.equal(
    M.resolveLocalMediaUrl('http://127.0.0.1:8765/media/tok_secret_abcdef123456/oaiy-output/clip.wav', '/data/app'),
    '/data/app/output/clip.wav',
  );
  assert.equal(
    M.resolveLocalMediaUrl('http://127.0.0.1:8765/media/oaiy-output/sub/clip.wav', '/data/app'),
    '/data/app/output/sub/clip.wav',
  );
  assert.throws(
    () => M.resolveLocalMediaUrl('http://127.0.0.1:8765/media/tok/oaiy-output/%2e%2e/secret.txt', '/data/app'),
    /path traversal/,
  );
  assert.equal(
    M.redactMediaToken('http://127.0.0.1:8765/media/tok_secret_abcdef123456/oaiy-output/clip.wav').includes('tok_secret'),
    false,
  );

  assert.equal(M.sanitizeAudioFilename('../../etc/evil', 'clip').includes('..'), false);
  assert.equal(M.sanitizeAudioFormat('wav;rm -rf /', 'wav'), 'wav');
  assert.equal(M.sanitizeVideoFilename('../out/bad', 'vid').includes('..'), false);
  assert.equal(M.sanitizeVideoFormat('mp4/../../x', 'mp4'), 'mp4');
  assert.equal(M.sanitizeFfmpegColor("white:text='injected'", 'white'), 'white');
  assert.equal(M.sanitizeFfmpegColor('rgba(255, 0, 128, 0.5)', 'black@0.7'), '0xff0080@0.5');

  // appendVideos stages remote http(s) URL via secureFetch and passes -protocol_whitelist file ONLY
  const secureFetched = [];
  let ffmpegArgs = null;
  const vidMod = M.VideoRuntime.createMethods({
    log() {},
    secureFetch: async (url) => {
      secureFetched.push(url);
      return mockResp(200, 'video-bytes', { 'content-type': 'video/mp4' });
    },
    tauri: {
      invoke: async (cmd, args) => {
        if (cmd === 'plugin:oaiy-filesystem|get_app_data_dir') return '/app';
        if (cmd === 'plugin:oaiy-filesystem|get_temp_dir') return '/tmp';
        if (cmd === 'plugin:oaiy-filesystem|run_command') {
          ffmpegArgs = args.args;
          return { code: 0, stdout: '', stderr: '' };
        }
        return null;
      },
    },
  });
  await vidMod.appendVideos(['https://cdn.example.com/a.mp4', '/app/b.mp4'], '../evil', 'mp4;bad', 'v1');
  assert.deepEqual(secureFetched, ['https://cdn.example.com/a.mp4'], 'remote video must be fetched through ctx.secureFetch');
  const wlIdx = ffmpegArgs.indexOf('-protocol_whitelist');
  assert.equal(ffmpegArgs[wlIdx + 1], 'file', 'FFmpeg -protocol_whitelist must only allow file (no http/https/tcp/tls)');
});

await check('23. core-database, core-agent, core-flow-control, core-utility, sql.ts, oaiyAgentTools, selection-analyzer & TrustDialog fixes', async () => {
  // core-database: abort returns __ABORT__, missing db throws, compiler avoids double-stringify
  const abortedAc = new AbortController();
  abortedAc.abort();
  const dbAborted = M.DatabaseRuntime.createMethods({ log() {}, abortSignal: abortedAc.signal });
  assert.equal(await dbAborted.execute('query', 'items', null, '{}', 'd1'), '__ABORT__');
  const dbMissing = M.DatabaseRuntime.createMethods({ log() {} });
  await assert.rejects(() => dbMissing.execute('query', 'items', null, '{}', 'd2'), /Database not available/);
  const dbCode = M.DatabaseCompiler.compileNode('database_query', {
    node: { id: 'q1', type: 'database_query', data: { collectionName: 'c' } },
    inputs: new Map([['filter', 'node_f_out']]),
    outputVar: 'node_q1_out',
    sanitizedId: 'q1',
    skipVarDeclaration: false,
    escapeString: (s) => s,
    sanitizeId: (s) => s,
    isInLoop: false,
  });
  assert.ok(dbCode.includes('typeof node_f_out === "string" ? node_f_out : JSON.stringify'));

  // core-agent: $& preserved literally; ask sets error status on failure
  assert.equal(M.taskText('cost is $& and $$10', 'Check: {{input}}', null), 'Check: cost is $& and $$10');
  const agentStatuses = [];
  const agentMod = M.AgentRuntime.createMethods({
    log() {},
    onNodeStatus: (id, s) => agentStatuses.push({ id, s }),
  });
  await assert.rejects(() => agentMod.ask('hi', '', null, 'main', 5, 'a1'));
  assert.deepEqual(agentStatuses, [{ id: 'a1', s: 'running' }, { id: 'a1', s: 'error' }]);

  // core-flow-control: missing callback throws, abort throws, self-recursion from currentFlowId caught
  const subNoCb = M.SubflowRuntime.createMethods({ log() {}, currentFlowId: 'flow-self' });
  await assert.rejects(() => subNoCb.execute('child', {}, 's1'), /No subflow callback/);
  const subSelf = M.SubflowRuntime.createMethods({
    log() {},
    currentFlowId: 'flow-self',
    runSubflow: async () => ({}),
  });
  await assert.rejects(() => subSelf.execute('flow-self', {}, 's2'), /Recursive subflow/);
  const subAbort = M.SubflowRuntime.createMethods({
    log() {},
    abortSignal: abortedAc.signal,
    runSubflow: async () => ({}),
  });
  await assert.rejects(() => subAbort.execute('child', {}, 's3'), /aborted/i);

  // core-utility: URL-encodes serviceId in getServiceUrl / ensureService
  const utilUrls = [];
  const utilMod = M.UtilityRuntime.createMethods({
    log() {},
    secureFetch: async (url) => {
      utilUrls.push(url);
      return mockResp(404, {});
    },
  });
  await utilMod.getServiceUrl('../admin?x=1');
  await utilMod.ensureService('../admin#hash');
  assert.ok(utilUrls[0].includes('%2Fadmin%3Fx%3D1'), `expected encoded serviceId: ${utilUrls[0]}`);
  assert.ok(utilUrls[1].includes('%2Fadmin%23hash'), `expected encoded serviceId: ${utilUrls[1]}`);

  // sql.ts: $1name is not rewritten while $1 is
  assert.equal(M.normalizeSql('SELECT $1name, $1, $2_col, $2 FROM t'), 'SELECT $1name, ?1, $2_col, ?2 FROM t');

  // oaiyAgentTools: withdrawTool / withdrawHook throw on !resp.ok
  const origFetch = globalThis.fetch;
  globalThis.window.__OAIY_DESKTOP__ = { origin: 'http://127.0.0.1:17972', token: 'tok' };
  globalThis.fetch = async () => mockResp(500, 'server error');
  try {
    const fakeFlow = { id: 'f1', name: 'F1', createdAt: '', updatedAt: '', graph: { nodes: [], edges: [] } };
    await assert.rejects(() => M.withdrawTool(fakeFlow), /HTTP 500/);
    await assert.rejects(() => M.withdrawHook(fakeFlow), /HTTP 500/);
  } finally {
    globalThis.fetch = origFetch;
    delete globalThis.window.__OAIY_DESKTOP__;
  }

  // selection-analyzer: populates _macroName, _macroInputs, _macroOutputs & unique parentEdges IDs
  const graph = {
    nodes: [
      { id: 'src', type: 'input_text', position: { x: 0, y: 0 }, data: {} },
      { id: 'mid', type: 'template', position: { x: 200, y: 0 }, data: {} },
      { id: 'dst', type: 'output', position: { x: 400, y: 0 }, data: {} },
    ],
    edges: [
      { id: 'e1', source: 'src', sourceHandle: 'text', target: 'mid', targetHandle: 'input' },
      { id: 'e2', source: 'src', sourceHandle: 'text', target: 'mid', targetHandle: 'input2' },
      { id: 'e3', source: 'mid', sourceHandle: 'result', target: 'dst', targetHandle: 'result' },
    ],
  };
  const analysis = M.analyzeSelection(graph.nodes, graph.edges, ['mid']);
  const created = M.convertSelectionToMacro(analysis, 'macro-1', 'MyMacro', {}, {});
  assert.equal(created.macroNode.data._macroName, 'MyMacro');
  assert.equal(created.macroNode.data._macroInputs.length, 2);
  assert.equal(created.macroNode.data._macroOutputs.length, 1);
  const edgeIds = created.parentEdges.map((e) => e.id);
  assert.equal(new Set(edgeIds).size, edgeIds.length, 'parentEdges IDs must be unique');

  // customNodeCompiler blocks all @tauri-apps/* & TrustDialog high-risk + diffusion
  const compilerSrc = fs.readFileSync(path.join(UI, 'src/services/customNodeCompiler.ts'), 'utf8');
  assert.ok(compilerSrc.includes('/^@tauri-apps(?:\\/|$)/'), 'customNodeCompiler must block all @tauri-apps/* imports');
  const trustSrc = fs.readFileSync(path.join(UI, 'src/components/PackageManager/TrustDialog.tsx'), 'utf8');
  assert.ok(trustSrc.includes("'filesystem:read'") && trustSrc.includes("'network:local'"), 'TrustDialog must include filesystem:read & network:local in HIGH_RISK_PERMISSIONS');
  assert.ok(/diffusion\s*:/.test(trustSrc), 'TrustDialog must define PERMISSION_INFO entry for diffusion');
});

finish();
