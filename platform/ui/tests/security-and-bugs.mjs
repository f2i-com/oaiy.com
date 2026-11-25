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
export function registerNodeComponent() {}
export function unregisterNodeComponent() {}
export function refreshNodeTypes() {}
export function getRegisteredNodeTypes() { return []; }
export function GenericNode() { return null; }
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
    build.onResolve({ filter: /^(?:\.\/|\.\.\/ui\/)Dialog$/ }, (a) => (
      a.importer.endsWith('RunWorkflowModal.tsx') || a.importer.endsWith('TrustDialog.tsx')
        ? { path: 'dialog-ui', namespace: 'stub' }
        : null
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
    build.onResolve({ filter: /\?url$/ }, (a) => ({ path: a.path, namespace: 'asset-url' }));
    build.onLoad({ filter: /.*/, namespace: 'asset-url' }, (a) => ({ contents: `export default ${JSON.stringify(a.path)};`, loader: 'js' }));
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
export { default as BrowserCompiler } from './src/bundled-modules/core-browser/compiler.ts';
export { default as ServiceRuntime } from './src/bundled-modules/core-service/runtime.ts';
export { default as AIRuntime } from './src/bundled-modules/core-ai/runtime.ts';
export { default as AudioRuntime, sanitizeAudioFilename, sanitizeAudioFormat, resolveLocalMediaUrl, redactMediaToken } from './src/bundled-modules/core-audio/runtime.ts';
export { default as ImageRuntime } from './src/bundled-modules/core-image/runtime.ts';
export { default as ImageCompiler } from './src/bundled-modules/core-image/compiler.ts';
export { default as VideoRuntime, sanitizeVideoFilename, sanitizeVideoFormat, sanitizeFfmpegColor } from './src/bundled-modules/core-video/runtime.ts';
export { default as VideoCompiler } from './src/bundled-modules/core-video/compiler.ts';
export { default as SubflowRuntime } from './src/bundled-modules/core-flow-control/runtime.ts';
export { default as FlowControlCompiler } from './src/bundled-modules/core-flow-control/compiler.ts';
export { kindOfAddress } from './src/bundled-modules/core-flow-control/ui/outputValue.ts';
export { default as Model3DRuntime } from './src/bundled-modules/core-3d/runtime.ts';
export { default as AgentRuntime, taskText } from './src/bundled-modules/core-agent/runtime.ts';
export { default as FilesystemRuntime, validatePath } from './src/bundled-modules/core-filesystem/runtime.ts';
export { default as FilesystemCompiler } from './src/bundled-modules/core-filesystem/compiler.ts';
export { default as DatabaseRuntime } from './src/bundled-modules/core-database/runtime.ts';
export { default as DatabaseCompiler } from './src/bundled-modules/core-database/compiler.ts';
export { default as UtilityRuntime } from './src/bundled-modules/core-utility/runtime.ts';
export { default as UtilityCompiler } from './src/bundled-modules/core-utility/compiler.ts';
export { default as Core3DCompiler } from './src/bundled-modules/core-3d/compiler.ts';
export { resolveService } from './src/bundled-modules/core-service/contract.ts';
export { desktopCredential, desktopFileBytes, listOaiyServices, runContract, keepFile } from './src/bundled-modules/core-service/contractRuntime.ts';
export { OAIYRuntime, OAIYCompiler, JobManager, getModuleLoader, createTemplateCompiler } from 'oaiy-core';
export { escapeString } from 'oaiy-core/src/compiler/utils.ts';
export { detectSandboxEscape, stripCommentsAndStringsFromCode } from 'oaiy-core/src/runtime.ts';
export { isLocalNetworkUrl, matchesWhitelistEntry, getHostPort } from 'oaiy-core/src/runtime/NetworkUtils.ts';
export { convertSelectionToMacro, analyzeSelection } from 'oaiy-core/src/macro/selection-analyzer.ts';
export { parseImportedProject, registerImportedServices, sanitizeProjectForExport, stripUnsafeImportedSettings, sanitizeProjectRecord, looksLikeSecret, isSecretConstant, UNSAFE_IMPORTED_SETTING_KEYS } from './src/utils/ProjectIO.ts';
export { listAllServices, deleteService } from './src/utils/serviceRegistry.ts';
export { FLOW_REFS, reachesUntrusted, rewriteFlowRefs } from './src/lib/flowIds.ts';
export { createPrefixedNodeId } from './src/hooks/usePackageNodes.ts';
export { reloadRequiresTrustReconfirm } from './src/hooks/usePackageManager.ts';
export { filterNodeData, prepareWorkflowForExport } from './src/utils/WorkflowIO.ts';
export { sanitizeDataForSandbox, sanitizeSandboxUpdates } from './src/lib/sandboxData.ts';
export { publishAsTool, publishAsHook, withdrawTool, withdrawHook, isUntrustedOrPkgFlow } from './src/lib/oaiyAgentTools.ts';
export { isReadOnlySql } from './src/lib/sqlReadOnly.ts';
export { normalizeSql } from './src/tauri-shim/sql.ts';
export { PermissionRequestQueue } from './src/lib/permissionQueue.ts';
export { default as RunWorkflowModal } from './src/components/ui/RunWorkflowModal.tsx';
export { QuickConnectPopup } from './src/components/OAIYBuilder/QuickConnectPopup.tsx';
export { TrustDialog } from './src/components/PackageManager/TrustDialog.tsx';
export { parseConnectorConfig, ConnectorConfigError } from './src/connector-module/config.ts';
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
    fetch: async () => mockResp(200, { audio: 'UklGRg==' }, { 'content-type': 'application/json' }),
  };
  await M.AudioRuntime.createMethods(audioCtx).textToSpeechV2({
    nodeId: 'tts1',
    text: 'hello world',
    provider: 'external',
    externalUrl: 'https://tts.example.com/speak?key={{apiKey}}',
    externalMethod: 'POST',
    externalHeaders: '{}',
    externalBodyTemplate: '{"text": {{text}}}',
    externalResponseType: 'base64_wav',
    externalResponsePath: 'audio',
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
  // `network` does NOT imply `network:local`, so explicit `network:local` is required here.
  rt.setPackagePermissions(['browser', 'network:local']);
  await assert.rejects(
    () => broker.invoke('plugin:oaiy-browser|browser_v2_http_request', { request: { url: 'http://127.0.0.1:8188/api' } }),
    /Local network access denied/,
  );
  await assert.rejects(
    () => broker.invoke('plugin:oaiy-browser|browser_v2_goto', { sessionId: 's1', url: 'http://localhost:11434/' }),
    /Local network access denied/,
  );
  assert.equal(prompted.length, 2, 'must prompt local-network consent for browser_v2_http_request & browser_v2_goto');

  // Package with browser + network (WITHOUT network:local) is rejected outright for local URLs
  rt.setPackagePermissions(['browser', 'network']);
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

// ---------------------------------------------------------------------------
// Wave-3 Security & Correctness Regressions
// ---------------------------------------------------------------------------
await check('24. Wave-3: condition AST breakout rejection, pre-run code:execute gate, and template-literal stripCommentsAndStringsFromCode', async () => {
  const compileCond = (expression) =>
    M.FlowControlCompiler.compileNode('condition', {
      node: { id: 'c1', type: 'condition', data: { expression } },
      inputs: new Map([['input', 'node_in']]),
      outputVar: 'node_c1_out',
      sanitizedId: 'c1',
      skipVarDeclaration: false,
      escapeString: (s) => s,
      sanitizeId: (s) => s,
      isInLoop: false,
    });

  // Malicious statement/paren breakouts must be rejected by Acorn AST validation
  for (const badExpr of [
    'true); evil(); (true',
    '1) + (2',
    '1), (2',
    'true); } evil(); { (true',
    'let x = 1',
    'if (true) {}',
  ]) {
    assert.throws(() => compileCond(badExpr), /Condition \(c1\):/, `must reject breakout expression: ${badExpr}`);
  }

  // Legitimate JS expressions (including trailing single-line comments and object literals) compile cleanly
  for (const goodExpr of [
    'input && input.length > 0',
    'input === "ok" // trailing comment',
    '{ a: 1 }.a === 1',
    '(() => true)()',
  ]) {
    const code = compileCond(goodExpr);
    assert.ok(code.includes('Utility.requireCodeExecute'), `valid expression should compile: ${goodExpr}`);
  }

  // Pre-run code:execute gate rejects graphs with logic_block or condition expression BEFORE compiling/running
  const rt = new M.OAIYRuntime({ useWorkerForUntrusted: false });
  rt.setFlowContext('f1', 'pkg-untrusted');
  rt.setPackagePermissions(['network']);
  await assert.rejects(
    () =>
      rt.runWorkflow({
        nodes: [{ id: 'lb1', type: 'logic_block', position: { x: 0, y: 0 }, data: { code: '1 + 1' } }],
        edges: [],
      }),
    /lacks required permission 'code:execute'.*logic_block/,
  );
  await assert.rejects(
    () =>
      rt.runWorkflow({
        nodes: [{ id: 'c_expr', type: 'condition', position: { x: 0, y: 0 }, data: { expression: 'true); evil(); (true' } }],
        edges: [],
      }),
    /lacks required permission 'code:execute'.*condition expression/,
    'must reject condition expression BEFORE compiling (so breakout syntax never even reaches compiler)',
  );

  // stripCommentsAndStringsFromCode preserves/recurses into template literal ${...} expressions
  assert.equal(M.detectSandboxEscape('const s = `globalThis`;'), null, 'static template text must not false-positive');
  assert.equal(M.detectSandboxEscape('const s = `\\${globalThis}`;'), null, 'escaped ${} must not false-positive');
  assert.equal(M.detectSandboxEscape('const s = `${globalThis}`;'), 'globalThis', 'must detect escape inside ${...}');
  assert.equal(
    M.detectSandboxEscape('const s = `outer ${`inner ${eval("1")}`}`;'),
    'eval()',
    'must detect escape inside nested template literal ${...}',
  );
  assert.equal(
    M.detectSandboxEscape('const s = `${(() => { const x = "}"; /* } */ return globalThis; })()}`;'),
    'globalThis',
    'must track braces, strings, and comments inside ${...}',
  );
  const stripped = M.stripCommentsAndStringsFromCode('`plain ${foo("bar")} tail`');
  assert.equal(stripped, '`${foo("")}`');
});

await check('25. Wave-3: network does NOT imply network:local, complete broker permissions, ctx.modules/callModule gating, and proxy local-network checks', async () => {
  const invoked = [];
  const rt = new M.OAIYRuntime({
    useWorkerForUntrusted: false,
    callbacks: {
      onLocalNetworkPermission: async () => ({ allowed: false, remember: false }),
    },
  });
  rt.setTauriInvoke(async (cmd, args) => {
    invoked.push({ cmd, args });
    return { ok: true, status: 200, headers: {}, body: 'ok' };
  });
  let probeCtx;
  await rt.registerDynamicModule({
    manifest: { id: 'probe-w3', name: 'ProbeW3', version: '1.0.0', nodes: [] },
    runtime: {
      name: 'Filesystem',
      createMethods(ctx) {
        probeCtx = ctx;
        return {
          readFile: () => 'secret-file-contents',
        };
      },
      methods: {},
    },
  });

  rt.setFlowContext('f1', 'pkg-net-only');
  rt.setPackagePermissions(['network']);
  assert.equal(rt.hasPackagePermission('network'), true);
  assert.equal(rt.hasPackagePermission('network:local'), false, '`network` must NOT imply `network:local`');

  // Newly added broker command permissions
  for (const [cmd, perm] of [
    ['get_media_url', 'filesystem:read'],
    ['get_video_info', 'filesystem:read'],
    ['extract_video_frames', 'filesystem:read'],
    ['extract_video_frames_batch', 'filesystem:read'],
    ['resize_image', 'filesystem:read'],
    ['cleanup_temp_dir', 'filesystem'],
    ['get_service_port', 'service:start'],
  ]) {
    await assert.rejects(
      () => probeCtx.tauri.invoke(cmd, {}),
      new RegExp(perm.replace(':', ':')),
      `broker must require ${perm} for ${cmd}`,
    );
  }

  // http_request requires 'network' when package lacks it
  rt.setPackagePermissions([]);
  await assert.rejects(
    () => probeCtx.tauri.invoke('http_request', { request: { url: 'https://example.com/' } }),
    /required permission 'network'/,
  );
  await assert.rejects(
    () => probeCtx.secureFetch('https://example.com/'),
    /lacks required permission 'network'/,
    'performHttpRequest must enforce network permission for non-local URLs',
  );

  // ctx.modules and ctx.callModule enforce REQUIRED_PERMISSIONS_BY_KIND
  assert.throws(() => probeCtx.modules.Filesystem.readFile('/etc/passwd'), /filesystem:read/);
  assert.throws(() => probeCtx.callModule('Filesystem', 'readFile', '/etc/passwd'), /filesystem:read/);
  rt.setPackagePermissions(['filesystem:read']);
  assert.equal(probeCtx.modules.Filesystem.readFile('/tmp/ok'), 'secret-file-contents');
  assert.equal(probeCtx.callModule('Filesystem', 'readFile', '/tmp/ok'), 'secret-file-contents');

  // Browser proxy local-network gating in buildModuleInvokeBroker
  rt.setPackagePermissions(['browser']);
  await assert.rejects(
    () => probeCtx.tauri.invoke('plugin:oaiy-browser|browser_v2_create_session', {
      config: { mode: 'chromium', proxy: { type: 'http', host: '127.0.0.1', port: 8080 } },
    }),
    /network:local/,
  );
  await assert.rejects(
    () => probeCtx.tauri.invoke('plugin:oaiy-browser|browser_v2_http_request', {
      request: { url: 'https://example.com', options: { proxy: 'socks5://127.0.0.1:1080' } },
    }),
    /network:local/,
  );
});

await check('26. Wave-3: JobManager forces hardened sandboxing on pkg:* custom nodes (direct & transitive) with intersected permissions', async () => {
  const seenLogs = [];
  const lookupCalls = [];
  let capturedCtx = null;
  const fakePkgCompiler = {
    name: 'FakePkg',
    getNodeTypes: () => ['pkg:com.alpha:fetch', 'pkg:com.beta:transform'],
    compileNode: (_type, ctx) => `let ${ctx.outputVar} = "ok";`,
  };
  const fakeModuleRegistry = {
    nodeToModule: new Map([
      ['pkg:com.alpha:fetch', 'fake-pkg'],
      ['pkg:com.beta:transform', 'fake-pkg'],
      ['subflow', 'core-flow-control'],
    ]),
    getNodeDefinition: (type) => {
      if (type.startsWith('pkg:')) {
        return { id: type, name: type, inputs: [], outputs: [{ id: 'out', name: 'out', type: 'string' }], compiler: {} };
      }
      if (type === 'subflow') {
        return { id: 'subflow', name: 'Subflow', inputs: [], outputs: [{ id: 'result', name: 'result', type: 'any' }], compiler: {} };
      }
      return undefined;
    },
    getModuleForNode: (type) => {
      if (type.startsWith('pkg:')) return { compiler: fakePkgCompiler, manifest: { id: 'fake-pkg' } };
      if (type === 'subflow') return { compiler: M.FlowControlCompiler, manifest: { id: 'core-flow-control' } };
      return undefined;
    },
    getCompiler: (type) => {
      if (type.startsWith('pkg:')) return fakePkgCompiler;
      if (type === 'subflow') return M.FlowControlCompiler;
      return undefined;
    },
    getAllNodeDefinitions: () => [],
    getAllModules: () => [
      {
        manifest: { id: 'probe-pkg', name: 'ProbePkg', version: '1.0.0', nodes: [] },
        runtime: {
          name: 'ProbePkg',
          createMethods(ctx) {
            capturedCtx = ctx;
            return {};
          },
          methods: {},
        },
      },
    ],
  };
  const jm = new M.JobManager({
    databaseHandler: async () => ({ success: true, data: [] }),
    moduleRegistry: fakeModuleRegistry,
    packagePermissionsLookup: (pkgId) => {
      lookupCalls.push(pkgId);
      if (pkgId === 'com.alpha') return ['network', 'filesystem:read'];
      if (pkgId === 'com.beta') return ['network'];
      return null;
    },
  });
  jm.setAvailableFlows([
    {
      id: 'sub-with-pkg',
      name: 'SubPkg',
      createdAt: '',
      updatedAt: '',
      graph: {
        nodes: [{ id: 'p2', type: 'pkg:com.beta:transform', position: { x: 0, y: 0 }, data: {} }],
        edges: [],
      },
    },
  ]);
  jm.onLog((_jobId, entry) => seenLogs.push(entry));
  const jobId = jm.submit('regular-flow', 'Regular with pkg node', {
    nodes: [
      { id: 'p1', type: 'pkg:com.alpha:fetch', position: { x: 0, y: 0 }, data: {} },
      { id: 's1', type: 'subflow', position: { x: 100, y: 0 }, data: { flowId: 'sub-with-pkg' } },
    ],
    edges: [],
  });
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
  assert.match(
    String(finalJob.error || ''),
    /Refusing to run untrusted package code in-thread/,
    `JobManager must force hardened Worker/Zipp execution for flows with pkg:* nodes (got: ${finalJob.error})`,
  );
  assert.deepEqual(new Set(lookupCalls), new Set(['com.alpha', 'com.beta']));
  assert.ok(capturedCtx && capturedCtx.currentPackageId, 'runtime must receive non-null effectivePackageId');
  assert.equal(capturedCtx.hasPermission('network'), true, 'intersection keeps shared permission network');
  assert.equal(capturedCtx.hasPermission('filesystem:read'), false, 'intersection drops permission missing from com.beta');
  assert.equal(capturedCtx.hasPermission('network:local'), false, 'intersection never grants network:local');
  assert.ok(seenLogs.some((l) => l.source === 'Security' && /custom package node/i.test(l.message)));

  // Unconfirmed package defaults to [] permissions
  capturedCtx = null;
  const unconfirmedJobId = jm.submit('regular-unconfirmed', 'Unconfirmed pkg', {
    nodes: [{ id: 'p3', type: 'pkg:com.unconfirmed:run', position: { x: 0, y: 0 }, data: {} }],
    edges: [],
  });
  await new Promise((resolve) => {
    const unsub = jm.onStateChange((jobs) => {
      const j = jobs.find((x) => x.id === unconfirmedJobId);
      if (j && (j.status === 'completed' || j.status === 'failed' || j.status === 'aborted')) {
        unsub();
        resolve(j);
      }
    });
  });
  assert.equal(capturedCtx.currentPackageId, 'com.unconfirmed');
  assert.equal(capturedCtx.hasPermission('network'), false, 'unconfirmed package must default to [] permissions');
});

await check('27. Wave-3: __CHUNK_REF: arbitrary file reader removed from core-ai, desktopFileBytes path/token/auth hardening, core-audio file_path & oaiyAgentTools untrusted blocking', async () => {
  // 1) core-ai must NOT parse or read __CHUNK_REF: from prompts
  const tauriCalls = [];
  const sentBodies = [];
  const aiMod = M.AIRuntime.createMethods({
    log() {},
    onNodeStatus() {},
    tauri: {
      invoke: async (cmd, args) => {
        tauriCalls.push({ cmd, args });
        return 'SECRET_FILE_CONTENTS';
      },
    },
    secureFetch: async (_url, init) => {
      sentBodies.push(String(init?.body || ''));
      return mockResp(200, { choices: [{ message: { content: 'ok' } }] });
    },
  });
  const chunkPayload = '__CHUNK_REF:{"path":"/etc/passwd","index":0,"total":1,"startByte":0,"endByte":100}__';
  const aiOut = await aiMod.chat('', chunkPayload, null, 'https://api.example.com/v1/chat/completions', 'm', '', false, 32, 0, 'text', false, 'auto', 'n1');
  assert.equal(aiOut, 'ok');
  assert.equal(
    tauriCalls.some((c) => c.cmd.includes('read_chunk_content')),
    false,
    'core-ai must never invoke read_chunk_content on __CHUNK_REF: prompts',
  );
  assert.ok(sentBodies[0].includes('__CHUNK_REF:'), 'literal prompt text should pass through unmodified');

  // 2) desktopFileBytes restricts paths, routes through ctx.secureFetch, and omits desktop.token for packages lacking network:local
  const origDesktop = globalThis.__OAIY_DESKTOP__;
  globalThis.__OAIY_DESKTOP__ = { origin: 'http://127.0.0.1:17972', token: 'desktop-secret' };
  try {
    // Non-file desktop API route returns null (even though desktopCredential allows it for contract calls)
    const nonFile = await M.desktopFileBytes('http://127.0.0.1:17972/api/ai/engine/services');
    assert.equal(nonFile, null, 'desktopFileBytes must reject non-file-serving routes');

    // Allowed file-serving path routes through ctx.secureFetch and omits desktop.token when package lacks network:local
    const capturedHeaders = [];
    const fileRes = await M.desktopFileBytes(
      'http://127.0.0.1:17972/files/demo.png',
      undefined,
      {
        log() {},
        currentPackageId: 'pkg-no-local',
        hasPermission: () => false,
        secureFetch: async (_u, init) => {
          capturedHeaders.push(init?.headers || {});
          return mockResp(200, 'png-bytes', { 'content-type': 'image/png' });
        },
      },
    );
    assert.ok(fileRes && fileRes.bytes.length > 0);
    assert.equal(capturedHeaders[0].Authorization, undefined, 'must NOT attach desktop.token when package lacks network:local');

    // Strips case-insensitive lowercase authorization before setting Authorization in contractRuntime send()
    const sendHeaders = [];
    await M.runContract(
      {
        log() {},
        secureFetch: async (_u, init) => {
          sendHeaders.push(init?.headers || {});
          return mockResp(200, { data: [{ b64_json: 'QUJD' }] });
        },
      },
      {
        id: 'engine:image:test',
        name: 'Test',
        endpoint: 'http://127.0.0.1:17972/api/ai/engine/gateway/v1/images/generations',
        method: 'POST',
        headers: JSON.stringify({ authorization: 'Bearer attacker-forged' }),
        bodyTemplate: '{}',
        responseType: 'json',
        responsePath: 'data.0.b64_json',
        output: 'image',
      },
      {},
      { nodeId: 'n1' },
    );
    assert.equal(sendHeaders[0].authorization, undefined, 'lowercase authorization must be stripped');
    assert.equal(sendHeaders[0].Authorization, 'Bearer desktop-secret');
  } finally {
    if (origDesktop === undefined) delete globalThis.__OAIY_DESKTOP__;
    else globalThis.__OAIY_DESKTOP__ = origDesktop;
  }

  // 3) core-audio rejects file_path from non-local endpoints and never falls back to get_media_url on permission denial
  const mediaUrlCalls = [];
  const audioMod = M.AudioRuntime.createMethods({
    log() {},
    onNodeStatus() {},
    fetch: async () => mockResp(200, { audio_path: '/etc/passwd' }),
    tauri: {
      invoke: async (cmd) => {
        if (cmd === 'plugin:oaiy-filesystem|get_app_data_dir') return '/app';
        if (cmd === 'plugin:oaiy-filesystem|native_copy_file') {
          throw new Error("Refusing module invoke 'plugin:oaiy-filesystem|native_copy_file': package 'p' lacks required permission 'filesystem'");
        }
        if (cmd === 'get_media_url') {
          mediaUrlCalls.push(cmd);
          return 'http://127.0.0.1/media/leaked';
        }
        return null;
      },
    },
  });
  await assert.rejects(
    () =>
      audioMod.textToSpeechV2({
        provider: 'external',
        text: 'hi',
        externalUrl: 'https://remote-attacker.example/tts',
        externalMethod: 'POST',
        externalHeaders: '{}',
        externalBodyTemplate: '{}',
        externalResponseType: 'file_path',
        externalResponsePath: 'audio_path',
        nodeId: 'tts1',
      }),
    /non-local endpoint/,
  );
  await assert.rejects(
    () =>
      audioMod.textToSpeechV2({
        provider: 'external',
        text: 'hi',
        externalUrl: 'http://127.0.0.1:8765/tts',
        externalMethod: 'POST',
        externalHeaders: '{}',
        externalBodyTemplate: '{}',
        externalResponseType: 'file_path',
        externalResponsePath: 'audio_path',
        nodeId: 'tts2',
      }),
    /lacks required permission/,
  );
  assert.equal(mediaUrlCalls.length, 0, 'core-audio must NEVER fall back to get_media_url on permission denial');

  // 4) oaiyAgentTools blocks untrusted, transitive-untrusted, and pkg:* flows
  const untrustedFlow = { id: 'u1', name: 'Untrusted', untrusted: true, createdAt: '', updatedAt: '', graph: { nodes: [], edges: [] } };
  const pkgFlow = { id: 'p1', name: 'Pkg', createdAt: '', updatedAt: '', graph: { nodes: [{ id: 'n1', type: 'pkg:evil:run', position: { x: 0, y: 0 }, data: {} }], edges: [] } };
  const callerFlow = { id: 'c1', name: 'Caller', createdAt: '', updatedAt: '', graph: { nodes: [{ id: 's1', type: 'subflow', position: { x: 0, y: 0 }, data: { flowId: 'u1' } }], edges: [] } };
  assert.equal(M.isUntrustedOrPkgFlow(untrustedFlow), true);
  assert.equal(M.isUntrustedOrPkgFlow(pkgFlow), true);
  assert.equal(M.isUntrustedOrPkgFlow(callerFlow, [untrustedFlow, callerFlow]), true);
  await assert.rejects(() => M.publishAsTool(untrustedFlow, { name: 't', description: 'd' }), /Untrusted flows/);
  await assert.rejects(() => M.publishAsHook(pkgFlow, { tool: 'write_file', mode: 'before' }), /Untrusted flows/);
  await assert.rejects(() => M.publishAsTool(callerFlow, { name: 't', description: 'd' }, [untrustedFlow, callerFlow]), /Untrusted flows/);
});

await check('28. Wave-3: outputValue malformed %ZZ, ProjectIO non-array resilience, and compiler-sandbox LockManager/locks isolation', async () => {
  // outputValue.ts must not throw on malformed percent-encoding in ComfyUI filename query
  assert.doesNotThrow(() => M.kindOfAddress('http://127.0.0.1:8188/view?filename=%ZZ'));
  assert.equal(M.kindOfAddress('http://127.0.0.1:8188/view?filename=%ZZ.png'), 'image');
  assert.equal(M.kindOfAddress('http://127.0.0.1:8188/view?filename=%ZZ'), null);

  // ProjectIO.parseImportedProject handles non-array / null / corrupted endpoint & constant bags
  const malformedProject = JSON.stringify({
    name: 'Corrupt Bags',
    flows: [null, 'bad', { id: 'f1', name: 'Valid', graph: { nodes: [], edges: [] } }],
    llmEndpoints: 'not-an-array',
    imageGenEndpoints: { bad: true },
    httpPresets: 42,
    constants: [null, 123, { noKey: true }, { id: 'c1', key: 'GOOD_KEY', value: 'ok', category: 'custom', isSecret: false }],
  });
  const parsed = M.parseImportedProject(malformedProject);
  assert.equal(parsed.flows.length, 1);
  assert.equal(parsed.flows[0].untrusted, true);
  assert.ok(Array.isArray(parsed.llmEndpoints) && parsed.llmEndpoints.length > 0);
  assert.ok(Array.isArray(parsed.imageGenEndpoints) && parsed.imageGenEndpoints.length > 0);
  assert.ok(Array.isArray(parsed.httpPresets) && parsed.httpPresets.length > 0);
  assert.ok(parsed.constants.some((c) => c.key === 'GOOD_KEY'));

  // compiler-sandbox-worker neutralizes LockManager, locks, and navigator.locks across prototypes
  const workerSrc = fs.readFileSync(path.join(UI, 'src/sandbox/compiler-sandbox-worker.ts'), 'utf8');
  for (const name of ['LockManager', 'locks', 'storage']) {
    assert.ok(workerSrc.includes(`'${name}'`), `compiler-sandbox-worker must neutralize ${name}`);
  }
  const sandboxClientSrc = fs.readFileSync(path.join(UI, 'src/services/compilerSandbox.ts'), 'utf8');
  assert.ok(sandboxClientSrc.includes('w.terminate()'), 'compilerSandbox must terminate ephemeral worker per compile');
});

// ---------------------------------------------------------------------------
// Wave-4 Security & Correctness Regressions
// ---------------------------------------------------------------------------
await check('29. Wave-4: escapeString non-string coercion, compiler call sites & createTemplateCompiler single-pass substitution', async () => {
  // escapeString must coerce arrays/objects/null/numbers before JSON.stringify(...).slice(1, -1)
  const payloadArr = [1, '"; evil(); //'];
  const escapedArr = M.escapeString(payloadArr);
  assert.equal(/(?<!\\)"/.test(escapedArr), false, `unescaped quote breakout leaked: ${escapedArr}`);
  assert.equal(new Function(`return "${escapedArr}";`)(), String(payloadArr));
  assert.equal(JSON.parse(`"${escapedArr}"`), String(payloadArr));
  assert.equal(M.escapeString(null), '');
  assert.equal(M.escapeString(undefined), '');
  assert.equal(JSON.parse(`"${M.escapeString({ a: '"; evil()' })}"`), '[object Object]');
  assert.equal(M.escapeString(123), '123');

  // core-image & core-video ComfyUI config arrays with crafted non-string elements stay valid JS literals
  const imgCode = M.ImageCompiler.compileNode('image_gen', {
    node: {
      id: 'img1',
      type: 'image_gen',
      data: {
        apiFormat: 'comfyui',
        comfyImageInputNodeIds: [['1', '"); evil(); ("']],
        comfyImageInputConfigs: [{ nodeId: ['1', '"); evil()'], title: { x: 1 }, nodeType: ['LoadImage'] }],
        comfyAllImageNodeIds: [['9', '"); evil()']],
      },
    },
    inputs: new Map([['prompt', 'node_in']]),
    outputVar: 'node_img1_out',
    sanitizedId: 'img1',
    skipVarDeclaration: false,
    escapeString: M.escapeString,
    sanitizeId: (s) => s.replace(/[^a-zA-Z0-9_]/g, '_'),
    isInLoop: false,
  });
  assert.doesNotThrow(() => new Function(`async function f(){ let node_in = "p"; let workflow_context = {}; ${imgCode} }`), 'core-image compiled code must parse cleanly');
  assert.equal(/(?<!\\)"\); evil\(\)/.test(imgCode), false);

  const vidCode = M.VideoCompiler.compileNode('video_captions', {
    node: { id: 'pkg:com.x:cap.1', type: 'video_captions', data: { text: 'hi' } },
    inputs: new Map([['video', 'v_in'], ['segments', 'seg_in']]),
    outputVar: 'node_cap_out',
    sanitizedId: 'pkg_com_x_cap_1',
    skipVarDeclaration: false,
    escapeString: M.escapeString,
    sanitizeId: (s) => s.replace(/[^a-zA-Z0-9_]/g, '_'),
    isInLoop: false,
  });
  assert.doesNotThrow(() => new Function(`async function f(){ let v_in, seg_in, workflow_context = {}; ${vidCode} }`), 'video_captions must use ctx.sanitizedId for dotted/colon node IDs');
  assert.ok(vidCode.includes('_segments_pkg_com_x_cap_1'));

  // core-flow-control macro node guards non-array _macroInputs/_macroOutputs and strips newlines in debug comments
  const macroCode = M.FlowControlCompiler.compileNode('macro', {
    node: {
      id: 'm1',
      type: 'macro',
      data: {
        _macroWorkflowId: 'wf-1',
        _macroName: 'Demo',
        _macroInputs: [{ id: 'in1\nevil()', name: ['x', '": evil()'] }],
        _macroOutputs: [{ id: 42, name: ['out', '"] = evil(); //'] }],
      },
    },
    inputs: new Map(),
    outputVar: 'node_m1_out',
    sanitizedId: 'm1',
    skipVarDeclaration: false,
    escapeString: M.escapeString,
    sanitizeId: (s) => s.replace(/[^a-zA-Z0-9_]/g, '_'),
    isInLoop: false,
    debugEnabled: true,
  });
  assert.doesNotThrow(() => new Function(`async function f(){ let workflow_context = {}; ${macroCode} }`), 'macro compiled code must parse cleanly');
  assert.equal(macroCode.includes('\nevil()'), false, 'debug comment must not allow newline breakout');

  // createTemplateCompiler single-pass substitution prevents props -> data second-pass expansion
  const defs = new Map([
    [
      'pkg:demo:echo',
      {
        id: 'pkg:demo:echo',
        name: 'Echo',
        inputs: [],
        outputs: [{ id: 'out', name: 'out', type: 'string' }],
        properties: [{ id: 'text', name: 'Text', type: 'string', default: '' }],
        compiler: {
          template: '{{outputVar}} = [{{props.text}}, {{data.payload}}];',
        },
      },
    ],
  ]);
  const tplCompiler = M.createTemplateCompiler('Demo', defs);
  const compiledTpl = tplCompiler.compileNode('pkg:demo:echo', {
    node: {
      id: 't1',
      type: 'pkg:demo:echo',
      data: {
        text: '{{data.payload}}',
        payload: '; globalThis.pwned = 1; //',
      },
    },
    inputs: new Map(),
    outputVar: 'node_t1_out',
    sanitizedId: 't1',
    skipVarDeclaration: false,
    escapeString: M.escapeString,
    sanitizeId: (s) => s.replace(/[^a-zA-Z0-9_]/g, '_'),
    isInLoop: false,
  });
  assert.ok(compiledTpl);
  const evaluated = new Function(`let workflow_context = {}; ${compiledTpl}\nreturn node_t1_out;`)();
  assert.deepEqual(evaluated, ['{{data.payload}}', '; globalThis.pwned = 1; //']);
});

await check('30. Wave-4: registerImportedServices blocks built-in/desktop/reserved prefixes & strips secrets; resolveService prevents shadowing', () => {
  const prevCustom = store.get('oaiy.customServices');
  const prevDesktop = store.get('oaiy.desktopServices');
  store.delete('oaiy.customServices');
  store.set('oaiy.desktopServices', JSON.stringify([
    { id: 'desktop-svc', name: 'Desktop', endpoint: 'http://127.0.0.1:17972/api/ai/engine/x', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
    { id: 'engine:llm:flash', name: 'Engine Flash', endpoint: 'http://127.0.0.1:17972/api/ai/providers/oaiy-engine/v1/chat/completions', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
    { id: 'voice:transcribe', name: 'Voice', endpoint: 'http://127.0.0.1:17972/api/voice/transcribe', method: 'POST', headers: '{}', bodyTemplate: '', responseType: 'json', responsePath: 'text' },
    { id: 'companion:rig', name: 'Rig', endpoint: 'http://127.0.0.1:9000/run', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
  ]));
  try {
    const res = M.registerImportedServices({
      services: [
        // Reserved prefixes must be skipped
        { id: 'engine:llm:evil', name: 'Evil', endpoint: 'https://evil.example/v1', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
        { id: 'voice:transcribe', name: 'Evil Voice', endpoint: 'https://evil.example/v', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
        { id: 'companion:rig', name: 'Evil Rig', endpoint: 'https://evil.example/r', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
        // Built-in ID collision must be skipped
        { id: 'ollama-chat', name: 'Shadow Ollama', endpoint: 'https://evil.example/ollama', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
        // Desktop ID collision must be skipped
        { id: 'desktop-svc', name: 'Shadow Desktop', endpoint: 'https://evil.example/d', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
        // Invalid endpoint schemes / relative desktop path must be skipped
        { id: 'bad-proto', name: 'Bad', endpoint: 'javascript:alert(1)', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
        { id: 'rel-path', name: 'Rel', endpoint: '/api/ai/engine/gateway/v1/images', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
        // Invalid ID format must be skipped
        { id: 'bad id:colon', name: 'Bad ID', endpoint: 'https://ok.example/v1', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
        // Legitimate custom service: accepted, but apiKeyConstant & auth/{{apiKey}} headers stripped
        {
          id: 'legit-imported-svc',
          name: 'Legit',
          endpoint: 'https://api.example.com/v1/run',
          method: 'POST',
          headers: JSON.stringify({ Authorization: 'Bearer {{apiKey}}', 'X-Api-Key': 'raw', 'X-Custom': 'val-{{apiKey}}', Accept: 'application/json' }),
          bodyTemplate: '{"q": {{input}}}',
          responseType: 'json',
          responsePath: 'out',
          apiKeyConstant: 'OPENAI_API_KEY',
          isBuiltIn: true,
          group: 'engine',
        },
      ],
    });
    assert.equal(res.added, 1, `expected only 1 valid service added, got ${JSON.stringify(res)}`);
    assert.equal(res.skipped, 8);
    const saved = M.listAllServices().find((s) => s.id === 'legit-imported-svc');
    assert.ok(saved);
    assert.equal(saved.apiKeyConstant, undefined, 'apiKeyConstant must be stripped');
    assert.equal(saved.isBuiltIn, false);
    assert.equal(saved.group, undefined);
    assert.deepEqual(JSON.parse(saved.headers), { Accept: 'application/json' }, 'auth and {{apiKey}} headers must be stripped');

    // resolveService must NEVER let custom entries shadow reserved desktop prefixes
    const spoofedCustom = [
      { id: 'engine:llm:flash', name: 'Spoof', endpoint: 'https://evil.example/e', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
      { id: 'voice:transcribe', name: 'Spoof', endpoint: 'https://evil.example/v', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
      { id: 'companion:rig', name: 'Spoof', endpoint: 'https://evil.example/c', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
      { id: 'engine:image:not-installed', name: 'Spoof', endpoint: 'https://evil.example/ni', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '' },
    ];
    assert.equal(M.resolveService('engine:llm:flash', { custom: spoofedCustom })?.endpoint, 'http://127.0.0.1:17972/api/ai/providers/oaiy-engine/v1/chat/completions');
    assert.equal(M.resolveService('voice:transcribe', { custom: spoofedCustom })?.endpoint, 'http://127.0.0.1:17972/api/voice/transcribe');
    assert.equal(M.resolveService('companion:rig', { custom: spoofedCustom })?.endpoint, 'http://127.0.0.1:9000/run');
    assert.equal(M.resolveService('engine:image:not-installed', { custom: spoofedCustom }), null);
  } finally {
    if (prevCustom === undefined) store.delete('oaiy.customServices');
    else store.set('oaiy.customServices', prevCustom);
    if (prevDesktop === undefined) store.delete('oaiy.desktopServices');
    else store.set('oaiy.desktopServices', prevDesktop);
  }
});

await check('31. Wave-4: PackageMacros isolation, executeSubflow fail-closed & createPrefixedNodeId / authoritative packageId', async () => {
  // 1) Non-package job must NOT see packageMacros even when a package macro shares an ID with a user macro
  const trustedMacro = {
    id: 'shared-macro-id',
    name: 'User Macro',
    isMacro: true,
    createdAt: '',
    updatedAt: '',
    graph: {
      nodes: [],
      edges: [],
    },
  };
  const evilPkgMacro = {
    id: 'shared-macro-id',
    name: 'Hijack Macro',
    isMacro: true,
    untrusted: true,
    packageId: 'com.evil',
    createdAt: '',
    updatedAt: '',
    graph: {
      nodes: [{ id: 'lb1', type: 'logic_block', position: { x: 0, y: 0 }, data: { code: 'return "HIJACKED"' } }],
      edges: [],
    },
  };
  const rtTrusted = new M.OAIYRuntime({ useWorkerForUntrusted: false });
  rtTrusted.setAvailableFlows([trustedMacro]);
  rtTrusted.setPackageMacros([evilPkgMacro]);
  await rtTrusted.registerDynamicModule({
    manifest: { id: 'core-flow-control', name: 'Subflow', version: '1.0.0', nodes: [] },
    runtime: M.SubflowRuntime,
  });
  // Calling shared-macro-id from trusted context (no currentPackageId) must resolve trustedMacro, NOT evilPkgMacro
  const resTrusted = await rtTrusted.executeScript('return await Subflow.execute("shared-macro-id", {});');
  assert.notEqual(resTrusted, 'HIJACKED', 'trusted context must never resolve packageMacros');

  // 2) Trusted context must fail closed if availableFlows contains an untrusted, packageId-tagged, or pkg:* subflow
  const rtGuard = new M.OAIYRuntime({ useWorkerForUntrusted: false });
  rtGuard.setAvailableFlows([
    { id: 'u-sub', name: 'Untrusted', untrusted: true, createdAt: '', updatedAt: '', graph: { nodes: [], edges: [] } },
    { id: 'p-sub', name: 'PkgTagged', packageId: 'com.evil', createdAt: '', updatedAt: '', graph: { nodes: [], edges: [] } },
    { id: 'n-sub', name: 'HasPkgNode', createdAt: '', updatedAt: '', graph: { nodes: [{ id: 'x', type: 'pkg:com.evil:n', position: { x: 0, y: 0 }, data: {} }], edges: [] } },
  ]);
  await rtGuard.registerDynamicModule({
    manifest: { id: 'core-flow-control', name: 'Subflow', version: '1.0.0', nodes: [] },
    runtime: M.SubflowRuntime,
  });
  for (const id of ['u-sub', 'p-sub', 'n-sub']) {
    await assert.rejects(
      () => rtGuard.executeScript(`return await Subflow.execute("${id}", {});`),
      /Refusing to execute untrusted or package subflow/,
      `trusted context must reject subflow ${id}`,
    );
  }

  // 3) JobManager does not attach packageMacros to non-package jobs
  const jm = new M.JobManager({ databaseHandler: async () => ({ success: true, data: [] }) });
  jm.setAvailableFlows([
    { id: 'user-only', name: 'User', createdAt: '', updatedAt: '', graph: { nodes: [], edges: [] } },
  ]);
  jm.setPackageMacros([evilPkgMacro]);
  const jobId = jm.submit('caller', 'Caller', { nodes: [], edges: [] });
  const doneJob = await new Promise((resolve) => {
    const unsub = jm.onStateChange((jobs) => {
      const j = jobs.find((x) => x.id === jobId);
      if (j && (j.status === 'completed' || j.status === 'failed' || j.status === 'aborted')) {
        unsub();
        resolve(j);
      }
    });
  });
  assert.equal(doneJob.status, 'completed', `non-package job should complete cleanly without packageMacros contamination (got ${doneJob.error})`);

  // 4) createPrefixedNodeId enforces verified packageId and rejects colons
  assert.equal(M.createPrefixedNodeId('com.real', 'my_node'), 'pkg:com.real:my_node');
  assert.throws(() => M.createPrefixedNodeId('com.trusted:spoof', 'my_node'), /Invalid packageId/);
  assert.throws(() => M.createPrefixedNodeId('com.real', 'bad:node'), /Invalid nodeId/);

  // 5) JobManager resolves authoritative packageId from moduleRegistry.getModuleForNode().path over spoofed node type prefix
  const lookedUp = [];
  const spoofRegistry = {
    getNodeDefinition: () => ({ id: 'pkg:com.trusted:steal', name: 'Steal', inputs: [], outputs: [{ id: 'out', name: 'out', type: 'string' }], compiler: {} }),
    getModuleForNode: () => ({
      path: 'package:com.actual-evil/nodes',
      compiler: { name: 'Evil', getNodeTypes: () => ['pkg:com.trusted:steal'], compileNode: (_t, c) => `let ${c.outputVar} = 1;` },
      manifest: { id: 'pkg-evil' },
    }),
    getCompiler: () => ({ name: 'Evil', getNodeTypes: () => ['pkg:com.trusted:steal'], compileNode: (_t, c) => `let ${c.outputVar} = 1;` }),
    getAllNodeDefinitions: () => [],
    getAllModules: () => [],
  };
  const jmSpoof = new M.JobManager({
    databaseHandler: async () => ({ success: true, data: [] }),
    moduleRegistry: spoofRegistry,
    packagePermissionsLookup: (pkgId) => {
      lookedUp.push(pkgId);
      return pkgId === 'com.trusted' ? ['filesystem', 'process:execute'] : [];
    },
  });
  const spoofJobId = jmSpoof.submit('f-spoof', 'Spoof', {
    nodes: [{ id: 's1', type: 'pkg:com.trusted:steal', position: { x: 0, y: 0 }, data: {} }],
    edges: [],
  });
  await new Promise((resolve) => {
    const unsub = jmSpoof.onStateChange((jobs) => {
      const j = jobs.find((x) => x.id === spoofJobId);
      if (j && (j.status === 'completed' || j.status === 'failed' || j.status === 'aborted')) {
        unsub();
        resolve(j);
      }
    });
  });
  assert.deepEqual(lookedUp, ['com.actual-evil'], 'JobManager must look up authoritative owning packageId from module path, not spoofed prefix');
});

await check('32. Wave-4: detectSandboxEscape Acorn AST walk defeats regex-literal bypass and fails closed on parse errors', () => {
  // Regex-literal bypass that previously broke the char-scanner
  assert.equal(M.detectSandboxEscape('/"/; globalThis; /"/;'), 'globalThis');
  assert.equal(M.detectSandboxEscape('/\'/; eval("1"); /\'/;'), 'eval()');
  assert.equal(M.detectSandboxEscape('/`/; Function("return this")(); /`/;'), 'Function constructor');
  assert.ok(M.stripCommentsAndStringsFromCode('/"/; globalThis; /"/;').includes('globalThis'));

  // AST catches computed constructor chains, indirect eval, new Function, import(), import.meta, arrows & IIFEs
  assert.equal(M.detectSandboxEscape('const x = ({})["constructor"]["constructor"]("return this")();'), 'constructor.constructor chain');
  assert.equal(M.detectSandboxEscape('const x = (() => {}).constructor("return this")();'), 'constructor.constructor chain');
  assert.equal(M.detectSandboxEscape('const x = (0, eval)("1");'), 'eval()');
  assert.equal(M.detectSandboxEscape('const x = new Function("return 1");'), 'Function constructor');
  assert.equal(M.detectSandboxEscape('const x = import("./mod.js");'), 'dynamic import()');
  assert.equal(M.detectSandboxEscape('const x = import.meta.url;'), 'import.meta');
  assert.equal(M.detectSandboxEscape('const x = (function(){ return this; })();'), '(function(){return this})()');
  assert.equal(M.detectSandboxEscape('const x = () => this;'), '() => this arrow');

  // Fails closed on syntax / wrapper-breakout errors
  assert.equal(M.detectSandboxEscape('const x = 1 +;'), 'unparseable script');
  assert.equal(M.detectSandboxEscape('}; globalThis; {'), 'unparseable script');

  // Legitimate compiled workflow code (await, yield, top-level return, division, strings) stays allowed
  assert.equal(M.detectSandboxEscape('const a = 10 / 2; const s = "globalThis"; yield Utility.requireCodeExecute([]); return await Foo.bar(a, s);'), null);
});

await check('33. Wave-4: NetworkUtils aligns with shared/capabilities/local.ts (.internal, .local, .lan, .home.arpa, single-label, NAT64, 6to4, Teredo)', () => {
  for (const localUrl of [
    'http://metadata.google.internal/computeMetadata/v1/',
    'http://printer.local:631/',
    'http://router.lan/',
    'http://nas.home.arpa/',
    'http://intranet:8080/',
    'http://[64:ff9b::127.0.0.1]/',
    'http://[64:ff9b::169.254.169.254]/',
    'http://[64:ff9b:1::1]/',
    'http://[2002:7f00:0001::1]/',
    'http://[2002:a9fe:a9fe::1]/',
    // Teredo encoding client 127.0.0.1 => inverted 0x7f00 -> 0x80ff, 0x0001 -> 0xfffe
    'http://[2001:0000:4136:e378:8000:63bf:80ff:fffe]/',
  ]) {
    assert.equal(M.isLocalNetworkUrl(localUrl), true, `expected local/private: ${localUrl}`);
  }
  for (const pubUrl of [
    'https://api.openai.com/v1/chat/completions',
    'http://[2606:4700:4700::1111]/',
    'http://[64:ff9b::8.8.8.8]/',
    'http://[2002:0808:0808::1]/',
  ]) {
    assert.equal(M.isLocalNetworkUrl(pubUrl), false, `expected public: ${pubUrl}`);
  }
});

await check('34. Wave-4: core-utility template {{toString}}, core-browser quoted CSS selector & flowIds FLOW_REFS', async () => {
  // template {{toString}} / {{constructor}} / {{__proto__}} must not inherit Object.prototype methods
  const tmplCode = M.UtilityCompiler.compileNode('template', {
    node: { id: 'tpl1', type: 'template', data: { template: 'Hi {{input}} {{toString}} {{constructor}} {{__proto__}}' } },
    inputs: new Map([['input', 'node_in']]),
    outputVar: 'node_tpl1_out',
    sanitizedId: 'tpl1',
    skipVarDeclaration: false,
    escapeString: M.escapeString,
    sanitizeId: (s) => s.replace(/[^a-zA-Z0-9_]/g, '_'),
    isInLoop: false,
  });
  const tmplOut = new Function(`let node_in = "World"; let workflow_context = {}; ${tmplCode}\nreturn node_tpl1_out;`)();
  assert.equal(tmplOut, 'Hi World {{toString}} {{constructor}} {{__proto__}}');

  // core-browser quoted CSS selector on session page must NOT double-escape quotes
  const browserCode = M.BrowserCompiler.compileNode('browser_extract', {
    node: { id: 'bx1', type: 'browser_extract', data: { type: 'selector', selector: 'a[href="https://example.com"]', attribute: 'data-x="1"' } },
    inputs: new Map([['session', 'sess_var']]),
    outputVar: 'node_bx1_out',
    sanitizedId: 'bx1',
    skipVarDeclaration: false,
    escapeString: M.escapeString,
    sanitizeId: (s) => s.replace(/[^a-zA-Z0-9_]/g, '_'),
    isInLoop: false,
  });
  let capturedProbe = '';
  const evalFn = new Function(
    'Browser',
    'sess_var',
    `return (async () => { let workflow_context = {}; ${browserCode}; return node_bx1_out; })();`,
  );
  await evalFn(
    { evaluateJson: async (_id, script) => { capturedProbe = script; return 'ok'; } },
    { id: 's1' },
  );
  // Evaluating the probe's selector assignment must yield the exact unescaped CSS selector
  assert.ok(capturedProbe.includes('const sel = "a[href=\\"https://example.com\\"]";'), `probe selector was corrupted: ${capturedProbe}`);
  const extractedSel = new Function(`return ${capturedProbe.replace(/const r =[\s\S]*/, 'return [sel, attr]; })()')};`)();
  assert.deepEqual(extractedSel, ['a[href="https://example.com"]', 'data-x="1"']);

  // flowIds.ts FLOW_REFS covers all 5 flow/macro reference fields
  assert.deepEqual([...M.FLOW_REFS], ['flowId', 'targetFlowId', '_macroWorkflowId', 'macroWorkflowId', 'macroId']);
  const untrustedTarget = { id: 'u1', untrusted: true, graph: { nodes: [], edges: [] } };
  for (const field of ['flowId', 'targetFlowId', '_macroWorkflowId', 'macroWorkflowId', 'macroId']) {
    const g = { nodes: [{ id: 'n1', type: 'macro', data: { [field]: 'u1' } }] };
    assert.equal(M.reachesUntrusted('caller', g, [untrustedTarget]), true, `reachesUntrusted must trace ${field}`);
  }
});

await check('35. Wave-4: core-ai extractVideoFrames guards, contractRuntime cache/path hardening, non-local file_path & UI/registry fixes', async () => {
  // 1) core-ai extractVideoFrames rejects file:// without filesystem:read and path traversal
  const aiPkg = M.AIRuntime.createMethods({
    log() {},
    onNodeStatus() {},
    currentPackageId: 'pkg-no-fs',
    hasPermission: () => false,
    secureFetch: async () => mockResp(200, { choices: [{ message: { content: 'ok' } }] }),
  });
  await assert.rejects(
    () => aiPkg.chat('', 'describe', null, 'https://api.example.com/v1/chat/completions', 'm', '', false, 64, 0, 'text', false, 'auto', 'n1', [], '', 0, 0, false, '', 'file:///etc/passwd', 2),
    /lacks required permission 'filesystem:read'/,
  );
  const aiTrusted = M.AIRuntime.createMethods({
    log() {},
    onNodeStatus() {},
    secureFetch: async () => mockResp(200, { choices: [{ message: { content: 'ok' } }] }),
  });
  await assert.rejects(
    () => aiTrusted.chat('', 'describe', null, 'https://api.example.com/v1/chat/completions', 'm', '', false, 64, 0, 'text', false, 'auto', 'n1', [], '', 0, 0, false, '', 'file:///tmp/../etc/passwd', 2),
    /Path traversal/,
  );

  // 2) contractRuntime listOaiyServices checks network:local BEFORE 30s cache & runContract rejects non-local file paths
  const origDesktop = globalThis.__OAIY_DESKTOP__;
  globalThis.__OAIY_DESKTOP__ = { origin: 'http://127.0.0.1:17972', token: 'tok' };
  try {
    // Warm the 30s cache as a trusted caller
    const warm = await M.listOaiyServices({
      log() {},
      secureFetch: async () => mockResp(200, { services: [{ id: 'engine:llm:m', endpoint: '/api/ai/providers/oaiy-engine/v1/chat/completions' }] }),
    });
    assert.equal(warm.length, 1);
    // Package lacking network:local MUST still be denied even though the cache is warm
    await assert.rejects(
      () => M.listOaiyServices({ log() {}, currentPackageId: 'pkg-no-local', hasPermission: () => false }),
      /lacks required permission 'network:local'/,
    );
  } finally {
    if (origDesktop === undefined) delete globalThis.__OAIY_DESKTOP__;
    else globalThis.__OAIY_DESKTOP__ = origDesktop;
  }

  await assert.rejects(
    () => M.runContract(
      { log() {}, secureFetch: async () => mockResp(200, { path: '/etc/passwd' }) },
      { id: 'remote-vid', name: 'RemoteVid', endpoint: 'https://remote.example.com/make', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: 'path', output: 'video' },
      {},
      { nodeId: 'v1' },
    ),
    /non-local endpoint/,
  );

  // 3) core-video refuses non-local file_path and does NOT call ensure_service_ready_by_port for remote URLs
  const autoStartPorts = [];
  const vidMod = M.VideoRuntime.createMethods({
    log() {},
    onNodeStatus() {},
    tauri: {
      invoke: async (cmd, args) => {
        if (cmd === 'ensure_service_ready_by_port') autoStartPorts.push(args);
        return { success: true, port: 8768 };
      },
    },
    fetch: async () => mockResp(200, { success: true, video_path: '/etc/passwd', file_path: '/etc/passwd' }),
  });
  await assert.rejects(
    () => vidMod.generateAvatar('/tmp/a.png', '/tmp/a.wav', 'hi', 'https://remote.example.com:8768/generate', 5, 20, 'va1'),
    /non-local endpoint/,
  );
  assert.equal(autoStartPorts.length, 0, 'remote URL must never trigger ensure_service_ready_by_port');

  // 4) oaiyAgentTools checks project-level untrusted state
  const prevProj = store.get('oaiy_project');
  store.set('oaiy_project', JSON.stringify({ id: 'p1', name: 'Imported', untrusted: true, flows: [] }));
  try {
    const cleanFlow = { id: 'f-clean', name: 'Clean', createdAt: '', updatedAt: '', graph: { nodes: [], edges: [] } };
    assert.equal(M.isUntrustedOrPkgFlow(cleanFlow), true, 'project-level untrusted must block isUntrustedOrPkgFlow');
  } finally {
    if (prevProj === undefined) store.delete('oaiy_project');
    else store.set('oaiy_project', prevProj);
  }

  // 5) reloadRequiresTrustReconfirm re-prompts on contentHash change for unverified or active-component packages
  const baseManifest = { id: 'com.demo', name: 'Demo', version: '1.0.0', entryFlow: 'main.json', flows: ['main.json'], permissions: [] };
  assert.equal(M.reloadRequiresTrustReconfirm(baseManifest, baseManifest, 'hash_matched', 'hash_matched', 'hash-a', 'hash-b'), true);
  assert.equal(M.reloadRequiresTrustReconfirm(baseManifest, baseManifest, 'signature_verified', 'signature_verified', 'hash-a', 'hash-b'), false);
  const withMacros = { ...baseManifest, macros: ['m.json'] };
  assert.equal(M.reloadRequiresTrustReconfirm(withMacros, withMacros, 'signature_verified', 'signature_verified', 'hash-a', 'hash-b'), true);

  // 6) Static assertions for PackageManager oversize-cancel, useDesktopFlows mirror guard, core-audio token redaction, customNodeRegistry unload
  const pmSrc = fs.readFileSync(path.join(UI, 'src/components/PackageManager/PackageManager.tsx'), 'utf8');
  assert.match(pmSrc, /if \(!ok\) \{\s*setPendingInstall\(null\);\s*return;\s*\}/);
  const dfSrc = fs.readFileSync(path.join(UI, 'src/hooks/useDesktopFlows.ts'), 'utf8');
  assert.ok(dfSrc.includes('!isUntrustedOrPkgFlow(f, flows)'), 'useDesktopFlows must guard tool/hook mirroring with !isUntrustedOrPkgFlow');
  const audioSrc = fs.readFileSync(path.join(UI, 'src/bundled-modules/core-audio/runtime.ts'), 'utf8');
  assert.ok(audioSrc.includes('redactMediaToken(audioUrl)'), 'core-audio must redact media token in TTS Media URL log');
  const regSrc = fs.readFileSync(path.join(UI, 'src/services/customNodeRegistry.ts'), 'utf8');
  assert.ok(regSrc.includes('moduleLoader.unloadModule(moduleId)'), 'customNodeRegistry.unregisterPackageNodes must unload ModuleLoader modules');
});

// ---------------------------------------------------------------------------
// Wave-5 Security, AST Sandbox, Core & UI/UX Regressions
// ---------------------------------------------------------------------------

await check('36. Wave-5: detectSandboxEscape blocks destructured/aliased/computed constructor, ClassExpression, TaggedTemplate & sequence arrows', () => {
  // ObjectPattern destructuring (static Identifier, string literal, computed literal, template, params, assignment)
  const blockedPatterns = [
    'const { constructor: F } = () => {};',
    'const { "constructor": F } = () => {};',
    'const { ["constructor"]: F } = () => {};',
    'const { [`constructor`]: F } = () => {};',
    'const { [(0, "constructor")]: F } = () => {};',
    'function leak({ constructor: F }) { return F; }',
    'const leak = ({ ["constructor"]: F }) => F;',
    'let F; ({ constructor: F } = () => {});',
    'let F; ([{ ["constructor"]: F }] = [() => {}]);',
    'const { __proto__: P } = {};',
    // SequenceExpression & ClassExpression & aliased MemberExpression
    'const x = (0, () => {}).constructor("return this")();',
    'const x = (0, a.constructor).constructor("return this")();',
    'const x = (class {}).constructor("return this")();',
    'const f = () => {}; const x = f.constructor("return this")();',
    'const c = ({}).constructor; const x = c.constructor;',
    'const x = obj["constructor"];',
    'const x = obj[`constructor`];',
    'const x = obj[(0, "constructor")];',
    'const x = obj.__proto__;',
    // TaggedTemplateExpression
    'const x = (function(){ return this; })``;',
    'const x = (0, function(){ return this; })``;',
    'const x = (() => 1)``;',
    'const x = Function`return this`;',
    'const x = eval`1`;',
    // SequenceExpression arrow body returning this
    'const x = () => (0, this);',
    'const x = () => (foo(), bar(), this);',
  ];
  for (const code of blockedPatterns) {
    assert.ok(M.detectSandboxEscape(code), `expected sandbox escape detection for: ${code}`);
  }
  // Specific labels preserved for chain/tagged/arrow patterns
  assert.equal(M.detectSandboxEscape('const x = (class {}).constructor("return this")();'), 'constructor.constructor chain');
  assert.equal(M.detectSandboxEscape('const x = (0, () => {}).constructor("return this")();'), 'constructor.constructor chain');
  assert.equal(M.detectSandboxEscape('const x = (function(){ return this; })``;'), '(function(){return this})()');
  assert.equal(M.detectSandboxEscape('const x = () => (0, this);'), '() => this arrow');

  // Legitimate workflow code remains permitted
  assert.equal(
    M.detectSandboxEscape('const { name, count } = input; let total = count / 2; const fn = (a, b) => a + b; return await Foo.run(name, fn(total, 1));'),
    null,
  );
});

await check('37. Wave-5: executeSubflow refuses cross-package pkg:* nodes even when currentPackageId is already set', async () => {
  const mkRuntime = async (registry) => {
    const rt = new M.OAIYRuntime({ useWorkerForUntrusted: false });
    rt.setFlowContext('flow-a', 'pkgA');
    rt.setPackagePermissions(['network']);
    if (registry) rt.setModuleRegistry(registry);
    await rt.registerDynamicModule({
      manifest: { id: 'core-flow-control', name: 'Subflow', version: '1.0.0', nodes: [] },
      runtime: M.SubflowRuntime,
    });
    return rt;
  };

  // Direct cross-package node inside a subflow with no flow.packageId
  const rt1 = await mkRuntime();
  rt1.setAvailableFlows([
    {
      id: 'sneaky-sub',
      name: 'Sneaky',
      createdAt: '',
      updatedAt: '',
      graph: { nodes: [{ id: 'n1', type: 'pkg:pkgB:steal', position: { x: 0, y: 0 }, data: {} }], edges: [] },
    },
    {
      id: 'wrapper-sub',
      name: 'Wrapper',
      createdAt: '',
      updatedAt: '',
      graph: { nodes: [{ id: 's1', type: 'subflow', position: { x: 0, y: 0 }, data: { flowId: 'sneaky-sub' } }], edges: [] },
    },
  ]);
  await assert.rejects(
    () => rt1.executeScript('return await Subflow.execute("sneaky-sub", {});'),
    /cross-package node/,
    'direct cross-package pkg:pkgB:* node must be rejected under pkgA',
  );
  await assert.rejects(
    () => rt1.executeScript('return await Subflow.execute("wrapper-sub", {});'),
    /cross-package node/,
    'nested cross-package pkg:pkgB:* node must be rejected under pkgA',
  );

  // Authoritative packageId mismatch via moduleRegistry even if node prefix spoofs pkgA
  const spoofReg = {
    getNodeDefinition: () => ({ id: 'pkg:pkgA:spoof', name: 'Spoof', inputs: [], outputs: [{ id: 'out', name: 'out', type: 'string' }], compiler: {} }),
    getModuleForNode: () => ({
      path: 'package:pkgB/nodes',
      compiler: { name: 'PkgB', getNodeTypes: () => ['pkg:pkgA:spoof'], compileNode: (_t, c) => `let ${c.outputVar} = 1;` },
      manifest: { id: 'pkgB' },
    }),
    getCompiler: () => ({ name: 'PkgB', getNodeTypes: () => ['pkg:pkgA:spoof'], compileNode: (_t, c) => `let ${c.outputVar} = 1;` }),
    getAllNodeDefinitions: () => [],
    getAllModules: () => [],
  };
  const rt2 = await mkRuntime(spoofReg);
  rt2.setAvailableFlows([
    {
      id: 'spoof-sub',
      name: 'SpoofSub',
      createdAt: '',
      updatedAt: '',
      graph: { nodes: [{ id: 'n1', type: 'pkg:pkgA:spoof', position: { x: 0, y: 0 }, data: {} }], edges: [] },
    },
  ]);
  await assert.rejects(
    () => rt2.executeScript('return await Subflow.execute("spoof-sub", {});'),
    /cross-package node/,
    'authoritative module path mismatch must be rejected',
  );

  // Same-package node (pkgA calling pkg:pkgA:own) stays allowed
  const ownReg = {
    getNodeDefinition: () => ({ id: 'pkg:pkgA:own', name: 'Own', inputs: [], outputs: [{ id: 'out', name: 'out', type: 'string' }], compiler: {} }),
    getModuleForNode: () => ({
      path: 'package:pkgA/nodes',
      compiler: { name: 'PkgA', getNodeTypes: () => ['pkg:pkgA:own'], compileNode: (_t, c) => `let ${c.outputVar} = "ok-from-pkgA";` },
      manifest: { id: 'pkgA' },
    }),
    getCompiler: () => ({ name: 'PkgA', getNodeTypes: () => ['pkg:pkgA:own'], compileNode: (_t, c) => `let ${c.outputVar} = "ok-from-pkgA";` }),
    getAllNodeDefinitions: () => [],
    getAllModules: () => [],
  };
  const rt3 = await mkRuntime(ownReg);
  rt3.setAvailableFlows([
    {
      id: 'own-sub',
      name: 'OwnSub',
      packageId: 'pkgA',
      createdAt: '',
      updatedAt: '',
      graph: { nodes: [{ id: 'n1', type: 'pkg:pkgA:own', position: { x: 0, y: 0 }, data: {} }], edges: [] },
    },
  ]);
  await rt3.executeScript('return await Subflow.execute("own-sub", {});');
});

await check('38. Wave-5: registerImportedServices strips {{apiKeyRaw}} + bodyTemplate placeholders; reloadRequiresTrustReconfirm & unload await', () => {
  const prevCustom = store.get('oaiy.customServices');
  store.delete('oaiy.customServices');
  try {
    const res = M.registerImportedServices({
      services: [
        {
          id: 'wave5-raw-placeholder-svc',
          name: 'Wave5 Svc',
          endpoint: 'https://api.example.com/v1/run',
          method: 'POST',
          headers: JSON.stringify({
            'X-Raw': 'Token {{apiKeyRaw}}',
            'X-Spaced': 'Token {{ apiKeyRaw }}',
            'X-Keep': 'ok',
          }),
          bodyTemplate: '{"k1":"{{apiKey}}","k2":"{{ apiKeyRaw }}","q":{{input}}}',
          responseType: 'json',
          responsePath: 'out',
        },
      ],
    });
    assert.equal(res.added, 1);
    const saved = M.listAllServices().find((s) => s.id === 'wave5-raw-placeholder-svc');
    assert.ok(saved);
    assert.deepEqual(JSON.parse(saved.headers), { 'X-Keep': 'ok' }, '{{apiKeyRaw}} headers must be stripped');
    assert.equal(/apiKey/i.test(saved.bodyTemplate), false, `bodyTemplate leaked apiKey placeholder: ${saved.bodyTemplate}`);
    assert.ok(saved.bodyTemplate.includes('{{input}}'), 'legitimate {{input}} placeholder must be preserved');
  } finally {
    if (prevCustom === undefined) store.delete('oaiy.customServices');
    else store.set('oaiy.customServices', prevCustom);
  }

  // reloadRequiresTrustReconfirm fails closed when prevContentHash was set and nextContentHash is missing/empty
  const baseManifest = { id: 'com.demo', name: 'Demo', version: '1.0.0', entryFlow: 'main.json', flows: ['main.json'], permissions: [] };
  assert.equal(M.reloadRequiresTrustReconfirm(baseManifest, baseManifest, 'signature_verified', 'signature_verified', 'hash-a', ''), true);
  assert.equal(M.reloadRequiresTrustReconfirm(baseManifest, baseManifest, 'signature_verified', 'signature_verified', 'hash-a', undefined), true);
  assert.equal(M.reloadRequiresTrustReconfirm(baseManifest, baseManifest, 'signature_verified', 'signature_verified', 'hash-a', 'hash-a'), false);

  // Structural checks for async unloadModule await, loadedIds filtering, and packagePermissions cleanup
  const regSrc = fs.readFileSync(path.join(UI, 'src/services/customNodeRegistry.ts'), 'utf8');
  assert.match(regSrc, /export async function unregisterPackageNodes\(/);
  assert.ok(regSrc.includes('await moduleLoader.unloadModule(moduleId)'));
  const pkgNodesSrc = fs.readFileSync(path.join(UI, 'src/hooks/usePackageNodes.ts'), 'utf8');
  assert.ok(pkgNodesSrc.includes('await unregisterCustomNodes(packageId)'));
  const pkgMgrSrc = fs.readFileSync(path.join(UI, 'src/hooks/usePackageManager.ts'), 'utf8');
  assert.ok(pkgMgrSrc.includes('embeddedResults?.customNodes?.loadedIds'));
  assert.match(pkgMgrSrc, /handleClosePackage[\s\S]*?setPackagePermissions/);
});

await check('39. Wave-5: escapeString hostile objects, template-compiler U+2028/U+2029/__proto__, and connector-module ConnectorConfigError guards', () => {
  // Hostile objects must not throw TypeError in escapeString
  const hostile = { toString: 123, valueOf: 456 };
  assert.equal(typeof M.escapeString(hostile), 'string');
  assert.ok(M.escapeString(hostile).includes('toString'));
  const nullProto = Object.create(null);
  nullProto.k = 'v';
  assert.equal(typeof M.escapeString(nullProto), 'string');
  const circularNull = Object.create(null);
  circularNull.self = circularNull;
  assert.equal(M.escapeString(circularNull), '');

  // createTemplateCompiler escapes U+2028/U+2029 and neutralizes __proto__ keys for non-string props/data/defaults
  const protoPayload = JSON.parse('{"__proto__":{"polluted":true},"s":"a\\u2028b\\u2029c"}');
  const tplDef = {
    id: 'w5_tpl',
    name: 'W5 Tpl',
    inputs: [],
    outputs: [{ id: 'out', name: 'out', type: 'any' }],
    properties: [{ id: 'defObj', name: 'defObj', type: 'json', default: protoPayload }],
    compiler: { template: '{{outputVar}} = [{{props.obj}}, {{data.obj}}, {{props.defObj}}];' },
  };
  const compiler = M.createTemplateCompiler('w5', new Map([['w5_tpl', tplDef]]));
  const generated = compiler.compileNode('w5_tpl', {
    node: { id: 'n1', type: 'w5_tpl', data: { obj: protoPayload } },
    inputs: new Map(),
    outputVar: 'node_n1_out',
    sanitizedId: 'n1',
    skipVarDeclaration: false,
    escapeString: M.escapeString,
    sanitizeId: (s) => s.replace(/[^a-zA-Z0-9_]/g, '_'),
    isInLoop: false,
  });
  assert.ok(generated);
  assert.equal(generated.includes('\u2028') || generated.includes('\u2029'), false, 'raw U+2028/U+2029 must be escaped in compiled output');
  const evaluated = new Function(`let workflow_context = {}; ${generated}\nreturn node_n1_out;`)();
  assert.equal(({}).polluted, undefined, 'Object.prototype must not be polluted');
  assert.equal(evaluated[0].s, 'a\u2028b\u2029c');
  assert.equal(Object.getPrototypeOf(evaluated[0]), Object.prototype, 'object literal prototype must not be reparented by __proto__');
  assert.deepEqual(Object.getOwnPropertyDescriptor(evaluated[0], '__proto__')?.value, { polluted: true });

  // parseConnectorConfig throws ConnectorConfigError (never raw Error/TypeError) for invalid listResult/filters/limit/fields
  const baseCfg = (nodeOverride) => ({
    baseUrl: 'https://api.example.com',
    credential: 'MY_TOKEN',
    nodes: [{ nodeType: 'list_items', operation: 'listRecords', path: '/items', ...nodeOverride }],
  });
  const badConfigs = [
    baseCfg({ listResult: 'not-an-object' }),
    baseCfg({ listResult: null }),
    baseCfg({ filters: 'not-an-object' }),
    baseCfg({ filters: null }),
    baseCfg({ filters: { field: 'f', opKey: 'o', fieldKey: 'fk', valueKey: 'vk', ops: [] } }),
    baseCfg({ filters: { field: 'f', opKey: 'o', fieldKey: 'fk', valueKey: 'vk', ops: [null] } }),
    baseCfg({ limit: 'not-an-object' }),
    baseCfg({ limit: { field: 'l', param: 'limit', default: 0, max: 10 } }),
    baseCfg({ recordField: 'rec' }),
    { baseUrl: 'https://api.example.com', credential: 'T', nodes: [{ nodeType: 'get_item', operation: 'getRecord', path: '/items/:id', listResult: { items: 'x' } }] },
    { baseUrl: 'https://api.example.com', credential: 'T', nodes: [{ nodeType: 'get_item', operation: 'getRecord', path: '/items/:id', filters: { field: 'f', opKey: 'o', fieldKey: 'fk', valueKey: 'vk', ops: [{ op: '=', param: 'eq' }] } }] },
    { baseUrl: 'https://api.example.com', credential: 'T', nodes: [{ nodeType: 'get_item', operation: 'getRecord', path: '/items/:id', limit: { field: 'l', param: 'limit', default: 10, max: 50 } }] },
  ];
  for (const cfg of badConfigs) {
    assert.throws(
      () => M.parseConnectorConfig(cfg),
      (err) => err instanceof M.ConnectorConfigError,
      `expected ConnectorConfigError for ${JSON.stringify(cfg.nodes[0])}`,
    );
  }
});

await check('40. Wave-5: UI/UX fixes — QuickConnectPopup truncation, TrustDialog badges/icons, UnknownNode L/R handles, OutputNode & FlowsSidebar', () => {
  // QuickConnectPopup shows truncation footer when > 15 matches
  const twentyNodes = Array.from({ length: 20 }, (_, i) => ({
    category: 'Utility',
    definition: { id: `n_${i}`, name: `Node ${i}`, color: 'blue', icon: 'box' },
  }));
  const qcHtml = M.renderToStaticMarkup(
    M.createElement(M.QuickConnectPopup, {
      state: { isVisible: true, position: { x: 100, y: 100 }, searchQuery: '' },
      handlePosition: null,
      compatibleNodes: twentyNodes,
      onNodeSelect() {},
      onClose() {},
      onSearchChange() {},
    }),
  );
  assert.ok(qcHtml.includes('Showing 15 of 20 — refine search'), `expected truncation footer, got: ${qcHtml.slice(-200)}`);

  // TrustDialog verification badge states + High risk pill + zero musical-note SVG
  const sampleManifest = {
    id: 'com.test.pkg',
    name: 'Test Pkg',
    version: '1.0.0',
    entryFlow: 'main.json',
    flows: ['main.json'],
    permissions: ['network:local', 'constants:read'],
  };
  const verifiedHtml = M.renderToStaticMarkup(
    M.createElement(M.TrustDialog, { manifest: sampleManifest, verification: 'signature_verified', onConfirm() {}, onCancel() {} }),
  );
  assert.ok(verifiedHtml.includes('>Verified<'), 'signature_verified must render Verified badge');
  assert.ok(verifiedHtml.includes('High risk'), 'high-risk permission must render High risk pill');
  const hashHtml = M.renderToStaticMarkup(
    M.createElement(M.TrustDialog, { manifest: sampleManifest, verification: 'hash_matched', onConfirm() {}, onCancel() {} }),
  );
  assert.ok(hashHtml.includes('Hash verified'), 'hash_matched must render Hash verified badge');
  const trustSrc = fs.readFileSync(path.join(UI, 'src/components/PackageManager/TrustDialog.tsx'), 'utf8');
  assert.equal(trustSrc.includes('M9 19V6l12-3v13'), false, 'musical-note SVG path must be removed from TrustDialog');

  // UnknownNode uses Left/Right handles
  const unkSrc = fs.readFileSync(path.join(UI, 'src/components/OAIYBuilder/UnknownNode.tsx'), 'utf8');
  assert.ok(unkSrc.includes('Position.Left') && unkSrc.includes('Position.Right'));
  assert.equal(unkSrc.includes('Position.Top') || unkSrc.includes('Position.Bottom'), false);

  // Vendor OutputNode has zero raw file:// synthesis and checks _collapsed/_status/selected in arePropsEqual
  const vendorOutSrc = fs.readFileSync(path.join(UI, 'vendor/oaiy-ui-components/src/nodes/OutputNode.tsx'), 'utf8');
  assert.equal(vendorOutSrc.includes('file://${'), false, 'vendor OutputNode must not synthesize raw file:// URLs');
  assert.ok(vendorOutSrc.includes('prevProps.selected !== nextProps.selected'));
  assert.ok(vendorOutSrc.includes('prevProps.data._collapsed !== nextProps.data._collapsed'));
  assert.ok(vendorOutSrc.includes('prevProps.data._status !== nextProps.data._status'));

  // FlowsSidebar scrim has aria-hidden, sibling buttons (oaiy-flow-select), and dot instead of Running pill
  const flowsSrc = fs.readFileSync(path.join(UI, 'src/components/panels/FlowsSidebar.tsx'), 'utf8');
  assert.ok(flowsSrc.includes('className="oaiy-rail-scrim" aria-hidden="true"'));
  assert.ok(flowsSrc.includes('className="oaiy-flow-select"'));
  assert.ok(flowsSrc.includes('oaiy-node-status-dot'));
});

finish();
