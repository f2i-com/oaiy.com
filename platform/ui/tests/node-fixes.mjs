/**
 * Two nodes that did less than they said.
 *
 *     npm run test:node-fixes
 *
 * - AI Chat to Anthropic's Messages API (core-ai/runtime.ts): the history and
 *   the pictures reach the model. The request went out with the node's system
 *   prompt alone (a history given as text, folded into the prompt on every
 *   other path, was left out; a history's own system turns were dropped), and
 *   with pictures in OpenAI's `image_url` shape, which Anthropic refuses. Here
 *   the chat is sent to a stand-in server on this computer, which records what
 *   it was sent and answers as Anthropic does.
 * - Image View (core-image/ui/ImageViewNode.tsx): its Max Height is the
 *   preview's height. The property was in the node's settings and changed
 *   nothing. Rendered with react-dom/server.
 *
 * The modules are TypeScript with aliases only the bundler resolves, so they are
 * bundled for Node with esbuild (as engine-endpoint.mjs does); oaiy-ui-components
 * is stubbed (the node's wrapper renders its children).
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import * as esbuild from 'esbuild';

const UI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

let pass = 0;
const failures = [];
async function check(name, fn) {
  try {
    await fn();
    pass++;
    console.log(`  ok  ${name}`);
  } catch (e) {
    failures.push(`${name} — ${e.message}`);
    console.log(`  FAIL ${name} — ${e.stack || e.message}`);
  }
}

// A plain page: storage, and no OAIY around it (nothing here may ask a desktop).
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
  location: { protocol: 'https:', origin: 'https://oaiy.com' },
};
// Only the stand-in server is reached: OAIY's own ports on this computer never are.
const realFetch = globalThis.fetch;
const OWN_PORTS = /:(7860|8080|8783|179[7-9]\d)(\/|$)/;
globalThis.fetch = async (url, init) => {
  const u = String(url);
  if (OWN_PORTS.test(u)) throw new Error(`this test may not reach ${u}`);
  return realFetch(url, init);
};

// (loaded from here, so the stub's own `react` is this package's, the one react-dom renders with)
const UI_STUB = `import { createElement } from 'react';
export function invalidateDynamicOptions() {}
export function subscribeToDynamicOptionsInvalidation() { return () => {}; }
export function CollapsibleNodeWrapper({ children }) { return createElement('div', { 'data-node': '' }, children); }
`;
const aliases = {
  name: 'oaiy-aliases',
  setup(build) {
    build.onResolve({ filter: /^oaiy-ui-components$/ }, () => ({ path: 'oaiy-ui-components', namespace: 'stub' }));
    build.onLoad({ filter: /^oaiy-ui-components$/, namespace: 'stub' }, () => ({ contents: UI_STUB, loader: 'js', resolveDir: UI }));
    build.onResolve({ filter: /^oaiy-core\/modules\// }, (a) => ({ path: path.join(UI, 'src/bundled-modules', `${a.path.slice('oaiy-core/modules/'.length)}.ts`) }));
    build.onResolve({ filter: /^oaiy-core$/ }, () => ({ path: path.join(UI, 'vendor/oaiy-core/src/index.ts') }));
    build.onResolve({ filter: /^oaiy-core\// }, (a) => {
      const rest = a.path.slice('oaiy-core/'.length);
      const p = path.join(UI, 'vendor/oaiy-core', rest);
      return { path: fs.existsSync(p) ? p : `${p}.ts` };
    });
    // (the canvas library: only its Position names are read)
    build.onResolve({ filter: /^@xyflow\/react$/ }, () => ({ path: 'xyflow-stub', namespace: 'stub' }));
    build.onLoad({ filter: /^xyflow-stub$/, namespace: 'stub' }, () => ({ contents: "export const Position = { Left: 'left', Right: 'right', Top: 'top', Bottom: 'bottom' };", loader: 'js' }));
    build.onResolve({ filter: /^\.\/ffmpeg$/ }, () => ({ path: './ffmpeg', external: true }));
  },
};
const bundlePath = path.join(os.tmpdir(), `oaiy-node-fixes-${process.pid}.mjs`);
await esbuild.build({
  stdin: {
    contents: `export { default as AIRuntime } from './src/bundled-modules/core-ai/runtime.ts';
export { default as ImageViewNode, previewHeights } from './src/bundled-modules/core-image/ui/ImageViewNode.tsx';
export { createElement } from 'react';
export { renderToStaticMarkup } from 'react-dom/server';`,
    resolveDir: UI,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: bundlePath,
  logLevel: 'silent',
  jsx: 'automatic',
  // (react-dom/server is CommonJS and requires Node's own modules)
  banner: { js: "import { createRequire as __oaiyRequire } from 'node:module'; const require = __oaiyRequire(import.meta.url);" },
  plugins: [aliases],
});
const M = await import(pathToFileURL(bundlePath).href);
fs.rmSync(bundlePath, { force: true });

// ---------------------------------------------------------------------------
// AI Chat to Anthropic
// ---------------------------------------------------------------------------

/** A stand-in for Anthropic's Messages API: what it is sent is kept, and it answers in two text blocks. */
const received = [];
const server = http.createServer((req, res) => {
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', () => {
    received.push({ url: req.url, headers: req.headers, body: JSON.parse(body || '{}') });
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ content: [{ type: 'thinking', thinking: 'hm' }, { type: 'text', text: 'A red ' }, { type: 'text', text: 'square.' }] }));
  });
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
// (`localhost`, so the run does not ask a desktop to start a service on this port first)
const ENDPOINT = `http://localhost:${server.address().port}/anthropic/v1/messages`;

const ctx = {
  log() {},
  onNodeStatus() {},
  abortSignal: undefined,
  getModuleSetting: () => undefined,
  getConstant: () => undefined,
  secureFetch: (url, init) => globalThis.fetch(url, { method: init.method, headers: init.headers, body: init.body }),
};
const PICTURE = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
/** chat()'s arguments, in its order, with what each test changes. */
const chatArgs = (o) => [
  o.system ?? '', o.prompt ?? 'What is this?', o.input ?? null, ENDPOINT, 'claude-test', 'anthropic-test-key', false, 256, 0, 'text',
  o.images ?? false, 'auto', 'n1', [], o.history ?? '',
];
const send = async (o) => {
  received.length = 0;
  const answer = await M.AIRuntime.createMethods(ctx).chat(...chatArgs(o));
  assert.equal(received.length, 1, 'one request reached the stand-in');
  return { answer, sent: received[0].body };
};

await check("a history given as text reaches Anthropic's system prompt, beside the node's own", async () => {
  const { sent } = await send({ system: 'Be brief.', history: 'Step 1: looked at the file.' });
  assert.match(sent.system, /Be brief\./);
  assert.match(sent.system, /Step 1: looked at the file\./, 'the history is in the system prompt');
  assert.ok(sent.messages.every((m) => m.role !== 'system'), 'no system turn among the messages');
  assert.equal(sent.messages.at(-1).content, 'What is this?');
});

await check("a history given as turns is sent as turns, and its system turn goes into Anthropic's system prompt", async () => {
  const history = [
    { role: 'system', content: 'You answer about pictures.' },
    { role: 'user', content: 'Hello' },
    { role: 'assistant', content: 'Hi, send me a picture.' },
  ];
  const { sent } = await send({ system: 'Be brief.', history });
  assert.deepEqual(sent.messages.map((m) => [m.role, m.content]), [['user', 'Hello'], ['assistant', 'Hi, send me a picture.'], ['user', 'What is this?']]);
  assert.match(sent.system, /You answer about pictures\./);
  assert.match(sent.system, /Be brief\./);
});

await check("a picture goes as Anthropic's image block: its bytes, with the type its data URL names", async () => {
  const { answer, sent } = await send({ input: PICTURE, images: true });
  assert.equal(JSON.stringify(sent).includes('image_url'), false, "no OpenAI image_url part is sent");
  const last = sent.messages.at(-1);
  assert.equal(last.role, 'user');
  assert.deepEqual(last.content[0], { type: 'text', text: 'What is this?' });
  assert.deepEqual(last.content[1], { type: 'image', source: { type: 'base64', media_type: 'image/png', data: PICTURE.slice('data:image/png;base64,'.length) } });
  assert.equal(received[0].headers['x-api-key'], 'anthropic-test-key');
  assert.equal(received[0].headers['anthropic-version'], '2023-06-01');
  assert.equal(answer, 'A red square.', 'every text block of the answer, in order (the thinking block has none)');
});

await check("a picture in a history's turn given in OpenAI's shape by its address goes as an image block by its address", async () => {
  const history = [{ role: 'user', content: [{ type: 'text', text: 'Earlier:' }, { type: 'image_url', image_url: { url: 'https://example.com/a.png' } }] }, { role: 'assistant', content: 'Seen.' }];
  const { sent } = await send({ history });
  assert.deepEqual(sent.messages[0].content, [{ type: 'text', text: 'Earlier:' }, { type: 'image', source: { type: 'url', url: 'https://example.com/a.png' } }]);
});

await new Promise((resolve) => server.close(resolve));

// ---------------------------------------------------------------------------
// Image View's Max Height
// ---------------------------------------------------------------------------

/** The node rendered with `data`, and its preview box's height. */
const render = (data) => {
  const html = M.renderToStaticMarkup(M.createElement(M.ImageViewNode, { data: { imageUrl: PICTURE, ...data } }));
  const box = /<div[^>]*data-preview-box[^>]*>/.exec(html)?.[0];
  assert.ok(box, `the preview box is rendered: ${html.slice(0, 300)}`);
  return { html, height: /height:\s*(\d+)px/.exec(box)?.[1], larger: /Larger/.test(html) };
};

await check('Image View shows its picture in a box as tall as its Max Height allows', async () => {
  const tall = render({ maxHeight: 600 });
  assert.equal(tall.height, '128', 'first shown at the small height');
  assert.equal(tall.larger, true, 'Larger is offered, up to 600 pixels');
  assert.deepEqual(M.previewHeights(600), { small: 128, large: 600 });
  const plain = render({});
  assert.equal(plain.height, '128');
  assert.deepEqual(M.previewHeights(undefined), { small: 128, large: 300 }, "the property's default");
  const short = render({ maxHeight: 80 });
  assert.equal(short.height, '80', 'no taller than its Max Height');
  assert.equal(short.larger, false, 'nothing for Larger to show');
});

await check("Image View's Max Height is held to the property's range", async () => {
  assert.deepEqual(M.previewHeights(5), { small: 50, large: 50 });
  assert.deepEqual(M.previewHeights(99999), { small: 128, large: 1000 });
  assert.deepEqual(M.previewHeights('450'), { small: 128, large: 450 }, 'a number written as text');
  assert.deepEqual(M.previewHeights('tall'), { small: 128, large: 300 });
});

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
