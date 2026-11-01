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
 * - AI Chat when a provider's 200 holds no answer (no choice, an empty answer cut
 *   off at the limit, an error body, a thought that never closes): the step
 *   fails. It passed, the provider's JSON as its answer, or nothing. A reply that
 *   holds an answer of another shape does not fail: the table of reply shapes
 *   below is each one, what the step gives now and what it gave before.
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

/** A stand-in for Anthropic's Messages API: what it is sent is kept, and it answers in two text blocks, or with `answerWith`. */
const received = [];
let answerWith = null;
const server = http.createServer((req, res) => {
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', () => {
    received.push({ url: req.url, headers: req.headers, body: JSON.parse(body || '{}') });
    res.writeHead(200, { 'content-type': 'application/json' });
    if (answerWith !== null) return res.end(typeof answerWith === 'string' ? answerWith : JSON.stringify(answerWith));
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

// ---------------------------------------------------------------------------
// A 200 that holds no answer
// ---------------------------------------------------------------------------

/** chat() against a stand-in that answers `body` (an object as JSON, a string as it is); what it gave, or what it threw. */
const asked = async (body, { endpoint = OPENAI, template = '', path = '', host = {} } = {}) => {
  answerWith = body;
  const statuses = [];
  const args = chatArgs({});
  args[3] = endpoint;
  try {
    const methods = M.AIRuntime.createMethods({ ...ctx, ...host, onNodeStatus: (_id, status) => statuses.push(status) });
    const answer = await methods.chat(...args, 0, 0, false, '', null, 8, template, path, '');
    return { answer, status: statuses.at(-1) };
  } catch (e) {
    return { failed: e.message, status: statuses.at(-1) };
  } finally {
    answerWith = null;
  }
};
const OPENAI = ENDPOINT.replace('/anthropic/v1/messages', '/v1/chat/completions');
const TEMPLATE = '{"messages": {{messagesJsonRaw}}}';
const fails = async (name, body, words, where) => {
  await check(`${name} fails the step`, async () => {
    const r = await asked(body, where);
    assert.match(r.failed ?? `no failure, the answer was ${JSON.stringify(r.answer)}`, words);
    assert.equal(r.status, 'error');
  });
};
const answers = async (name, body, answer, where) => {
  await check(name, async () => {
    const r = await asked(body, where);
    assert.equal(r.failed, undefined);
    assert.equal(r.answer, answer);
    assert.equal(r.status, 'completed');
  });
};
const SPENT = /token limit.*Raise Max Tokens/;
const cutOff = { choices: [{ message: { content: '', reasoning_content: 'Let me think.' }, finish_reason: 'length' }] };

await fails('an OpenAI reply with no choice', { choices: [], id: 'x' }, /without a message/);
await fails('an empty answer cut off at the limit', cutOff, SPENT);
await fails('an error body with a status of 200', { error: { message: 'model overloaded' } }, /model overloaded/);
await fails('an error named in its choice', { choices: [{ error: { message: 'upstream gone' }, message: { content: '' } }] }, /upstream gone/);
await fails('an error beside the start of an answer', { choices: [{ error: { message: 'upstream gone' }, message: { content: 'Par' }, finish_reason: 'error' }] }, /upstream gone/);
await fails('a thought that never closes', { choices: [{ message: { content: '<think>Still going' }, finish_reason: 'length' }] }, SPENT);
await fails('a thought that never closes, with no reason given', { choices: [{ message: { content: '<think>Still going' } }] }, SPENT);
await fails('a thought that closes on nothing', { choices: [{ message: { content: '<think>Hm.</think>' }, finish_reason: 'stop' }] }, /without a message/);
await fails('an empty answer', { choices: [{ message: { content: '' }, finish_reason: 'stop' }] }, /without a message/);
await fails('an answer of white space', { choices: [{ message: { content: ' \n' }, finish_reason: 'stop' }] }, /without a message/);
await answers('an answer after its thought is the answer', { choices: [{ message: { content: '<think>Hm.</think>\nParis' } }] }, 'Paris');
await answers("a reply cut off after its answer began is that answer", { choices: [{ message: { content: 'Par' }, finish_reason: 'length' }] }, 'Par');
await answers('the older local shape is an answer', { response: 'Paris' }, 'Paris');

await fails("Anthropic's reply cut off in its thinking", { content: [{ type: 'thinking', thinking: 'hm' }], stop_reason: 'max_tokens' }, SPENT, { endpoint: ENDPOINT });
await fails("Anthropic's error body", { type: 'error', error: { type: 'overloaded_error', message: 'Overloaded' } }, /Overloaded/, { endpoint: ENDPOINT });
await fails("Anthropic's reply with no text", { content: [], stop_reason: 'end_turn' }, /without a message/, { endpoint: ENDPOINT });

// A service's own template: what it answers is its own, and an empty text may be its answer.
const CUSTOM = { template: TEMPLATE };
const AT_PATH = { template: TEMPLATE, path: 'choices.0.message.content' };
await answers("a template's empty answer at its path is its answer", { choices: [{ message: { content: '' }, finish_reason: 'stop' }] }, '', AT_PATH);
await answers("a template's empty text is its answer", { text: '' }, '', CUSTOM);
await answers("a template's endpoint of a shape of its own is read by the flow", { result: { a: 1 } }, '{"result":{"a":1}}', CUSTOM);
await answers("a template's endpoint that answers in plain text", 'Paris', 'Paris', CUSTOM);
await answers('an answer at its path stands beside an error member', { text: 'Paris', error: 'none' }, 'Paris', { template: TEMPLATE, path: 'text' });
await fails('a template with no choice at its path', { choices: [] }, /nothing at choices\.0\.message\.content/, AT_PATH);
await fails('a template cut off at the limit', cutOff, SPENT, AT_PATH);
await fails('a template with no path cut off at the limit', cutOff, SPENT, CUSTOM);
await fails('a template with an error body at a path', { error: { message: 'quota spent' } }, /quota spent/, AT_PATH);
await fails('a template with an error body and no path', { error: 'quota spent' }, /quota spent/, CUSTOM);
await fails('a template with no choice and no path', { choices: [] }, /without a message/, CUSTOM);
await fails("a template's thought that never closes", { text: '<think>Still going' }, SPENT, CUSTOM);

// Every shape of reply, what the step gives for it, and what it gave before a 200 was checked for an answer (4000bd22).
// A string is the answer, JSON is the reply's own JSON handed to the flow to read, and a pattern is a failure's words.
const JSON_REPLY = Symbol('the reply as JSON');
const STRUCTURED = /text\.replace is not a function/;
const openai = (message, finish_reason = 'stop', extra = {}) => ({ choices: [{ message, finish_reason }], ...extra });
const CALL = { id: 'call_1', type: 'function', function: { name: 'look_up', arguments: '{}' } };
const TOOL_TURN = openai({ content: null, tool_calls: [CALL] }, 'tool_calls');
const TOOL_TURN_BLANK = openai({ content: '', tool_calls: [CALL] }, 'tool_calls');
const PICTURES = [{ type: 'image_url', image_url: { url: PICTURE } }];
const OWN_LIST = { choices: ['yes', 'no'], picked: 'yes' };
const AT_CLAUDE = { endpoint: ENDPOINT.replace('/anthropic/v1/messages', '/claude/v1/chat/completions') };
const AT_ANTHROPIC = { endpoint: ENDPOINT };
const AT_TOOL_CALLS = { template: TEMPLATE, path: 'choices.0.message.tool_calls' };

const SHAPES = [
  // A plain OpenAI endpoint.
  ['an answer', {}, openai({ content: 'Paris' }), 'Paris', 'Paris'],
  ['an answer cut off', {}, openai({ content: 'Par' }, 'length'), 'Par', 'Par'],
  ['the older local shape', {}, { response: 'Paris' }, 'Paris', 'Paris'],
  ["Ollama's own chat", {}, { model: 'm', message: { role: 'assistant', content: 'Paris' }, done: true }, 'Paris', JSON_REPLY],
  ['a turn that only calls tools', {}, TOOL_TURN, JSON_REPLY, JSON_REPLY],
  ['a turn that only calls tools, its text blank', {}, TOOL_TURN_BLANK, JSON_REPLY, JSON_REPLY],
  ['pictures and no text', {}, openai({ content: '', images: PICTURES }), JSON_REPLY, JSON_REPLY],
  ['an empty error beside an answer', {}, openai({ content: 'Paris' }, 'stop', { error: {} }), 'Paris', 'Paris'],
  ['an empty error beside nothing', {}, openai({ content: '' }, 'stop', { error: {} }), /without a message \(finish_reason: stop\)/, JSON_REPLY],
  ['an error beside a whole answer', {}, openai({ content: 'Paris' }, 'stop', { error: { code: 0, message: 'none' } }), 'Paris', 'Paris'],
  ['a refusal', {}, openai({ content: null, refusal: 'I cannot help with that.' }), /The model refused: I cannot help with that\. \(finish_reason: stop\)/, JSON_REPLY],
  ['a content filter', {}, openai({ content: null }, 'content_filter'), /without a message \(finish_reason: content_filter\)/, JSON_REPLY],
  ['nothing, cut off at the limit', {}, cutOff, SPENT, JSON_REPLY],
  ['nothing but reasoning', {}, openai({ content: '', reasoning_content: 'Let me think.' }), /without a message \(finish_reason: stop\)/, JSON_REPLY],
  ['no choice', {}, { choices: [], id: 'x' }, /without a message/, JSON_REPLY],
  ['an empty answer', {}, openai({ content: '' }), /without a message \(finish_reason: stop\)/, JSON_REPLY],
  ['an error body', {}, { error: { message: 'model overloaded' } }, /model overloaded/, JSON_REPLY],
  ['an error beside the start of an answer', {}, { choices: [{ error: { message: 'upstream gone' }, message: { content: 'Par' }, finish_reason: 'error' }] }, /upstream gone/, 'Par'],
  ['an answer in parts (not read, then or now)', {}, openai({ content: [{ type: 'text', text: 'Paris' }] }), STRUCTURED, STRUCTURED],

  // An address with Anthropic in it, or `claude`.
  ['text blocks', AT_ANTHROPIC, { content: [{ type: 'thinking', thinking: 'hm' }, { type: 'text', text: 'Par' }, { type: 'text', text: 'is' }], stop_reason: 'end_turn' }, 'Paris', 'Paris'],
  ['an OpenAI reply at a proxy with claude in its URL', AT_CLAUDE, openai({ content: 'Paris' }), 'Paris', JSON_REPLY],
  ['no text at the end of a turn', AT_ANTHROPIC, { content: [], stop_reason: 'end_turn' }, /without a message \(stop_reason: end_turn\)/, JSON_REPLY],
  ['a refusal by its stop reason', AT_ANTHROPIC, { content: [], stop_reason: 'refusal' }, /without a message \(stop_reason: refusal\)/, JSON_REPLY],
  ['thinking to the limit', AT_ANTHROPIC, { content: [{ type: 'thinking', thinking: 'hm' }], stop_reason: 'max_tokens' }, SPENT, JSON_REPLY],
  ['a turn that only calls a tool', AT_ANTHROPIC, { content: [{ type: 'tool_use', id: 't', name: 'look_up', input: {} }], stop_reason: 'tool_use' }, JSON_REPLY, JSON_REPLY],
  ["Anthropic's error body", AT_ANTHROPIC, { type: 'error', error: { type: 'overloaded_error', message: 'Overloaded' } }, /Overloaded/, JSON_REPLY],

  // A service's own template with no response path.
  ['a template: a turn that only calls tools', CUSTOM, TOOL_TURN, JSON_REPLY, JSON_REPLY],
  ['a template: a turn that only calls tools, its text blank', CUSTOM, TOOL_TURN_BLANK, JSON_REPLY, ''],
  ['a template: pictures and no text', CUSTOM, openai({ content: '', images: PICTURES }), JSON_REPLY, ''],
  ["a template: an endpoint's own `choices` list", CUSTOM, OWN_LIST, JSON_REPLY, JSON_REPLY],
  ['a template: no choice', CUSTOM, { choices: [] }, /without a message/, JSON_REPLY],
  ['a template: an error body', CUSTOM, { error: 'quota spent' }, /quota spent/, JSON_REPLY],
  ['a template: an empty error beside a shape of its own', CUSTOM, { error: {}, result: 1 }, JSON_REPLY, JSON_REPLY],
  ['a template: a refusal', CUSTOM, openai({ content: null, refusal: 'No.' }), /The model refused: No\. \(finish_reason: stop\)/, JSON_REPLY],
  ['a template: a content filter, blank', CUSTOM, openai({ content: '' }, 'content_filter'), /finish_reason: content_filter/, ''],
  ['a template: a shape of its own', CUSTOM, { result: { a: 1 } }, JSON_REPLY, JSON_REPLY],
  ['a template: plain text', CUSTOM, 'Paris', 'Paris', 'Paris'],

  // The same with a response path.
  ['at a path: a turn that only calls tools', AT_PATH, TOOL_TURN, JSON_REPLY, JSON_REPLY],
  ['at a path: the tool calls themselves', AT_TOOL_CALLS, TOOL_TURN, JSON.stringify([CALL]), JSON.stringify([CALL])],
  ['at a path: nothing there', AT_PATH, { choices: [] }, /nothing at choices\.0\.message\.content/, JSON_REPLY],
  ['at a path: an empty answer', AT_PATH, openai({ content: '' }), '', ''],
  ["at a path: an endpoint's own `choices` list", { template: TEMPLATE, path: 'picked' }, OWN_LIST, 'yes', 'yes'],
];

for (const [name, where, body, now] of SHAPES) {
  await check(`reply shape: ${name}`, async () => {
    const r = await asked(body, where);
    if (now instanceof RegExp) {
      assert.match(r.failed ?? `no failure, the answer was ${JSON.stringify(r.answer)}`, now);
      assert.equal(r.status, 'error');
      return;
    }
    assert.equal(r.failed, undefined);
    assert.equal(r.answer, now === JSON_REPLY ? JSON.stringify(body) : now);
    assert.equal(r.status, 'completed');
  });
}

// The two answers that come by no provider.
const localModel = (text) => ({ endpoint: 'oaiy-llm://local', host: { tauri: { invoke: async (command) => (command.endsWith('|llm_chat') ? { text, modelId: 'm', backend: 'cpu', promptTokens: 1, generatedTokens: 1 } : 0) } } });
await fails('a local model that only thought', null, SPENT, localModel('<think>Still going'));
await answers("a local model's answer after its thought", null, 'Paris', localModel('<think>Hm.</think>Paris'));
await fails('an external AI that only thought', null, SPENT, { host: { useClaudeForAI: true, yieldForAI: async () => '<think>Still going' } });

await new Promise((resolve) => server.close(resolve));

// ---------------------------------------------------------------------------
// A 401, and whose it is
// ---------------------------------------------------------------------------

await check("a 401 from the run's own OAIY is not put down to an API key, and a provider's still is", async () => {
  const refuses = http.createServer((req, res) => {
    req.resume();
    res.writeHead(401, { 'content-type': 'application/json' });
    res.end('{"error":{"code":"token_invalid","message":"The credential is not valid."}}');
  });
  await new Promise((resolve) => refuses.listen(0, '127.0.0.1', resolve));
  const base = `http://localhost:${refuses.address().port}`;
  const endpoint = `${base}/api/ai/providers/oaiy-engine/v1/chat/completions`;
  try {
    Object.assign(process.env, { OAIY_SERVER_URL: base, OAIY_SERVER_TOKEN: 'oaiyrun_x' });
    const own = await asked('', { endpoint });
    assert.doesNotMatch(own.failed, /Check your API key/);
    assert.match(own.failed, /This OAIY refused the run's own credential for AI chat.*token_invalid/);
    process.env.OAIY_SERVER_URL = 'http://localhost:1';
    assert.match((await asked('', { endpoint })).failed, /Check your API key/, "another address is a provider's");
  } finally {
    delete process.env.OAIY_SERVER_URL;
    delete process.env.OAIY_SERVER_TOKEN;
    await new Promise((resolve) => refuses.close(resolve));
  }
});

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
