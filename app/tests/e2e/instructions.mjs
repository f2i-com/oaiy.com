// What the person's own conversations are told and given, in the browser: a fake OAIY Desktop with the phone and the
// calendar on (the business: Green Lawns, its receptionist Aokie) and its control API's real tool list (as agentctl.mjs
// serves it), and a scripted model behind a provider of OAIY's engine's kind, in OAIY's window. Each request's tools are
// read as the model gets them. The Front desk's runner is told who answers the phone and has the business's tools and
// OAIY's read tools; a kept project is not told, and has the business's tools and all of OAIY's until its Instructions
// dialog turns the business's off (then OAIY's own tools but those for the business's data); an incognito project has
// none of it, nor any of OAIY's tools. The person's instructions (Agent settings) end their own conversations' system
// prompts, a project may take its own instead or none, and a call or a text never gets them. Each step asks OAIY's engine
// to think as the chip says, as long as the Agent's settings say, and the chat keeps the count its engine says as the
// thought goes on; a call does not think, and a text thinks as the engine is set, as long as the Agent's settings say.
// One of the desktop's AI providers, and a paired computer's engine, are asked as far as their server is known to take
// it. Nothing here reaches a real desktop, phone or model.
//   node tests/e2e/instructions.mjs [screenshot-dir]
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { createServer as createHttpServer } from 'node:http';
import { join } from 'node:path';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const shotsDir = process.argv[2] ?? null;
const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].filter(Boolean).find((p) => existsSync(p));
if (!executablePath) {
  console.error('no Chrome found; set CHROME');
  process.exit(2);
}
if (shotsDir) mkdirSync(shotsDir, { recursive: true });
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const cors = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*', 'Access-Control-Allow-Methods': '*', 'Access-Control-Allow-Private-Network': 'true' };
const readBody = (req) => new Promise((resolve) => {
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', () => resolve(body));
});
async function until(what, fn, ms = 20_000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (await fn()) return;
    await wait(100);
  }
  throw new Error(`timed out waiting for ${what}`);
}

// --- OAIY's control tools, as the desktop lists them (captured from a real oaiy-server) -------------------------------
const { tools: CONTROL_TOOLS, runner: READ_TOOLS } = JSON.parse(readFileSync(new URL('../fixtures/control-tools.json', import.meta.url), 'utf8'));
const CONTROL = CONTROL_TOOLS.map((t) => t.name);
/** Those that read, change or describe the business's data: a project whose business tools are off is not offered them. */
const BUSINESS = ['calendar_settings_get', 'calendar_settings_set', 'contacts_list', 'contact_get', 'contact_set', 'contact_forget_fact', 'link_status', 'link_sync_now', 'plugin_settings_get', 'plugin_settings_set', 'plugin_command', 'logs_tail'];
/** The desktop's flow tools the app's own stand in for, where it has them (desktop/mcp.ts COVERED): not counted either way. */
const STOOD_IN_FOR = new Set(['flows_list', 'flow_get', 'flow_create', 'flow_update', 'flow_run']);
/** OAIY's control tools among a request's tools (by their desktop names: one whose name was taken has `oaiy_` in front). */
const controlIn = (tools) => [...new Set(tools.map((n) => n.replace(/^oaiy_/, '')))].filter((n) => CONTROL.includes(n) && !STOOD_IN_FOR.has(n)).sort();
const counted = (names) => names.filter((n) => !STOOD_IN_FOR.has(n)).sort();
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// --- the scripted model --------------------------------------------------------------
/** Each request: whose it is (a call's, by its system prompt, or the person's own), what it was sent and how it was asked to think. */
const asked = [];
const textOf = (m) => (typeof m.content === 'string' ? m.content : (m.content ?? []).map((p) => p.text ?? '').join(''));
const sse = (value) => `data: ${JSON.stringify(value)}\n\n`;
const chunks = (text) => (text.match(/.{1,6}/gs) ?? []).map((piece) => sse({ choices: [{ index: 0, delta: { content: piece } }] })).join('');
const end = (calls) => {
  let out = '';
  (calls ?? []).forEach((call, i) => {
    out += sse({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: `call_${asked.length}_${i}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.input) } }] } }] });
  });
  return `${out}${sse({ choices: [{ index: 0, delta: {}, finish_reason: calls ? 'tool_calls' : 'stop' }] })}data: [DONE]\n\n`;
};
/** While the slow thought is between its words: the chat's status is read then. */
let thoughtPaused = false;
/**
 * A request answered: recorded, and its answer streamed. "Think it over." thinks slowly first, with the count OAIY's
 * engine says (4 of 512), then more words of the thought, a pause, and the answer.
 */
async function answer(req, res, route) {
  const body = JSON.parse(await readBody(req));
  res.setHeader('content-type', 'text/event-stream');
  // A warm (the model reads the prompt ahead, a few words at most asked for): nothing to say.
  if ((body.max_tokens ?? body.max_completion_tokens) <= 32) {
    if (process.env.E2E_DEBUG) console.log(`     (a warm on ${route}: ${body.max_tokens ?? body.max_completion_tokens})`);
    return res.end(end());
  }
  const system = textOf(body.messages.find((m) => m.role === 'system') ?? { content: '' });
  const at = body.messages.map((m) => m.role === 'user').lastIndexOf(true);
  const user = textOf(body.messages[at] ?? { content: '' });
  const steps = body.messages.slice(at + 1).filter((m) => m.role === 'assistant').length;
  // (by how a call's and a text thread's system prompts begin: the Front desk's own speaks of its calls and texts too)
  const kind = /live phone call/.test(system) ? 'call' : /This conversation is a text-message thread with/.test(system) ? 'sms' : 'own';
  asked.push({
    kind,
    route,
    system,
    user,
    messages: JSON.stringify(body.messages),
    tools: (body.tools ?? []).map((t) => t.function?.name ?? t.name),
    effort: body.reasoning_effort,
    thinking: body.thinking,
    template: body.chat_template_kwargs,
    maxTokens: body.max_tokens,
  });
  if (process.env.E2E_DEBUG) console.log(`     (${kind} on ${route}: "${user.slice(0, 40)}", ${body.messages.length} messages)`);
  if (kind === 'call') return res.end(chunks('Yes, we are open on Saturday.') + end());
  if (/Think it over/.test(user) && steps === 0) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
    res.write(sse({ choices: [{ index: 0, delta: { reasoning_content: 'Let me think about this.' } }] }));
    res.write(sse({ id: 'x', choices: [], oaiy_thinking: { used: 4, budget: 512, done: false } }));
    await wait(1200);
    res.write(sse({ choices: [{ index: 0, delta: { reasoning_content: ' And a little more.' } }] }));
    thoughtPaused = true;
    await wait(1500);
    thoughtPaused = false;
    res.write(sse({ id: 'x', choices: [], oaiy_thinking: { used: 9, budget: 512, done: true } }));
    return res.end(chunks('Done.') + end());
  }
  return res.end(chunks('Done.') + end());
}
const model = createHttpServer(async (req, res) => {
  for (const [k, v] of Object.entries(cors)) res.setHeader(k, v);
  if (req.method === 'OPTIONS') return res.end();
  if (req.url.endsWith('/models')) {
    res.setHeader('content-type', 'application/json');
    return res.end(JSON.stringify({ data: [{ id: 'test-model', context_length: 65_536 }] }));
  }
  if (req.method === 'GET') {
    res.statusCode = 404;
    return res.end();
  }
  return answer(req, res, 'engine');
});
await new Promise((r) => model.listen(0, '127.0.0.1', r));
const modelUrl = `http://127.0.0.1:${model.address().port}`;

// --- a fake OAIY Desktop: the phone and the calendar on, a business and its receptionist -------------------
const TOKEN = 'instructions-e2e';
const MODULES = {
  revision: 1,
  modules: [
    { id: 'phone', name: 'Phone', enabled: true, provider: { pluginId: 'aokie', name: 'AI Receptionist', state: 'running', declared: true } },
    { id: 'calendar', name: 'Calendar', enabled: true, provider: { pluginId: 'aokie', name: 'AI Receptionist', state: 'running', declared: true } },
  ],
  warnings: [],
};
/** What the desktop says the Agent runs on (Settings → Agent): its engine, until the checks of its routes. */
let preference = { model: { source: 'engine' } };
/** The desktop's AI providers, as GET /api/ai/providers lists them (never a key). */
let aiProviders = [];
const voiceClients = new Set();
/** The phone's events (a text received), in the order the page reads them. */
const bridgeEvents = [];
/** What the page asked of the desktop, latest last (for a failure's message). */
const desktopSeen = [];
const desktopServer = createHttpServer(async (req, res) => {
  for (const [k, v] of Object.entries(cors)) res.setHeader(k, v);
  if (req.method === 'OPTIONS') return res.end();
  const path = new URL(req.url, 'http://x').pathname;
  desktopSeen.push(`${req.method} ${path}`);
  // The desktop's gateway to one of its AI providers, and to a paired computer's engine: the request goes on as it is.
  const routed = /^\/api\/(?:ai\/providers\/[^/]+|network\/nodes\/[^/]+)\/v1\/(models|chat\/completions)$/.exec(path);
  if (routed && routed[1] === 'chat/completions') {
    return answer(req, res, path).catch((error) => {
      console.log(`     (the desktop's ${path} could not answer: ${error.message})`);
      res.statusCode = 500;
      res.end();
    });
  }
  const body = await readBody(req);
  const json = (value, status = 200) => {
    res.statusCode = status;
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify(value));
  };
  if (routed) return json({ object: 'list', data: [{ id: 'box-model', context_length: 65_536 }] });
  const stream = (events) => {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
    for (const e of events) res.write(`data: ${JSON.stringify(e)}\n\n`);
  };
  if (path === '/api/health') return json({ product: 'oaiy-desktop', protocol: 'oaiy-bridge/1', version: 'instructions' });
  if (path === '/api/modules') return json(MODULES);
  if (path === '/api/modules/events') return stream([MODULES]);
  if (path === '/api/agent/events') return stream([]);
  if (path === '/api/agent/preferences') return json(preference);
  if (path === '/api/ai/providers') return json({ providers: aiProviders });
  if (path === '/api/voice/events') {
    stream([{ type: 'hello', calls: [] }]);
    voiceClients.add(res);
    req.on('close', () => voiceClients.delete(res));
    return;
  }
  if (path === '/api/bridge/events') {
    const since = Number(new URL(req.url, 'http://x').searchParams.get('since') ?? 0);
    return json({ events: bridgeEvents.slice(since).map((envelope, i) => ({ seq: since + i + 1, envelope })), next: bridgeEvents.length });
  }
  if (path.startsWith('/api/bridge/leases/')) return json({ granted: true, holder: (JSON.parse(body || '{}').holder ?? '') });
  if (path.startsWith('/api/bridge/connectors/')) {
    const { command } = JSON.parse(body || '{}');
    return json({ ok: true, result: { ok: true, data: command === 'phone.status' ? { connected: true } : command === 'settings.get' ? { value: '' } : {} } });
  }
  if (path === '/api/bridge/flows') return json({ flows: [] });
  if (path === '/api/plugins') return json({ plugins: [{ id: 'aokie', state: 'running' }] });
  // The business's name; no receptionistName (a desktop that has not set one says none): the receptionist is Aokie.
  if (path.startsWith('/api/calendar')) return json({ available: true, settings: { business: 'Green Lawns' }, appointments: [], now: new Date().toISOString() });
  if (path.startsWith('/api/voice/calls/')) return json({ result: { ok: true, output: {} } });
  if (path === '/api/voice/callers') return json({});
  if (path === '/api/contacts') return json({ contacts: [], total: 0 });
  if (path.startsWith('/api/contacts/')) return json({ error: { code: 'no_contact', message: 'none' } }, 404);
  // The control API: the desktop's session rule (project and setup all of its tools, the runner its read ones, a call,
  // a text or a flow's task none), with its real list.
  if (path === '/api/mcp') {
    const message = JSON.parse(body || '{}');
    if (message.id === undefined) {
      res.statusCode = 202;
      return res.end();
    }
    const session = req.headers['x-oaiy-session'] ?? 'project';
    const ok = (result) => json({ jsonrpc: '2.0', id: message.id, result });
    if (message.method === 'initialize') return ok({ protocolVersion: '2025-06-18', capabilities: { tools: {} }, serverInfo: { name: 'oaiy', version: 'e2e' } });
    const offered = session === 'project' || session === 'setup' ? CONTROL_TOOLS : session === 'runner' ? CONTROL_TOOLS.filter((t) => READ_TOOLS.includes(t.name)) : [];
    if (message.method === 'tools/list') return ok({ tools: offered });
    return json({ jsonrpc: '2.0', id: message.id, error: { code: -32601, message: `method not found: ${message.method}` } });
  }
  return json({ error: { message: 'not here' } }, 404);
});
await new Promise((r) => desktopServer.listen(0, '127.0.0.1', r));
const desktopUrl = `http://127.0.0.1:${desktopServer.address().port}`;
const voice = (event) => {
  for (const c of voiceClients) c.write(`data: ${JSON.stringify(event)}\n\n`);
};

// --- the app ---------------------------------------------------------------------------
const server = await createServer({ server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const base = `http://127.0.0.1:${server.httpServer.address().port}`;
const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-first-run'] });
let failures = 0;
const check = async (name, fn) => {
  try {
    await fn();
    console.log(`ok   ${name}`);
  } catch (error) {
    failures++;
    console.log(`FAIL ${name}\n     ${String(error.message).split('\n').join('\n     ')}`);
  }
};
const expect = (c, m) => {
  if (!c) throw new Error(m);
};
const shots = [];
/** The page in each theme (and, with `before`, as that sets it up each time): only with a screenshot folder. */
async function shoot(page, scene, before, clip) {
  if (!shotsDir) return;
  for (const theme of ['light', 'dark']) {
    await page.evaluate((t) => window.__oaiySetTheme?.(t), theme);
    await wait(350);
    if (before) await before();
    const file = join(shotsDir, `instructions-${scene}-${theme}.png`);
    await page.screenshot({ path: file, ...(clip ? { clip: await clip() } : {}) });
    shots.push(file);
  }
  await page.evaluate(() => window.__oaiySetTheme?.('light'));
}

const idle = (page) => page.waitForFunction(() => !document.querySelector('section.chat')?.classList.contains('busy'), { timeout: 30_000 });
async function say(page, text) {
  await idle(page);
  await page.focus('.chat-input');
  await page.keyboard.type(text);
  await page.keyboard.press('Enter');
}
/** The open conversation asked something short: its request (with the tools it was given). */
async function ask(page, text) {
  const from = asked.length;
  await say(page, text);
  await until(`the request for "${text}"`, () => asked.slice(from).some((a) => a.kind === 'own')).catch(async (error) => {
    // (what the chat said instead, for the failure's message)
    const said = await page.evaluate(() => [...document.querySelectorAll('div.msg.system pre, div.msg.error pre')].slice(-3).map((p) => p.textContent).join(' | '));
    throw new Error(`${error.message}; the chat said: ${said}; the desktop was last asked: ${desktopSeen.filter((p) => !/events|leases|voice/.test(p)).slice(-6).join(', ')}`);
  });
  await idle(page);
  return asked.slice(from).find((a) => a.kind === 'own');
}
/** The chat's last note, once it says what `pattern` matches. */
async function noteSaying(page, pattern) {
  await page.waitForFunction((source) => new RegExp(source).test([...document.querySelectorAll('div.msg.system pre')].at(-1)?.textContent ?? ''), { timeout: 5000 }, pattern.source).catch(() => {});
  return page.evaluate(() => [...document.querySelectorAll('div.msg.system pre')].at(-1)?.textContent ?? '');
}
/** The Agent's settings as this page keeps them. */
const kept = (page) => page.evaluate(async () => (await (await import('/src/settings.ts')).loadSettings()).agent);
/** The Agent's settings once they are kept as `ok` says (they are written as the dialog closes). */
async function keptAs(page, ok) {
  let last = null;
  await until('the settings kept', async () => ok((last = await kept(page))), 5000).catch(() => {
    throw new Error(JSON.stringify(last));
  });
  return last;
}
async function openProject(page, name) {
  await page.evaluate((name) => {
    const select = document.querySelector('.project-select');
    const option = [...select.options].find((o) => o.textContent.includes(name));
    select.value = option.value;
    select.dispatchEvent(new Event('change'));
  }, name);
  await page.waitForFunction((name) => document.querySelector('.project-select')?.selectedOptions[0]?.textContent.includes(name), { timeout: 10_000 }, name);
  await wait(300);
}
/** The Instructions dialog: opened from its chip, a choice made, saved. */
async function setInstructions(page, use, text, business) {
  await page.click('button.chip.instr');
  await page.waitForSelector('dialog.modal[open] .instructions-use', { timeout: 10_000 });
  // (after its opening animation, which moves it: a click during it can land beside what it was for)
  await wait(300);
  await page.click(`dialog.modal[open] input[name="instructions-use"][value="${use}"]`);
  await page.waitForFunction((use) => document.querySelector(`dialog.modal[open] input[name="instructions-use"][value="${use}"]`)?.checked, { timeout: 3000 }, use);
  if (text !== undefined) {
    await page.evaluate(() => {
      const t = document.querySelector('dialog.modal[open] textarea.instructions-text');
      t.value = '';
    });
    await page.type('dialog.modal[open] textarea.instructions-text', text);
  }
  if (business !== undefined) {
    const box = await page.$('dialog.modal[open] .instructions-business input');
    if ((await box.evaluate((b) => b.checked)) !== business) await box.click();
    await page.waitForFunction((on) => document.querySelector('dialog.modal[open] .instructions-business input')?.checked === on, { timeout: 3000 }, business);
  }
  await page.click('dialog.modal[open] button.primary');
  await page.waitForFunction(() => !document.querySelector('dialog.modal[open]'), { timeout: 5000 });
  // (what was chosen is what the chip says at once)
  await page.waitForFunction((use) => document.querySelector('button.chip.instr')?.textContent === `Instructions: ${use === 'these' ? 'project' : use}`, { timeout: 1000 }, use);
}
/** The Agent's settings: the person's instructions and the thinking a step may take, typed in and saved. */
async function setSettings(page, { instructions, tokens, unlimited }) {
  await page.click('button.settings-button');
  await page.waitForSelector('dialog.settings[open] .agent-instructions textarea', { timeout: 10_000 });
  if (instructions !== undefined) {
    await page.evaluate(() => {
      document.querySelector('dialog.settings[open] .agent-instructions textarea').value = '';
    });
    await page.type('dialog.settings[open] .agent-instructions textarea', instructions);
  }
  // (each in the middle of the dialog first: scrolled only into view, it sat under the dialog's sticky buttons, which took the click)
  const middle = (selector) => page.$eval(selector, (el) => el.scrollIntoView({ block: 'center' }));
  if (unlimited !== undefined) {
    await middle('dialog.settings[open] .agent-think-unlimited input');
    const box = await page.$('dialog.settings[open] .agent-think-unlimited input');
    if ((await box.evaluate((b) => b.checked)) !== unlimited) await box.click();
  }
  if (tokens !== undefined) {
    // (the field's number replaced as a person does it: all of it chosen, the new one typed, the field left)
    await middle('dialog.settings[open] input.think-tokens');
    await page.click('dialog.settings[open] input.think-tokens');
    await page.keyboard.down('Control');
    await page.keyboard.press('KeyA');
    await page.keyboard.up('Control');
    await page.keyboard.type(String(tokens));
    await page.keyboard.press('Tab');
  }
  await page.click('dialog.settings[open] .dialog-buttons button.primary');
  await page.waitForFunction(() => !document.querySelector('dialog.settings[open]'), { timeout: 5000 });
}
/** The chip's thinking turned round to `to`, a click at a time: what the chat said of the last click. */
async function thinkingTo(page, to) {
  for (let i = 0; i < 3; i++) {
    if ((await page.$eval('button.chip.think', (b) => b.dataset.think)) === to && i > 0) break;
    await page.click('button.chip.think');
    await wait(150);
  }
  await page.waitForFunction((to) => document.querySelector('button.chip.think')?.dataset.think === to, { timeout: 5000 }, to);
  return page.evaluate(() => [...document.querySelectorAll('div.msg.system pre')].at(-1)?.textContent ?? '');
}
/** The window brought back to the front: the page reads again what the Agent runs on (at most every few seconds). */
let lastReturn = 0;
async function backToFront(page) {
  await wait(Math.max(0, lastReturn + 5300 - Date.now()));
  lastReturn = Date.now();
  await page.evaluate(() => window.dispatchEvent(new Event('focus')));
}
const NOTE = 'The phone is answered as Aokie, the receptionist for Green Lawns.';
const MINE = 'Call me Sam. Keep answers short.';
const PROJECT_NOTE = 'OAIY itself: the OAIY tools';
const PLAIN_NOTE = "The business's tools are off in this project";

try {
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  page.on('dialog', (d) => d.accept());
  await page.setViewport({ width: 1280, height: 800 });
  // The model as a provider of OAIY's engine's kind (every tool in each request: none found later), the phone answering
  // calls, a kept project, and the Front desk open.
  await page.goto(`${base}/tests/e2e/harness.html`);
  await page.evaluate(async (modelUrl) => {
    const P = await import('/src/vfs/projects.ts');
    const ST = await import('/src/settings.ts');
    await P.OpenProject.openFrontDesk();
    const kept = await P.createProject('Kept work');
    const project = await P.OpenProject.open(kept);
    project.vfs.writeFile('/notes.md', '# Notes\n', { parents: true });
    await project.close();
    await ST.saveProviders([{ id: 'engine', type: 'custom', serverKind: 'oaiy', name: 'OAIY', baseUrl: modelUrl, modelId: 'test-model', apiKey: '', contextTokens: 65_536 }], 'engine');
    await ST.saveMessages({ ...ST.DEFAULT_MESSAGE_SETTINGS, calls: true, answer: true, country: 'AU' });
    await ST.saveLastProject('front-desk');
  }, modelUrl);
  await page.evaluateOnNewDocument((origin, token) => {
    window.__OAIY_DESKTOP__ = { origin, token };
    window.__OAIY_THEME__ = 'light';
  }, desktopUrl, TOKEN);
  await page.goto(base);
  await page.waitForSelector('.tree-row', { timeout: 60_000 });
  await page.waitForFunction(() => document.querySelector('.project-select')?.value === 'front-desk', { timeout: 20_000 });
  await wait(1500);

  await check("the Front desk's runner is told who answers the phone, has the business's tools and OAIY's read tools, and thinks as the engine is set, at most 2,048 tokens", async () => {
    const first = await ask(page, 'List your tools.');
    expect(first.system.includes(NOTE), first.system.slice(-1500));
    expect(first.tools.includes('start_outreach') && first.tools.includes('calendar_free_times'), first.tools.join(', '));
    expect(same(controlIn(first.tools), counted(READ_TOOLS)), `${controlIn(first.tools).join(', ')}\n     vs ${counted(READ_TOOLS).join(', ')}`);
    expect(first.tools.includes('contacts_list') && !first.tools.includes('contact_set') && !first.tools.includes('calendar_settings_set'), first.tools.join(', '));
    expect(first.effort === undefined && same(first.thinking, { budget_tokens: 2048 }), JSON.stringify([first.effort, first.thinking]));
    expect(!first.system.includes('<user_instructions>'), 'instructions of the person were given before any were written');
  });

  await check("the person's instructions and a thinking budget, set in the Agent's settings, reach the runner's next step", async () => {
    await setSettings(page, { instructions: MINE, tokens: 512 });
    await keptAs(page, (s) => s.instructions === MINE && s.thinkTokens === 512);
    const step = await ask(page, 'Hello.');
    expect(step.system.endsWith(`<user_instructions>\n${MINE}\n</user_instructions>`), step.system.slice(-600));
    expect(same(step.thinking, { budget_tokens: 512 }), JSON.stringify(step.thinking));
  });
  await check('a number typed past what a step may think is brought into range as the field is left', async () => {
    await setSettings(page, { tokens: 99_999 });
    await keptAs(page, (s) => s.thinkTokens === 32_768);
    await setSettings(page, { tokens: 100 });
    await keptAs(page, (s) => s.thinkTokens === 256);
    await setSettings(page, { tokens: 512 });
    await keptAs(page, (s) => s.thinkTokens === 512);
  });
  await shoot(page, 'settings', async () => {
    if (!(await page.$('dialog.settings[open]'))) await page.click('button.settings-button');
    await page.waitForSelector('dialog.settings[open] .agent-instructions textarea');
    await page.evaluate(() => document.querySelector('dialog.settings[open] .agent-settings')?.scrollIntoView({ block: 'start' }));
    await wait(200);
  });
  await page.keyboard.press('Escape');
  await page.waitForFunction(() => !document.querySelector('dialog.settings[open]'), { timeout: 5000 }).catch(() => {});

  await check("a kept project is not told who answers the phone (nor the names), has the business's tools and all of OAIY's, takes the person's instructions, and thinks briefly when Thinking is on", async () => {
    await openProject(page, 'Kept work');
    await thinkingTo(page, 'on');
    const chip = await page.$eval('button.chip.instr', (b) => ({ text: b.textContent, state: b.dataset.instr }));
    expect(chip.text === 'Instructions: mine' && chip.state === 'mine', JSON.stringify(chip));
    const first = await ask(page, 'List your tools.');
    expect(!first.system.includes('The phone is answered as') && !/Aokie|Green Lawns/.test(first.system), first.system.slice(-1500));
    expect(first.tools.includes('start_outreach') && first.tools.includes('calendar_free_times'), first.tools.join(', '));
    expect(same(controlIn(first.tools), counted(CONTROL)), `${controlIn(first.tools).join(', ')}`);
    expect(first.system.includes(PROJECT_NOTE) && !first.system.includes(PLAIN_NOTE), first.system.slice(-2500));
    expect(first.system.endsWith(`<user_instructions>\n${MINE}\n</user_instructions>`), first.system.slice(-600));
    expect(first.effort === 25 && same(first.thinking, { type: 'enabled', budget_tokens: 512 }), JSON.stringify([first.effort, first.thinking]));
  });

  await check('the chat keeps how far the thought has gone, as its engine counts it, while the thought goes on', async () => {
    const from = asked.length;
    await say(page, 'Think it over.');
    await until('the thought between its words', () => thoughtPaused, 15_000);
    await wait(400);
    const status = await page.$eval('.chat-status', (s) => s.textContent);
    expect(status === 'thinking… 4 / 512 tokens', `the status after the thought's next words: "${status}"`);
    const label = await page.evaluate(() => [...document.querySelectorAll('.thinking-label')].at(-1)?.textContent ?? '');
    expect(label === 'Thinking… 4 / 512 tokens', label);
    await idle(page);
    expect(asked.slice(from).filter((a) => a.kind === 'own').length === 1, 'one request');
  });

  await check("its Instructions dialog gives it its own instead and turns its business tools off: OAIY's own tools stay, none for the business's data; both are kept with the project", async () => {
    await setInstructions(page, 'these', 'Use metric units.', false);
    const chip = await page.$eval('button.chip.instr', (b) => ({ text: b.textContent, state: b.dataset.instr, title: b.title }));
    expect(chip.text === 'Instructions: project' && chip.state === 'project' && /Business tools are off here/.test(chip.title), JSON.stringify(chip));
    // (said once the choice is written with the project)
    const said = await noteSaying(page, /Business tools are off here/);
    expect(/Business tools are off here: no phone, calendar, contacts or outreach, nor OAIY's tools for them\./.test(said), said);
    const first = await ask(page, 'List your tools.');
    expect(first.system.endsWith('<user_instructions>\nUse metric units.\n</user_instructions>') && !first.system.includes(MINE), first.system.slice(-600));
    expect(!first.tools.includes('start_outreach') && !first.tools.includes('calendar_free_times') && !first.tools.includes('calendar_book'), first.tools.join(', '));
    const control = controlIn(first.tools);
    expect(same(control, counted(CONTROL.filter((n) => !BUSINESS.includes(n)))), control.join(', '));
    expect(BUSINESS.every((n) => !first.tools.includes(n) && !first.tools.includes(`oaiy_${n}`)), first.tools.join(', '));
    expect(['status', 'models_list', 'plugins_list', 'services_list'].every((n) => first.tools.includes(n)), first.tools.join(', '));
    expect(first.system.includes(PLAIN_NOTE) && !first.system.includes("plugins and their settings, flows, and the calendar's settings"), first.system.slice(-2500));
    const kept = await page.evaluate(async () => {
      const projects = await (await navigator.storage.getDirectory()).getDirectoryHandle('projects');
      for await (const [, handle] of projects) {
        if (handle.kind !== 'directory') continue;
        const meta = JSON.parse(await (await (await handle.getFileHandle('project.json')).getFile()).text());
        if (meta.name === 'Kept work') return meta;
      }
      return null;
    });
    expect(same(kept?.instructions, { use: 'these', text: 'Use metric units.' }) && kept?.businessTools === false, JSON.stringify(kept));
  });
  // (opened for the first picture, and left open for the second: a click on the chip behind it closes it)
  await shoot(page, 'dialog', async () => {
    if (!(await page.$('dialog.modal[open]'))) await page.click('button.chip.instr');
    await page.waitForSelector('dialog.modal[open] .instructions-use');
    await wait(300);
  });
  await page.keyboard.press('Escape');
  await page.waitForFunction(() => !document.querySelector('dialog.modal[open]'), { timeout: 5000 }).catch(() => {});
  await shoot(page, 'chips', null, () => page.evaluate(() => {
    const r = document.querySelector('header.topbar').getBoundingClientRect();
    return { x: r.x, y: r.y, width: r.width, height: r.height };
  }));

  await check('"None" gives it no instructions of the person\'s; with no limit a step thinks at most half of what it may write; off, not at all', async () => {
    await setInstructions(page, 'none', undefined, true);
    await setSettings(page, { unlimited: true });
    const step = await ask(page, 'Hello again.');
    expect(!step.system.includes('<user_instructions>'), step.system.slice(-600));
    expect(same(step.thinking, { type: 'enabled', budget_tokens: Math.floor(step.maxTokens / 2) }) && step.maxTokens === 16_384, JSON.stringify([step.thinking, step.maxTokens]));
    // (its business tools on again: all of OAIY's tools are back)
    expect(same(controlIn(step.tools), counted(CONTROL)), controlIn(step.tools).join(', '));
    const said = await thinkingTo(page, 'off');
    expect(said === 'Thinking off: the model answers each step at once.', said);
    const off = await ask(page, 'And once more.');
    expect(off.effort === 'none' && off.thinking === undefined && off.template === undefined, JSON.stringify([off.effort, off.thinking, off.template]));
  });

  await check("one of OAIY Desktop's AI providers is asked of the thinking only as far as the desktop says what server it is, and the chip says so", async () => {
    // The desktop says the Agent runs on one of its AI providers, and lists none: what kind of server it is, unknown.
    aiProviders = [];
    preference = { model: { source: 'provider', provider: 'lan-box', model: 'box-model' }, providerName: 'LAN box' };
    await backToFront(page);
    // (the move said, once the desktop's providers are read: the chip's next clicks say their own after it)
    const moved = await noteSaying(page, /^The Agent now runs on LAN box \(box-model\)/);
    expect(/^The Agent now runs on LAN box \(box-model\)/.test(moved), moved);
    await page.waitForFunction(() => document.querySelector('button.chip.model')?.title.includes('LAN box (box-model)') && document.querySelector('button.chip.think')?.dataset.reach === 'unknown', { timeout: 10_000 });
    const title = await page.$eval('button.chip.think', (b) => b.title);
    expect(title.includes('(OAIY Desktop does not say what kind of server LAN box is, so it is not asked: it thinks as it is set.)'), title);
    const said = await thinkingTo(page, 'off');
    expect(said === 'Thinking off: OAIY Desktop does not say what kind of server LAN box is, so it is not asked: it thinks as it is set.', said);
    const unknown = await ask(page, 'Hello there.');
    expect(unknown.route === '/api/ai/providers/lan-box/v1/chat/completions' && unknown.user.endsWith('Hello there.'), `${unknown.route}: ${unknown.user.slice(-80)}`);
    expect(unknown.effort === undefined && unknown.thinking === undefined && unknown.template === undefined, JSON.stringify([unknown.effort, unknown.thinking, unknown.template]));
    // The desktop lists it as a server the person marked local (LM Studio, Ollama, llama.cpp): off reaches it, through its chat template.
    aiProviders = [{ id: 'lan-box', name: 'LAN box', category: 'local', protocol: 'openai', baseUrl: 'http://192.168.1.20:1234/v1', model: null, capabilities: [], enabled: true, allowLocal: true, hasKey: false }];
    await backToFront(page);
    await page.waitForFunction(() => document.querySelector('button.chip.think')?.dataset.reach === 'template', { timeout: 10_000 });
    const local = await ask(page, 'Hello once more.');
    expect(local.route === '/api/ai/providers/lan-box/v1/chat/completions', local.route);
    expect(same(local.template, { enable_thinking: false }) && local.effort === undefined && local.thinking === undefined, JSON.stringify([local.effort, local.thinking, local.template]));
    // An online API (an address the person did not mark local): nothing.
    aiProviders = [{ ...aiProviders[0], baseUrl: 'https://api.example.com/v1', allowLocal: false }];
    await backToFront(page);
    await page.waitForFunction(() => document.querySelector('button.chip.think')?.dataset.reach === 'none', { timeout: 10_000 });
    const api = await ask(page, 'Hello, online.');
    expect(api.effort === undefined && api.thinking === undefined && api.template === undefined, JSON.stringify([api.effort, api.thinking, api.template]));
  });

  await check("a paired computer's OAIY engine (OAIY Network) is asked as this computer's", async () => {
    preference = { model: { source: 'engine', model: 'engine:llm:node.abc123/worker-model' }, providerName: 'Rig' };
    await backToFront(page);
    await page.waitForFunction(() => document.querySelector('button.chip.model')?.title.includes('over OAIY Network') && document.querySelector('button.chip.think')?.dataset.reach === 'oaiy', { timeout: 10_000 });
    const off = await ask(page, 'Hello, far away.');
    expect(off.route === '/api/network/nodes/abc123/v1/chat/completions' && off.user.endsWith('Hello, far away.'), `${off.route}: ${off.user.slice(-80)}`);
    expect(off.effort === 'none' && off.thinking === undefined && off.template === undefined, JSON.stringify([off.effort, off.thinking, off.template]));
    await thinkingTo(page, 'on');
    const on = await ask(page, 'And think a little.');
    expect(on.effort === 25 && same(on.thinking, { type: 'enabled', budget_tokens: Math.floor(on.maxTokens / 2) }), JSON.stringify([on.effort, on.thinking, on.maxTokens]));
    preference = { model: { source: 'engine' } };
    await backToFront(page);
    await page.waitForFunction(() => !document.querySelector('button.chip.model')?.title.includes('over OAIY Network'), { timeout: 10_000 });
  });

  await check("an incognito project knows nothing of the business: no note, no phone, calendar or outreach tools, none of OAIY's own tools, and its dialog asks nothing of them", async () => {
    await page.click('button.menu-toggle');
    await page.click('button.incognito-toggle');
    await page.waitForFunction(() => document.body.classList.contains('incognito'), { timeout: 10_000 });
    await wait(300);
    const first = await ask(page, 'List your tools.');
    expect(!first.system.includes('The phone is answered as') && !/Aokie|Green Lawns/.test(first.system), first.system.slice(-1500));
    expect(!first.tools.includes('start_outreach') && !first.tools.includes('calendar_free_times') && !first.tools.includes('calendar_list'), first.tools.join(', '));
    expect(controlIn(first.tools).length === 0 && !first.tools.some((n) => n.startsWith('oaiy_')), controlIn(first.tools).join(', '));
    expect(!first.system.includes(PROJECT_NOTE) && !first.system.includes(PLAIN_NOTE), first.system.slice(-2500));
    // (the person's own instructions are theirs, private or not)
    expect(first.system.endsWith(`<user_instructions>\n${MINE}\n</user_instructions>`), first.system.slice(-600));
    await page.click('button.chip.instr');
    await page.waitForSelector('dialog.modal[open] .instructions-use', { timeout: 10_000 });
    const dialog = await page.evaluate(() => ({ box: !!document.querySelector('dialog.modal[open] .instructions-business'), note: document.querySelector('dialog.modal[open] .instructions-note')?.textContent ?? '' }));
    expect(!dialog.box && dialog.note === "Incognito: no business tools or business details here, and none of OAIY's own tools.", JSON.stringify(dialog));
    await page.keyboard.press('Escape');
  });
  await shoot(page, 'incognito-dialog', async () => {
    if (!(await page.$('dialog.modal[open]'))) await page.click('button.chip.instr');
    await page.waitForSelector('dialog.modal[open] .instructions-use');
    await wait(300);
  });
  await page.keyboard.press('Escape');
  await page.waitForFunction(() => !document.querySelector('dialog.modal[open]'), { timeout: 5000 }).catch(() => {});

  await check("a call is never given the person's instructions, and answers at once", async () => {
    voice({ type: 'call.started', callId: 'live-1', from: '0491570006', name: '', greeting: 'Hi, thanks for calling Green Lawns. How can I help?' });
    await wait(500);
    voice({ type: 'call.caller', callId: 'live-1', text: 'Hi, are you open on Saturday?', startMs: 3_100, endMs: 5_400 });
    await until('the call\'s request', () => asked.some((a) => a.kind === 'call'));
    const call = asked.find((a) => a.kind === 'call');
    expect(call.system.includes('You are Aokie, the receptionist for Green Lawns.'), call.system.slice(0, 800));
    expect(!call.system.includes('<user_instructions>') && !call.system.includes(MINE) && !call.messages.includes(MINE), call.system.slice(-800));
    expect(call.effort === 'none' && call.thinking === undefined, JSON.stringify([call.effort, call.thinking]));
    // (and none of OAIY's control tools: the desktop offers a call none)
    expect(controlIn(call.tools).length === 0, controlIn(call.tools).join(', '));
    voice({ type: 'call.ended', callId: 'live-1' });
  });

  await check("a text is never given the person's instructions nor the chat's thinking, and thinks at most as long as the Agent's settings say", async () => {
    // (the chip says on, and the settings no limit, from the checks before: a limit of the person's again)
    await setSettings(page, { unlimited: false, tokens: 768 });
    await keptAs(page, (s) => s.thinkTokens === 768);
    bridgeEvents.push({ name: 'aokie.sms.received', source: 'aokie', correlationId: 't1', idempotencyKey: 't1', occurredAt: new Date().toISOString(), data: { from: '+61415000123', name: 'Dee', body: 'Do you mow on Sundays?', handle: '0400000000000077' } });
    await until("the text's request", () => asked.some((a) => a.kind === 'sms'));
    const text = asked.find((a) => a.kind === 'sms');
    expect(text.user.includes('Do you mow on Sundays?'), text.user.slice(-200));
    expect(!text.system.includes('<user_instructions>') && !text.system.includes(MINE) && !text.messages.includes(MINE), text.system.slice(-800));
    // (as the engine is set: not asked to think as the chip says, only stopped where the person's limit is)
    expect(text.effort === undefined && same(text.thinking, { budget_tokens: 768 }), JSON.stringify([text.effort, text.thinking]));
    expect(controlIn(text.tools).length === 0, controlIn(text.tools).join(', '));
  });

  // A phone's width: the chip gives way to the project's name, and its dialog is in the project menu.
  await page.setViewport({ width: 390, height: 844 });
  await wait(500);
  await check('on a phone the Instructions chip gives way to the project\'s name, and the project menu opens its dialog', async () => {
    const bar = await page.evaluate(() => {
      const b = document.querySelector('header.topbar');
      return { overflow: b.scrollWidth - b.clientWidth, chip: document.querySelector('button.chip.instr').getClientRects().length, project: Math.round(document.querySelector('.project-picker .combo-button').getBoundingClientRect().width) };
    });
    expect(bar.overflow <= 0 && bar.chip === 0 && bar.project >= 80, JSON.stringify(bar));
    await page.click('button.menu-toggle');
    await page.waitForSelector('#project-menu button.instructions-item', { visible: true, timeout: 5000 });
    await page.click('#project-menu button.instructions-item');
    await page.waitForSelector('dialog.modal[open] .instructions-use', { timeout: 5000 });
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !document.querySelector('dialog.modal[open]'), { timeout: 5000 });
  });
  // (the bar, and the project menu open under it with Instructions… in it)
  await shoot(page, 'phone-menu', async () => {
    if (!(await page.evaluate(() => document.querySelector('header.topbar')?.classList.contains('menu-open')))) await page.click('button.menu-toggle');
    await page.waitForSelector('#project-menu button.instructions-item', { visible: true });
    await wait(200);
  }, () => ({ x: 0, y: 0, width: 390, height: 600 }));
  if (await page.evaluate(() => document.querySelector('header.topbar')?.classList.contains('menu-open'))) await page.keyboard.press('Escape');

  await check('the page raised no errors', async () => {
    expect(!pageErrors.length, pageErrors.join('; '));
  });
  await context.close();
} finally {
  await browser.close();
  await server.close();
  model.close();
  desktopServer.close();
}
if (shots.length) console.log(`screenshots:\n  ${shots.join('\n  ')}`);
console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
