// The Agent sets OAIY up through OAIY Desktop's control API, on ChatGPT: a fake
// desktop (its MCP server keeping the desktop's session rule, with the real
// tool list; the Agent's model preference; OAIY's ChatGPT connector as a
// scripted model on its generic and live-call routes) and the app in OAIY's
// window. The wizard's intent opens "Set up OAIY"; the model checks OAIY's
// status, shows the person a plugin's pairing step, and checks it after; the
// model chip says ChatGPT; a phone call runs on the live-call route with no
// control tools; signed out, the Agent says where to sign in. Nothing here
// reaches a real desktop, ChatGPT or model.
//   node tests/e2e/agentctl.mjs [screenshot-dir]
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
const TOKEN = 'agentctl-token';
const { tools: CONTROL_TOOLS, runner: READ_TOOLS } = JSON.parse(readFileSync(new URL('../fixtures/control-tools.json', import.meta.url), 'utf8'));
const CONTROL_NAMES = new Set(CONTROL_TOOLS.map((t) => t.name));

// --- the engine's model (the app's own provider): it must not be asked while the Agent runs on ChatGPT ---
const engineAsked = [];
const engine = createHttpServer(async (req, res) => {
  for (const [k, v] of Object.entries(cors)) res.setHeader(k, v);
  if (req.method === 'OPTIONS') return res.end();
  if (req.url.endsWith('/models')) {
    res.setHeader('content-type', 'application/json');
    return res.end(JSON.stringify({ data: [{ id: 'Qwen3.8-Flash-Next' }] }));
  }
  engineAsked.push(req.url);
  res.statusCode = 500;
  res.end(JSON.stringify({ error: { message: 'the engine was asked' } }));
});
await new Promise((r) => engine.listen(0, '127.0.0.1', r));
const engineUrl = `http://127.0.0.1:${engine.address().port}`;

// --- the fake desktop -------------------------------------------------------------
const MODULES = {
  revision: 1,
  modules: [
    { id: 'phone', name: 'Phone', enabled: true, provider: { pluginId: 'aokie', name: 'AI Receptionist', state: 'running', declared: true } },
    { id: 'calendar', name: 'Calendar', enabled: true, provider: { pluginId: 'aokie', name: 'AI Receptionist', state: 'running', declared: true } },
  ],
  warnings: [],
};
/** Every MCP request: its method, its session header, and the tool it called. */
const mcp = [];
/** Every request to OAIY's ChatGPT connector: its route, key, model, tools and messages. */
const chatgpt = [];
const preference = { model: { source: 'chatgpt' } };
let signedIn = true;
let refusedSignedOut = 0;
const voiceClients = new Set();
const spoken = [];

/** The desktop's answers to its tools, for a desk whose phone plugin needs pairing. */
function toolAnswer(name, args) {
  const text = (value) => ({ content: [{ type: 'text', text: typeof value === 'string' ? value : JSON.stringify(value) }], structuredContent: typeof value === 'string' ? undefined : value });
  switch (name) {
    case 'status':
      return text({ agentMayChange: true, agentModel: { source: 'chatgpt' }, chatgpt: { available: true, signedIn: true }, engines: { running: false }, plugins: [{ id: 'aokie', name: 'AI Receptionist', state: 'running', needsSetup: true }], modules: MODULES.modules.map((m) => ({ id: m.id, enabled: m.enabled })), setup: { firstRunFinished: false, pluginsNeedingSetup: ['aokie'] } });
    case 'setup_status':
      return text({ firstRun: { finished: false, chosenPlugins: ['aokie'] }, plugins: [{ id: 'aokie', needsSetup: true }] });
    case 'plugin_setup_open':
      return text(`The dashboard shows ${args.pluginId}'s setup at the ${args.stepId ?? 'first'} step.`);
    case 'plugin_setup_status':
      return text({ pluginId: args.pluginId, needsSetup: false, outstanding: [], steps: [{ id: 'permissions', complete: true }, { id: 'pair', kind: 'screen', complete: true }] });
    default:
      return text(`${name} done`);
  }
}

const desktopServer = createHttpServer(async (req, res) => {
  for (const [k, v] of Object.entries(cors)) res.setHeader(k, v);
  if (req.method === 'OPTIONS') return res.end();
  const path = new URL(req.url, 'http://x').pathname;
  const body = await readBody(req);
  const json = (value, status = 200) => {
    res.statusCode = status;
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify(value));
  };
  const stream = (events) => {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
    for (const e of events) res.write(`data: ${JSON.stringify(e)}\n\n`);
  };
  const authorized = req.headers.authorization === `Bearer ${TOKEN}`;

  // The control API: the desktop's session rule (project and setup all, runner reads, call/sms/task none).
  if (path === '/api/mcp') {
    if (!authorized) return json({ error: 'origin not allowed' }, 403);
    const message = JSON.parse(body || '{}');
    const session = req.headers['x-oaiy-session'] ?? 'project';
    mcp.push({ method: message.method, session, tool: message.params?.name ?? null });
    if (message.id === undefined) {
      res.statusCode = 202;
      return res.end();
    }
    const ok = (result) => json({ jsonrpc: '2.0', id: message.id, result });
    if (message.method === 'initialize') return ok({ protocolVersion: '2025-06-18', capabilities: { tools: {} }, serverInfo: { name: 'oaiy', version: 'e2e' } });
    const offered = session === 'project' || session === 'setup' ? CONTROL_TOOLS : session === 'runner' ? CONTROL_TOOLS.filter((t) => READ_TOOLS.includes(t.name)) : [];
    if (message.method === 'tools/list') return ok({ tools: offered });
    if (message.method === 'tools/call') {
      const name = message.params?.name;
      if (!offered.some((t) => t.name === name)) return ok({ content: [{ type: 'text', text: `${name} is not offered to ${session} sessions.` }], isError: true });
      return ok(toolAnswer(name, message.params?.arguments ?? {}));
    }
    return json({ jsonrpc: '2.0', id: message.id, error: { code: -32601, message: `method not found: ${message.method}` } });
  }
  if (path === '/api/agent/preferences') return authorized ? json(preference) : json({ error: 'origin not allowed' }, 403);

  // OAIY's ChatGPT connector: its model list, the generic route and the live-call alias.
  const route = /^\/api\/ai\/providers\/([^/]+)\/v1\/(models|chat\/completions)$/.exec(path);
  if (route) {
    if (!authorized) return json({ error: 'origin not allowed' }, 403);
    if (!signedIn) refusedSignedOut++;
    if (!signedIn) return json({ error: { code: 'codex_not_authenticated', message: 'not signed in to ChatGPT — start a login from OAIY Desktop → Providers' } }, 428);
    if (route[2] === 'models') return json({ object: 'list', data: [{ id: 'gpt-5.6-luna', object: 'model', displayName: 'Luna' }, { id: 'gpt-5.5', object: 'model', displayName: 'GPT-5.5', isDefault: true }] });
    const request = JSON.parse(body);
    chatgpt.push({ route: route[1], model: request.model, tools: (request.tools ?? []).map((t) => t.function.name), messages: request.messages, stream: request.stream, reasoning: request.reasoning_effort, template: request.chat_template_kwargs });
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    return res.end(sse(answer(route[1], request)));
  }

  if (path === '/api/health') return json({ product: 'oaiy-desktop', protocol: 'oaiy-bridge/1', version: 'e2e' });
  if (path === '/api/modules') return json(MODULES);
  if (path === '/api/modules/events') return stream([MODULES]);
  if (path === '/api/agent/events') return stream([]);
  if (path === '/api/voice/events') {
    stream([{ type: 'hello', calls: [] }]);
    voiceClients.add(res);
    req.on('close', () => voiceClients.delete(res));
    return;
  }
  if (path === '/api/bridge/events') return json({ events: [], next: 0 });
  if (path.startsWith('/api/bridge/leases/')) return json({ granted: true, holder: (JSON.parse(body || '{}').holder ?? '') });
  if (path.startsWith('/api/bridge/connectors/')) {
    const { command } = JSON.parse(body || '{}');
    return json({ ok: true, result: { ok: true, data: command === 'phone.status' ? { connected: true } : command === 'settings.get' ? { value: '' } : {} } });
  }
  if (path === '/api/bridge/flows') return json({ flows: [] });
  if (path === '/api/plugins') return json({ plugins: [{ id: 'aokie', state: 'running' }] });
  if (path.startsWith('/api/calendar')) return json({ available: true, settings: {}, appointments: [], now: new Date().toISOString() });
  if (path.startsWith('/api/voice/calls/')) {
    if (path.endsWith('/say')) spoken.push(JSON.parse(body || '{}').text);
    return json({ result: { ok: true, output: {} } });
  }
  if (path === '/api/voice/callers') return json({});
  return json({ error: { message: 'not here' } }, 404);
});
await new Promise((r) => desktopServer.listen(0, '127.0.0.1', r));
const desktopUrl = `http://127.0.0.1:${desktopServer.address().port}`;
const voice = (event) => {
  for (const c of voiceClients) c.write(`data: ${JSON.stringify(event)}\n\n`);
};

// --- the scripted ChatGPT ---------------------------------------------------------
let n = 0;
function sse(step) {
  n++;
  const events = [];
  for (const piece of step.text?.match(/.{1,12}/gs) ?? []) events.push({ choices: [{ index: 0, delta: { content: piece } }] });
  (step.calls ?? []).forEach((call, i) => {
    events.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: `call_${n}_${i}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.input) } }] } }] });
  });
  events.push({ choices: [{ index: 0, delta: {}, finish_reason: step.calls ? 'tool_calls' : 'stop' }] });
  return `${events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('')}data: [DONE]\n\n`;
}
const textOf = (m) => (typeof m.content === 'string' ? m.content : (m.content ?? []).map((p) => p.text ?? '').join(''));
function answer(routeId, body) {
  const system = textOf(body.messages.find((m) => m.role === 'system') ?? { content: '' });
  if (routeId !== 'openai-codex-agent' || /live phone call/.test(system)) return { text: 'Thanks for calling. How can I help?' };
  // Since the person's last message: how many steps the model has taken.
  const lastUser = body.messages.map((m) => m.role === 'user' && !textOf(m).startsWith('[')).lastIndexOf(true);
  const said = textOf(body.messages[lastUser] ?? { content: '' });
  const steps = body.messages.slice(lastUser + 1).filter((m) => m.role === 'assistant').length;
  if (/business phone/i.test(said)) {
    if (steps === 0) return { text: "Let me see how OAIY is set up.", calls: [{ name: 'status', input: {} }, { name: 'setup_status', input: {} }] };
    if (steps === 1) return { text: 'The AI Receptionist is installed and running, and it needs your phone paired with it.', calls: [{ name: 'plugin_setup_open', input: { pluginId: 'aokie', stepId: 'pair' } }] };
    return { text: "I've opened the pairing step in OAIY's dashboard. On your phone, open Bluetooth, choose **OAIY**, and accept the code that shows on both screens. Tell me when it's done." };
  }
  if (/paired/i.test(said)) {
    if (steps === 0) return { calls: [{ name: 'plugin_setup_status', input: { pluginId: 'aokie' } }] };
    return { text: 'Your phone is paired and the receptionist is ready to answer calls and texts. Shall I set your opening hours next?' };
  }
  return { text: 'Done.' };
}

// --- the app ----------------------------------------------------------------------
const server = await createServer({ server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const base = `http://127.0.0.1:${server.httpServer.address().port}`;
const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-first-run'] });
const failures = [];
const checks = [];
const check = async (name, fn) => {
  try {
    await fn();
    checks.push(`ok   ${name}`);
  } catch (error) {
    checks.push(`FAIL ${name}: ${error.message}`);
    failures.push(name);
  }
};
const expect = (c, m) => {
  if (!c) throw new Error(m);
};
const shots = [];
/** The header, where the model chip is. */
const header = (page) => async () => {
  const box = await page.$eval('header.topbar', (el) => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; });
  return { ...box, height: Math.ceil(box.height) + 1 };
};
async function shoot(page, scene, { clip, before } = {}) {
  if (!shotsDir) return;
  for (const theme of ['light', 'dark']) {
    await page.evaluate((t) => window.__oaiySetTheme?.(t), theme);
    await wait(350);
    if (before) await before();
    const file = join(shotsDir, `agentctl-${scene}-${theme}-1058.png`);
    await page.screenshot({ path: file, ...(clip ? { clip: await clip() } : {}) });
    shots.push(file);
  }
  await page.evaluate(() => window.__oaiySetTheme?.('light'));
}

try {
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  page.on('dialog', (d) => d.accept());
  await page.setViewport({ width: 1058, height: 688 });
  // The engine's provider as the app keeps it, and answering calls on (from the app's own storage code).
  await page.goto(`${base}/tests/e2e/harness.html`);
  await page.evaluate(async (engineUrl) => {
    const ST = await import('/src/settings.ts');
    await ST.saveProviders([{ id: 'engine', type: 'local', serverKind: 'other', name: 'OAIY', baseUrl: engineUrl, modelId: 'Qwen3.8-Flash-Next', apiKey: '' }], 'engine');
    await ST.saveMessages({ ...ST.DEFAULT_MESSAGE_SETTINGS, calls: true });
  }, engineUrl);
  // OAIY's window: the desktop is given, with its token.
  await page.evaluateOnNewDocument((origin, token) => {
    window.__OAIY_DESKTOP__ = { origin, token };
    window.__OAIY_THEME__ = 'light';
  }, desktopUrl, TOKEN);
  await page.goto(base);
  await page.waitForSelector('.tree-row', { timeout: 60_000 });

  await check('the model chip says ChatGPT, with the model Codex runs by default, when the desktop says the Agent runs on ChatGPT', async () => {
    await page.waitForFunction(() => document.querySelector('.chip.model')?.textContent === 'ChatGPT · gpt-5.5', { timeout: 15_000 }).catch(() => {});
    const chip = await page.$eval('.chip.model', (el) => ({ text: el.textContent, source: el.dataset.source, title: el.title }));
    expect(chip.text === 'ChatGPT · gpt-5.5' && chip.source === 'chatgpt' && /Settings → Agent/.test(chip.title), JSON.stringify(chip));
  });
  await check('the control API is started once, and listed for the person\'s conversations with their session headers', async () => {
    await page.waitForFunction(() => true);
    for (let i = 0; i < 50 && !mcp.some((m) => m.method === 'tools/list' && m.session === 'runner'); i++) await wait(100);
    const inits = mcp.filter((m) => m.method === 'initialize').length;
    const lists = new Set(mcp.filter((m) => m.method === 'tools/list').map((m) => m.session));
    expect(inits === 1 && mcp.some((m) => m.method === 'notifications/initialized'), `initialize ${inits}×`);
    expect(lists.has('project') && lists.has('setup') && lists.has('runner') && !lists.has('call') && !lists.has('sms') && !lists.has('task'), [...lists].join(','));
  });
  await shoot(page, 'chip', { clip: header(page) });

  await check('the wizard\'s setupWithAgent opens "Set up OAIY", with its own empty state and suggestions', async () => {
    const taken = await page.evaluate(() => window.__oaiyIntent?.('setupWithAgent'));
    expect(taken === true, `the intent was not taken: ${taken}`);
    await page.waitForFunction(() => document.querySelector('.chat-empty h2')?.textContent === 'Set up OAIY', { timeout: 15_000 });
    // The project list is drawn last, once the project is open.
    await page.waitForFunction(() => document.querySelector('.project-select')?.value === 'oaiy-setup', { timeout: 5000 }).catch(() => {});
    const state = await page.evaluate(() => ({
      project: document.querySelector('.project-picker .combo-button')?.textContent ?? '',
      select: document.querySelector('.project-select')?.value,
      ideas: [...document.querySelectorAll('.chat-empty .suggestion')].map((b) => b.textContent),
      placeholder: document.querySelector('.chat-input')?.placeholder,
      readme: [...document.querySelectorAll('.tree-row')].map((r) => r.getAttribute('title')).includes('/README.md'),
      rename: [...document.querySelectorAll('.actions button')].find((b) => b.textContent === 'Rename')?.disabled,
      menu: [...document.querySelectorAll('.actions button')][0]?.textContent,
    }));
    expect(state.project.includes('Set up OAIY') && state.select === 'oaiy-setup', JSON.stringify(state));
    expect(['Set up my business phone', 'What can OAIY do on this computer?', 'Use ChatGPT instead of a local model'].every((i) => state.ideas.includes(i)), JSON.stringify(state.ideas));
    expect(state.placeholder === 'Tell the Agent what OAIY should do…' && state.readme && state.rename === true && state.menu === 'Set up OAIY', JSON.stringify(state));
  });
  await shoot(page, 'setup-empty');

  await check('the setup conversation runs on ChatGPT: it checks OAIY\'s status, shows the pairing step, and checks it once the person says it is done', async () => {
    await page.evaluate(() => [...document.querySelectorAll('.chat-empty .suggestion')].find((b) => b.textContent === 'Set up my business phone').click());
    expect((await page.$eval('.chat-input', (el) => el.value)) === 'Set up my business phone', 'the suggestion did not fill the box');
    await page.focus('.chat-input');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes("Tell me when it's done"), { timeout: 20_000 });
    await wait(400);
  });
  // The run of tools opened, so its rows show.
  await shoot(page, 'setup-midway', { before: () => page.evaluate(() => document.querySelectorAll('.chat-log details.tool-group').forEach((g) => { g.open = true; })) });
  await check('the tool rows say in plain words what the Agent did', async () => {
    const rows = await page.evaluate(() => [...document.querySelectorAll('.chat-log details.tool')].map((d) => ({ tool: d.dataset.tool, name: d.querySelector('.tool-name')?.textContent, arg: d.querySelector('.tool-arg')?.textContent ?? '', failed: d.classList.contains('failed') })));
    const by = (tool) => rows.find((r) => r.tool === tool);
    expect(by('status')?.name === "Checked OAIY's status" && by('setup_status')?.name === 'Checked the setup', JSON.stringify(rows));
    expect(by('plugin_setup_open')?.name === 'Showed you a setup step' && by('plugin_setup_open').arg.includes('aokie · pair'), JSON.stringify(rows));
    expect(rows.every((r) => !r.failed), JSON.stringify(rows));
  });
  await check('then plugin_setup_status, after the person says it is paired', async () => {
    await page.type('.chat-input', "Done, it's paired.");
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('ready to answer calls and texts'), { timeout: 20_000 });
    const calls = mcp.filter((m) => m.method === 'tools/call');
    expect(JSON.stringify(calls.map((m) => m.tool)) === JSON.stringify(['status', 'setup_status', 'plugin_setup_open', 'plugin_setup_status']), JSON.stringify(calls));
    expect(calls.every((m) => m.session === 'setup'), JSON.stringify(calls));
    const row = await page.evaluate(() => document.querySelector('.chat-log details.tool[data-tool="plugin_setup_status"] .tool-name')?.textContent);
    expect(row === "Checked a plugin's setup", `row: ${row}`);
  });
  await check('every request went to ChatGPT\'s generic route with the desktop token\'s key, the default model and the setup instructions; none to the engine', async () => {
    const setup = chatgpt.filter((c) => c.route === 'openai-codex-agent');
    expect(setup.length === 5, `${setup.length} requests`);
    expect(setup.every((c) => c.model === 'gpt-5.5' && c.stream === true && c.reasoning === undefined && c.template === undefined), JSON.stringify(setup.map((c) => [c.model, c.stream, c.reasoning])));
    const system = setup[0].messages.find((m) => m.role === 'system').content;
    expect(/"Set up OAIY"/.test(system) && /agentMayChange is false/.test(system) && /Settings → Agent/.test(system) && /plugin_setup_open/.test(system) && /setup_finish/.test(system), 'the setup instructions are not in the system prompt');
    const names = setup[0].tools;
    expect(['status', 'setup_status', 'plugin_setup_open', 'plugin_setup_status', 'setup_finish', 'flow_delete', 'flow_write', 'flow_run'].every((t) => names.includes(t)), names.join(' '));
    expect(!names.includes('flows_list') && !names.includes('flow_create') && new Set(names).size === names.length, 'a covered or repeated tool reached the model');
    expect(engineAsked.length === 0, `the engine was asked: ${engineAsked.join(', ')}`);
  });

  await check("a phone call runs on ChatGPT's live-call route, with its call tools and none of OAIY's control tools", async () => {
    voice({ type: 'call.started', callId: 'live-1', from: '+61455666777', name: 'Dave', greeting: 'Hi, thanks for calling. How can I help?' });
    await wait(1200);
    voice({ type: 'call.caller', callId: 'live-1', text: 'Hi, can someone look at my hedge this week?', startMs: 3100, endMs: 6400 });
    for (let i = 0; i < 100 && !chatgpt.some((c) => c.route !== 'openai-codex-agent'); i++) await wait(100);
    const call = chatgpt.find((c) => c.route !== 'openai-codex-agent');
    expect(call, `no call request: ${JSON.stringify(chatgpt.map((c) => c.route))}`);
    expect(call.route === 'openai-codex-agent-none', call.route);
    expect(call.tools.includes('end_call') && call.stream === true, JSON.stringify(call.tools));
    const control = call.tools.filter((t) => CONTROL_NAMES.has(t));
    expect(!control.length, `control tools on a call: ${control.join(', ')}`);
    expect(!mcp.some((m) => m.session === 'call' || m.session === 'sms' || m.session === 'task'), 'the control API was asked for a call, a text or a task');
    for (let i = 0; i < 50 && !spoken.length; i++) await wait(100);
    expect(spoken.join(' ').includes('How can I help'), `said: ${JSON.stringify(spoken)}`);
    voice({ type: 'call.ended', callId: 'live-1' });
    await wait(600);
  });

  await check('signed out of ChatGPT, the Agent says where to sign in, the chip says so, and nothing else is asked', async () => {
    // Back to "Set up OAIY" (the call showed its own conversation).
    await page.evaluate(() => {
      const select = document.querySelector('.project-select');
      if (select.value !== 'oaiy-setup') {
        select.value = 'oaiy-setup';
        select.dispatchEvent(new Event('change'));
      }
    });
    await wait(500);
    const own = await page.evaluate(() => document.querySelector('.chat [aria-haspopup="listbox"]')?.textContent ?? '');
    if (!own.includes('Set up OAIY')) {
      await page.evaluate(() => document.querySelector('.chat [aria-haspopup="listbox"]')?.click());
      await wait(200);
      await page.evaluate(() => [...document.querySelectorAll('.chat [role="option"]')].find((o) => o.textContent.includes('Set up OAIY'))?.click());
      await wait(400);
    }
    signedIn = false;
    const before = chatgpt.length;
    await page.type('.chat-input', 'Set my opening hours to nine to five.');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Sign in to ChatGPT in OAIY'), { timeout: 20_000 });
    await page.waitForFunction(() => document.querySelector('.chip.model')?.textContent === 'ChatGPT · sign in needed', { timeout: 5000 });
    expect(chatgpt.length === before && refusedSignedOut === 1, `answered ${chatgpt.length - before}, refused ${refusedSignedOut}`);
    expect(engineAsked.length === 0, `the engine was asked: ${engineAsked.join(', ')}`);
    await shoot(page, 'chip-signin', { clip: header(page) });
    await shoot(page, 'setup-signin');
    signedIn = true;
  });

  await check('the page raised no errors', async () => {
    expect(!pageErrors.length, pageErrors.join('; '));
  });
  await page.close();

  await check('on a computer with no AI provider of its own, an Agent that runs on ChatGPT is not told to set one up', async () => {
    // A fresh browser profile: nothing saved, so no provider in Settings.
    const context = await browser.createBrowserContext();
    const fresh = await context.newPage();
    const errors = [];
    fresh.on('pageerror', (e) => errors.push(e.message));
    await fresh.setViewport({ width: 1058, height: 688 });
    await fresh.evaluateOnNewDocument((origin, token) => {
      window.__OAIY_DESKTOP__ = { origin, token };
      window.__OAIY_THEME__ = 'light';
    }, desktopUrl, TOKEN);
    await fresh.goto(base);
    await fresh.waitForSelector('.tree-row', { timeout: 60_000 });
    await fresh.waitForFunction(() => document.querySelector('.chip.model')?.textContent === 'ChatGPT · gpt-5.5', { timeout: 15_000 });
    await wait(1500);
    const log = await fresh.evaluate(() => document.querySelector('.chat-log')?.textContent ?? '');
    await context.close();
    expect(!/Set up an AI provider/.test(log), 'the welcome asked for a provider');
    expect(!errors.length, errors.join('; '));
  });
} finally {
  await browser.close();
  await server.close();
  engine.close();
  desktopServer.close();
}
console.log(checks.join('\n'));
if (shots.length) console.log(`${shots.length} screenshots in ${shotsDir}`);
if (failures.length) {
  console.error(`agentctl: ${failures.length} failed`);
  process.exit(1);
}
console.log('agentctl: all passed');
process.exit(0);
