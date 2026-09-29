// Contacts, in the browser: a fake OAIY Desktop that keeps contacts (Lance,
// named by the person, with their notes for the receptionist and what it
// remembered), its control API (recording what the Agent's page asks), and a
// scripted model; the Front desk seeded as the live one was kept (its own
// facts about Lance in callers.json). In OAIY's window: the facts are moved to
// the desktop once, the runner knows who answers the phone, a call from Lance
// has his contact in its context (the business's notes first) and says who it
// is, what the call's agent remembers is written to his contact, and his
// conversation's "Contact" opens the dashboard on him (ui_open). In a browser
// tab paired with the same desktop: the facts are not moved twice, and
// "Contact" shows his contact read only. Nothing here reaches a real desktop,
// phone or model.
//   node tests/e2e/contacts.mjs [screenshot-dir]
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
const { tools: CONTROL_TOOLS, runner: READ_TOOLS } = JSON.parse(readFileSync(new URL('../fixtures/control-tools.json', import.meta.url), 'utf8'));

// --- the scripted model --------------------------------------------------------------
/** Each request: what kind of conversation asked (by its system prompt), the prompt, and the messages. */
const asked = [];
const textOf = (m) => (typeof m.content === 'string' ? m.content : (m.content ?? []).map((p) => p.text ?? '').join(''));
const chunks = (text) => (text.match(/.{1,6}/gs) ?? []).map((piece) => `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: piece } }] })}\n\n`).join('');
const end = (calls) => {
  let out = '';
  (calls ?? []).forEach((call, i) => {
    out += `data: ${JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: `call_${asked.length}_${i}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.input) } }] } }] })}\n\n`;
  });
  return `${out}data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: calls ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`;
};
function answer(body) {
  // A warm (Agent.warm: the model reads the prompt ahead, and a few words at most are asked for): nothing to say.
  if ((body.max_tokens ?? body.max_completion_tokens) <= 32) return { text: '' };
  const system = textOf(body.messages.find((m) => m.role === 'system') ?? { content: '' });
  const lastUser = body.messages.map((m) => m.role === 'user').lastIndexOf(true);
  const steps = body.messages.slice(lastUser + 1).filter((m) => m.role === 'assistant').length;
  const kind = /live phone call/.test(system) ? 'call' : /text-message thread/.test(system) ? 'sms' : 'other';
  asked.push({ kind, system, messages: JSON.stringify(body.messages) });
  if (kind === 'call') return steps === 0 ? { calls: [{ name: 'remember', input: { fact: 'Gate code 4821' } }] } : { text: 'Hi Lance! How can I help?' };
  return { text: 'Done.' };
}
const model = createHttpServer(async (req, res) => {
  for (const [k, v] of Object.entries(cors)) res.setHeader(k, v);
  if (req.method === 'OPTIONS') return res.end();
  if (req.url.endsWith('/models')) {
    res.setHeader('content-type', 'application/json');
    return res.end(JSON.stringify({ data: [{ id: 'Qwen3.8-Flash-Next', context_length: 131072 }] }));
  }
  if (req.method === 'GET') {
    res.statusCode = 404;
    return res.end();
  }
  const step = answer(JSON.parse(await readBody(req)));
  res.setHeader('content-type', 'text/event-stream');
  res.end(chunks(step.text ?? '') + end(step.calls));
});
await new Promise((r) => model.listen(0, '127.0.0.1', r));
const modelUrl = `http://127.0.0.1:${model.address().port}`;

// --- a fake OAIY Desktop: the phone on, its contacts, its control API -------------------
const TOKEN = 'contacts-e2e';
const MODULES = {
  revision: 1,
  modules: [
    { id: 'phone', name: 'Phone', enabled: true, provider: { pluginId: 'aokie', name: 'AI Receptionist', state: 'running', declared: true } },
    { id: 'calendar', name: 'Calendar', enabled: true, provider: { pluginId: 'aokie', name: 'AI Receptionist', state: 'running', declared: true } },
  ],
  warnings: [],
};
const keyOf = (number) => {
  const digits = String(number).replace(/\D/g, '');
  return digits.length >= 8 ? digits.slice(-9) : '';
};
const NOTES = 'Call him Lance, never Mr Smith. Always offer the loyalty discount.';
/** The desktop's contacts, by key: Lance, named by the person, with their notes and what the receptionist remembered. */
const contacts = new Map([
  ['491570006', { key: '491570006', number: '+61491570006', name: 'Lance', nameBy: 'owner', notes: NOTES, facts: [{ text: 'Invoices go to the body corporate', at: '2026-09-28T00:00:00Z', by: 'owner' }, { text: 'Has a dog called Max', at: '2026-09-28T00:00:00Z', by: 'agent' }], createdAt: '2026-09-28T00:00:00Z', updatedAt: '2026-09-28T00:00:00Z' }],
]);
/** Every fact posted to a contact: its number, words and who wrote it. */
const posted = [];
/** Every MCP request: its method, its session header, the tool and its arguments. */
const mcp = [];
const voiceClients = new Set();
let liveCalls = [];
const desktopServer = createHttpServer(async (req, res) => {
  for (const [k, v] of Object.entries(cors)) res.setHeader(k, v);
  if (req.method === 'OPTIONS') return res.end();
  const url = new URL(req.url, 'http://x');
  const path = url.pathname;
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
  if (path === '/api/health') return json({ product: 'oaiy-desktop', protocol: 'oaiy-bridge/1', version: 'contacts' });
  if (path === '/api/modules') return json(MODULES);
  if (path === '/api/modules/events') return stream([MODULES]);
  if (path === '/api/agent/events') return stream([]);
  if (path === '/api/agent/preferences') return json({ model: { source: 'engine' } });
  if (path === '/api/voice/events') {
    stream([{ type: 'hello', calls: liveCalls }]);
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
  // The business's name; no receptionistName (a desktop that has not set one says none): the receptionist is Aokie.
  if (path.startsWith('/api/calendar')) return json({ available: true, settings: { business: 'Green Lawns' }, appointments: [], now: new Date().toISOString() });
  if (path.startsWith('/api/voice/calls/')) return json({ result: { ok: true, output: {} } });
  if (path === '/api/voice/callers') {
    const { number, name } = JSON.parse(body || '{}');
    const c = contacts.get(keyOf(number));
    return json({ number, name: c?.nameBy === 'owner' ? c.name : name });
  }
  // The contacts (the desktop's Contacts), as its routes answer.
  if (path === '/api/contacts') return json({ contacts: [...contacts.values()], total: contacts.size });
  const one = /^\/api\/contacts\/([^/]+)(\/facts(?:\/(\d+))?)?$/.exec(path);
  if (one) {
    const key = keyOf(decodeURIComponent(one[1]));
    if (!key) return json({ error: { code: 'no_number', message: 'not a phone number' } }, 400);
    let c = contacts.get(key);
    if (!one[2] && req.method === 'GET') return c ? json(c) : json({ error: { code: 'no_contact', message: `there is no contact whose number ends ${key}` } }, 404);
    if (one[2] && !one[3] && req.method === 'POST') {
      const { text, by = 'agent' } = JSON.parse(body || '{}');
      posted.push({ number: decodeURIComponent(one[1]), text, by });
      if (!c) contacts.set(key, (c = { key, number: '', name: '', nameBy: null, notes: '', facts: [], createdAt: '', updatedAt: '' }));
      const there = c.facts.some((f) => f.text.toLowerCase() === String(text).toLowerCase());
      if (!there) c.facts.push({ text, at: new Date().toISOString(), by });
      return json({ contact: c, added: !there, dropped: null }, there ? 200 : 201);
    }
    if (one[3] && req.method === 'DELETE' && c) {
      const [fact] = c.facts.splice(Number(one[3]), 1);
      return json({ contact: c, forgotten: fact });
    }
    return json({ error: { code: 'no_contact', message: 'no' } }, 404);
  }
  // The control API: the desktop's session rule (project and setup all, runner reads, call/sms/task none).
  if (path === '/api/mcp') {
    if (req.headers.authorization !== `Bearer ${TOKEN}`) return json({ error: 'origin not allowed' }, 403);
    const message = JSON.parse(body || '{}');
    const session = req.headers['x-oaiy-session'] ?? 'project';
    mcp.push({ method: message.method, session, tool: message.params?.name ?? null, args: message.params?.arguments ?? null });
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
      const args = message.params?.arguments ?? {};
      return ok({ content: [{ type: 'text', text: `The dashboard shows ${args.view}.` }], structuredContent: args });
    }
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
async function shoot(page, scene, before) {
  if (!shotsDir) return;
  for (const theme of ['light', 'dark']) {
    await page.evaluate((t) => window.__oaiySetTheme?.(t), theme);
    await wait(350);
    if (before) await before();
    const file = join(shotsDir, `contacts-${scene}-${theme}-1058.png`);
    await page.screenshot({ path: file });
    shots.push(file);
  }
  await page.evaluate(() => window.__oaiySetTheme?.('light'));
}

/** The Front desk as the live one was kept (Lance's own facts in callers.json), the model, and the phone on. */
async function seed(page, paired) {
  await page.goto(`${base}/tests/e2e/harness.html`);
  await page.evaluate(async (modelUrl, paired) => {
    const P = await import('/src/vfs/projects.ts');
    const ST = await import('/src/settings.ts');
    const F = await import('/tests/fixtures/liveFrontDesk.ts');
    const desk = await P.OpenProject.openFrontDesk();
    const live = F.liveFrontDesk();
    await desk.saveSessions(live.index);
    for (const [id, turns] of Object.entries(live.chats)) await desk.saveSessionChat(id, turns);
    await desk.saveCallers(live.callers);
    await ST.saveProviders([{ id: 'model', type: 'local', serverKind: 'other', name: 'OAIY', baseUrl: modelUrl, modelId: 'Qwen3.8-Flash-Next', apiKey: '' }], 'model');
    await ST.saveMessages({ ...ST.DEFAULT_MESSAGE_SETTINGS, calls: true, answer: true, country: 'AU' });
    await ST.saveLastProject('front-desk');
    if (paired) await ST.saveDesktop(paired);
  }, modelUrl, paired);
}
async function openPicker(page) {
  await page.evaluate(() => {
    const button = document.querySelector('.chat .combo-button');
    if (button?.getAttribute('aria-expanded') !== 'true') button?.click();
  });
  await wait(200);
}
/** What the Front desk keeps, read from its storage. */
const stored = (page) => page.evaluate(async () => {
  const desk = await (await navigator.storage.getDirectory()).getDirectoryHandle('front-desk');
  const read = async (name) => {
    try {
      return JSON.parse(await (await (await desk.getFileHandle(name)).getFile()).text());
    } catch {
      return null;
    }
  };
  return { callers: await read('callers.json'), moved: await read('contacts-moved.json') };
});

try {
  // ---- OAIY's window ----
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  page.on('dialog', (d) => d.accept());
  await page.setViewport({ width: 1058, height: 688 });
  await seed(page, null);
  await page.evaluateOnNewDocument((origin, token) => {
    window.__OAIY_DESKTOP__ = { origin, token };
    window.__OAIY_THEME__ = 'light';
  }, desktopUrl, TOKEN);
  await page.goto(base);
  await page.waitForSelector('.tree-row', { timeout: 60_000 });
  await page.waitForFunction(() => /\d+ others/.test(document.querySelector('.chat .combo-button')?.textContent ?? ''), { timeout: 20_000 });

  await check("the Front desk's facts are moved to the desktop's contacts once, and callers.json is kept (with a mark that holds it as it was)", async () => {
    for (let i = 0; i < 100 && !posted.some((p) => p.text === 'Prefers afternoons'); i++) await wait(100);
    await wait(500);
    const texts = posted.map((p) => p.text);
    expect(texts.filter((t) => /lawn mowing, fortnightly/i.test(t)).length === 1 && texts.filter((t) => t === 'Prefers afternoons').length === 1, JSON.stringify(posted));
    expect(posted.every((p) => p.by === 'agent' && p.number === '+61491570006'), JSON.stringify(posted));
    const kept = await stored(page);
    expect(kept.moved && kept.moved.sent === 2 && kept.moved.callers.length === 2, JSON.stringify(kept.moved));
    expect(Array.isArray(kept.callers) && kept.callers.some((c) => c.number === '+61491570006'), JSON.stringify(kept.callers));
  });

  await check("his contact is read into the Front desk's copy: the business's name for him, its notes, what was remembered", async () => {
    await page.waitForFunction(() => [...document.querySelectorAll('.chat .combo-option .combo-name')].some((n) => n.textContent === 'Lance') || document.querySelector('.chat .combo-button')?.textContent.includes('Lance'), { timeout: 10_000 }).catch(() => {});
    const kept = await stored(page);
    const lance = kept.callers.find((c) => c.number === '+61491570006');
    expect(lance?.name === 'Lance' && lance.nameBy === 'owner' && lance.notes === 'Call him Lance, never Mr Smith. Always offer the loyalty discount.', JSON.stringify(lance));
    expect(lance.ownerFacts?.join() === 'Invoices go to the body corporate' && lance.facts.includes('Has a dog called Max') && lance.facts.includes('Prefers afternoons'), JSON.stringify(lance));
  });

  await check('the runner knows who answers the phone, and for whom', async () => {
    await page.focus('.chat-input');
    await page.keyboard.type('Who answers the phone?');
    await page.keyboard.press('Enter');
    for (let i = 0; i < 100 && !asked.some((a) => a.kind === 'other'); i++) await wait(100);
    const runner = asked.find((a) => a.kind === 'other');
    expect(runner, 'the runner was not asked');
    expect(runner.system.includes('The phone is answered as Aokie, the receptionist for Green Lawns.'), runner.system.slice(-1500));
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Done.'), { timeout: 10_000 });
  });

  await check("a call from Lance has his contact in the call's context (the business's notes first, and winning) and says who it is", async () => {
    liveCalls = ['live-1'];
    voice({ type: 'call.started', callId: 'live-1', from: '0491570006', name: '', greeting: 'Hi, thanks for calling Green Lawns. How can I help?' });
    await page.waitForFunction(() => !document.querySelector('.call-live')?.hidden, { timeout: 10_000 });
    await wait(500);
    voice({ type: 'call.caller', callId: 'live-1', text: "Hi, it's Lance. The gate code is 4821.", startMs: 3_100, endMs: 5_400 });
    for (let i = 0; i < 100 && !asked.some((a) => a.kind === 'call'); i++) await wait(100);
    const call = asked.find((a) => a.kind === 'call');
    expect(call, 'the call was not answered');
    expect(call.messages.includes('Notes from the business: Call him Lance, never Mr Smith. Always offer the loyalty discount.'), call.messages.slice(0, 1500));
    expect(call.messages.includes('- Invoices go to the body corporate') && call.messages.includes('- Has a dog called Max'), call.messages.slice(0, 1500));
    expect(call.messages.includes('the notes win'), 'the call is not told the notes win');
    expect(call.messages.includes('Name: Lance (the name the business has them by: use it)'), call.messages.slice(0, 1500));
    expect(call.system.includes('You are Aokie, the receptionist for Green Lawns.') && call.system.includes('never "OAIY", never "your person"'), call.system.slice(0, 1200));
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Hi Lance! How can I help?'), { timeout: 10_000 });
    const title = await page.evaluate(() => document.querySelector('.chat .combo-button .convo-name')?.textContent ?? '');
    expect(/^Lance/.test(title), title);
  });

  await check("what the call's agent remembers is written to his contact, as the receptionist's", async () => {
    for (let i = 0; i < 50 && !posted.some((p) => p.text === 'Gate code 4821'); i++) await wait(100);
    const fact = posted.find((p) => p.text === 'Gate code 4821');
    expect(fact && fact.by === 'agent' && fact.number === '+61491570006', JSON.stringify(posted));
    expect(contacts.get('491570006').facts.some((f) => f.text === 'Gate code 4821' && f.by === 'agent'), JSON.stringify(contacts.get('491570006').facts));
  });
  await shoot(page, 'call');

  await check('"Contact" in his conversation opens the dashboard\'s Contacts page on him: ui_open, as the person\'s own (project)', async () => {
    const shown = await page.evaluate(() => {
      const b = document.querySelector('.chat .convo-contact');
      return { visible: !!b && !b.hidden && b.getBoundingClientRect().width > 0, text: b?.textContent ?? '' };
    });
    expect(shown.visible && shown.text === 'Contact', JSON.stringify(shown));
    const before = mcp.length;
    await page.click('.chat .convo-contact');
    for (let i = 0; i < 50 && !mcp.slice(before).some((m) => m.method === 'tools/call'); i++) await wait(100);
    const opened = mcp.slice(before).filter((m) => m.method === 'tools/call');
    expect(opened.length === 1 && opened[0].tool === 'ui_open' && opened[0].session === 'project', JSON.stringify(opened));
    expect(JSON.stringify(opened[0].args) === JSON.stringify({ view: 'contacts', contact: '491570006' }), JSON.stringify(opened[0].args));
    await wait(300);
    expect(!(await page.$('.contact-pop')), 'a card showed as well as the dashboard');
  });

  await check('the runner\'s own conversation and a flow\'s have no Contact', async () => {
    await openPicker(page);
    await page.evaluate(() => [...document.querySelectorAll('.chat .combo-option')].find((o) => o.dataset.kind === 'runner')?.click());
    await wait(400);
    const hidden = await page.evaluate(() => document.querySelector('.chat .convo-contact')?.hidden);
    expect(hidden === true, `hidden: ${hidden}`);
  });

  voice({ type: 'call.ended', callId: 'live-1' });
  liveCalls = [];
  await wait(500);
  await check('the page raised no errors', async () => {
    expect(!pageErrors.length, pageErrors.join('; '));
  });
  await context.close();

  // ---- a browser tab, paired with the same desktop ----
  const tabContext = await browser.createBrowserContext();
  const tab = await tabContext.newPage();
  const tabErrors = [];
  tab.on('pageerror', (e) => tabErrors.push(e.message));
  tab.on('dialog', (d) => d.accept());
  await tab.setViewport({ width: 1058, height: 688 });
  await seed(tab, { origin: desktopUrl, token: TOKEN });
  await tab.evaluateOnNewDocument(() => {
    window.__OAIY_THEME__ = 'light';
  });
  const postedBefore = posted.length;
  await tab.goto(base);
  await tab.waitForSelector('.tree-row', { timeout: 60_000 });
  await tab.waitForFunction(() => /\d+ others/.test(document.querySelector('.chat .combo-button')?.textContent ?? ''), { timeout: 20_000 });

  await check('in another page the facts are moved again: nothing the desktop has is added twice', async () => {
    for (let i = 0; i < 50 && !(await stored(tab)).moved; i++) await wait(100);
    const lance = contacts.get('491570006').facts.map((f) => f.text.toLowerCase());
    expect(new Set(lance).size === lance.length, JSON.stringify(lance));
    expect(posted.length - postedBefore === 2 && (await stored(tab)).moved?.there === 2, JSON.stringify(posted.slice(postedBefore)));
  });

  await check('"Contact" in a browser tab shows his contact read only: the name, the notes for the receptionist, what it remembered', async () => {
    // His conversation takes the name the business has him by, once his contact is read.
    await openPicker(tab);
    await tab.waitForFunction(() => [...document.querySelectorAll('.chat .combo-option .combo-name')].some((n) => n.textContent === 'Lance'), { timeout: 10_000 });
    await tab.evaluate(() => [...document.querySelectorAll('.chat .combo-option')].find((o) => o.querySelector('.combo-name')?.textContent === 'Lance')?.click());
    await wait(500);
    const before = mcp.length;
    await tab.click('.chat .convo-contact');
    await tab.waitForFunction(() => document.querySelector('.contact-pop .contact-name'), { timeout: 10_000 });
    const card = await tab.evaluate(() => {
      const pop = document.querySelector('.contact-pop');
      return {
        name: pop.querySelector('.contact-name')?.textContent,
        number: pop.querySelector('.contact-number')?.textContent,
        heads: [...pop.querySelectorAll('h4')].map((e) => e.textContent),
        notes: pop.querySelector('.contact-notes')?.textContent,
        facts: [...pop.querySelectorAll('.contact-facts li')].map((e) => e.textContent),
        inputs: pop.querySelectorAll('input, textarea, [contenteditable]').length,
      };
    });
    expect(card.name === 'Lance' && /0491 570 006/.test(card.number) && /named by you/.test(card.number), JSON.stringify(card));
    expect(card.heads.join('|') === 'Notes for the receptionist|What the receptionist remembered' && card.notes === 'Call him Lance, never Mr Smith. Always offer the loyalty discount.', JSON.stringify(card));
    expect(['Invoices go to the body corporate', 'Has a dog called Max', 'Gate code 4821'].every((f) => card.facts.includes(f)) && card.inputs === 0, JSON.stringify(card));
    expect(!mcp.slice(before).some((m) => m.method === 'tools/call'), 'the tab asked the control API');
  });
  await shoot(tab, 'card', async () => {
    if (!(await tab.$('.contact-pop'))) await tab.click('.chat .convo-contact');
    await tab.waitForSelector('.contact-pop .contact-name');
  });

  await check('Escape closes the card', async () => {
    if (!(await tab.$('.contact-pop'))) await tab.click('.chat .convo-contact');
    await tab.waitForSelector('.contact-pop');
    await tab.keyboard.press('Escape');
    await wait(200);
    expect(!(await tab.$('.contact-pop')), 'the card stayed');
  });

  await check('the browser tab raised no errors', async () => {
    expect(!tabErrors.length, tabErrors.join('; '));
  });
  await tabContext.close();
} finally {
  await browser.close();
  await server.close();
  model.close();
  desktopServer.close();
}
console.log(checks.join('\n'));
if (shots.length) console.log(`${shots.length} screenshots in ${shotsDir}`);
if (failures.length) {
  console.error(`contacts: ${failures.length} failed`);
  process.exit(1);
}
console.log('contacts: all passed');
process.exit(0);
