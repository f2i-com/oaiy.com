// One conversation per person, in the browser: OAIY's window with a fake OAIY
// Desktop (the phone on, calls and texts on demand) and a scripted model, and
// the Front desk seeded as the live one was kept before (Lance's calls under
// 0491570006, his texts under +61491570006). The app merges them on its first
// start (keeping a backup), shows one conversation for him in order, and a
// call from 0491570006 then a text from +61491570006 land in it, the call's
// live header and reply going on through the text. Nothing here reaches a
// real desktop, phone or model.
//   node tests/e2e/oneperson.mjs [screenshot-dir]
import { existsSync, mkdirSync } from 'node:fs';
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

// --- the scripted model: a call's receptionist, and the texts' -------------------------
const held = [];
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
  if ((body.max_tokens ?? body.max_completion_tokens) === 1) return { text: '' };
  const system = textOf(body.messages.find((m) => m.role === 'system') ?? { content: '' });
  const lastUser = body.messages.map((m) => m.role === 'user').lastIndexOf(true);
  const last = textOf(body.messages[lastUser] ?? { content: '' });
  const steps = body.messages.slice(lastUser + 1).filter((m) => m.role === 'assistant').length;
  if (/live phone call/.test(system)) {
    asked.push('call');
    if (/Tuesday/i.test(last)) return { text: 'Tuesday at ten is free. Shall I ask the team for it?', hold: 14 };
    if (/please do/i.test(last)) return { text: 'Done, Lance. Anything else?' };
    // The goodbye, then a word after it that is never said (a model often writes "Done." after end_call).
    if (/that's all/i.test(last)) return steps === 0 ? { calls: [{ name: 'end_call', input: { goodbye: 'Bye, Lance!' } }] } : { text: 'Done.' };
    return { text: 'Hi Lance! How can I help?' };
  }
  if (/text-message thread/.test(system)) {
    asked.push('sms');
    // A text that comes as a short call ends: its agent is still at work when the call is over.
    if (/Running late/.test(last)) return steps === 0 ? { calls: [{ name: 'send_text_message', input: { body: 'No worries, see you at 10:15.' } }] } : { text: 'Told him that is fine.', hold: 5 };
    if (steps === 0) return { calls: [{ name: 'send_text_message', input: { body: 'Thanks Lance, got the gate code. See you Tuesday!' } }] };
    return { text: 'Replied.' };
  }
  asked.push('other');
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
  if (step.hold) {
    // The reply begins, then waits: the call's agent is still writing when the text comes.
    res.write(chunks(step.text.slice(0, step.hold)));
    held.push(() => res.end(chunks(step.text.slice(step.hold)) + end(step.calls)));
    return;
  }
  res.end(chunks(step.text ?? '') + end(step.calls));
});
await new Promise((r) => model.listen(0, '127.0.0.1', r));
const modelUrl = `http://127.0.0.1:${model.address().port}`;

// --- a fake OAIY Desktop: the phone on, calls and texts on demand ------------------------
const TOKEN = 'oneperson';
const MODULES = {
  revision: 1,
  modules: [
    { id: 'phone', name: 'Phone', enabled: true, provider: { pluginId: 'aokie', name: 'AI Receptionist', state: 'running', declared: true } },
    { id: 'calendar', name: 'Calendar', enabled: true, provider: { pluginId: 'aokie', name: 'AI Receptionist', state: 'running', declared: true } },
  ],
  warnings: [],
};
const voiceClients = new Set();
let liveCalls = [];
const bridgeEvents = [];
const sent = [];
const spoken = [];
const named = [];
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
  if (path === '/api/health') return json({ product: 'oaiy-desktop', protocol: 'oaiy-bridge/1', version: 'oneperson' });
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
  if (path === '/api/bridge/events') {
    const since = Number(url.searchParams.get('since') ?? 0);
    return json({ events: bridgeEvents.slice(since).map((envelope, i) => ({ seq: since + i + 1, envelope })), next: bridgeEvents.length });
  }
  if (path.startsWith('/api/bridge/leases/')) return json({ granted: true, holder: (JSON.parse(body || '{}').holder ?? '') });
  if (path.startsWith('/api/bridge/connectors/')) {
    const { command, payload } = JSON.parse(body || '{}');
    if (command === 'sms.send') sent.push(payload);
    return json({ ok: true, result: { ok: true, data: command === 'phone.status' ? { connected: true } : command === 'settings.get' ? { value: '' } : command === 'sms.send' ? { messageId: `m${sent.length}` } : {} } });
  }
  if (path === '/api/bridge/flows') return json({ flows: [] });
  if (path === '/api/plugins') return json({ plugins: [{ id: 'aokie', state: 'running' }] });
  if (path.startsWith('/api/calendar')) return json({ available: true, settings: {}, appointments: [], now: new Date().toISOString() });
  if (path.startsWith('/api/voice/calls/')) {
    if (path.endsWith('/say')) spoken.push(JSON.parse(body || '{}').text);
    return json({ result: { ok: true, output: {} } });
  }
  if (path === '/api/voice/callers') {
    if (req.method === 'PUT') named.push(JSON.parse(body || '{}'));
    return json({});
  }
  if (path === '/api/mcp') return json({ jsonrpc: '2.0', id: JSON.parse(body || '{}').id ?? null, result: { tools: [] } });
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
    const file = join(shotsDir, `oneperson-${scene}-${theme}-1058.png`);
    await page.screenshot({ path: file });
    shots.push(file);
  }
  await page.evaluate(() => window.__oaiySetTheme?.('light'));
}

/** What the chat's log shows, in order: its dividers (with their way), and who said what. */
const feed = (page) => page.evaluate(() => [...document.querySelectorAll('.chat-feed .day-divider, .chat-feed .msg-body, .chat-feed .sms-body, .chat-feed .call-end')].map((e) => {
  if (e.matches('.day-divider')) return `[${e.dataset.way ?? 'call'}] ${e.textContent.trim()}`;
  if (e.matches('.call-end')) return '[ended]';
  const who = e.closest('.run')?.dataset.speaker ?? '';
  return `${who}${e.matches('.sms-body') ? ' texted' : ''}: ${e.textContent.trim()}`;
}));
const logState = (page) => page.evaluate(() => {
  const log = document.querySelector('.chat-log');
  return { fromBottom: Math.round(log.scrollHeight - log.scrollTop - log.clientHeight) };
});
const pickerOptions = (page) => page.evaluate(() => [...document.querySelectorAll('.chat .combo-option')].map((o) => ({ kind: o.dataset.kind, name: o.querySelector('.combo-name')?.textContent, detail: o.querySelector('.combo-detail')?.textContent, icons: o.querySelectorAll('.combo-detail-icons .icon').length, meta: o.querySelector('.combo-meta')?.textContent })));
async function openPicker(page) {
  await page.evaluate(() => {
    const button = document.querySelector('.chat .combo-button');
    if (button?.getAttribute('aria-expanded') !== 'true') button?.click();
  });
  await wait(200);
}
async function closePicker(page) {
  await page.keyboard.press('Escape');
  await page.evaluate(() => document.activeElement?.blur?.());
  await wait(150);
}

try {
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  page.on('dialog', (d) => d.accept());
  await page.setViewport({ width: 1058, height: 688 });
  // The Front desk as the live one was kept, before a person's calls and texts were one (through the app's own storage code).
  await page.goto(`${base}/tests/e2e/harness.html`);
  await page.evaluate(async (modelUrl) => {
    const P = await import('/src/vfs/projects.ts');
    const ST = await import('/src/settings.ts');
    const F = await import('/tests/fixtures/liveFrontDesk.ts');
    const desk = await P.OpenProject.openFrontDesk();
    desk.vfs.writeFile('/knowledge/services.md', '# Services\n\n- Lawn mowing, from $45\n', { parents: true });
    await desk.flush();
    const live = F.liveFrontDesk();
    await desk.saveSessions(live.index);
    for (const [id, turns] of Object.entries(live.chats)) await desk.saveSessionChat(id, turns);
    await desk.saveCallers(live.callers);
    await ST.saveProviders([{ id: 'model', type: 'local', serverKind: 'other', name: 'OAIY', baseUrl: modelUrl, modelId: 'Qwen3.8-Flash-Next', apiKey: '' }], 'model');
    await ST.saveMessages({ ...ST.DEFAULT_MESSAGE_SETTINGS, calls: true, answer: true, country: 'AU' });
    await ST.saveLastProject('front-desk');
  }, modelUrl);
  await page.evaluateOnNewDocument((origin, token) => {
    window.__OAIY_DESKTOP__ = { origin, token };
    window.__OAIY_THEME__ = 'light';
  }, desktopUrl, TOKEN);
  await page.goto(base);
  await page.waitForSelector('.tree-row', { timeout: 60_000 });
  await page.waitForFunction(() => /\d+ others/.test(document.querySelector('.chat .combo-button')?.textContent ?? ''), { timeout: 20_000 });
  await wait(800);

  await check('the Calls (0491570006) and Texts (+61491570006) kept for Lance are merged on the first start: one conversation, the files kept in a backup', async () => {
    const stored = await page.evaluate(async () => {
      const desk = await (await navigator.storage.getDirectory()).getDirectoryHandle('front-desk');
      const names = async (dir) => {
        const out = [];
        for await (const [name] of dir) out.push(name);
        return out.sort();
      };
      const top = await names(desk);
      const backup = top.find((n) => n.startsWith('.backup-'));
      const sessions = await desk.getDirectoryHandle('sessions');
      const merged = JSON.parse(await (await (await sessions.getFileHandle('person-61491570006.json')).getFile()).text());
      const index = JSON.parse(await (await (await sessions.getFileHandle('index.json')).getFile()).text());
      return { backup, backedUp: backup ? await names(await (await desk.getDirectoryHandle(backup)).getDirectoryHandle('sessions')) : [], files: await names(sessions), merged: merged.map((t) => t.via), index: index.map((i) => [i.id, i.key, i.thread]) };
    });
    expect(/^\.backup-\d{4}-\d{2}-\d{2}$/.test(stored.backup ?? ''), JSON.stringify(stored.backup));
    expect(['call-0491570006.json', 'sms-61491570006.json', 'index.json'].every((f) => stored.backedUp.includes(f)), JSON.stringify(stored.backedUp));
    expect(!stored.files.includes('call-0491570006.json') && !stored.files.includes('sms-61491570006.json'), JSON.stringify(stored.files));
    expect(stored.merged.join(',') === [...Array(7).fill('call'), ...Array(5).fill('sms'), ...Array(14).fill('call')].join(','), stored.merged.join(','));
    expect(stored.index.filter(([, key]) => key === '+61491570006').every(([, , thread]) => thread === 'person-61491570006') && stored.index.filter(([, key]) => key === '+61491570006').length === 2, JSON.stringify(stored.index));
  });

  await check('the picker lists Lance once, under Calls and texts, with both ways, his number as it is dialled here, and when he was last in touch', async () => {
    await openPicker(page);
    const options = await pickerOptions(page);
    const lance = options.filter((o) => o.name?.startsWith('Lance'));
    expect(lance.length === 1, JSON.stringify(options));
    expect(lance[0].kind === 'person' && lance[0].icons === 2 && lance[0].detail === '0491 570 006 · Calls and texts' && !!lance[0].meta, JSON.stringify(lance[0]));
    const groups = await page.evaluate(() => [...document.querySelectorAll('.chat .combo-group-label, .chat [role="presentation"]')].map((g) => g.textContent.trim()).filter(Boolean));
    expect(groups.some((g) => g.startsWith('Calls and texts')), JSON.stringify(groups));
    expect(!options.some((o) => o.kind === 'call' || o.kind === 'sms'), JSON.stringify(options));
  });
  await shoot(page, 'picker', async () => openPicker(page));
  await closePicker(page);

  await check("Lance's conversation shows his calls and texts in the order they happened: yesterday's call, this morning's texts, the call after them", async () => {
    await openPicker(page);
    await page.evaluate(() => [...document.querySelectorAll('.chat .combo-option')].find((o) => o.querySelector('.combo-name')?.textContent.startsWith('Lance'))?.click());
    await wait(600);
    const shown = await feed(page);
    const dividers = shown.filter((l) => l.startsWith('['));
    expect(dividers.length >= 4, JSON.stringify(dividers));
    expect(dividers[0].startsWith('[call]') && /Call from Lance/.test(dividers[0]), JSON.stringify(dividers));
    expect(dividers[1] === '[ended]' && dividers[2].startsWith('[sms]') && /Texts/.test(dividers[2]) && dividers[3].startsWith('[call]'), JSON.stringify(dividers));
    const texts = shown.filter((l) => l.startsWith('texter: '));
    expect(texts.join('|') === 'texter: Testing|texter: Hello|texter: Hello|texter: Hello|texter: Hello', JSON.stringify(texts));
    const at = (line) => shown.indexOf(line);
    expect(at('caller: Great, thanks. Bye.') < at('texter: Testing') && at('texter: Testing') < at('caller: Hi there, I was hoping to book a lawn mow.'), JSON.stringify(shown));
    const s = await logState(page);
    expect(s.fromBottom <= 2, JSON.stringify(s));
  });
  await shoot(page, 'merged', async () => {
    await page.evaluate(() => {
      const log = document.querySelector('.chat-log');
      const divider = document.querySelector('.chat-feed .day-divider.way-sms');
      if (divider) log.scrollTop += divider.getBoundingClientRect().top - log.getBoundingClientRect().top - 150;
    });
    await wait(250);
  });

  await check('a call from 0491570006 opens in the same conversation, with the live header, and follows at the bottom', async () => {
    liveCalls = ['live-1'];
    voice({ type: 'call.started', callId: 'live-1', from: '0491570006', name: '', greeting: 'Hi, thanks for calling Greenline Gardens. How can I help?' });
    await page.waitForFunction(() => !document.querySelector('.call-live')?.hidden, { timeout: 10_000 });
    await wait(600);
    voice({ type: 'call.caller', callId: 'live-1', text: 'Hi, can you come Tuesday at ten?', startMs: 3_100, endMs: 5_400 });
    for (let i = 0; i < 100 && !held.length; i++) await wait(100);
    expect(held.length === 1, `the call's reply did not begin: ${asked.join(',')}`);
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Tuesday at te'), { timeout: 10_000 });
    const state = await page.evaluate(() => ({ live: document.querySelector('.call-live')?.textContent ?? '', button: document.querySelector('.chat .combo-button')?.textContent ?? '', conversations: document.querySelectorAll('.chat .combo-option').length }));
    expect(/Live call/.test(state.live) && /Lance/.test(state.live), JSON.stringify(state));
    expect(/Lance/.test(state.button) && /Live/.test(state.button), JSON.stringify(state));
    const s = await logState(page);
    expect(s.fromBottom <= 2, JSON.stringify(s));
  });

  await check('a text from +61491570006 during the call lands in the same conversation, is answered, and the call goes on', async () => {
    bridgeEvents.push({ name: 'aokie.sms.received', source: 'aokie', correlationId: 'c1', idempotencyKey: 'k1', occurredAt: new Date().toISOString(), data: { from: '+61491570006', name: 'Lance', body: 'The gate code is 4821', handle: '0400000000000040' } });
    for (let i = 0; i < 100 && !sent.length; i++) await wait(100);
    expect(sent.length === 1 && sent[0].to === '+61491570006' && /gate code/.test(sent[0].body), JSON.stringify(sent));
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('The gate code is 4821') && document.querySelector('.chat-log .sms-out'), { timeout: 10_000 });
    // The call's reply is still being written: it goes on in its own box, and is said whole.
    held.splice(0).forEach((release) => release());
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Shall I ask the team for it?'), { timeout: 10_000 });
    for (let i = 0; i < 50 && !spoken.some((s) => /Shall I ask/.test(s)); i++) await wait(100);
    expect(spoken.join(' ').includes('Tuesday at ten is free. Shall I ask the team for it?'), JSON.stringify(spoken));
    const replies = await page.evaluate(() => [...document.querySelectorAll('.chat-feed .msg.assistant .msg-body')].map((e) => e.textContent.trim()).filter((t) => /Tuesday at ten/.test(t)));
    expect(replies.length === 1 && replies[0] === 'Tuesday at ten is free. Shall I ask the team for it?', JSON.stringify(replies));
    voice({ type: 'call.caller', callId: 'live-1', text: 'Yes please do.', startMs: 14_000, endMs: 15_200 });
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Done, Lance. Anything else?'), { timeout: 10_000 });
    await wait(500);
    const shown = await feed(page);
    const tail = shown.slice(shown.lastIndexOf(shown.find((l) => l.startsWith('[call]') && /Today|Call from/.test(l) && shown.indexOf(l) > shown.indexOf('texter: Hello'))));
    const at = (pattern) => tail.findIndex((l) => pattern.test(l));
    expect(at(/^caller: Hi, can you come Tuesday/) < at(/^\[sms\] /) && at(/^\[sms\] /) < at(/^texter: The gate code is 4821/) && at(/^texter: The gate code/) < at(/^agent texted: Thanks Lance/) && at(/^agent texted: Thanks Lance/) < at(/^\[call\] .*On the call/) && at(/On the call/) < at(/^caller: Yes please do\./) && at(/^caller: Yes please do\./) < at(/^agent: Done, Lance/), JSON.stringify(tail));
    const state = await page.evaluate(() => ({ live: !document.querySelector('.call-live')?.hidden, people: [...document.querySelectorAll('.chat .combo-option')].filter((o) => o.querySelector('.combo-name')?.textContent.startsWith('Lance')).length }));
    expect(state.live, 'the live header went');
    const s = await logState(page);
    expect(s.fromBottom <= 2, JSON.stringify(s));
  });
  // The call going on, the text that came during it, and the call's words after it.
  await shoot(page, 'live', async () => {
    await page.evaluate(() => {
      const log = document.querySelector('.chat-log');
      const divider = [...document.querySelectorAll('.chat-feed .day-divider.way-sms')].at(-1);
      if (divider) log.scrollTop += divider.getBoundingClientRect().top - log.getBoundingClientRect().top - 110;
    });
    await wait(250);
  });
  await page.evaluate(() => document.querySelector('.to-latest')?.click());
  await wait(400);
  await openPicker(page);
  await check('the picker still has one conversation for Lance, live', async () => {
    const lance = (await pickerOptions(page)).filter((o) => o.name?.startsWith('Lance'));
    expect(lance.length === 1 && lance[0].detail === 'On a call now', JSON.stringify(lance));
  });
  await shoot(page, 'picker-live', async () => openPicker(page));
  await closePicker(page);

  await check('the desktop is given each name once, under the E.164 number', async () => {
    const lance = named.filter((n) => /Lance/.test(n.name));
    expect(lance.length >= 1 && lance.every((n) => n.number === '+61491570006'), JSON.stringify(named));
  });

  await check('the call\'s goodbye is said, and "Done." written after end_call is a quiet note, never a line of the receptionist\'s', async () => {
    const before = spoken.length;
    voice({ type: 'call.caller', callId: 'live-1', text: "No, that's all. Bye!", startMs: 20_000, endMs: 21_200 });
    await page.waitForFunction(() => [...document.querySelectorAll('.chat-feed .msg.unsaid')].some((n) => /Done\./.test(n.textContent)), { timeout: 10_000 });
    await wait(400);
    const drawn = await page.evaluate(() => ({
      replies: [...document.querySelectorAll('.chat-feed .msg.assistant .msg-body')].map((e) => e.textContent.trim()),
      unsaid: [...document.querySelectorAll('.chat-feed .msg.unsaid')].map((e) => ({ label: e.querySelector('.unsaid-label')?.textContent, text: e.querySelector('.unsaid-text')?.textContent, inRun: !!e.closest('.run') })),
      ended: !!document.querySelector('.chat-feed details.tool[data-tool="end_call"]'),
    }));
    expect(!drawn.replies.includes('Done.'), JSON.stringify(drawn.replies.slice(-4)));
    const done = drawn.unsaid.find((u) => u.text === 'Done.');
    expect(done && /^Not said/.test(done.label) && !done.inRun && drawn.ended, JSON.stringify(drawn));
    // Replied. (the texts' agent's own words after send_text_message) is a note too.
    expect(drawn.unsaid.some((u) => u.text === 'Replied.' && /^Not sent/.test(u.label)), JSON.stringify(drawn.unsaid));
    expect(!spoken.slice(before).some((s) => /Done/.test(s)), JSON.stringify(spoken.slice(before)));
  });
  await shoot(page, 'unsaid');

  voice({ type: 'call.ended', callId: 'live-1' });
  liveCalls = [];
  await wait(800);
  await check('the call ends: the live header goes, the conversation stays one', async () => {
    const state = await page.evaluate(() => ({ live: !document.querySelector('.call-live')?.hidden }));
    expect(!state.live, JSON.stringify(state));
    await openPicker(page);
    const lance = (await pickerOptions(page)).filter((o) => o.name?.startsWith('Lance'));
    expect(lance.length === 1 && /Calls and texts/.test(lance[0].detail), JSON.stringify(lance));
    await closePicker(page);
  });

  await check('a text answered while a short call ends: its reply is drawn whole, the text it sent included, when it is done', async () => {
    liveCalls = ['live-2'];
    voice({ type: 'call.started', callId: 'live-2', from: '0491570006', name: '' });
    await page.waitForFunction(() => !document.querySelector('.call-live')?.hidden, { timeout: 10_000 });
    bridgeEvents.push({ name: 'aokie.sms.received', source: 'aokie', correlationId: 'c2', idempotencyKey: 'k2', occurredAt: new Date().toISOString(), data: { from: '+61491570006', name: 'Lance', body: 'Running late, there at 10:15', handle: '0400000000000041' } });
    for (let i = 0; i < 100 && (sent.length < 2 || !held.length); i++) await wait(100);
    expect(sent.length === 2 && sent[1].body === 'No worries, see you at 10:15.' && held.length === 1, JSON.stringify({ sent, held: held.length }));
    // The call ends while the texts' agent is still writing.
    voice({ type: 'call.ended', callId: 'live-2' });
    liveCalls = [];
    await page.waitForFunction(() => document.querySelector('.call-live')?.hidden, { timeout: 10_000 });
    held.splice(0).forEach((release) => release());
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Told him that is fine.'), { timeout: 10_000 });
    await wait(400);
    const drawn = await page.evaluate(() => ({
      sent: [...document.querySelectorAll('.chat-feed .sms-out .sms-body')].map((e) => e.textContent).filter((t) => /10:15/.test(t)),
      // The texts' agent's own words (not a text it sent): a quiet note, not a reply.
      replies: [...document.querySelectorAll('.chat-feed .msg.unsaid .unsaid-text')].map((e) => e.textContent.trim()).filter((t) => /Told him|that is fine/.test(t)),
      bubbles: [...document.querySelectorAll('.chat-feed .msg.assistant .msg-body')].map((e) => e.textContent.trim()).filter((t) => /Told him|that is fine/.test(t)).length,
      text: [...document.querySelectorAll('.chat-feed .msg.incoming.texter .msg-body')].map((e) => e.textContent).filter((t) => /Running late/.test(t)).length,
    }));
    expect(drawn.sent.length === 1 && drawn.replies.length === 1 && drawn.replies[0] === 'Told him that is fine.' && drawn.bubbles === 0 && drawn.text === 1, JSON.stringify(drawn));
    await openPicker(page);
    const lance = (await pickerOptions(page)).filter((o) => o.name?.startsWith('Lance'));
    expect(lance.length === 1, JSON.stringify(lance));
    await closePicker(page);
  });

  await check('the page raised no errors', async () => {
    expect(!pageErrors.length, pageErrors.join('; '));
  });
} finally {
  await browser.close();
  await server.close();
  model.close();
  desktopServer.close();
}
console.log(checks.join('\n'));
if (shots.length) console.log(`${shots.length} screenshots in ${shotsDir}`);
if (failures.length) {
  console.error(`oneperson: ${failures.length} failed`);
  process.exit(1);
}
console.log('oneperson: all passed');
process.exit(0);
