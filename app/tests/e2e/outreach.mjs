// Outreach in the browser: OAIY's window with a fake OAIY Desktop (the phone
// on, Aokie's dial, text and hang-up commands recorded, its events and the
// voice stream played by the test) and a scripted model. The runner is asked
// to call three people: the person approves it once in the dialog, then it
// runs by itself: #1 answers and confirms (record_result, one goodbye, and
// "Done." after it drawn as a quiet note), #2 does not answer and is rung
// again after the gap, #3 is a voicemail it hangs up on without a word; the
// report comes back to the runner, and the results are in the Front desk.
// Then texts (a reply answered and recorded, a STOP never answered), a
// reload in the middle of a call, and a report that waits for the Front desk
// to open. Nothing here reaches a real desktop, phone or model.
//   node tests/e2e/outreach.mjs [screenshot-dir]
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
const until = async (what, fn, ms = 15_000) => {
  for (let t = 0; t < ms; t += 100) {
    if (await fn()) return;
    await wait(100);
  }
  throw new Error(`timed out waiting for ${what}`);
};

// --- what the runner is asked to start -----------------------------------------------
const WINDOW = { from: '00:00', to: '24:00' };
const CALLS = {
  kind: 'call',
  name: 'Confirm Friday bookings',
  objective: "Confirm they're still coming on Friday 2 Oct, or find a better time.",
  openingLine: "Hi {first_name}, it's Greenleaf Lawns about your {service} on {appointment}. Have you got a minute?",
  collect: [
    { key: 'coming', question: 'Still coming Friday?', type: 'yes_no' },
    { key: 'new_time', question: 'If not, what day and time suits?', optional: true },
  ],
  people: [
    { name: 'Jane Smith', number: '0412 345 678', fields: { service: 'lawn mow', appointment: 'Fri 2 Oct 10:30' } },
    { name: 'Bob Jones', number: '0413 000 111', fields: { service: 'hedge trim', appointment: 'Fri 2 Oct 1:00' } },
    { name: 'Cara Lee', number: '0414 222 333', fields: { service: 'lawn mow', appointment: 'Fri 2 Oct 3:00' } },
    { name: 'Jane S', number: '+61412345678', fields: { gate: '4821' } },
    { name: 'Old number', number: '0412 000' },
  ],
  retries: { times: 1, gapMinutes: 30 },
  window: WINDOW,
  afterwards: "Move anyone who can't come to the time they gave.",
};
const TEXTS = {
  kind: 'text',
  name: 'Friday reminders',
  objective: 'Check they are still coming on Friday.',
  textTemplate: 'Hi {first_name}, Greenleaf Lawns here: still right for Friday? Reply YES or NO.',
  collect: [{ key: 'coming', question: 'Still coming?', type: 'yes_no' }],
  people: [{ name: 'Dee Park', number: '0415 000 101' }, { name: 'Eve Quinn', number: '0415 000 102' }],
  window: WINDOW,
};
const RELOAD = { ...CALLS, name: 'Hal check', people: [{ name: 'Hal Ray', number: '0416 000 201', fields: { service: 'mow', appointment: 'Sat' } }], retries: { times: 0, gapMinutes: 30 }, afterwards: '' };
const CLOSED = { ...TEXTS, name: 'Gus reminder', people: [{ name: 'Gus Moss', number: '0417 000 301' }] };

// --- the scripted model: the runner, a call's agent, a text thread's agent ------------
const asked = { runner: [], call: [], sms: [] };
const reports = [];
const privacy = [];
const textOf = (m) => (typeof m.content === 'string' ? m.content : (m.content ?? []).map((p) => p.text ?? '').join(''));
const chunks = (text) => (text.match(/.{1,6}/gs) ?? []).map((piece) => `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: piece } }] })}\n\n`).join('');
const end = (calls, n) => {
  let out = '';
  (calls ?? []).forEach((call, i) => {
    out += `data: ${JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: `call_${n}_${i}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.input) } }] } }] })}\n\n`;
  });
  return `${out}data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: calls ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`;
};
let served = 0;
function answer(body) {
  if ((body.max_tokens ?? body.max_completion_tokens) === 1) return { text: '' };
  const system = textOf(body.messages.find((m) => m.role === 'system') ?? { content: '' });
  const tools = (body.tools ?? []).map((t) => t.function.name);
  const lastUser = body.messages.map((m) => m.role === 'user').lastIndexOf(true);
  const last = textOf(body.messages[lastUser] ?? { content: '' });
  const after = body.messages.slice(lastUser + 1);
  const steps = after.filter((m) => m.role === 'assistant').length;
  const results = after.filter((m) => m.role === 'tool').map(textOf);
  if (/live phone call/.test(system)) {
    asked.call.push({ tools, system, last });
    const first = /Who: (\S+)/.exec(system)?.[1] ?? 'there';
    if (/leave a message|after the tone/i.test(last)) return steps === 0 ? { calls: [{ name: 'record_result', input: { outcome: 'voicemail', summary: 'Their voicemail greeting; no message left.' } }, { name: 'end_call', input: { silent: true } }] } : { text: '' };
    if (/still coming/i.test(last)) {
      // Words and end_call in one reply (as seen live): the words are the one goodbye; "Done." after it is never said.
      return steps === 0
        ? { text: `Lovely, thanks ${first}, see you Friday.`, calls: [{ name: 'record_result', input: { outcome: 'completed', answers: { coming: true }, summary: 'Still coming on Friday at 10:30.' } }, { name: 'end_call', input: { goodbye: 'You are welcome, have a great day!' } }] }
        : { text: 'Done.' };
    }
    return { text: 'Sorry, could you say that again?' };
  }
  if (/text-message thread/.test(system)) {
    asked.sms.push({ tools, system, last });
    const first = /Text message from (\S+)/.exec(last)?.[1] ?? 'there';
    if (steps === 0) return { calls: [{ name: 'read_file', input: { path: '/outreach/confirm-friday-bookings/results.csv' } }] };
    if (steps === 1) {
      privacy.push(results.at(-1) ?? '');
      return { calls: [{ name: 'record_result', input: { outcome: 'completed', answers: { coming: true }, summary: 'Yes, still coming Friday.' } }, { name: 'send_text_message', input: { body: `Thanks ${first}, see you Friday!` } }] };
    }
    return { text: 'Recorded and replied.' };
  }
  asked.runner.push({ tools, last });
  if (/\[OAIY\] Outreach "[^"]+" is finished/.test(last)) {
    reports.push(last);
    const name = /Outreach "([^"]+)"/.exec(last)[1];
    return { text: `"${name}" is done. I have the results; tell me if you want anything changed.` };
  }
  const start = (input) => (steps === 0 ? { calls: [{ name: 'start_outreach', input }] } : { text: /declined/.test(results.at(-1) ?? '') ? 'Okay, I will leave it.' : 'Started: I will work through the list and report back here.' });
  if (/call these people/i.test(last)) return start(CALLS);
  if (/text these people/i.test(last)) return start(TEXTS);
  if (/check on Hal/i.test(last)) return start(RELOAD);
  if (/remind Gus/i.test(last)) return start(CLOSED);
  return { text: 'Okay.' };
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
  res.end(chunks(step.text ?? '') + end(step.calls, ++served));
});
await new Promise((r) => model.listen(0, '127.0.0.1', r));
const modelUrl = `http://127.0.0.1:${model.address().port}`;

// --- a fake OAIY Desktop -----------------------------------------------------------------
const TOKEN = 'outreach';
const MODULES = {
  revision: 1,
  modules: [
    { id: 'phone', name: 'Phone', enabled: true, provider: { pluginId: 'aokie', name: 'AI Receptionist', state: 'running', declared: true } },
    { id: 'calendar', name: 'Calendar', enabled: true, provider: { pluginId: 'aokie', name: 'AI Receptionist', state: 'running', declared: true } },
  ],
  warnings: [],
};
const aokie = { outboundEnabled: false, quietHoursStart: 0, quietHoursEnd: 0, maxDailyDials: 20, realtimeVoiceMode: 'desktop_realtime', realtimeVoiceEndpoint: 'http://127.0.0.1:17972/api/ai/providers/oaiy/realtime', acceptPattern: '', blockedNumbers: '', rejectPrivate: false };
const voiceClients = new Set();
let liveCalls = [];
const bridgeEvents = [];
const commands = [];
const spoken = [];
const finished = [];
let dialN = 0;
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
  if (path === '/api/health') return json({ product: 'oaiy-desktop', protocol: 'oaiy-bridge/1', version: 'outreach' });
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
  if (path.startsWith('/api/bridge/leases/')) return json({ granted: true, holder: JSON.parse(body || '{}').holder ?? '' });
  if (path.startsWith('/api/bridge/connectors/')) {
    const { command, payload, idempotencyKey } = JSON.parse(body || '{}');
    const ok = (data) => json({ ok: true, result: { ok: true, data } });
    if (command === 'phone.status') return ok({ connected: true });
    if (command === 'settings.get') return payload?.key ? ok({ key: payload.key, value: aokie[payload.key] ?? null }) : ok({ settings: { ...aokie } });
    commands.push({ command, payload, key: idempotencyKey });
    if (command === 'settings.set') {
      Object.assign(aokie, payload);
      return ok({ saved: true });
    }
    if (command === 'call.dial') {
      dialN++;
      return ok({ accepted: true, queued: true, operationId: `op_${dialN}`, via: 'radio', callId: `call_${dialN}`, to: payload.number, dialsToday: dialN, maxDailyDials: aokie.maxDailyDials });
    }
    if (command === 'sms.send') return ok({ messageId: payload.messageId ?? `m${commands.length}`, to: payload.to, status: 'queued', via: 'radio' });
    return ok({ accepted: true });
  }
  if (path === '/api/bridge/flows') return json({ flows: [] });
  if (path === '/api/plugins') return json({ plugins: [{ id: 'aokie', state: 'running' }] });
  if (path.startsWith('/api/calendar')) return json({ available: true, settings: {}, appointments: [], now: new Date().toISOString() });
  if (path.startsWith('/api/voice/calls/')) {
    const said = JSON.parse(body || '{}');
    if (path.endsWith('/say')) spoken.push(said.text);
    if (path.endsWith('/finish')) finished.push(said.goodbye);
    return json({ result: { ok: true, output: {} } });
  }
  if (path === '/api/voice/callers') return json({});
  if (path === '/api/mcp') return json({ jsonrpc: '2.0', id: JSON.parse(body || '{}').id ?? null, result: { tools: [] } });
  return json({ error: { message: 'not here' } }, 404);
});
await new Promise((r) => desktopServer.listen(0, '127.0.0.1', r));
const desktopUrl = `http://127.0.0.1:${desktopServer.address().port}`;
const voice = (event) => {
  for (const c of voiceClients) c.write(`data: ${JSON.stringify(event)}\n\n`);
};
let eventN = 0;
const bridge = (name, data, correlationId = '') => bridgeEvents.push({ name, source: 'aokie', correlationId, idempotencyKey: `k${++eventN}`, occurredAt: new Date().toISOString(), data });
const dials = () => commands.filter((c) => c.command === 'call.dial');
const texts = () => commands.filter((c) => c.command === 'sms.send');

// --- the app -------------------------------------------------------------------------------
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
async function shoot(page, scene, { clip, before } = {}) {
  if (!shotsDir) return;
  for (const theme of ['light', 'dark']) {
    await page.evaluate((t) => window.__oaiySetTheme?.(t), theme);
    await wait(350);
    if (before) await before();
    const file = join(shotsDir, `outreach-${scene}-${theme}-1058.png`);
    await page.screenshot({ path: file, ...(clip ? { clip: await clip() } : {}) });
    shots.push(file);
  }
  await page.evaluate(() => window.__oaiySetTheme?.('light'));
}
const header = (page) => async () => {
  const box = await page.$eval('header.topbar', (el) => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; });
  return { ...box, height: Math.ceil(box.height) + 1 };
};

/** How far outreach's clock has been moved on (put back after a reload, which starts it afresh). */
let skewed = 0;
/** Move outreach's clock on (a gap, a deadline) and have it look now. */
const tick = (page, ms = 0) => {
  skewed += ms;
  return page.evaluate((ms) => window.__oaiyOutreachTick(ms), ms);
};
/** The agent shown is not working. */
const idle = (page) => page.waitForFunction(() => !document.querySelector('section.chat')?.classList.contains('busy'), { timeout: 20_000 });
/** The person's own conversation (the runner's), shown. */
async function showOwn(page) {
  await page.evaluate(() => {
    const button = document.querySelector('.chat .combo-button');
    if (button?.getAttribute('aria-expanded') !== 'true') button?.click();
  });
  await wait(200);
  await page.evaluate(() => [...document.querySelectorAll('.chat .combo-option')].find((o) => o.dataset.kind === 'runner' || o.dataset.kind === 'project')?.click());
  await wait(500);
}
/** Someone's conversation, shown. */
async function showPerson(page, name) {
  await page.evaluate(() => {
    const button = document.querySelector('.chat .combo-button');
    if (button?.getAttribute('aria-expanded') !== 'true') button?.click();
  });
  await wait(200);
  await page.evaluate((name) => [...document.querySelectorAll('.chat .combo-option')].find((o) => o.querySelector('.combo-name')?.textContent.startsWith(name))?.click(), name);
  await wait(500);
}
async function say(page, text) {
  await idle(page);
  await page.focus('.chat-input');
  await page.keyboard.type(text);
  await page.keyboard.press('Enter');
}
const log = (page) => page.evaluate(() => document.querySelector('.chat-log')?.textContent ?? '');
/** A file of the Front desk, as it is kept in this browser. */
const deskFile = (page, path) => page.evaluate(async (path) => {
  try {
    let dir = await (await (await navigator.storage.getDirectory()).getDirectoryHandle('front-desk')).getDirectoryHandle('files');
    const parts = path.split('/').filter(Boolean);
    for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p);
    return await (await (await dir.getFileHandle(parts.at(-1))).getFile()).text();
  } catch {
    return null;
  }
}, path);
const campaigns = (page) => page.evaluate(async () => {
  try {
    const dir = await (await (await navigator.storage.getDirectory()).getDirectoryHandle('front-desk')).getDirectoryHandle('outreach');
    const ids = JSON.parse(await (await (await dir.getFileHandle('index.json')).getFile()).text());
    const out = [];
    for (const id of ids) out.push(JSON.parse(await (await (await dir.getFileHandle(`${id}.json`)).getFile()).text()));
    return out;
  } catch {
    return [];
  }
});
/** One call, as the phone and the desktop's voice tell it: answered, the caller's words, then its end. */
async function answered(callId, from, greeting) {
  bridge('aokie.call.outbound.dialing', { callId, to: from }, callId);
  bridge('aokie.call.answered', { at: new Date().toISOString() }, callId);
  liveCalls = [callId];
  voice({ type: 'call.started', callId, from, direction: 'outbound', greeting, purpose: 'x' });
}
function hungUp(callId, outcome, reason) {
  liveCalls = [];
  voice({ type: 'call.ended', callId, reason });
  bridge('aokie.call.ended', { callId, outcome, reason, direction: 'outbound', from: '' }, callId);
}

let page;
try {
  page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  await page.setViewport({ width: 1058, height: 688 });
  await page.goto(`${base}/tests/e2e/harness.html`);
  await page.evaluate(async (modelUrl) => {
    const P = await import('/src/vfs/projects.ts');
    const ST = await import('/src/settings.ts');
    const desk = await P.OpenProject.openFrontDesk();
    desk.vfs.writeFile('/knowledge/services.md', '# Services\n\n- Lawn mowing, from $45\n', { parents: true });
    await desk.flush();
    await P.createProject('Garden notes');
    await ST.saveProviders([{ id: 'model', type: 'local', serverKind: 'other', name: 'OAIY', baseUrl: modelUrl, modelId: 'Qwen3.8-Flash-Next', apiKey: '' }], 'model');
    await ST.saveMessages({ ...ST.DEFAULT_MESSAGE_SETTINGS, calls: true, answer: false, country: 'AU' });
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

  // ---- 1. calls ------------------------------------------------------------------------
  await check('the runner is asked to call a list: start_outreach, and the person is asked once, in a dialog that shows the plan', async () => {
    await say(page, 'Please call these people and confirm Friday.');
    await page.waitForSelector('dialog.modal[open]', { timeout: 20_000 });
    const shown = await page.evaluate(() => {
      const d = document.querySelector('dialog.modal[open]');
      return { title: d.querySelector('h2')?.textContent, text: d.textContent, ok: d.querySelector('button.primary')?.textContent };
    });
    expect(shown.title === 'Call 3 people?' && shown.ok === 'Start calling', JSON.stringify(shown));
    expect(shown.text.includes("Hi Jane, it's Greenleaf Lawns about your lawn mow on Fri 2 Oct 10:30. Have you got a minute?"), 'the first opening line is not shown');
    expect(/Old number · 0412 000: not a full phone number/.test(shown.text) && /1 duplicate merged/.test(shown.text), 'the skipped and merged are not shown');
    expect(/This also turns on outbound calling on the phone\./.test(shown.text) && /Only contact people who expect to hear from you/.test(shown.text), 'the notes are missing');
    expect(/the phone allows 20 calls a day, shared with call backs, so about 1 day/.test(shown.text), 'the timing is missing');
    expect(dials().length === 0, 'it dialled before it was approved');
    expect(asked.runner.some((r) => r.tools.includes('start_outreach')), 'start_outreach was not offered to the runner');
  });
  await shoot(page, 'confirm');

  await check('approved: outbound calling is turned on, and #1 is dialled at once with the filled opening line; the header shows the calls', async () => {
    await page.click('dialog.modal[open] button.primary');
    await until('the first dial', () => dials().length === 1);
    expect(commands.some((c) => c.command === 'settings.set' && c.payload.outboundEnabled === true), 'outbound calling was not turned on');
    const d = dials()[0];
    expect(d.payload.number === '+61412345678' && d.payload.openingLine === "Hi Jane, it's Greenleaf Lawns about your lawn mow on Fri 2 Oct 10:30. Have you got a minute?" && /:p1:1$/.test(d.key), JSON.stringify(d));
    await page.waitForFunction(() => /Started: I will work through the list/.test(document.querySelector('.chat-log')?.textContent ?? ''), { timeout: 10_000 });
    await page.waitForFunction(() => !document.querySelector('.chip.outreach')?.hidden, { timeout: 10_000 });
    const chip = await page.$eval('.chip.outreach', (el) => el.textContent);
    expect(/^Calling 0\/3/.test(chip), chip);
  });

  await check("#1 answers and confirms: record_result, one goodbye (the agent's own words), then the phone hangs up; \"Done.\" after it is a quiet note", async () => {
    await answered('call_1', '+61412345678', "Hi Jane, it's Greenleaf Lawns about your lawn mow on Fri 2 Oct 10:30. Have you got a minute?");
    await page.waitForFunction(() => !document.querySelector('.call-live')?.hidden, { timeout: 10_000 });
    await wait(500);
    voice({ type: 'call.caller', callId: 'call_1', text: "Yes, I'm still coming on Friday.", startMs: 6_000, endMs: 8_000 });
    await until('the goodbye', () => finished.length === 1);
    expect(finished[0] === 'Lovely, thanks Jane, see you Friday.', JSON.stringify(finished));
    expect(!spoken.some((s) => /Lovely|welcome|Done/.test(s)), JSON.stringify(spoken));
    const call = asked.call[0];
    expect(call.tools.includes('record_result') && call.tools.includes('end_call') && !call.tools.includes('start_outreach'), JSON.stringify(call.tools));
    expect(/This is a call YOU placed for your person's outreach "Confirm Friday bookings"/.test(call.system), 'the objective is not in the call\'s instructions');
    await page.waitForFunction(() => [...document.querySelectorAll('.chat-feed .msg.unsaid')].some((n) => /Done\./.test(n.textContent)), { timeout: 10_000 });
    hungUp('call_1', 'completed', 'agent_hangup');
    await until('Jane done', async () => (await campaigns(page))[0]?.people[0]?.state === 'done');
    const c = (await campaigns(page))[0];
    expect(c.people[0].outcome === 'completed' && c.people[0].answers.coming === true && c.people[0].fields.gate === '4821', JSON.stringify(c.people[0]));
  });
  // Her conversation: the outreach call, its recorded result, and the note never said.
  await wait(600);
  await shoot(page, 'conversation', {
    before: async () => {
      await page.evaluate(() => {
        // The tools it used opened (the result it recorded), and the log from her answer to the call's end.
        document.querySelectorAll('.chat-log details.tool-group').forEach((g) => { g.open = true; });
        const log = document.querySelector('.chat-log');
        const said = [...document.querySelectorAll('.chat-feed .msg.incoming.caller')].at(-1);
        if (said) log.scrollTop += said.getBoundingClientRect().top - log.getBoundingClientRect().top - 40;
      });
      await wait(250);
    },
  });

  await check('#2 does not answer: rung again after the gap, with the next key; #3 is a voicemail, hung up on without a word', async () => {
    await tick(page, 61_000);
    await until('the second dial', () => dials().length === 2);
    expect(dials()[1].payload.number === '+61413000111' && /:p2:1$/.test(dials()[1].key), JSON.stringify(dials()[1]));
    bridge('aokie.call.ringing', { at: new Date().toISOString() }, 'call_2');
    bridge('aokie.call.ended', { callId: 'call_2', outcome: 'no_answer', reason: 'no_answer', direction: 'outbound' }, 'call_2');
    await until('Bob waits for a retry', async () => (await campaigns(page))[0]?.people[1]?.state === 'waiting');
    await tick(page, 61_000);
    await until('the third dial', () => dials().length === 3);
    expect(dials()[2].payload.number === '+61414222333', JSON.stringify(dials()[2]));
    await answered('call_3', '+61414222333', "Hi Cara, it's Greenleaf Lawns about your lawn mow on Fri 2 Oct 3:00. Have you got a minute?");
    await wait(600);
    voice({ type: 'call.caller', callId: 'call_3', text: "Hi, you've reached Cara. Please leave a message after the tone.", startMs: 5_000, endMs: 9_000 });
    await until('the silent hang-up', () => commands.some((c) => c.command === 'call.hangup'));
    const hangup = commands.find((c) => c.command === 'call.hangup');
    expect(hangup.payload.callId === 'call_3' && /^oaiy:outreach-hangup:call_3$/.test(hangup.key), JSON.stringify(hangup));
    expect(finished.length === 1, `a goodbye was said to the voicemail: ${JSON.stringify(finished)}`);
    hungUp('call_3', 'completed', 'local_hangup');
    await until('Cara waits for a retry', async () => (await campaigns(page))[0]?.people[2]?.state === 'waiting');
  });

  await check('the runner shows the live card: a row a person, where each is, and their answers', async () => {
    await showOwn(page);
    await page.waitForSelector('.outreach-card .outreach-person', { timeout: 10_000 });
    const card = await page.evaluate(() => ({
      title: document.querySelector('.outreach-card .outreach-title')?.textContent,
      rows: [...document.querySelectorAll('.outreach-card .outreach-person')].map((r) => `${r.querySelector('.outreach-person-name')?.firstChild?.textContent} | ${r.querySelector('.outreach-person-state')?.textContent} | ${r.querySelector('.outreach-person-answers')?.textContent}`),
      buttons: [...document.querySelectorAll('.outreach-card .outreach-actions button')].map((b) => b.textContent),
      lines: [...document.querySelectorAll('.chat-feed .outreach-line')].map((l) => l.textContent),
    }));
    expect(card.title === 'Confirm Friday bookings', JSON.stringify(card));
    expect(/^Jane Smith \| 0412 345 678 · completed \| coming: yes/.test(card.rows[0]) && /^Bob Jones \| 0413 000 111 · again at/.test(card.rows[1]) && /^Cara Lee \| 0414 222 333 · again at/.test(card.rows[2]), JSON.stringify(card.rows));
    expect(card.buttons.includes('Pause') && card.buttons.includes('Stop') && card.buttons.includes('/outreach/confirm-friday-bookings/results.md'), JSON.stringify(card.buttons));
    expect(card.lines.some((l) => /Jane Smith: completed\. coming: yes\./.test(l)), JSON.stringify(card.lines));
  });
  await shoot(page, 'card', {
    before: async () => {
      await page.evaluate(() => {
        const log = document.querySelector('.chat-log');
        const card = document.querySelector('.outreach-card');
        if (card) log.scrollTop += card.getBoundingClientRect().top - log.getBoundingClientRect().top - 90;
      });
      await wait(200);
    },
  });
  await shoot(page, 'chip', { clip: header(page) });

  await check('the retries: #2 rung again (key :2) and not answered, #3 a voicemail again; then the report reaches the runner, with the results in the Front desk', async () => {
    await tick(page, 31 * 60_000);
    await until('the retry of #2', () => dials().length === 4);
    expect(dials()[3].payload.number === '+61413000111' && /:p2:2$/.test(dials()[3].key), JSON.stringify(dials()[3]));
    bridge('aokie.call.ended', { callId: 'call_4', outcome: 'no_answer', reason: 'no_answer', direction: 'outbound' }, 'call_4');
    await until('Bob done', async () => (await campaigns(page))[0]?.people[1]?.state === 'done');
    await tick(page, 61_000);
    await until('the retry of #3', () => dials().length === 5);
    await answered('call_5', '+61414222333', 'Hi Cara');
    await wait(600);
    voice({ type: 'call.caller', callId: 'call_5', text: 'Please leave a message after the tone.', startMs: 4_000, endMs: 6_000 });
    await until('the second silent hang-up', () => commands.filter((c) => c.command === 'call.hangup').length === 2);
    hungUp('call_5', 'completed', 'local_hangup');
    await until('the report', () => reports.length === 1, 20_000);
    const c = (await campaigns(page))[0];
    expect(c.state === 'done' && c.people.map((p) => p.outcome).join(',') === 'completed,no_answer,voicemail', JSON.stringify(c.people.map((p) => [p.state, p.outcome])));
    expect(/Reached 1 of 3\./.test(reports[0]) && /Jane Smith \| completed \| coming: yes \| Still coming on Friday at 10:30\./.test(reports[0]) && /Move anyone who can't come/.test(reports[0]), reports[0]);
    const md = await deskFile(page, '/outreach/confirm-friday-bookings/results.md');
    expect(md && /\| Jane Smith \| 0412 345 678 \| completed \| yes \|/.test(md) && /Old number/.test(md), md);
    expect(/^name,number,outcome,coming,new_time,summary/.test(await deskFile(page, '/outreach/confirm-friday-bookings/results.csv')), 'no csv');
  });

  await check('the results open in the editor from the card', async () => {
    await showOwn(page);
    await page.waitForSelector('.outreach-card .outreach-path', { timeout: 10_000 });
    await page.evaluate(() => document.querySelector('.outreach-card .outreach-path').click());
    await page.waitForFunction(() => /\| Jane Smith \|/.test(document.querySelector('.cm-content')?.textContent ?? ''), { timeout: 10_000 });
  });
  await shoot(page, 'results');

  // ---- 2. texts ------------------------------------------------------------------------
  await check('texts: two sent with their own message ids; a reply answered and recorded (while answering is off); STOP kept and never answered', async () => {
    await showOwn(page);
    await say(page, 'Now text these people about Friday.');
    await page.waitForSelector('dialog.modal[open]', { timeout: 20_000 });
    const title = await page.$eval('dialog.modal[open] h2', (el) => el.textContent);
    expect(title === 'Text 2 people?', title);
    await page.click('dialog.modal[open] button.primary');
    await until('the first text', () => texts().length === 1);
    await tick(page, 21_000);
    await until('the second text', () => texts().length === 2);
    const [dee, eve] = texts();
    expect(dee.payload.to === '+61415000101' && dee.payload.body === 'Hi Dee, Greenleaf Lawns here: still right for Friday? Reply YES or NO.' && /^oaiy-out\.out-[\w-]+\.p1\.1$/.test(dee.payload.messageId) && dee.key === `oaiy:outreach-sms:${dee.payload.messageId}`, JSON.stringify(dee));
    bridge('aokie.sms.sent', { messageId: dee.payload.messageId, to: dee.payload.to }, dee.payload.messageId);
    bridge('aokie.sms.sent', { messageId: eve.payload.messageId, to: eve.payload.to }, eve.payload.messageId);
    bridge('aokie.sms.received', { from: '+61415000101', name: 'Dee', body: 'Yes, see you Friday', handle: 'h1' }, 'c-h1');
    await until('the reply to Dee', () => texts().length === 3, 20_000);
    expect(texts()[2].payload.to === '+61415000101' && texts()[2].payload.body === 'Thanks Dee, see you Friday!', JSON.stringify(texts()[2]));
    const sms = asked.sms[0];
    expect(sms.tools.includes('record_result') && !sms.tools.includes('start_outreach') && /You texted them for your person's outreach "Friday reminders"/.test(sms.system), JSON.stringify(sms.tools));
    bridge('aokie.sms.received', { from: '+61415000102', name: 'Eve', body: 'STOP', handle: 'h2' }, 'c-h2');
    await until('Eve opted out', async () => (await campaigns(page)).find((c) => c.name === 'Friday reminders')?.people[1]?.outcome === 'opted_out');
    await wait(1000);
    expect(texts().length === 3 && asked.sms.every((a) => !/STOP/.test(a.last)), 'STOP was answered');
    await until('the texts report', () => reports.length === 2, 20_000);
    expect(/Friday reminders" is finished/.test(reports[1]) && /Dee Park \| completed \| coming: yes/.test(reports[1]) && /Eve Quinn \| opted out/.test(reports[1]), reports[1]);
  });

  await check("the phone's agents cannot read /outreach: a text's read_file of the results is refused", async () => {
    expect(privacy.length >= 1 && privacy.every((r) => /ENOENT|no such file/.test(r)), JSON.stringify(privacy));
  });

  // ---- 3. a reload in the middle -----------------------------------------------------------
  await check('a reload during a call: its end, which came while the page reloaded, is settled, and no one is rung twice', async () => {
    await showOwn(page);
    await say(page, 'Please check on Hal.');
    await page.waitForSelector('dialog.modal[open]', { timeout: 20_000 });
    await page.click('dialog.modal[open] button.primary');
    await tick(page, 61_000);
    await until('the dial to Hal', () => dials().some((d) => d.payload.number === '+61416000201'), 20_000);
    const before = dials().length;
    const hal = dials().at(-1);
    const callId = `call_${dials().length}`;
    await page.reload();
    // While it reloads: the phone says the call was not answered.
    bridge('aokie.call.ended', { callId, outcome: 'no_answer', reason: 'no_answer', direction: 'outbound' }, callId);
    await page.waitForSelector('.tree-row', { timeout: 60_000 });
    await until('Hal settled', async () => (await campaigns(page)).find((c) => c.name === 'Hal check')?.people[0]?.outcome === 'no_answer', 20_000);
    // The page's clock for outreach starts afresh: put it back where the test had moved it.
    await page.waitForFunction(() => typeof window.__oaiyOutreachTick === 'function', { timeout: 10_000 });
    const back = skewed;
    skewed = 0;
    await tick(page, back);
    await tick(page, 61_000);
    await wait(800);
    expect(dials().length === before, `rung again: ${JSON.stringify(dials().slice(before))}`);
    expect(/:p1:1$/.test(hal.key), hal.key);
    await until('the Hal report', () => reports.some((r) => /Hal check" is finished/.test(r)), 20_000);
  });

  // ---- 4. a report that waits for the Front desk ------------------------------------------
  await check('a report that finishes while the Front desk is closed waits there, with the chip saying so, and runs when it opens', async () => {
    await page.waitForFunction(() => document.querySelector('.project-select')?.value === 'front-desk', { timeout: 20_000 });
    await showOwn(page);
    await say(page, 'Please remind Gus about Friday.');
    await page.waitForSelector('dialog.modal[open]', { timeout: 20_000 });
    await page.click('dialog.modal[open] button.primary');
    await tick(page, 21_000);
    await until('the text to Gus', () => texts().some((t) => t.payload.to === '+61417000301'), 20_000);
    await page.waitForFunction(() => /Started: I will work through the list/.test([...document.querySelectorAll('.chat-feed .msg.assistant')].at(-1)?.textContent ?? ''), { timeout: 10_000 });
    await wait(500);
    // Another project opens: the Front desk is closed.
    await page.evaluate(() => {
      const select = document.querySelector('.project-select');
      const other = [...select.options].find((o) => o.textContent.includes('Garden notes'));
      select.value = other.value;
      select.dispatchEvent(new Event('change'));
    });
    await page.waitForFunction(() => document.querySelector('.project-select')?.selectedOptions[0]?.textContent.includes('Garden notes'), { timeout: 10_000 });
    const gus = texts().find((t) => t.payload.to === '+61417000301');
    bridge('aokie.sms.sent', { messageId: gus.payload.messageId, to: gus.payload.to }, gus.payload.messageId);
    bridge('aokie.sms.received', { from: '+61417000301', name: 'Gus', body: 'Yes all good', handle: 'h3' }, 'c-h3');
    await until('Gus recorded', async () => (await campaigns(page)).find((c) => c.name === 'Gus reminder')?.state === 'done', 20_000);
    await page.waitForFunction(() => /Report waiting/.test(document.querySelector('.chip.outreach')?.textContent ?? ''), { timeout: 10_000 });
    const waiting = reports.filter((r) => /Gus reminder/.test(r)).length;
    expect(waiting === 0, 'the report ran while the Front desk was closed');
    expect(/"Gus reminder" finished: 1 of 1 done \(1 completed\)\. Its report waits in Front desk/.test(await log(page)), 'the notice is missing');
  });
  await shoot(page, 'chip-report', { clip: header(page) });

  await check('the chip opens the Front desk, and the report runs there', async () => {
    await page.click('.chip.outreach');
    await page.waitForFunction(() => document.querySelector('.project-select')?.value === 'front-desk', { timeout: 10_000 });
    await until('the Gus report', () => reports.some((r) => /Gus reminder" is finished/.test(r)), 20_000);
    await page.waitForFunction(() => [...document.querySelectorAll('.chat-feed .outreach-report')].some((r) => /Gus reminder/.test(r.textContent)), { timeout: 10_000 });
    await page.waitForFunction(() => document.querySelector('.chip.outreach')?.hidden, { timeout: 10_000 });
  });
  await shoot(page, 'report');

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
  console.error(`outreach: ${failures.length} failed`);
  process.exit(1);
}
console.log('outreach: all passed');
process.exit(0);
