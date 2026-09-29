// Screenshots of the Agent page, for its design: OAIY's window (a fake OAIY
// Desktop with the phone on, and a live call on demand) and a browser tab, a
// scripted model, and projects and conversations seeded through the app's own
// storage code. Nothing here reaches a real desktop or model.
//   node tests/e2e/shots-agentui.mjs <out-dir> [prefix]      (SCENES=runner,call… to pick)
import { existsSync, mkdirSync } from 'node:fs';
import { createServer as createHttpServer } from 'node:http';
import { join } from 'node:path';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const out = process.argv[2] ?? 'shots';
const prefix = process.argv[3] ?? 'agentui';
const only = process.env.SCENES ? process.env.SCENES.split(',') : null;
const PORT = Number(process.env.PORT ?? 5390);
const SIZES = (process.env.SIZES ?? '1058x688,1440x900').split(',').map((s) => s.split('x').map(Number));
const THEMES = (process.env.THEMES ?? 'light,dark').split(',');
mkdirSync(out, { recursive: true });
const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium'].filter(Boolean).find((p) => existsSync(p));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const cors = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*', 'Access-Control-Allow-Methods': '*', 'Access-Control-Allow-Private-Network': 'true' };
const readBody = (req) => new Promise((resolve) => {
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', () => resolve(body));
});

// --- the scripted model --------------------------------------------------------
const CHART = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100"><rect x="10" y="38" width="30" height="62" fill="#2457e6"/><rect x="50" y="52" width="30" height="48" fill="#2457e6"/></svg>\n';
const held = [];
let calls = 0;
function sse(step) {
  const n = ++calls;
  const events = [];
  for (const piece of step.text?.match(/.{1,8}/gs) ?? []) events.push({ choices: [{ index: 0, delta: { content: piece } }] });
  (step.calls ?? []).forEach((call, i) => {
    events.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: `call_${n}_${i}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.input) } }] } }] });
  });
  events.push({ choices: [{ index: 0, delta: {}, finish_reason: step.calls ? 'tool_calls' : 'stop' }] });
  return events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n';
}
const textOf = (m) => (typeof m.content === 'string' ? m.content : (m.content ?? []).map((p) => p.text ?? '').join(''));
function answer(body) {
  if ((body.max_tokens ?? body.max_completion_tokens) === 1) return { text: '' };
  const system = textOf(body.messages.find((m) => m.role === 'system') ?? { content: '' });
  const last = textOf([...body.messages].reverse().find((m) => m.role === 'user') ?? { content: '' });
  if (/live phone call/.test(system)) {
    if (/Thursday/i.test(last)) return { text: "Thursday morning is open. Would nine o'clock suit you?" };
    if (/hedge/i.test(last)) return { text: 'Sure, Dave, we can help with that. Which day suits you best?' };
    return { text: '' };
  }
  const at = body.messages.map((m) => m.role === 'user' && /chart it/i.test(textOf(m))).lastIndexOf(true);
  if (at >= 0) {
    const steps = body.messages.slice(at + 1).filter((m) => m.role === 'assistant').length;
    if (steps === 0) return { text: "Sure. I'll read the numbers again, then draw a bar chart.", calls: [{ name: 'read_file', input: { path: 'data/weather.csv' } }, { name: 'glob', input: { pattern: '**/*.svg' } }] };
    if (steps === 1) return { calls: [{ name: 'write_file', input: { path: 'chart.svg', content: CHART } }] };
    if (steps === 2) return { calls: [{ name: 'sandbox_shell', input: { command: 'ls -la && wc -c chart.svg' } }] };
    return 'hold';
  }
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
  const body = JSON.parse(await readBody(req));
  const step = answer(body);
  res.setHeader('content-type', 'text/event-stream');
  if (step === 'hold') {
    // The reply begins (a code block half written), then waits: the chat shows the agent at work.
    const begun = "Here is the chart: **chart.svg**, one bar a city, warmest first. To draw it again:\n\n```python\nimport csv\n\nrows = list(csv.DictReader(open('data/weather.csv')))\nrows.sort(key=lambda r: -int(r['temp_c']))\nfor i, r in enumerate(rows):\n    print(i, r['city'], r['temp_c'])\n";
    const pieces = begun.match(/.{1,8}/gs).map((piece) => `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: piece } }] })}\n\n`);
    res.write(pieces.join(''));
    held.push(() => res.end(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: '```\n\nOslo is the shortest bar.' } }] })}\n\ndata: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`));
    return;
  }
  res.end(sse(step));
});
await new Promise((r) => model.listen(0, '127.0.0.1', r));
const modelUrl = `http://127.0.0.1:${model.address().port}`;

// --- a fake OAIY Desktop: the phone and the calendar on, calls on demand --------
const MODULES = {
  revision: 1,
  modules: [
    { id: 'phone', name: 'Phone', enabled: true, provider: { pluginId: 'aokie', name: 'Aokie', state: 'running', declared: true } },
    { id: 'calendar', name: 'Calendar', enabled: true, provider: { pluginId: 'aokie', name: 'Aokie', state: 'running', declared: true } },
  ],
  warnings: [],
};
const voiceClients = new Set();
let liveCalls = [];
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
  if (path === '/api/health') return json({ product: 'oaiy-desktop', protocol: 'oaiy-bridge/1', version: 'shots' });
  if (path === '/api/modules') return json(MODULES);
  if (path === '/api/modules/events') return stream([MODULES]);
  if (path === '/api/agent/events') return stream([]);
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
  if (path.startsWith('/api/calendar')) return json({ available: true, settings: {}, appointments: [], now: new Date().toISOString() });
  if (path.startsWith('/api/voice/calls/')) return json({ result: { ok: true, output: {} } });
  if (path === '/api/voice/callers') return json({});
  return json({ error: { message: 'not here' } }, 404);
});
await new Promise((r) => desktopServer.listen(0, '127.0.0.1', r));
const desktopUrl = `http://127.0.0.1:${desktopServer.address().port}`;
const voice = (event) => {
  for (const c of voiceClients) c.write(`data: ${JSON.stringify(event)}\n\n`);
};

// --- the app ----------------------------------------------------------------------
const server = await createServer({ server: { port: PORT, strictPort: true, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const base = `http://127.0.0.1:${PORT}`;
const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-first-run'] });

/** Projects, files and conversations, through the app's own storage (from a page of the same origin that does not run the app). */
async function seed(page) {
  await page.goto(`${base}/tests/e2e/harness.html`);
  return page.evaluate(async (modelUrl, desktopUrl) => {
    const P = await import('/src/vfs/projects.ts');
    const S = await import('/src/sessions.ts');
    const ST = await import('/src/settings.ts');
    const now = Date.now();
    const min = 60_000;
    const hour = 60 * min;
    const day = 24 * hour;

    // A project of the person's own: folders, code, and a conversation with tools, markdown and code.
    const meta = await P.createProject('Weather report');
    const project = await P.OpenProject.open(meta);
    const files = {
      'README.md': '# Weather report\n\nA small study of city temperatures.\n\n- `data/` the readings\n- `src/` the scripts\n',
      'data/weather.csv': 'city,temp_c\nParis,18\nRome,24\nOslo,9\nCairo,31\nLima,17\n',
      'data/stations.json': JSON.stringify({ stations: [{ id: 'PAR', city: 'Paris' }, { id: 'ROM', city: 'Rome' }] }, null, 2) + '\n',
      'src/summary.py': 'import csv\nimport statistics\n\n\ndef load(path="data/weather.csv"):\n    """Read the readings: one city and its temperature a row."""\n    with open(path) as f:\n        return [(row["city"], int(row["temp_c"])) for row in csv.DictReader(f)]\n\n\nrows = load()\ntemps = [t for _, t in rows]\nwarmest = max(rows, key=lambda r: r[1])\nprint(f"mean {statistics.mean(temps):.1f}")\nprint(f"warmest {warmest[0]} {warmest[1]}")\n',
      'src/chart.js': "const bars = (rows) => rows.map(([city, t], i) => `<rect x=\"${i * 40}\" height=\"${t * 2}\"/>`);\nexport default bars;\n",
      'notes/todo.md': '- [ ] add humidity\n- [x] mean temperature\n',
      'index.html': '<!doctype html>\n<title>Weather</title>\n<link rel="stylesheet" href="styles.css">\n<h1>Weather</h1>\n',
      'styles.css': 'body { font-family: system-ui; margin: 2rem; }\n',
      'assets/logo.svg': '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="#2457e6"/></svg>\n',
    };
    for (const [path, text] of Object.entries(files)) project.vfs.writeFile(`/${path}`, text, { parents: true });
    await project.flush();
    const code = "import statistics\nrows = [l.split(',') for l in open('data/weather.csv').read().splitlines()[1:]]\ntemps = [int(t) for _, t in rows]\nprint('mean', statistics.mean(temps))\nprint('warmest', max(rows, key=lambda r: int(r[1])))";
    await project.saveChat([
      { role: 'user', text: 'Summarise the weather data: the warmest city, the mean, and a small table. Save it as report.md.' },
      { role: 'assistant', text: "I'll look at the data first.", calls: [{ id: 'w1', name: 'list_files', input: { path: 'data' } }, { id: 'w2', name: 'read_file', input: { path: 'data/weather.csv' } }] },
      { role: 'tool', results: [{ id: 'w1', name: 'list_files', content: 'data/stations.json  96 bytes\ndata/weather.csv  64 bytes', isError: false }, { id: 'w2', name: 'read_file', content: files['data/weather.csv'], isError: false }] },
      { role: 'assistant', text: '', calls: [{ id: 'w3', name: 'code_run', input: { language: 'python', code } }] },
      { role: 'tool', results: [{ id: 'w3', name: 'code_run', content: "exit 0\nmean 19.8\nwarmest ['Cairo', '31']", isError: false }] },
      { role: 'assistant', text: '', calls: [{ id: 'w4', name: 'sandbox_shell', input: { command: 'python src/plot.py' } }, { id: 'w5', name: 'write_file', input: { path: 'report.md', content: '# Weather report\n\nWarmest: Cairo, 31 °C. Mean: 19.8 °C.\n' } }] },
      { role: 'tool', results: [{ id: 'w4', name: 'sandbox_shell', content: "python: can't open file 'src/plot.py': No such file", isError: true }, { id: 'w5', name: 'write_file', content: 'Wrote report.md (61 bytes).', isError: false }] },
      { role: 'assistant', text: '## Weather summary\n\nThe **warmest** city is Cairo at `31 °C`, and the mean across the five cities is **19.8 °C**.\n\n| City | °C |\n|---|---:|\n| Cairo | 31 |\n| Rome | 24 |\n| Paris | 18 |\n| Lima | 17 |\n| Oslo | 9 |\n\nIt is saved in `report.md`. To work it out again:\n\n```python\nimport statistics\n\ntemps = [18, 24, 9, 31, 17]\nprint(statistics.mean(temps))  # 19.8\n```\n\n1. Oslo is the coldest, at 9 °C\n2. Two cities are above the mean', calls: [] },
    ]);

    // A long conversation (older turns are drawn a page at a time as it is scrolled).
    const longMeta = await P.createProject('Long chat');
    const long = await P.OpenProject.open(longMeta);
    long.vfs.writeFile('/notes.md', '# Notes\n', { parents: true });
    await long.flush();
    // Seven turns an exchange (two messages of the person's, two steps with a tool each, a reply): the chat's pages
    // (60 turns, from a message of the person's) then begin at the second message, so each page's edge falls
    // between two of the person's messages, which must stay one run under one header.
    const turns = [];
    for (let i = 1; i <= 30; i++) {
      turns.push({ role: 'user', text: `Request ${i}: add line ${i} to notes.md` });
      turns.push({ role: 'user', text: `And read it back, please (${i}).` });
      turns.push({ role: 'assistant', text: '', calls: [{ id: `a${i}`, name: 'append_file', input: { path: 'notes.md', content: `line ${i}\n` } }] });
      turns.push({ role: 'tool', results: [{ id: `a${i}`, name: 'append_file', content: 'Appended.', isError: false }] });
      turns.push({ role: 'assistant', text: '', calls: [{ id: `b${i}`, name: 'read_file', input: { path: 'notes.md' } }] });
      turns.push({ role: 'tool', results: [{ id: `b${i}`, name: 'read_file', content: `line ${i}`, isError: false }] });
      turns.push({ role: 'assistant', text: `Added line ${i}.`, calls: [] });
    }
    await long.saveChat(turns);

    // The Front desk: its knowledge, the runner's (empty) chat, and the phone's conversations.
    const desk = await P.OpenProject.openFrontDesk();
    desk.vfs.writeFile('/knowledge/services.md', '# Services\n\n- Lawn mowing, from $45\n- Hedge trimming, from $60\n- Garden clean-ups, quoted on site\n', { parents: true });
    desk.vfs.writeFile('/knowledge/areas.md', '# Areas we cover\n\nThe inner west and the north shore.\n', { parents: true });
    desk.vfs.writeFile('/knowledge/faq.md', '# Common questions\n\n**Do you take green waste?** Yes, at no charge.\n', { parents: true });
    desk.vfs.writeFile('/uploads/price-list.md', '# Price list\n\nMowing $45 · Hedges $60\n', { parents: true });
    await desk.flush();
    await desk.saveChat([]);
    const note = (number, name, facts = []) => ({ number, name, facts, updatedAt: now });
    const callers = [note('+61491570006', 'Lance', ['Prefers afternoons', 'Lawn mowing, fortnightly']), note('+61400111222', 'Priya Shah'), note('+61455666777', 'Dave'), note('+61488999000', 'Mia Chen')];
    await desk.saveCallers(callers);
    const greeting = 'Hi, thanks for calling Greenline Gardens. How can I help?';
    const start = (who, at, number) => ({ role: 'user', automatic: true, fresh: true, text: S.callStartNote(who, greeting, S.knownText(callers.find((c) => c.number === number)), new Date(at)) });
    const said = (text, calls = []) => ({ role: 'assistant', text, calls });
    const results = (...list) => ({ role: 'tool', results: list.map(([id, name, content, isError = false]) => ({ id, name, content, isError })) });
    const caller = (text, when, youSaid) => ({ role: 'user', text: S.callerLine(text, when, youSaid) });
    const ended = { role: 'user', automatic: true, text: '[OAIY] 📞 The call ended.' };
    const text = (title, number, body) => ({ role: 'user', text: S.textMessage(title, number, body) });
    const sms = (id, body) => [said('', [{ id, name: 'send_text_message', input: { body } }]), results([id, 'send_text_message', `Sent to them (message m_${id}).`])];

    const lanceYesterday = now - day - 2 * hour;
    const lanceToday = now - 2 * hour;
    const sessions = [
      { info: { id: 'call-61491570006', kind: 'call', key: '+61491570006', title: 'Lance', lastAt: lanceToday + 50_000, unread: 0 }, turns: [
        start('Lance (+61491570006)', lanceYesterday, '+61491570006'),
        caller('Hi, just checking you got my message about the mowing?', { startMs: 3_400 }),
        said('Yes, we did. The team has you down for next week.'),
        caller('Great, thanks. Bye.', { startMs: 11_900 }),
        said('', [{ id: 'y1', name: 'end_call', input: { goodbye: 'Bye Lance!' } }]),
        results(['y1', 'end_call', 'The goodbye is being said, then the call ends. Write nothing more.']),
        ended,
        start('Lance (+61491570006)', lanceToday, '+61491570006'),
        caller('Hi there, I was hoping to book a lawn mow.', { startMs: 4_200 }),
        said('Sure thing. What day suits you best?'),
        { role: 'user', text: [S.callerLine('mm-hmm', { startMs: 9_100, over: true }, 'Sure thing.'), S.callerLine('Next Tuesday, maybe in the afternoon?', { startMs: 11_800 })].join('\n') },
        said('Let me check.', [{ id: 'l1', name: 'lookup_business_data', input: { question: 'Free times next Tuesday afternoon' } }]),
        results(['l1', 'lookup_business_data', S.LOOKUP_ASKED]),
        { role: 'user', text: '[OAIY] The answer to your lookup "Free times next Tuesday afternoon":\n{"free": ["13:00", "15:30"]}' },
        said('Tuesday at 1 pm or 3:30 pm are free. Which would you like?'),
        caller("One o'clock is great. Oh, and it's Lance, by the way.", { startMs: 24_000, cut: true }, 'Which would you like?'),
        said('', [{ id: 'l2', name: 'remember', input: { name: 'Lance' } }, { id: 'l3', name: 'request_appointment', input: { callerName: 'Lance', service: 'Lawn mowing', date: '2026-10-06', time: '13:00', agreementPhrase: "One o'clock is great" } }]),
        results(['l2', 'remember', 'Saved.'], ['l3', 'request_appointment', '{\n "requestId": "apt_31",\n "status": "requested"\n}']),
        said("Thanks, Lance. I've requested Tuesday the 6th at 1 pm for a lawn mow; the team will confirm by text."),
        caller('Perfect, thanks. Bye!', { startMs: 41_000 }),
        said('', [{ id: 'l4', name: 'end_call', input: { goodbye: 'Bye Lance, have a great day!' } }]),
        results(['l4', 'end_call', 'The goodbye is being said, then the call ends. Write nothing more.']),
        ended,
      ] },
      { info: { id: 'sms-61491570006', kind: 'sms', key: '+61491570006', title: 'Lance', lastAt: now - 20 * min, unread: 2 }, turns: [
        text('Lance', '+61491570006', 'Hi! Can you come a bit earlier on Tuesday?'),
        said('', [{ id: 's0', name: 'calendar_free_times', input: { from: '2026-10-06', days: 1 } }]),
        results(['s0', 'calendar_free_times', '2026-10-06: 10:30, 13:00, 15:30']),
        ...sms('s1', 'Hi Lance, 10:30 am on Tuesday is free. Shall I request that instead of 1 pm?'),
        text('Lance', '+61491570006', 'Yes please'),
        text('Lance', '+61491570006', 'And can they do the edges too?'),
        ...sms('s2', "Done: I've asked for 10:30 on Tuesday, edges included. The team will confirm by text."),
        { role: 'user', text: 'Tell him the invoice goes to his email as well.' },
        ...sms('s3', 'Also, the invoice will come to your email as usual. Thanks Lance!'),
        text('Lance', '+61491570006', 'Thanks 👍'),
      ] },
      { info: { id: 'call-61400111222', kind: 'call', key: '+61400111222', title: 'Priya Shah', lastAt: now - day - 5 * hour, unread: 0 }, turns: [
        start('Priya Shah (+61400111222)', now - day - 5 * hour, '+61400111222'),
        caller('Do you do hedges on the north shore?', { startMs: 2_900 }),
        said('We do, from $60 a hedge. Would you like a time?'),
        caller('Not yet, thanks.', { startMs: 9_000 }),
        ended,
      ] },
      { info: { id: 'sms-61400333444', kind: 'sms', key: '+61400333444', title: '+61400333444', lastAt: now - 3 * day, unread: 1 }, turns: [text('+61400333444', '+61400333444', 'Is this Greenline? Do you take green waste?')] },
      { info: { id: 'sms-test', kind: 'sms', key: 'test', title: 'Test', lastAt: now - 5 * day, unread: 0 }, turns: [text('Test', 'test', 'What are your hours?'), ...sms('t1', "We're open 8 am to 5 pm, Monday to Saturday.")] },
      { info: { id: 'task-morning-summary-1', kind: 'task', key: 'Morning summary', title: 'Morning summary', lastAt: now - 6 * hour, unread: 0 }, turns: [
        { role: 'user', text: '[OAIY] Your flow "Morning summary" asks: Summarise yesterday\'s calls and texts in three short bullet points.' },
        said('- Lance booked a lawn mow for Tuesday at 1 pm\n- Priya asked about hedges on the north shore\n- One new text asks about green waste'),
      ] },
      { info: { id: 'call-61455666777', kind: 'call', key: '+61455666777', title: 'Dave', lastAt: now - 4 * day, unread: 0 }, turns: [
        start('Dave (+61455666777)', now - 4 * day, '+61455666777'),
        caller('Just ringing to say thanks for last week.', { startMs: 2_000 }),
        said("That's lovely to hear, Dave. Thanks for calling!"),
        ended,
      ] },
      { info: { id: 'sms-61488999000', kind: 'sms', key: '+61488999000', title: 'Mia Chen', lastAt: now - 2 * day, unread: 0 }, turns: [text('Mia Chen', '+61488999000', 'Can I move Friday to Saturday?'), ...sms('m1', 'Hi Mia, Saturday at 9 am is free. Shall I request it?')] },
      { info: { id: 'call-8841', kind: 'call', key: '8841', title: 'Hidden number', lastAt: now - 7 * day, unread: 0 }, turns: [start('Hidden number', now - 7 * day, '8841'), caller('Sorry, wrong number.', { startMs: 1_500 }), ended] },
      { info: { id: 'task-new-lead-triage-1', kind: 'task', key: 'New lead triage', title: 'New lead triage', lastAt: now - 9 * day, unread: 0 }, turns: [{ role: 'user', text: '[OAIY] Your flow "New lead triage" asks: Is this lead worth a call back? "Need a quote for a big garden clean-up"' }, said('Yes: a clean-up is quoted on site, so a call back is worth it.')] },
    ];
    await desk.saveSessions(sessions.map((s) => s.info));
    for (const s of sessions) await desk.saveSessionChat(s.info.id, s.turns);

    await ST.saveProviders([{ id: 'shots', type: 'local', serverKind: 'other', name: 'OAIY', baseUrl: modelUrl, modelId: 'Qwen3.8-Flash-Next', apiKey: '' }], 'shots');
    await ST.saveMessages({ ...ST.DEFAULT_MESSAGE_SETTINGS, calls: true });
    await ST.saveDesktop({ origin: desktopUrl, token: 'shots' });
    await ST.saveLastProject('front-desk');
    return { project: meta.id, long: longMeta.id };
  }, modelUrl, desktopUrl);
}

const shots = [];
const checks = [];
async function shoot(page, scene, { themes = THEMES, sizes = SIZES, mode = 'oaiy', before } = {}) {
  if (only && !only.includes(scene)) return;
  for (const theme of themes) {
    if (mode === 'oaiy') await page.evaluate((t) => window.__oaiySetTheme?.(t), theme);
    else await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: theme }]);
    for (const [width, height] of sizes) {
      await page.setViewport({ width, height });
      await wait(350);
      if (before) await before();
      const file = join(out, `${prefix}-${mode === 'oaiy' ? '' : `${mode}-`}${scene}-${theme}-${width}.png`);
      await page.screenshot({ path: file });
      shots.push(file);
    }
  }
}

/** Open the conversations' list, and (with `pick`) choose the one of that kind and name. */
async function conversation(page, kind, name) {
  await page.evaluate(() => {
    const button = document.querySelector('.chat [aria-haspopup="listbox"]');
    if (button?.getAttribute('aria-expanded') !== 'true') button?.click();
  });
  await wait(150);
  const found = await page.evaluate((kind, name) => {
    const marks = { call: '📞', sms: '💬', task: '🔀' };
    const option = [...document.querySelectorAll('.chat [role="option"]')].find((o) => (o.dataset.kind ? o.dataset.kind === kind : o.textContent.includes(marks[kind])) && o.textContent.includes(name));
    option?.click();
    return !!option;
  }, kind, name);
  if (!found) throw new Error(`no ${kind} conversation with ${name}`);
  await wait(400);
}

/** Where the chat's log is scrolled: how far from its bottom, whether "Latest" shows, and what it says. */
const logState = (page) => page.evaluate(() => {
  const log = document.querySelector('.chat-log');
  const jump = document.querySelector('.to-latest');
  return { fromBottom: Math.round(log.scrollHeight - log.scrollTop - log.clientHeight), scrollTop: Math.round(log.scrollTop), scrollHeight: log.scrollHeight, jump: !!jump && !jump.hidden, jumpText: jump?.textContent ?? '' };
});

async function closePicker(page) {
  await page.keyboard.press('Escape');
  await page.evaluate(() => document.activeElement?.blur?.());
  await wait(150);
}

try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  page.on('dialog', (d) => d.accept());
  await page.setViewport({ width: SIZES[0][0], height: SIZES[0][1] });
  const { project, long } = await seed(page);
  // OAIY's window: the desktop is given, and it says the theme.
  await page.evaluateOnNewDocument((origin) => {
    window.__OAIY_DESKTOP__ = { origin, token: 'shots' };
    window.__OAIY_THEME__ = 'light';
  }, desktopUrl);
  await page.goto(base);
  await page.waitForSelector('.tree-row', { timeout: 60_000 });
  await page.waitForFunction(() => /\d+ others/.test(document.querySelector('.chat [aria-haspopup="listbox"]')?.textContent ?? ''), { timeout: 20_000 });
  await wait(800);

  await shoot(page, 'runner');

  await shoot(page, 'picker', {
    before: async () => {
      await page.evaluate(() => {
        const button = document.querySelector('.chat [aria-haspopup="listbox"]');
        if (button?.getAttribute('aria-expanded') !== 'true') button?.click();
      });
      await wait(150);
      const search = await page.$('.chat input[role="combobox"], .chat input[type="search"]');
      if (search) {
        await search.click({ clickCount: 3 });
        await page.keyboard.type('la');
      }
      await wait(200);
    },
  });
  await closePicker(page);

  // The pickers and the tree work from the keyboard (the new ones only).
  const verify = async (name, fn) => {
    try {
      await fn();
      checks.push(`ok   ${name}`);
    } catch (error) {
      checks.push(`FAIL ${name}: ${error.message}`);
    }
  };
  const expect = (c, m) => {
    if (!c) throw new Error(m);
  };
  const state = () => page.evaluate(() => {
    const search = document.querySelector('.chat .combo-search');
    const button = document.querySelector('.chat .combo-button');
    const options = [...document.querySelectorAll('.chat .combo-option')];
    const active = document.getElementById(search?.getAttribute('aria-activedescendant') ?? '');
    return {
      open: button?.getAttribute('aria-expanded') === 'true',
      focus: document.activeElement === search ? 'search' : document.activeElement === button ? 'button' : document.activeElement?.className ?? '',
      value: search?.value ?? '',
      options: options.map((o) => o.querySelector('.combo-name')?.textContent),
      active: active?.querySelector('.combo-name')?.textContent ?? null,
      activeIndex: options.indexOf(active),
      empty: document.querySelector('.chat .combo-empty')?.hidden === false ? document.querySelector('.chat .combo-empty').textContent : null,
      button: button?.textContent ?? '',
    };
  });
  const clearSearch = async () => {
    await page.keyboard.down('Control');
    await page.keyboard.press('a');
    await page.keyboard.up('Control');
    await page.keyboard.press('Backspace');
  };
  if (await page.$('.chat .combo-button')) {
    await verify('typing on the closed conversations picker opens it with the search begun', async () => {
      await page.evaluate(() => document.querySelector('.chat .combo-button').focus());
      await page.keyboard.type('p');
      const s = await state();
      expect(s.open && s.focus === 'search' && s.value === 'p', JSON.stringify(s));
    });
    await verify('the search finds by name and by number, and says when nothing matches', async () => {
      await page.keyboard.type('riya');
      let s = await state();
      expect(s.options.length === 1 && s.options[0] === 'Priya Shah', JSON.stringify(s.options));
      await clearSearch();
      await page.keyboard.type('333 444');
      s = await state();
      expect(s.options.length === 1 && s.options[0] === '0400 333 444', JSON.stringify(s.options));
      await clearSearch();
      await page.keyboard.type('zebra');
      s = await state();
      expect(!s.options.length && s.empty === 'No matches for "zebra"', JSON.stringify(s));
      await clearSearch();
    });
    await verify('Up, Down, Home and End move through the list; Enter chooses, and the message box takes the focus (as the app does on a switch)', async () => {
      await page.keyboard.press('End');
      let s = await state();
      expect(s.activeIndex === s.options.length - 1, `End: ${JSON.stringify(s)}`);
      await page.keyboard.press('Home');
      s = await state();
      expect(s.activeIndex === 0, `Home: ${s.activeIndex}`);
      await page.keyboard.press('ArrowDown');
      await page.keyboard.press('ArrowDown');
      s = await state();
      expect(s.activeIndex === 2, `Down twice: ${s.activeIndex}`);
      await page.keyboard.press('ArrowUp');
      expect((await state()).activeIndex === 1, 'Up');
      await page.keyboard.type('mia');
      await page.keyboard.press('Enter');
      await wait(400);
      s = await state();
      expect(!s.open && s.focus.includes('chat-input') && s.button.includes('Mia Chen'), JSON.stringify(s));
    });
    await verify('Escape closes the picker and gives the focus back to its button', async () => {
      await page.evaluate(() => document.querySelector('.chat .combo-button').focus());
      await page.keyboard.press('ArrowDown');
      expect((await state()).open, 'ArrowDown on the button did not open it');
      await page.keyboard.press('Escape');
      const s = await state();
      expect(!s.open && s.focus === 'button', JSON.stringify(s));
    });
    await verify('the file tree moves with the arrow keys and opens and closes folders', async () => {
      await page.evaluate(() => document.querySelector('.tree-row[tabindex="0"]').focus());
      const row = () => page.evaluate(() => ({ title: document.activeElement?.getAttribute('title'), expanded: document.activeElement?.getAttribute('aria-expanded') }));
      expect((await row()).title === '/knowledge', JSON.stringify(await row()));
      await page.keyboard.press('ArrowRight');
      await wait(100);
      expect((await row()).expanded === 'true', `Right: ${JSON.stringify(await row())}`);
      await page.keyboard.press('ArrowDown');
      const child = await row();
      expect(child.title?.startsWith('/knowledge/'), `Down: ${JSON.stringify(child)}`);
      await page.keyboard.press('ArrowLeft');
      expect((await row()).title === '/knowledge', `Left to the folder: ${JSON.stringify(await row())}`);
      await page.keyboard.press('ArrowLeft');
      await wait(100);
      expect((await row()).expanded === 'false', `Left closes: ${JSON.stringify(await row())}`);
      await page.keyboard.press('End');
      expect((await row()).title === '/brief.md', `End: ${JSON.stringify(await row())}`);
    });
    // Back to the runner's own conversation for what follows.
    await page.evaluate(() => document.querySelector('.chat .combo-button').focus());
    await page.keyboard.type('runner');
    await page.keyboard.press('Enter');
    await wait(400);
  }

  await conversation(page, 'person', 'Lance');
  await closePicker(page);
  await verify("a person's conversation opens at its end, the latest at the bottom, their calls and then their texts in order", async () => {
    await wait(300);
    const s = await logState(page);
    expect(s.fromBottom <= 2 && !s.jump, JSON.stringify(s));
    const order = await page.evaluate(() => [...document.querySelectorAll('.chat-feed .msg-body, .chat-feed .day-divider, .chat-feed .call-end')].map((e) => e.textContent.trim().slice(0, 24)));
    const texts = order.indexOf('Texts');
    expect(order[0].startsWith('Yesterday') && order.indexOf('Perfect, thanks. Bye!') > order.indexOf('Hi there, I was hoping t') && texts > order.lastIndexOf('The call ended') && order.at(-1) === 'Thanks 👍', JSON.stringify(order));
  });
  await shoot(page, 'call');

  await conversation(page, 'person', 'Mia Chen');
  await closePicker(page);
  await shoot(page, 'sms');

  if (!only || only.includes('live')) {
    // A call comes in: the app shows it, and its agent answers (the scripted model).
    liveCalls = ['live-1'];
    voice({ type: 'call.started', callId: 'live-1', from: '+61455666777', name: 'Dave', greeting: 'Hi, thanks for calling Greenline Gardens. How can I help?' });
    await wait(1500);
    voice({ type: 'call.caller', callId: 'live-1', text: "Hi, it's Dave again. Could someone look at my hedge this week?", startMs: 3_100, endMs: 6_400 });
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Which day suits you best'), { timeout: 20_000 }).catch(() => console.log('  (no first reply on the call)'));
    await wait(600);
    voice({ type: 'call.caller', callId: 'live-1', text: 'yeah', startMs: 7_900, endMs: 8_200, over: true, backchannel: true });
    voice({ type: 'call.caller', callId: 'live-1', text: 'Thursday morning, if you can.', startMs: 9_800, endMs: 11_600 });
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes("nine o'clock"), { timeout: 20_000 }).catch(() => console.log('  (no second reply on the call)'));
    await wait(1200);
    await verify('a live call follows what is said at the bottom', async () => {
      const s = await logState(page);
      const last = await page.evaluate(() => [...document.querySelectorAll('.chat-feed .msg-body')].at(-1)?.textContent);
      expect(s.fromBottom <= 2 && /nine o'clock/.test(last ?? ''), JSON.stringify({ ...s, last }));
    });
    await shoot(page, 'live');
    voice({ type: 'call.ended', callId: 'live-1' });
    liveCalls = [];
    await wait(800);
  }

  // The person's own project, chosen from the project picker by keyboard (the select, before it): its conversation, then a request the agent is still working on.
  if (await page.$('.project-picker .combo-button')) {
    await verify('the project picker opens with Down, finds a project as it is typed, and switches to it with Enter', async () => {
      await page.evaluate(() => document.querySelector('.project-picker .combo-button').focus());
      await page.keyboard.press('ArrowDown');
      await page.keyboard.type('weath');
      const names = await page.$$eval('.project-picker .combo-option .combo-name', (els) => els.map((e) => e.textContent));
      expect(names.length === 1 && names[0] === 'Weather report', JSON.stringify(names));
      await page.keyboard.press('Enter');
      await page.waitForFunction(() => document.querySelector('.project-picker .combo-button')?.textContent.includes('Weather report') && document.querySelector('.project-select').value !== 'front-desk', { timeout: 10_000 });
    });
  } else await page.select('.project-select', project);
  await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Weather summary'), { timeout: 20_000 });
  await wait(600);
  await shoot(page, 'chat');
  // A run of tools opened, and one of them: its input and result.
  await shoot(page, 'tools', {
    sizes: [[1440, 900]],
    before: async () => {
      await page.evaluate(() => {
        const group = [...document.querySelectorAll('details.tool-group')].at(-1);
        if (!group) return;
        group.open = true;
        const run = group.querySelector('details.tool[data-tool="code_run"]');
        if (run) run.open = true;
        const failed = group.querySelector('details.tool.failed');
        if (failed) failed.open = true;
      });
      // Opened at the bottom, the log keeps to it; then the reader scrolls up to the rows ("Latest" shows).
      await wait(200);
      await page.evaluate(() => {
        const log = document.querySelector('.chat-log');
        const group = [...document.querySelectorAll('details.tool-group')].at(-1);
        if (group) log.scrollTop += group.getBoundingClientRect().top - log.getBoundingClientRect().top - 8;
      });
      await wait(250);
    },
  });
  if (!only || only.includes('busy')) {
    // Up the log (the tools opened): sending a message goes back down to where the reply comes.
    await page.evaluate(() => document.querySelector('.chat-log').scrollTo({ top: 0 }));
    await page.click('.chat-input');
    await page.keyboard.type('Also chart it as an SVG.');
    await page.keyboard.press('Enter');
    for (let i = 0; i < 100 && !held.length; i++) await wait(200);
    await wait(900);
    await verify('sending goes to the bottom, and the log keeps to it as the reply streams in', async () => {
      const s = await logState(page);
      const last = await page.evaluate(() => {
        const feed = document.querySelector('.chat-feed');
        const status = feed.lastElementChild;
        return { status: status.matches('.chat-status') ? status.textContent : 'not last', reply: [...feed.querySelectorAll('.msg.assistant')].at(-1)?.textContent.slice(0, 20) };
      });
      expect(s.fromBottom <= 2 && !s.jump && last.reply?.startsWith('Here is the chart'), JSON.stringify({ ...s, ...last }));
    });
    await shoot(page, 'busy');
    await verify('scrolled up, the reader stays put as the reply finishes and more comes, and "Latest" says how much is new', async () => {
      await page.evaluate(() => document.querySelector('.chat-log').scrollTo({ top: 200 }));
      await wait(200);
      const before = await logState(page);
      held.splice(0).forEach((release) => release());
      await page.waitForFunction(() => document.querySelector('.chat-log').textContent.includes('Oslo is the shortest bar.'), { timeout: 10_000 });
      await page.type('.chat-input', '/help');
      await page.keyboard.press('Enter');
      await wait(500);
      const after = await logState(page);
      expect(before.scrollTop === 200 && after.scrollTop === 200 && after.scrollHeight > before.scrollHeight && after.jump && /1 new/.test(after.jumpText), JSON.stringify({ before, after }));
      await page.click('.to-latest');
      await wait(900);
      const back = await logState(page);
      expect(back.fromBottom <= 2 && !back.jump, JSON.stringify(back));
    });
    await wait(300);
  }

  // A file open from the tree, in the editor.
  await page.evaluate(() => document.querySelector('.center-tabs button[data-pane="editor"]')?.click());
  await page.evaluate(() => [...document.querySelectorAll('.tree-row')].find((r) => r.getAttribute('title') === '/src')?.click());
  await wait(300);
  await page.evaluate(() => [...document.querySelectorAll('.tree-row')].find((r) => r.getAttribute('title') === '/src/summary.py')?.click());
  await wait(500);
  await page.click('.term-input');
  await page.keyboard.type('ls');
  await page.keyboard.press('Enter');
  await wait(2500);
  await shoot(page, 'files');
  if (!only || only.includes('terminal')) {
    const toggle = await page.$('.terminal .term-toggle');
    if (toggle) {
      await toggle.click();
      await wait(300);
      await shoot(page, 'terminal');
      await toggle.click();
    }
  }
  // A narrow window: one pane at a time.
  await shoot(page, 'narrow', { sizes: [[760, 688]], before: async () => page.evaluate(() => document.querySelector('nav.tabs button[data-view="agent"]')?.click()) });

  // A long conversation: its older turns come a page at a time as it is scrolled, each speaker's run going on across the pages.
  if (!only || only.includes('long')) {
    await page.setViewport({ width: 1440, height: 900 });
    await verify('a long conversation opens at its end, draws older turns as it is scrolled up with the reader held in place, one run per speaker across the pages', async () => {
      const errors = [];
      const onError = (e) => errors.push(e.message);
      page.on('pageerror', onError);
      await page.select('.project-select', long);
      await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Added line 30.'), { timeout: 20_000 });
      await wait(400);
      const opened = await logState(page);
      expect(opened.fromBottom <= 2, `opened away from the end: ${JSON.stringify(opened)}`);
      // Up to the first message drawn: older turns come in above it, and it stays where it was on screen.
      // (Measured from the log's own top: scrolling it must not move anything else on the page.)
      const held = await page.evaluate(() => {
        const log = document.querySelector('.chat-log');
        const first = document.querySelector('.chat-feed .msg.user');
        log.scrollTop += first.getBoundingClientRect().top - log.getBoundingClientRect().top - 10;
        return { text: first.textContent, top: first.getBoundingClientRect().top - log.getBoundingClientRect().top, scrollTop: log.scrollTop };
      });
      await wait(400);
      const moved = await page.evaluate((text) => {
        const log = document.querySelector('.chat-log');
        const el = [...document.querySelectorAll('.chat-feed .msg.user')].find((e) => e.textContent === text);
        return { top: el ? el.getBoundingClientRect().top - log.getBoundingClientRect().top : null, scrollTop: log.scrollTop, older: document.querySelector('.chat-feed .msg.user')?.textContent !== text };
      }, held.text);
      expect(moved.older && moved.scrollTop > 0 && Math.abs(moved.top - held.top) <= 2, `the reader's place moved: ${JSON.stringify({ held, moved })}`);
      for (let i = 0; i < 20 && (await page.$('.chat-log .show-earlier')); i++) {
        await page.evaluate(() => {
          document.querySelector('.chat-log').scrollTop = 0;
        });
        await wait(250);
      }
      page.off('pageerror', onError);
      const found = await page.evaluate(() => {
        const runs = [...document.querySelectorAll('.chat-feed > section.run')];
        const twice = runs.filter((r, i) => i && runs[i - 1] === r.previousElementSibling && runs[i - 1].dataset.speaker === r.dataset.speaker && runs[i - 1].dataset.who === r.dataset.who).length;
        const yours = runs.filter((r) => r.dataset.speaker === 'you').map((r) => r.querySelectorAll('.msg.user').length);
        return { older: !!document.querySelector('.chat-log .show-earlier'), requests: document.querySelectorAll('.chat-log .msg.user').length, tools: document.querySelectorAll('.chat-log details.tool').length, groups: document.querySelectorAll('.chat-log details.tool-group').length, twice, yours: yours.length, split: yours.filter((n) => n !== 2).length };
      });
      expect(!errors.length, errors.join('; '));
      expect(!found.older && found.requests === 60 && found.tools === 60 && found.groups === 30 && !found.twice && found.yours === 30 && !found.split, JSON.stringify(found));
    });
  }
  await page.close();

  // The same in a browser tab of its own (paired with the desktop, following the system's theme).
  if (!only || only.some((s) => s.startsWith('tab'))) {
    const tab = await browser.newPage();
    tab.on('pageerror', (e) => console.log('  [pageerror]', e.message));
    await tab.setViewport({ width: 1440, height: 900 });
    await tab.goto(base);
    await tab.waitForSelector('.tree-row', { timeout: 60_000 });
    await wait(1500);
    await shoot(tab, 'tab-chat', { mode: 'tab', sizes: [[1440, 900], [1058, 688]] });
    await tab.select('.project-select', 'front-desk');
    await tab.waitForFunction(() => /\d+ others/.test(document.querySelector('.chat [aria-haspopup="listbox"]')?.textContent ?? ''), { timeout: 20_000 }).catch(() => {});
    await wait(800);
    await conversation(tab, 'person', 'Lance').catch((e) => console.log('  ', e.message));
    await closePicker(tab);
    await shoot(tab, 'tab-call', { mode: 'tab', sizes: [[1440, 900]] });
    await tab.close();
  }
} finally {
  await browser.close();
  await server.close();
  model.close();
  desktopServer.close();
}
console.log(`${shots.length} screenshots in ${out}`);
if (checks.length) console.log(checks.join('\n'));
process.exit(0);
