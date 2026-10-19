// The Agent's header chooses OAIY's models: a mock OAIY (two language models, one set to run on a Mac's eGPU, one on
// the Mac) and a mock OAIY Desktop the page is given, as OAIY's window gives it. One menu lists the engine's models,
// each with where it runs, and chooses one in Engines (through the desktop), which loads it where it runs and has the
// other let go of what it held (as the engines do: one model loaded at a time); the menu says how its loading goes;
// chats then are answered by the model chosen. A computer paired over OAIY Network lends its language model: the menu
// lists it under that computer's name, choosing it makes it the engine's (the Agent then runs there, through the
// desktop), and choosing one of this Mac's takes the engine back. The menu follows what the Agent changes itself, a
// computer whose engines stop, one that switches its model, and one just paired; on an OAIY with no language model of
// its own, a paired computer's is offered; and the header fits a phone.
//   node tests/e2e/oaiy-models.mjs
import { existsSync } from 'node:fs';
import { createServer as createHttpServer } from 'node:http';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].filter(Boolean).find((p) => existsSync(p));
if (!executablePath) {
  console.error('no Chrome found; set CHROME');
  process.exit(2);
}

// What the engines have: the model chosen in Engines, the one the card holds and how its loading goes (it is ready
// at the second look after a start), and the one this Mac's engine holds (one chosen is held from the second look
// after the choice). The 27B is set to the eGPU, the 9B to the Mac. `idle`: they run no language model, and a model
// chosen then is only chosen (as OAIY's studio does: it loads one on a change only while it runs one).
const ON_CARD = 'qwen3.8-27b';
const engines = { default: 'qwen3.8-27b', card: 'qwen3.8-27b', state: 'ready', looks: 0, mac: null, macPending: null, macLooks: 0, down: false, idle: false };
const chats = [];
const asked = [];
// The paired computer (OAIY Network) and its language model, by the id the desktop's catalogue gives it; `chosen`: the
// engine's default is that one, in place of this Mac's. The desktop says a choice of it `lag` ms after it answers (it
// settles it over OAIY Network). `others`: more paired computers' models; `prefs`: the Agent's choice set in OAIY.
const NODE = 'a1b2c3d4e5f60718293a4b5c6d7e8f90';
const remoteId = (model) => `engine:llm:node.${NODE}/${model}`;
const REMOTE = remoteId('Qwen3.8-Flash-Next');
const network = { chosen: false, chosenAt: 0, lag: 0, model: 'Qwen3.8-Flash-Next', ready: true, nodeOnline: true, others: [], prefs: null, chats: [] };

function discovery(origin) {
  if (engines.state === 'starting' && ++engines.looks >= 2) engines.state = 'ready';
  if (engines.macPending && ++engines.macLooks >= 2) {
    engines.mac = engines.macPending;
    engines.macPending = null;
  }
  return {
    service: 'oaiy-studio',
    version: '0.1.0',
    base_url: origin,
    openai_base_url: `${origin}/v1`,
    auth: { type: 'none', required: false },
    endpoints: [{ name: 'chat', path: '/v1/chat/completions', url: `${origin}/v1/chat/completions`, spec: 'openai' }],
    models: {
      llm: ['qwen3.8-27b', 'qwen3.5-9b'].map((id) => ({ id, default: id === engines.default, loaded: id === engines.mac, egpu: id === ON_CARD })),
    },
    defaults: { llm: engines.default },
    llm: { context_tokens: 32768 },
    egpu: { available: true, enabled: true, engine: 'webgpu', state: engines.card ? engines.state : 'stopped', model: engines.card, models: [ON_CARD], lent: false, paused: false },
  };
}

/** One of this Mac's models chosen (in Engines, or by the Agent's tool): it loads where it runs, the other lets go. */
function chooseHere(model) {
  network.chosen = false;
  engines.default = model;
  const card = model === ON_CARD;
  Object.assign(engines, card ? { card: model, state: 'starting', looks: 0, mac: null, macPending: null } : { card: null, state: 'stopped', mac: null, macPending: model, macLooks: 0 });
}

const cors = (res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type, *');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
};
const reply = (res, status, body, type = 'application/json') => {
  res.statusCode = status;
  res.setHeader('content-type', type);
  res.end(type === 'application/json' ? JSON.stringify(body) : body);
};
const read = (req) => new Promise((r) => {
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', () => r(body));
});
const say = (e) => `data: ${JSON.stringify(e)}\n\n`;
const stream = (res, text, calls = []) => {
  res.setHeader('content-type', 'text/event-stream');
  const events = text ? [{ choices: [{ index: 0, delta: { content: text } }] }] : [];
  calls.forEach((c, i) => events.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: `call_${i}`, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.input) } }] } }] }));
  events.push({ choices: [{ index: 0, delta: {}, finish_reason: calls.length ? 'tool_calls' : 'stop' }] });
  res.end(events.map(say).join('') + 'data: [DONE]\n\n');
};
const lastUser = (ask) => {
  const m = [...ask.messages].reverse().find((x) => x.role === 'user');
  return typeof m?.content === 'string' ? m.content : (m?.content ?? []).map((p) => p.text ?? '').join('');
};

const oaiy = createHttpServer(async (req, res) => {
  // (the engines stopped: nothing answers, as a closed port does not)
  if (engines.down) return req.socket.destroy();
  cors(res);
  if (req.method === 'OPTIONS') return res.end();
  const body = await read(req);
  const url = req.url.split('?')[0];
  if (url === '/v1/discovery') return reply(res, 200, discovery(`http://${req.headers.host}`));
  if (url === '/v1/models') return reply(res, 200, { data: [{ id: 'qwen3.8-27b' }, { id: 'qwen3.5-9b' }] });
  if (url === '/v1/chat/completions') {
    const ask = JSON.parse(body);
    chats.push(ask);
    // The Agent asked to use the smaller model: it chooses it with OAIY's tool, then says so.
    if (ask.messages.at(-1)?.role === 'tool') return stream(res, 'Done: the engine runs the smaller model now.');
    if (lastUser(ask).includes('Use the smaller model')) return stream(res, '', [{ name: 'model_set_default', input: { group: 'llm', model: 'qwen3.5-9b' } }]);
    // (as OAIY's gateway: a request that names no model is the engine's chosen one's)
    return stream(res, `Answered by ${ask.model ?? engines.default}.`);
  }
  reply(res, 404, { error: { message: `no route ${req.method} ${url}` } });
});

// An OAIY with no language model of its own (it makes pictures only), for a page whose Agent can run only on a paired
// computer's.
const bare = createHttpServer((req, res) => {
  cors(res);
  if (req.method === 'OPTIONS') return res.end();
  const origin = `http://${req.headers.host}`;
  if (req.url.split('?')[0] === '/v1/discovery') {
    return reply(res, 200, { service: 'oaiy-studio', version: '0.1.0', base_url: origin, openai_base_url: `${origin}/v1`, auth: { type: 'none', required: false }, endpoints: [], models: { llm: [], image: ['flux-3-klein'] }, defaults: {} });
  }
  reply(res, 404, { error: { message: 'not here' } });
});

// OAIY's control tools, as the desktop offers them to the Agent: choosing a model is one.
const CONTROL_TOOLS = [{ name: 'model_set_default', title: 'Choose a model', description: "Choose the engine's model for a group.", inputSchema: { type: 'object', properties: { group: { type: 'string' }, model: { type: 'string' } }, required: ['group', 'model'] }, annotations: { readOnlyHint: false } }];

// OAIY Desktop: the engines' relay (choosing a model), its catalogue of paired computers' models, the Agent's choice,
// the route to a paired computer's model, and its control API; every other route not there. The engines load the
// model chosen where it runs and let the other go of what it held.
const desktop = createHttpServer(async (req, res) => {
  cors(res);
  if (req.method === 'OPTIONS') return res.end();
  const body = await read(req);
  const url = req.url.split('?')[0];
  asked.push({ method: req.method, url, auth: req.headers.authorization, body: body ? JSON.parse(body) : null });
  if (req.method === 'PUT' && url === '/api/engines/defaults' && JSON.parse(body).model.startsWith(`engine:llm:node.${NODE}/`)) {
    Object.assign(network, { chosen: true, chosenAt: Date.now() });
    return reply(res, 200, { group: 'llm', model: JSON.parse(body).model });
  }
  if (req.method === 'GET' && url === '/api/ai/engine/services') {
    return reply(res, 200, { running: true, services: [{ id: remoteId(network.model), kind: 'llm', model: network.model, nodeId: NODE, nodeName: 'Desktop', locality: 'network', ready: network.ready, nodeOnline: network.nodeOnline, contextTokens: 131072, default: network.chosen }, ...network.others] });
  }
  if (req.method === 'GET' && url === '/api/agent/preferences') {
    if (network.prefs) return reply(res, 200, network.prefs);
    const settled = network.chosen && Date.now() - network.chosenAt >= network.lag;
    return reply(res, 200, settled ? { model: { source: 'engine' }, engineDefault: remoteId(network.model), engineDefaultName: 'Desktop' } : { model: { source: 'engine' } });
  }
  if (req.method === 'POST' && url === `/api/network/nodes/${NODE}/v1/chat/completions`) {
    const ask = JSON.parse(body);
    network.chats.push(ask);
    // (as the worker does: a model asked for by name that it does not run now is refused, not answered by another)
    if (ask.model && ask.model !== network.model) return reply(res, 409, { error: { code: 'model_unavailable', message: `Desktop runs ${network.model} now, not ${ask.model}: choose its language model again` } });
    return stream(res, `Answered on Desktop by ${network.model}.`);
  }
  if (req.method === 'PUT' && url === '/api/engines/defaults') {
    // (one of this Mac's chosen: a paired computer's that was the default is not any more)
    chooseHere(JSON.parse(body).model);
    return reply(res, 200, { group: 'llm', model: engines.default });
  }
  // (its events: none, but it answers, so the page takes it as there)
  if (req.method === 'GET' && url === '/api/bridge/events') return reply(res, 200, { events: [], next: 0 });
  if (req.method === 'POST' && url === '/api/mcp') {
    const m = JSON.parse(body || '{}');
    if (!('id' in m)) {
      res.statusCode = 202;
      return res.end();
    }
    if (m.method === 'initialize') return reply(res, 200, { jsonrpc: '2.0', id: m.id, result: { protocolVersion: '2025-06-18', capabilities: { tools: {} }, serverInfo: { name: 'oaiy', version: '0.1.0' } } });
    if (m.method === 'tools/list') return reply(res, 200, { jsonrpc: '2.0', id: m.id, result: { tools: CONTROL_TOOLS } });
    if (m.method === 'tools/call' && m.params.name === 'model_set_default') {
      if (engines.idle) engines.default = m.params.arguments.model;
      else chooseHere(m.params.arguments.model);
      return reply(res, 200, { jsonrpc: '2.0', id: m.id, result: { content: [{ type: 'text', text: `The engine's llm model is now ${m.params.arguments.model}.` }], isError: false } });
    }
    return reply(res, 200, { jsonrpc: '2.0', id: m.id, error: { code: -32601, message: 'no such method' } });
  }
  reply(res, 404, { error: 'not here' });
});

for (const s of [oaiy, bare, desktop]) await new Promise((r) => s.listen(0, '127.0.0.1', r));
const oaiyUrl = `http://127.0.0.1:${oaiy.address().port}`;
const bareUrl = `http://127.0.0.1:${bare.address().port}`;
const desktopUrl = `http://127.0.0.1:${desktop.address().port}`;

const server = await createServer({ server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const base = `http://127.0.0.1:${server.httpServer.address().port}`;
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
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-first-run'] });
/** A page in OAIY's window (the desktop given), in a browser context of its own (its own storage), at `oaiyAt`. */
async function openWindow(oaiyAt, context = browser) {
  const page = await context.newPage();
  await page.setViewport({ width: 1400, height: 900 });
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  await page.evaluateOnNewDocument((origin) => {
    window.__OAIY_DESKTOP__ = { origin, token: 'desktop-token' };
  }, desktopUrl);
  await page.goto(`${base}/?oaiy=${encodeURIComponent(oaiyAt)}`);
  await page.waitForSelector('.tree-row', { timeout: 60_000 });
  return page;
}
try {
  const page = await openWindow(oaiyUrl);
  await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes("OAIY's engines are ready here"), { timeout: 15_000 });
  const menu = (cls) => page.$eval(`select.${cls}.pick`, (s) => ({ hidden: s.hidden, value: s.value, options: [...s.options].map((o) => o.textContent), shown: s.selectedOptions[0]?.textContent, state: s.dataset.state }));
  const log = (p = page) => p.$eval('.chat-log', (l) => l.textContent);
  const shown = (p, text) => p.waitForFunction((t) => document.querySelector('select.model.pick')?.selectedOptions[0]?.textContent === t, { timeout: 15_000 }, text).catch(async (e) => {
    throw new Error(`${e.message}\n${JSON.stringify(await p.$eval('select.model.pick', (s) => [...s.options].map((o) => o.textContent + (o.selected ? ' [selected]' : ''))))}`);
  });
  /** The window comes back to the front: the page reads OAIY again (at most every five seconds). */
  const backToFront = async (p = page) => {
    await wait(5100);
    await p.evaluate(() => window.dispatchEvent(new Event('focus')));
  };
  const send = async (p, text) => {
    await p.type('.chat-input', text);
    await p.keyboard.press('Enter');
  };

  await check("one menu lists the engine's language models, each with where it runs, the chosen one ready on the eGPU, in place of the provider chip", async () => {
    await page.waitForFunction(() => !document.querySelector('select.model.pick')?.hidden, { timeout: 10_000 });
    const m = await menu('model');
    expect(m.value === 'qwen3.8-27b' && m.shown === 'qwen3.8-27b · eGPU · ready', JSON.stringify(m));
    expect(m.options.includes('qwen3.5-9b · Mac') && m.options.includes('AI providers…'), JSON.stringify(m.options));
    expect(m.state === 'ready', 'the menu is not shown ready');
    expect(await page.$eval('button[title="AI provider"]', (b) => b.hidden), 'the provider chip is still shown beside it');
    expect(!(await page.$('select.egpu.pick')), 'a second menu, for the eGPU, is still there');
    // (OAIY's own window says what its engines offer, not their service's name or address)
    const text = await log();
    expect(!text.includes('oaiy-studio') && !text.includes(oaiyUrl), text.slice(0, 400));
  });

  await check("choosing a Mac's model chooses it in Engines, through the desktop: it loads on the Mac (the menu says so), the card lets go, and chats are answered by it", async () => {
    await page.select('select.model.pick', 'qwen3.5-9b');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes("OAIY's engine now runs qwen3.5-9b, on this Mac"), { timeout: 10_000 });
    expect(await page.$eval('.chat-log', (l) => l.textContent.includes('the eGPU lets go of its model')), 'the card letting go is not said');
    const put = asked.find((a) => a.url === '/api/engines/defaults');
    expect(put && put.auth === 'Bearer desktop-token' && put.body.group === 'llm' && put.body.model === 'qwen3.5-9b', JSON.stringify(asked));
    expect(!asked.some((a) => a.url === '/api/engines/egpu'), 'the eGPU was set apart from the choice');
    // (the Mac says only whether it holds it: loading until it does, then ready, and said)
    await shown(page, 'qwen3.5-9b · Mac · loading…');
    expect((await menu('model')).state === 'starting', JSON.stringify(await menu('model')));
    await shown(page, 'qwen3.5-9b · Mac · ready');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('qwen3.5-9b is loaded on this Mac.'), { timeout: 10_000 });
    const m = await menu('model');
    expect(m.options.includes('qwen3.8-27b · eGPU'), JSON.stringify(m.options));
    await send(page, 'Hello there.');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Answered by qwen3.5-9b.'), { timeout: 30_000 }).catch(async (e) => {
      throw new Error(`${e.message}\nchats: ${JSON.stringify(chats.map((c) => c.model))}\nlog: ${(await log()).slice(-600)}`);
    });
    // (the provider follows Engines: it names no model, and the engine answers with the one chosen)
    expect(chats.at(-1).model === undefined, chats.at(-1)?.model);
  });

  await check("choosing the eGPU's model loads it on the card, the Mac lets go, and the menu follows its loading", async () => {
    await page.select('select.model.pick', 'qwen3.8-27b');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes("OAIY's engine now runs qwen3.8-27b, on the eGPU"), { timeout: 10_000 });
    expect(await page.$eval('.chat-log', (l) => l.textContent.includes("this Mac's engine lets go of any model it held")), 'the Mac letting go is not said');
    expect(asked.at(-1).url === '/api/engines/defaults' && asked.at(-1).body.model === 'qwen3.8-27b', JSON.stringify(asked.at(-1)));
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('qwen3.8-27b is loaded on the eGPU.'), { timeout: 20_000 });
    const m = await menu('model');
    expect(m.value === 'qwen3.8-27b' && m.shown === 'qwen3.8-27b · eGPU · ready', JSON.stringify(m));
    expect(m.options.includes('qwen3.5-9b · Mac'), 'the Mac still shown holding its model: ' + JSON.stringify(m.options));
  });

  await check("a paired computer's language model is in the menu, under that computer's name, and the closed menu stays narrow", async () => {
    await page.waitForFunction(() => [...(document.querySelector('select.model.pick')?.options ?? [])].some((o) => o.textContent === 'Qwen3.8-Flash-Next · Desktop'), { timeout: 15_000 }).catch(async (e) => {
      throw new Error(`${e.message}\n${JSON.stringify(await menu('model'))}`);
    });
    const group = await page.$eval('select.model.pick', (s) => [...s.querySelectorAll('optgroup')].map((g) => g.label));
    expect(group.length === 1 && group[0] === 'On Desktop (OAIY Network)', JSON.stringify(group));
    const m = await menu('model');
    expect(m.shown === 'qwen3.8-27b · eGPU · ready' && m.options.at(-1) === 'AI providers…', JSON.stringify(m));
    // (as wide as the model chosen, up to about 280px: not as wide as its widest option)
    const width = await page.$eval('select.model.pick', (s) => s.getBoundingClientRect().width);
    expect(width <= 285, `the closed menu is ${width}px wide`);
  });

  await check("choosing it makes it the engine's, through the desktop, and the Agent runs there: said once, and the menu stays on it while the desktop settles it", async () => {
    network.lag = 3000;
    await page.select('select.model.pick', REMOTE);
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes("OAIY's engine now runs Qwen3.8-Flash-Next on Desktop, over OAIY Network"), { timeout: 10_000 });
    const put = asked.filter((a) => a.url === '/api/engines/defaults').at(-1);
    expect(put.body.group === 'llm' && put.body.model === REMOTE && put.auth === 'Bearer desktop-token', JSON.stringify(put));
    // (at once, though the desktop does not say so for a moment: a read before then does not take it back)
    expect((await menu('model')).shown === 'Qwen3.8-Flash-Next · Desktop', JSON.stringify(await menu('model')));
    await page.evaluate(() => window.dispatchEvent(new Event('focus')));
    await wait(1000);
    expect((await menu('model')).shown === 'Qwen3.8-Flash-Next · Desktop', `snapped back: ${JSON.stringify(await menu('model'))}`);
    expect(await page.$eval('button[title="AI provider"]', (b) => b.hidden), 'the provider chip took the menu\'s place');
    // the desktop has settled it: read again, nothing new to say
    await backToFront();
    await wait(800);
    const text = await log();
    expect(!text.includes("The Agent now runs on OAIY's engine") && !text.includes('The Agent now runs on Qwen3.8-Flash-Next') && !text.includes("OAIY's engine default is now"), `the change was said again: ${text.slice(-500)}`);
    expect((await menu('model')).state === 'ready', JSON.stringify(await menu('model')));
    network.lag = 0;
    const before = chats.length;
    await send(page, 'And now?');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Answered on Desktop by Qwen3.8-Flash-Next.'), { timeout: 30_000 }).catch(async (e) => {
      throw new Error(`${e.message}\nlog: ${(await log()).slice(-600)}`);
    });
    expect(network.chats.length === 1 && network.chats[0].model === 'Qwen3.8-Flash-Next' && chats.length === before, `remote ${JSON.stringify(network.chats.map((c) => c.model))}, local ${chats.length - before}`);
    // (the paired computer's window, as the desktop says it: no window of this Mac's model said for it)
    expect(!(await log()).includes('qwen3.8-27b: 32.8k tokens of context'), 'the window said is not the model\'s that answers');
  });

  await check('a paired computer whose engines stop is said so, muted, not as a failure; one just paired is in the menu when the window is back in front', async () => {
    Object.assign(network, { ready: false, nodeOnline: true });
    network.others = [{ id: 'engine:llm:node.c0ffee01/Gemma-4', kind: 'llm', model: 'Gemma-4', nodeId: 'c0ffee01', nodeName: 'Office PC', locality: 'network', ready: false, nodeOnline: false, default: false }];
    await backToFront();
    await shown(page, 'Qwen3.8-Flash-Next · Desktop · engines not running');
    const m = await menu('model');
    expect(m.state === 'offline', JSON.stringify(m));
    expect(m.options.includes('Gemma-4 · Office PC · offline'), JSON.stringify(m.options));
    const groups = await page.$eval('select.model.pick', (s) => [...s.querySelectorAll('optgroup')].map((g) => g.label));
    expect(groups.includes('On Office PC (OAIY Network)'), JSON.stringify(groups));
    expect((await page.$eval('select.model.pick', (s) => s.title)).includes('its engines are not running now'), await page.$eval('select.model.pick', (s) => s.title));
    Object.assign(network, { ready: true, others: [] });
  });

  await check('a paired computer that switched its model answers 409 model_unavailable: the page reads the change at once (said as the default moving, not as a choice), and Try again goes to the model it runs now', async () => {
    network.model = 'Gemma-4';
    await send(page, 'Still there?');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes("OAIY's engine default is now Gemma-4 on Desktop, over OAIY Network: the Agent runs there."), { timeout: 15_000 }).catch(async (e) => {
      throw new Error(`${e.message}\nlog: ${(await log()).slice(-700)}`);
    });
    expect(!(await log()).includes('as chosen in OAIY'), 'the default moving is said as a choice');
    await shown(page, 'Gemma-4 · Desktop');
    await page.waitForFunction(() => [...document.querySelectorAll('.chat-log button.retry')].some((b) => b.textContent.includes('Try again')), { timeout: 10_000 });
    await page.evaluate(() => [...document.querySelectorAll('.chat-log button.retry')].find((b) => b.textContent.includes('Try again')).click());
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Answered on Desktop by Gemma-4.'), { timeout: 20_000 }).catch(async (e) => {
      throw new Error(`${e.message}\nlog: ${(await log()).slice(-600)}`);
    });
    expect(network.chats.at(-1).model === 'Gemma-4', JSON.stringify(network.chats.map((c) => c.model)));
  });

  await check("choosing one of this Mac's again takes the engine back from the paired computer, at once and said once", async () => {
    await page.select('select.model.pick', 'qwen3.8-27b');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes("OAIY's engine now runs qwen3.8-27b, on the eGPU"), { timeout: 10_000 });
    expect(!network.chosen, 'the paired computer is still the default');
    await page.waitForFunction(() => document.querySelector('select.model.pick')?.selectedOptions[0]?.textContent.startsWith('qwen3.8-27b · eGPU'), { timeout: 10_000 }).catch(async (e) => {
      throw new Error(`${e.message}\n${JSON.stringify(await menu('model'))}`);
    });
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('qwen3.8-27b is loaded on the eGPU.'), { timeout: 20_000 }).catch(() => {});
    expect(!(await log()).includes("OAIY's engine default is this computer's own model again"), 'the choice made here was said again');
    const before = network.chats.length;
    await send(page, 'Back here?');
    await page.waitForFunction((n) => (document.querySelector('.chat-log')?.textContent.match(/Answered by qwen3\.8-27b\./g) ?? []).length >= n, { timeout: 30_000 }, 1);
    expect(network.chats.length === before, 'a chat still went to the paired computer');
  });

  await check("the Agent choosing a model itself (OAIY's model_set_default) moves the menu, and its loading is followed", async () => {
    await send(page, 'Use the smaller model from now on.');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Done: the engine runs the smaller model now.'), { timeout: 30_000 }).catch(async (e) => {
      throw new Error(`${e.message}\nlog: ${(await log()).slice(-600)}`);
    });
    await page.waitForFunction(() => document.querySelector('select.model.pick')?.selectedOptions[0]?.textContent.startsWith('qwen3.5-9b · Mac'), { timeout: 10_000 }).catch(async (e) => {
      throw new Error(`${e.message}\n${JSON.stringify(await menu('model'))}`);
    });
    await shown(page, 'qwen3.5-9b · Mac · ready');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('qwen3.5-9b is loaded on this Mac.'), { timeout: 15_000 });
    // (the row says what it chose, in words)
    const rows = await page.$$eval('.chat-log .tool', (r) => r.map((x) => x.textContent));
    expect(rows.some((t) => t.includes('llm → qwen3.5-9b')), JSON.stringify(rows));
  });

  await check('the header drawn again with nothing changed leaves its menu as it is (one open stays open)', async () => {
    await page.$eval('select.model.pick', (s) => { window.__modelOption = s.options[0]; });
    const before = asked.length;
    // the model it runs already: nothing to choose, the header drawn again (as the minute's timer draws it)
    await page.select('select.model.pick', 'qwen3.5-9b');
    await wait(300);
    expect(asked.length === before, `asked ${JSON.stringify(asked.slice(before))}`);
    expect(await page.evaluate(() => window.__modelOption.isConnected), "the menu's options were made again");
  });

  await check('a change made in OAIY just after a choice in the menu is taken (the Agent set to another provider), not held off as an answer from before the choice', async () => {
    // (the paired computer's model chosen here, which the desktop says only 20 s later)
    network.lag = 20_000;
    const said = (await log()).split("OAIY's engine now runs").length;
    await page.select('select.model.pick', remoteId(network.model));
    await page.waitForFunction((n) => document.querySelector('.chat-log')?.textContent.split("OAIY's engine now runs").length > n, { timeout: 10_000 }, said);
    // ...and, meanwhile, the Agent set to LM Studio in OAIY: the desktop says so at its next look
    network.prefs = { model: { source: 'provider', provider: 'lm-studio', model: 'qwen-coder-30b' }, providerName: 'LM Studio' };
    await backToFront();
    await page.waitForFunction(() => document.querySelector('button.chip.model:not([hidden])')?.textContent === 'LM Studio · qwen-coder-30b', { timeout: 10_000 }).catch(async (e) => {
      throw new Error(`${e.message}\nchip: ${await page.$eval('button.chip.model', (b) => `${b.hidden ? '(hidden) ' : ''}${b.textContent}`)}; menu: ${JSON.stringify(await menu('model'))}`);
    });
    // (back on this Mac's model, the engine's own)
    Object.assign(network, { prefs: null, chosen: false, lag: 0 });
    await backToFront();
    await page.waitForFunction(() => !document.querySelector('select.model.pick')?.hidden, { timeout: 10_000 });
    await shown(page, 'qwen3.5-9b · Mac · ready');
  });

  await check("the Agent choosing a model while OAIY's engines run none: the menu says it is chosen, not loading (nothing loads)", async () => {
    Object.assign(engines, { idle: true, default: 'qwen3.8-27b', card: null, state: 'stopped', mac: null, macPending: null });
    await backToFront();
    await shown(page, 'qwen3.8-27b · eGPU · not loaded');
    await send(page, 'Use the smaller model from now on.');
    await page.waitForFunction(() => (document.querySelector('.chat-log')?.textContent.match(/Done: the engine runs the smaller model now\./g) ?? []).length >= 2, { timeout: 30_000 });
    await shown(page, 'qwen3.5-9b · Mac · not loaded');
    const seen = new Set();
    for (let i = 0; i < 20; i++) {
      seen.add((await menu('model')).shown);
      await wait(200);
    }
    expect(![...seen].some((t) => /loading/.test(t)), JSON.stringify([...seen]));
    // (the engines run a model again: this Mac's, chosen; held from the second look)
    engines.idle = false;
    chooseHere('qwen3.5-9b');
    await backToFront();
    await backToFront();
    await shown(page, 'qwen3.5-9b · Mac · ready');
  });

  await check("OAIY's engines stopped: the run says they are not answering (no CORS advice, OAIY not Studio), the menu says so, and it is ready again once they answer", async () => {
    engines.down = true;
    await send(page, 'Are you still there?');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('engines are not answering'), { timeout: 30_000 }).catch(async (e) => {
      throw new Error(`${e.message}\nlog: ${(await log()).slice(-500)}`);
    });
    const text = await log();
    const said = text.slice(text.lastIndexOf('engines are not answering') - 8, text.lastIndexOf('engines are not answering') + 160);
    expect(said.includes(`OAIY’s engines are not answering (${oaiyUrl}). Start them in OAIY’s Engines page`) && !/CORS|cors_origins|Studio/.test(said), said);
    await shown(page, 'qwen3.5-9b · not answering');
    const m = await menu('model');
    expect(m.state === 'offline', JSON.stringify(m));
    expect((await page.$eval('select.model.pick', (s) => s.title)).includes("OAIY's engines are not answering now"), 'the menu does not say why');
    // (looked at again every few seconds: ready once they answer, without waiting for the minute's look)
    engines.down = false;
    await shown(page, 'qwen3.5-9b · Mac · ready');
    expect((await menu('model')).state === 'ready', JSON.stringify(await menu('model')));
  });

  await check("on one of OAIY's AI providers, chosen in OAIY's Settings → Agent: the chip and Settings say that is where it is chosen", async () => {
    network.prefs = { model: { source: 'provider', provider: 'lm-studio', model: 'qwen-coder-30b' }, providerName: 'LM Studio' };
    await backToFront();
    await page.waitForFunction(() => document.querySelector('button.chip.model:not([hidden])')?.textContent === 'LM Studio · qwen-coder-30b', { timeout: 10_000 });
    const title = await page.$eval('button.chip.model', (b) => b.title);
    expect(title.includes("as chosen in OAIY's Settings → Agent"), title);
    await page.click('button.chip.model');
    await page.waitForSelector('dialog.settings[open]');
    const note = await page.$eval('dialog.settings .agent-elsewhere', (n) => n.textContent);
    expect(note.includes('The Agent runs on LM Studio (qwen-coder-30b)') && note.includes("OAIY's Settings → Agent"), note);
    // (OAIY's own window reaches OAIY as it is: no word of letting a site in)
    const form = await page.$eval('dialog.settings .provider-form', (f) => f.textContent);
    expect(!form.includes('cors_origins') && form.includes('Settings → Images, video and audio'), form.slice(-300));
    await page.evaluate(() => [...document.querySelectorAll('dialog.settings button')].find((b) => b.textContent === 'Cancel').click());
    network.prefs = null;
    await backToFront();
    await page.waitForFunction(() => !document.querySelector('select.model.pick')?.hidden, { timeout: 10_000 });
  });

  await check('on a phone the header fits: the settings button on screen, the status pills round, the project named', async () => {
    await page.setViewport({ width: 390, height: 844, isMobile: true, hasTouch: true });
    await wait(500);
    const fit = await page.evaluate(() => {
      const r = (s) => document.querySelector(s)?.getBoundingClientRect();
      const bar = document.querySelector('header.topbar');
      return { gear: r('button.settings-button').right, gate: r('button[role="switch"]').width, pick: r('select.model.pick').width, project: r('.project-picker .combo-button').width, overflow: bar.scrollWidth - bar.clientWidth };
    });
    expect(fit.gear <= 390 && fit.overflow <= 0, JSON.stringify(fit));
    expect(fit.gate >= 28 && fit.pick >= 60 && fit.pick <= 120 && fit.project >= 80, JSON.stringify(fit));
    await page.setViewport({ width: 1400, height: 900 });
  });

  if (process.env.SHOT) await page.screenshot({ path: process.env.SHOT, clip: { x: 0, y: 0, width: 1400, height: 70 } });

  await check("an OAIY with no language model of its own: the menu offers the paired computer's, the first message says to choose it, and once chosen it answers", async () => {
    Object.assign(network, { model: 'Qwen3.8-Flash-Next', chosen: false });
    const context = await browser.createBrowserContext();
    const p = await openWindow(bareUrl, context);
    await p.waitForFunction(() => !document.querySelector('select.model.pick')?.hidden, { timeout: 15_000 }).catch(async (e) => {
      throw new Error(`${e.message}\nchip: ${await p.$eval('button.chip.model', (b) => b.textContent)}`);
    });
    expect(await p.$eval('button.chip.model', (b) => b.hidden), '"Set up AI…" is still shown');
    const m = await p.$eval('select.model.pick', (s) => ({ shown: s.selectedOptions[0]?.textContent, options: [...s.options].map((o) => o.textContent) }));
    expect(m.shown === 'No model on this computer' && m.options.includes('Qwen3.8-Flash-Next · Desktop'), JSON.stringify(m));
    await p.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes("a paired computer's can run the Agent"), { timeout: 10_000 }).catch(async (e) => {
      throw new Error(`${e.message}\nlog: ${(await log(p)).slice(-500)}`);
    });
    await send(p, 'Hello?');
    await p.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('choose one in the model menu at the top'), { timeout: 15_000 }).catch(async (e) => {
      throw new Error(`${e.message}\nlog: ${(await log(p)).slice(-500)}`);
    });
    expect(!(await log(p)).includes('No AI provider is set up yet'), 'the run asked for a provider in Settings');
    await p.select('select.model.pick', REMOTE);
    await p.waitForFunction(() => document.querySelector('select.model.pick')?.selectedOptions[0]?.textContent === 'Qwen3.8-Flash-Next · Desktop', { timeout: 10_000 }).catch(async (e) => {
      throw new Error(`${e.message}\n${JSON.stringify(await p.$eval('select.model.pick', (s) => [...s.options].map((o) => o.textContent)))}`);
    });
    await send(p, 'And now?');
    await p.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Answered on Desktop by Qwen3.8-Flash-Next.'), { timeout: 30_000 }).catch(async (e) => {
      throw new Error(`${e.message}\nlog: ${(await log(p)).slice(-500)}`);
    });
    await context.close();
  });

  await check("the model's window is said for the model that answers: the Agent's model is read again before it is said", async () => {
    // A page on this Mac's model, then a paired computer's made the default in OAIY (not here), a while later.
    chooseHere('qwen3.8-27b');
    Object.assign(engines, { state: 'ready' });
    network.lag = 0;
    const context = await browser.createBrowserContext();
    const p = await openWindow(oaiyUrl, context);
    await p.waitForFunction(() => document.querySelector('select.model.pick')?.selectedOptions[0]?.textContent.startsWith('qwen3.8-27b'), { timeout: 15_000 });
    Object.assign(network, { chosen: true, chosenAt: 0 });
    await wait(16_000);
    await send(p, 'Who answers?');
    await p.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Answered on Desktop by Qwen3.8-Flash-Next.'), { timeout: 30_000 }).catch(async (e) => {
      throw new Error(`${e.message}\nlog: ${(await log(p)).slice(-600)}`);
    });
    const text = await log(p);
    expect(!text.includes('qwen3.8-27b: 32.8k tokens of context'), `the window of a model that did not answer was said: ${text.slice(-600)}`);
    expect(text.includes("OAIY's engine default is now Qwen3.8-Flash-Next on Desktop") && !text.includes('as chosen in OAIY'), text.slice(-600));
    await context.close();
  });
} finally {
  await browser.close();
  await server.close();
  for (const s of [oaiy, bare, desktop]) s.close();
}
console.log(failures ? `${failures} failed` : 'all passed');
process.exit(failures ? 1 : 0);
