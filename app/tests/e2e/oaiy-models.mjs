// The Agent's header chooses OAIY's models: a mock OAIY (two language models, one set to run on a Mac's eGPU, one on
// the Mac) and a mock OAIY Desktop the page is given, as OAIY's window gives it. One menu lists the engine's models,
// each with where it runs, and chooses one in Engines (through the desktop), which loads it where it runs and has the
// other let go of what it held (as the engines do: one model loaded at a time); the menu says how its loading goes;
// chats then are answered by the model chosen.
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
// at the second look after a start), and the one this Mac's engine holds. The 27B is set to the eGPU, the 9B to the Mac.
const ON_CARD = 'qwen3.8-27b';
const engines = { default: 'qwen3.8-27b', card: 'qwen3.8-27b', state: 'ready', looks: 0, mac: null };
const chats = [];
const asked = [];

function discovery(origin) {
  if (engines.state === 'starting' && ++engines.looks >= 2) engines.state = 'ready';
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

const oaiy = createHttpServer(async (req, res) => {
  cors(res);
  if (req.method === 'OPTIONS') return res.end();
  const body = await read(req);
  const url = req.url.split('?')[0];
  if (url === '/v1/discovery') return reply(res, 200, discovery(`http://${req.headers.host}`));
  if (url === '/v1/models') return reply(res, 200, { data: [{ id: 'qwen3.8-27b' }, { id: 'qwen3.5-9b' }] });
  if (url === '/v1/chat/completions') {
    const ask = JSON.parse(body);
    chats.push(ask);
    res.setHeader('content-type', 'text/event-stream');
    const say = (e) => `data: ${JSON.stringify(e)}\n\n`;
    // (as OAIY's gateway: a request that names no model is the engine's chosen one's)
    const model = ask.model ?? engines.default;
    return res.end(say({ choices: [{ index: 0, delta: { content: `Answered by ${model}.` } }] }) + say({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }) + 'data: [DONE]\n\n');
  }
  reply(res, 404, { error: { message: `no route ${req.method} ${url}` } });
});

// OAIY Desktop: the engines' relay (choosing a model), every other route not there. The engines load the model chosen
// where it runs and let the other go of what it held.
const desktop = createHttpServer(async (req, res) => {
  cors(res);
  if (req.method === 'OPTIONS') return res.end();
  const body = await read(req);
  const url = req.url.split('?')[0];
  asked.push({ method: req.method, url, auth: req.headers.authorization, body: body ? JSON.parse(body) : null });
  if (req.method === 'PUT' && url === '/api/engines/defaults') {
    engines.default = JSON.parse(body).model;
    const card = engines.default === ON_CARD;
    Object.assign(engines, card ? { card: engines.default, state: 'starting', looks: 0, mac: null } : { card: null, state: 'stopped', mac: engines.default });
    return reply(res, 200, { group: 'llm', model: engines.default });
  }
  reply(res, 404, { error: 'not here' });
});

for (const s of [oaiy, desktop]) await new Promise((r) => s.listen(0, '127.0.0.1', r));
const oaiyUrl = `http://127.0.0.1:${oaiy.address().port}`;
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

const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-first-run'] });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  await page.evaluateOnNewDocument((origin) => {
    window.__OAIY_DESKTOP__ = { origin, token: 'desktop-token' };
  }, desktopUrl);
  await page.goto(`${base}/?oaiy=${encodeURIComponent(oaiyUrl)}`);
  await page.waitForSelector('.tree-row', { timeout: 60_000 });
  await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Found oaiy-studio 0.1.0'), { timeout: 15_000 });
  const menu = (cls) => page.$eval(`select.${cls}.pick`, (s) => ({ hidden: s.hidden, value: s.value, options: [...s.options].map((o) => o.textContent), shown: s.selectedOptions[0]?.textContent }));

  await check("one menu lists the engine's language models, each with where it runs, the chosen one ready on the eGPU, in place of the provider chip", async () => {
    await page.waitForFunction(() => !document.querySelector('select.model.pick')?.hidden, { timeout: 10_000 });
    const m = await menu('model');
    expect(m.value === 'qwen3.8-27b' && m.shown === 'OAIY · qwen3.8-27b · eGPU · ready', JSON.stringify(m));
    expect(m.options.includes('OAIY · qwen3.5-9b · Mac') && m.options.includes('AI providers…'), JSON.stringify(m.options));
    expect(await page.$eval('select.model.pick', (s) => s.dataset.state) === 'ready', 'the menu is not shown ready');
    expect(await page.$eval('button[title="AI provider"]', (b) => b.hidden), 'the provider chip is still shown beside it');
    expect(!(await page.$('select.egpu.pick')), 'a second menu, for the eGPU, is still there');
  });

  await check("choosing a Mac's model chooses it in Engines, through the desktop: it loads on the Mac, the card lets go, and chats are answered by it", async () => {
    await page.select('select.model.pick', 'qwen3.5-9b');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes("OAIY's engine now runs qwen3.5-9b, on this Mac"), { timeout: 10_000 });
    expect(await page.$eval('.chat-log', (l) => l.textContent.includes('the eGPU lets go of its model')), 'the card letting go is not said');
    const put = asked.find((a) => a.url === '/api/engines/defaults');
    expect(put && put.auth === 'Bearer desktop-token' && put.body.group === 'llm' && put.body.model === 'qwen3.5-9b', JSON.stringify(asked));
    expect(!asked.some((a) => a.url === '/api/engines/egpu'), 'the eGPU was set apart from the choice');
    await page.waitForFunction(() => document.querySelector('select.model.pick')?.selectedOptions[0]?.textContent === 'OAIY · qwen3.5-9b · Mac · ready', { timeout: 10_000 }).catch(async (e) => {
      throw new Error(`${e.message}\n${JSON.stringify(await menu('model'))}`);
    });
    const m = await menu('model');
    expect(m.options.includes('OAIY · qwen3.8-27b · eGPU'), JSON.stringify(m.options));
    await page.type('.chat-input', 'Hello there.');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Answered by qwen3.5-9b.'), { timeout: 30_000 }).catch(async (e) => {
      throw new Error(`${e.message}\nchats: ${JSON.stringify(chats.map((c) => c.model))}\nlog: ${(await page.$eval('.chat-log', (l) => l.textContent)).slice(-600)}`);
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
    expect(m.value === 'qwen3.8-27b' && m.shown === 'OAIY · qwen3.8-27b · eGPU · ready', JSON.stringify(m));
    expect(m.options.includes('OAIY · qwen3.5-9b · Mac'), 'the Mac still shown holding its model: ' + JSON.stringify(m.options));
  });

  await check('the header drawn again with nothing changed leaves its menu as it is (one open stays open)', async () => {
    await page.$eval('select.model.pick', (s) => { window.__modelOption = s.options[0]; });
    const before = asked.length;
    // the model it runs already: nothing to choose, the header drawn again (as the minute's timer draws it)
    await page.select('select.model.pick', 'qwen3.8-27b');
    await new Promise((r) => setTimeout(r, 300));
    expect(asked.length === before, `asked ${JSON.stringify(asked.slice(before))}`);
    expect(await page.evaluate(() => window.__modelOption.isConnected), "the menu's options were made again");
  });

  if (process.env.SHOT) await page.screenshot({ path: process.env.SHOT, clip: { x: 0, y: 0, width: 1400, height: 70 } });
} finally {
  await browser.close();
  await server.close();
  oaiy.close();
  desktop.close();
}
console.log(failures ? `${failures} failed` : 'all passed');
process.exit(failures ? 1 : 0);
