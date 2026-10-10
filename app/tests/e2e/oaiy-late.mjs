// OAIY's language model arrives after the page has looked for it.
//
// A first run of OAIY Desktop: the Agent's page is made, hidden, as the desktop
// starts, and looks at OAIY then. There is no language model yet (the setup
// wizard downloads it afterwards), so the page gets no provider to chat on.
// The model is then there, the person writes their first message, and it has
// to be answered by that model: the page once looked only as it loaded, and
// said "No AI provider is set up yet" with the model sitting in Engines.
//
// A mock OAIY whose discovery document lists no language model, then one.
//   node tests/e2e/oaiy-late.mjs
import { existsSync } from 'node:fs';
import { createServer as createHttpServer } from 'node:http';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].filter(Boolean).find((p) => existsSync(p));
if (!executablePath) {
  console.error('no Chrome found; set CHROME');
  process.exit(2);
}

const MODEL = 'qwen3.8-27b';
const REPLY = 'The model you downloaded is answering.';
/** Whether OAIY has its language model yet: off as the page loads, on once "the wizard has downloaded it". */
let hasModel = false;
const chats = [];

function discovery(origin) {
  return {
    service: 'oaiy-studio',
    version: '0.1.0',
    base_url: origin,
    openai_base_url: `${origin}/v1`,
    auth: { type: 'none', required: false },
    endpoints: ['chat:/v1/chat/completions', 'models:/v1/models'].map((e) => {
      const [name, path] = e.split(':');
      return { name, path, url: `${origin}${path}`, spec: 'openai' };
    }),
    models: { llm: hasModel ? [{ id: MODEL, default: true }] : [] },
    defaults: hasModel ? { llm: MODEL } : {},
    llm: { context_tokens: 32768 },
  };
}

const oaiy = createHttpServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type, *');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  if (req.method === 'OPTIONS') return res.end();
  const origin = `http://${req.headers.host}`;
  const send = (status, body) => {
    res.statusCode = status;
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify(body));
  };
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', () => {
    const url = req.url.split('?')[0];
    if (url === '/v1/discovery') return send(200, discovery(origin));
    if (url === '/v1/models') return send(200, { data: hasModel ? [{ id: MODEL, type: 'llm' }] : [] });
    if (url === '/v1/chat/completions') {
      chats.push(JSON.parse(body));
      res.setHeader('content-type', 'text/event-stream');
      const events = [
        { id: 'c1', choices: [{ index: 0, delta: { role: 'assistant', content: REPLY } }] },
        { id: 'c1', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
      ];
      return res.end(events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n');
    }
    send(404, { error: { message: `no route ${url}` } });
  });
});
await new Promise((r) => oaiy.listen(0, '127.0.0.1', r));
const oaiyUrl = `http://127.0.0.1:${oaiy.address().port}`;

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
  await page.goto(`${base}/?oaiy=${encodeURIComponent(oaiyUrl)}`);
  await page.waitForSelector('.tree-row', { timeout: 60_000 });
  const chip = () => page.$eval('button[title="AI provider"]', (b) => b.textContent);
  const log = () => page.$eval('.chat-log', (e) => e.textContent);

  await check('an OAIY with no language model yet gives the page nothing to chat on', async () => {
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Welcome! Set up an AI provider'), { timeout: 15_000 });
    const text = await log();
    expect(text.includes('Found oaiy-studio 0.1.0') && !text.includes('in the AI providers for chat'), text.slice(0, 400));
    expect((await chip()) === 'Set up AI…', await chip());
  });

  await check('the model that arrives afterwards answers the first message', async () => {
    hasModel = true;
    await page.type('.chat-input', 'Hello, are you there?');
    await page.keyboard.press('Enter');
    await page.waitForFunction((reply) => {
      const text = document.querySelector('.chat-log')?.textContent ?? '';
      return text.includes(reply) || text.includes('No AI provider is set up yet');
    }, { timeout: 30_000 }, REPLY);
    const text = await log();
    expect(!text.includes('No AI provider is set up yet'), 'the page said there is no AI provider, with a model in OAIY');
    expect(text.includes(REPLY), text.slice(-400));
    expect(text.includes('in the AI providers for chat too, and in use'), text.slice(0, 600));
    // (A provider that follows Engines names no model in its request: OAIY runs the one chosen there.)
    expect(chats.length >= 1 && chats[0].messages.some((m) => m.role === 'user' && JSON.stringify(m.content).includes('Hello, are you there?')), JSON.stringify(chats).slice(0, 300));
    expect((await chip()) === `OAIY · ${MODEL}`, await chip());
  });

  // The Clear button beside the context meter: asked first, then the conversation starts again.
  await check('Clear asks first, and Cancel leaves the conversation as it is', async () => {
    await page.waitForFunction(() => !document.querySelector('.chat-clear')?.disabled, { timeout: 15_000 });
    await page.click('.chat-clear');
    await page.waitForSelector('dialog.modal[open]', { timeout: 5_000 });
    expect((await page.$eval('dialog.modal h2', (e) => e.textContent)) === 'Clear this conversation?', 'the question is not the one for clearing');
    await page.click('dialog.modal .dialog-buttons button[type=button]');
    await page.waitForFunction(() => !document.querySelector('dialog.modal'), { timeout: 5_000 });
    expect((await log()).includes(REPLY), 'the reply went although the person said Cancel');
  });

  await check('Clear removes the messages, and the model is next sent none of them', async () => {
    await page.click('.chat-clear');
    await page.waitForSelector('dialog.modal[open]', { timeout: 5_000 });
    await page.click('dialog.modal .dialog-buttons button.danger');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('New conversation.'), { timeout: 10_000 });
    const text = await log();
    expect(!text.includes(REPLY) && !text.includes('Hello, are you there?'), text.slice(0, 400));
    expect(await page.$eval('.context-meter', (e) => e.hidden), 'the context meter still shows the conversation that was cleared');
    const before = chats.length;
    await page.type('.chat-input', 'A second question.');
    await page.keyboard.press('Enter');
    await page.waitForFunction((reply) => document.querySelector('.chat-log')?.textContent.includes(reply), { timeout: 30_000 }, REPLY);
    const sent = JSON.stringify(chats[before]?.messages ?? []);
    expect(sent.includes('A second question.'), sent.slice(0, 300));
    expect(!sent.includes('Hello, are you there?') && !sent.includes(REPLY), 'the cleared conversation was sent to the model again');
  });

  await check('and it is kept: a page opened again has the provider without looking twice', async () => {
    const again = await browser.newPage();
    await again.goto(`${base}/?oaiy=${encodeURIComponent(oaiyUrl)}`);
    await again.waitForSelector('.tree-row', { timeout: 60_000 });
    await again.waitForFunction((want) => document.querySelector('button[title="AI provider"]')?.textContent === want, { timeout: 15_000 }, `OAIY · ${MODEL}`);
    const text = await again.$eval('.chat-log', (e) => e.textContent);
    expect(!text.includes('Welcome! Set up an AI provider'), 'the welcome asks for a provider that is there');
    await again.close();
  });
  await page.close();
} finally {
  await browser.close();
  await server.close();
  oaiy.close();
}
console.log(failures ? `${failures} FAILED` : 'all passed');
process.exit(failures ? 1 : 0);
