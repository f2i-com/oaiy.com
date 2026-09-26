// nrob end to end: a mock nrob (discovery, scripted chat, images, a video job)
// is found on its own when the app opens, and the agent makes an image and a
// video with it. A second mock refuses the page, as nrob does for an origin it
// does not allow.
//   node tests/e2e/nrob.mjs
import { existsSync } from 'node:fs';
import { createServer as createHttpServer } from 'node:http';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].filter(Boolean).find((p) => existsSync(p));
if (!executablePath) {
  console.error('no Chrome found; set CHROME');
  process.exit(2);
}

const PNG = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==', 'base64');
const chats = [];
const images = [];
const videos = [];
let polls = 0;
const script = [
  { calls: [{ name: 'generate_image', input: { prompt: 'a red fox in snow, watercolour', path: 'art/fox.png' } }] },
  { calls: [{ name: 'generate_video', input: { prompt: 'the fox runs through the snow, camera follows', path: 'media/fox.mp4', seconds: 4, start_image: 'art/fox.png' } }] },
  { text: 'Made art/fox.png and media/fox.mp4.' },
];
function sse(step, n) {
  const events = [];
  for (const piece of step.text?.match(/.{1,6}/gs) ?? []) events.push({ choices: [{ index: 0, delta: { content: piece } }] });
  (step.calls ?? []).forEach((call, i) => {
    events.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: `call_${n}_${i}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.input) } }] } }] });
  });
  events.push({ choices: [{ index: 0, delta: {}, finish_reason: step.calls ? 'tool_calls' : 'stop' }] });
  return events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n';
}

function discovery(origin) {
  return {
    service: 'nrob-studio',
    version: '0.1.0',
    base_url: origin,
    openai_base_url: `${origin}/v1`,
    auth: { type: 'none', required: false },
    endpoints: ['chat:/v1/chat/completions', 'models:/v1/models', 'images:/v1/images/generations', 'edits:/v1/images/edits', 'videos:/v1/videos'].map((e) => {
      const [name, path] = e.split(':');
      return { name, path, url: `${origin}${path}`, spec: 'openai' };
    }),
    models: {
      llm: [{ id: 'qwen3.8-27b', default: true }],
      image: [{ id: 'qwen-image-turbo-q4', default: true, edits: true, max_references: 3, size_step: 32, default_size: '1024x1024' }],
      video: [{ id: 'sulphur-2', default: true, fps: 24, max_seconds: 5, max_side: 1024, start_image: true }],
    },
    defaults: { llm: 'qwen3.8-27b', image: 'qwen-image-turbo-q4', video: 'sulphur-2' },
    llm: { context_tokens: 32768 },
  };
}

const nrob = createHttpServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type, *');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  if (req.method === 'OPTIONS') return res.end();
  const origin = `http://${req.headers.host}`;
  const send = (status, body, type = 'application/json') => {
    res.statusCode = status;
    res.setHeader('content-type', type);
    res.end(type === 'application/json' ? JSON.stringify(body) : body);
  };
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', () => {
    const url = req.url.split('?')[0];
    if (url === '/v1/discovery') return send(200, discovery(origin));
    if (url === '/v1/models') return send(200, { data: [{ id: 'qwen3.8-27b', type: 'llm' }, { id: 'qwen-image-turbo-q4', type: 'image' }, { id: 'sulphur-2', type: 'video' }] });
    if (url === '/v1/chat/completions') {
      chats.push(JSON.parse(body));
      res.setHeader('content-type', 'text/event-stream');
      return res.end(sse(script[chats.length - 1] ?? { text: 'out of script' }, chats.length));
    }
    if (url === '/v1/images/generations') {
      images.push(JSON.parse(body));
      return send(200, { created: 1, data: [{ b64_json: PNG.toString('base64') }], size: '1024x1024', model: 'qwen-image-turbo-q4' });
    }
    if (url === '/v1/videos' && req.method === 'POST') {
      videos.push(JSON.parse(body));
      return send(200, { id: 'video_1', object: 'video', status: 'queued', progress: 0, model: 'sulphur-2' });
    }
    if (url === '/v1/videos/video_1') {
      polls++;
      return send(200, polls < 3 ? { id: 'video_1', status: 'in_progress', progress: polls * 40 } : { id: 'video_1', status: 'completed', progress: 100, seconds: '4', size: '768x512', model: 'sulphur-2' });
    }
    if (url === '/v1/videos/video_1/content') return send(200, Buffer.from('not really an mp4'), 'video/mp4');
    send(404, { error: { message: `no route ${req.method} ${url}` } });
  });
});
await new Promise((r) => nrob.listen(0, '127.0.0.1', r));
const nrobUrl = `http://127.0.0.1:${nrob.address().port}`;

// An nrob that does not allow this page, as it answers an origin missing from gateway.cors_origins.
const closed = createHttpServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.statusCode = 403;
  res.setHeader('content-type', 'application/json');
  res.end(JSON.stringify({ error: { message: `requests from ${req.headers.origin} are not allowed; add it to gateway.cors_origins, or set an API key`, type: 'permission_error' } }));
});
await new Promise((r) => closed.listen(0, '127.0.0.1', r));
const closedUrl = `http://127.0.0.1:${closed.address().port}`;

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
  await page.goto(`${base}/?nrob=${encodeURIComponent(nrobUrl)}`);
  await page.waitForSelector('.tree-row', { timeout: 60_000 });

  await check('nrob is found on its own: images, video and chat are set up from its discovery document', async () => {
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Found nrob-studio 0.1.0'), { timeout: 15_000 });
    const log = await page.$eval('.chat-log', (e) => e.textContent);
    expect(log.includes('images (qwen-image-turbo-q4) and video (sulphur-2)') && log.includes('in the AI providers for chat too, and in use'), log.slice(0, 500));
    expect(!log.includes('Welcome! Set up an AI provider'), 'the welcome still asks for a provider');
    const chip = await page.$eval('button[title="AI provider"]', (b) => b.textContent);
    expect(chip === 'nrob · qwen3.8-27b', chip);
    await page.click('button[title="AI providers"]');
    await page.waitForSelector('dialog.settings[open]');
    const media = await page.$eval('.media-settings', (s) => ({ text: s.textContent, inputs: [...s.querySelectorAll('input')].map((i) => i.value) }));
    expect(media.text.includes(`nrob-studio 0.1.0 at ${new URL(nrobUrl).origin} (images and video)`), media.text);
    expect(media.inputs.includes(`${nrobUrl}/v1`) && media.inputs.includes('qwen-image-turbo-q4') && media.inputs.includes('sulphur-2'), JSON.stringify(media.inputs));
    await page.evaluate(() => [...document.querySelectorAll('dialog.settings button')].find((b) => b.textContent === 'Cancel').click());
  });

  await check('the agent makes an image and a video with nrob; both land in the project and play in the chat', async () => {
    // Every status line the chat shows, to see the video's progress go by.
    await page.evaluate(() => {
      window.__statuses = [];
      new MutationObserver(() => window.__statuses.push(document.querySelector('.chat-status').textContent)).observe(document.querySelector('.chat-status'), { childList: true, characterData: true, subtree: true });
    });
    await page.type('.chat-input', 'Paint a fox and animate it.');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Made art/fox.png and media/fox.mp4.'), { timeout: 60_000 });
    const tools = chats[0].tools.map((t) => t.function.name);
    expect(tools.includes('generate_image') && tools.includes('generate_video'), `tools offered: ${tools}`);
    const imageTool = chats[0].tools.find((t) => t.function.name === 'generate_image').function;
    expect(imageTool.description.includes('qwen-image-turbo-q4 (default): default size 1024x1024'), imageTool.description);
    expect(images.length === 1 && images[0].model === 'qwen-image-turbo-q4' && images[0].response_format === 'b64_json' && images[0].prompt.includes('red fox'), JSON.stringify(images));
    expect(videos.length === 1 && videos[0].model === 'sulphur-2' && videos[0].seconds === '4' && /^data:image\/png;base64,/.test(videos[0].input_reference?.image_url), JSON.stringify(videos).slice(0, 300));
    const results = chats.slice(1).map((c) => c.messages.filter((m) => m.role === 'tool').map((m) => m.content).join('\n')).join('\n');
    expect(results.includes('Saved /art/fox.png: 1×1 px, made with qwen-image-turbo-q4') && results.includes('Saved /media/fox.mp4 (4 s, 768x512'), results);
    const shown = await page.$$eval('.msg.presented', (els) => els.map((e) => ({ img: !!e.querySelector('img'), video: !!e.querySelector('video'), text: e.textContent })));
    expect(shown.some((s) => s.img) && shown.some((s) => s.video || s.text.includes('fox.mp4')), JSON.stringify(shown));
    const statuses = await page.evaluate(() => window.__statuses);
    expect(statuses.some((s) => s.includes('making the video: 40%')) && statuses.some((s) => s.includes('generating an image with qwen-image-turbo-q4')), statuses.join(' | '));
    const names = await page.$$eval('.tree-row .name', (els) => els.map((e) => e.textContent));
    expect(names.includes('art') && names.includes('media'), `tree: ${names}`);
  });
  await page.close();

  await check('an nrob that refuses this page says how to allow it', async () => {
    const other = await browser.newPage();
    await other.goto(`${base}/?nrob=${encodeURIComponent(closedUrl)}`);
    await other.waitForSelector('.tree-row', { timeout: 60_000 });
    await other.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('gateway.cors_origins'), { timeout: 15_000 });
    const log = await other.$eval('.chat-log', (e) => e.textContent);
    expect(log.includes(`The origin to allow is ${base}`) && log.includes('Find nrob'), log.slice(0, 600));
    await other.close();
  });
} finally {
  await browser.close();
  await server.close();
  nrob.close();
  closed.close();
}

console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
