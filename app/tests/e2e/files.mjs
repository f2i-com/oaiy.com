// Attachments, images and big files, and SoftN apps as folders — through
// the real UI with a scripted model.   node tests/e2e/files.mjs
import { existsSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer as createHttpServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { deflateSync } from 'node:zlib';
import { unzipSync, zipSync } from 'fflate';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].filter(Boolean).find((p) => existsSync(p));
const softnInstalled = existsSync('public/softn/index.html');

// --- fixtures ----------------------------------------------------------------
const dir = mkdtempSync(join(tmpdir(), 'botc-files-'));
function png(width, height, pixel) {
  const crcTable = Array.from({ length: 256 }, (_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  const crc = (buf) => {
    let c = 0xffffffff;
    for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const body = Buffer.concat([Buffer.from(type), data]);
    const sum = Buffer.alloc(4);
    sum.writeUInt32BE(crc(body));
    return Buffer.concat([len, body, sum]);
  };
  const raw = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * 3 + 1)] = 0;
    for (let x = 0; x < width; x++) raw.set(pixel(x, y), y * (width * 3 + 1) + 1 + x * 3);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}
// A 2000×1200 gradient with a small red square at (1500, 900): only a zoom sees it clearly.
writeFileSync(join(dir, 'pattern.png'), png(2000, 1200, (x, y) => (x >= 1500 && x < 1520 && y >= 900 && y < 920 ? [255, 0, 0] : [Math.floor(x / 8) % 256, Math.floor(y / 5) % 256, 128])));
const lines = [];
for (let i = 1; i <= 50_000; i++) lines.push(i === 25_000 ? `line ${i} NEEDLE here` : i === 30_000 ? `${'x'.repeat(120_000)}END-OF-LONG-LINE` : `line ${i}`);
writeFileSync(join(dir, 'big.txt'), lines.join('\n'));
const enc = (s) => new TextEncoder().encode(s);
writeFileSync(join(dir, 'imported.softn'), zipSync({
  'manifest.json': enc(JSON.stringify({ name: 'Imported', version: '1.0.0', main: 'ui/main.ui', files: { ui: ['ui/main.ui'], logic: ['logic/main.logic'] } })),
  'ui/main.ui': enc('<logic src="../logic/main.logic" />\n<App theme="dark">\n  <Heading level={1}>Imported app</Heading>\n  <Text>{greeting()}</Text>\n</App>\n'),
  'logic/main.logic': enc('function greeting() { return "hello from an imported app" }\n'),
}));

// --- the scripted model ----------------------------------------------------
const requests = [];
const script = [
  { calls: [{ name: 'view_image', input: { path: 'uploads/pattern.png', x: 1400, y: 800, width: 300, height: 300, grid: true } }] },
  { calls: [{ name: 'file_info', input: { path: 'uploads/big.txt' } }] },
  { calls: [{ name: 'search_file', input: { path: 'uploads/big.txt', pattern: 'NEEDLE', context: 1 } }] },
  { calls: [{ name: 'read_file', input: { path: 'uploads/big.txt', offset: 30000, limit: 1 } }] },
  (body) => {
    const long = /char_start (\d+)/.exec(JSON.stringify(body.messages.at(-1)));
    return { calls: [{ name: 'read_file', input: { path: 'uploads/big.txt', char_start: Number(long?.[1] ?? 0) + 117_990, char_count: 40 } }] };
  },
  { calls: [{ name: 'softn_check', input: { app: 'Imported' } }] },
  { calls: [{ name: 'softn_import', input: { path: 'uploads/imported.softn', parent: 'copies' } }] },
  { text: 'All done.' },
];
const model = createHttpServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', '*');
  if (req.method === 'OPTIONS') return res.end();
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', () => {
    const parsed = JSON.parse(body);
    requests.push(parsed);
    const next = script[requests.length - 1] ?? { text: 'out of script' };
    const step = typeof next === 'function' ? next(parsed) : next;
    const events = [];
    for (const piece of step.text?.match(/.{1,6}/gs) ?? []) events.push({ choices: [{ index: 0, delta: { content: piece } }] });
    (step.calls ?? []).forEach((call, i) => events.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: `c${requests.length}_${i}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.input) } }] } }] }));
    events.push({ choices: [{ index: 0, delta: {}, finish_reason: step.calls ? 'tool_calls' : 'stop' }] });
    res.setHeader('content-type', 'text/event-stream');
    res.end(events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n');
  });
});
await new Promise((r) => model.listen(0, '127.0.0.1', r));

const server = await createServer({ server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const base = `http://127.0.0.1:${server.httpServer.address().port}`;
const browser = await puppeteer.launch({ executablePath, headless: true });
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
const toolText = (i) => JSON.stringify(requests[i].messages.filter((m) => m.role === 'tool').at(-1)?.content ?? '');

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  page.on('dialog', (d) => d.accept(d.defaultValue()));
  const downloads = mkdtempSync(join(tmpdir(), 'botc-dl-'));
  const cdp = await page.createCDPSession();
  await cdp.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: downloads });
  await page.goto(base);
  await page.waitForSelector('.tree-row', { timeout: 60_000 });
  // A provider pointed at the scripted model.
  await page.click('button[title="AI providers"]');
  await page.waitForSelector('dialog.settings[open]');
  await page.evaluate((url) => {
    [...document.querySelectorAll('dialog.settings button')].find((b) => b.textContent.includes('Add a provider')).click();
  }, '');
  await page.select('dialog.settings select', 'local-other');
  await page.evaluate((url) => {
    const input = [...document.querySelectorAll('dialog.settings label')].find((l) => l.firstChild.textContent === 'Address').querySelector('input');
    input.value = url;
    input.dispatchEvent(new Event('input'));
    const select = document.querySelector('.model-picker select');
    select.value = '\u0000other';
    select.dispatchEvent(new Event('change'));
    const custom = document.querySelector('.model-picker input');
    custom.value = 'mock';
    custom.dispatchEvent(new Event('input'));
    [...document.querySelectorAll('dialog.settings button')].find((b) => b.textContent === 'Save').click();
  }, `http://127.0.0.1:${model.address().port}`);

  await check('files attached in the chat are saved in the project, and the image goes to the model', async () => {
    const input = await page.$('.chat input[type=file]');
    await input.uploadFile(join(dir, 'pattern.png'), join(dir, 'big.txt'), join(dir, 'imported.softn'));
    await page.waitForFunction(() => document.querySelectorAll('.attachments .attachment').length === 3);
    await page.type('.chat-input', 'Look at these.');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('All done.'), { timeout: 120_000 });
    const first = JSON.stringify(requests[0].messages.at(-1));
    expect(first.includes('image_url') && first.includes('data:image/'), 'the image did not reach the model');
    expect(first.includes('uploads/pattern.png') && first.includes('2000×1200') && first.includes('uploads/big.txt') && first.includes('uploads/imported.softn: unpacked into Imported/') && first.includes('main ui/main.ui') && first.includes('logic/main.logic'), first.slice(0, 900));
    const tree = await page.$$eval('.tree-row .name', (els) => els.map((e) => e.textContent));
    expect(tree.includes('uploads') && tree.includes('Imported'), `tree: ${tree}`);
  });

  await check('view_image zooms into the original pixels and shows the region to the model', async () => {
    const tool = toolText(1);
    expect(tool.includes('2000×1200') && tool.includes('region x=1400, y=800, 300×300'), tool);
    const after = requests[1].messages.at(-1);
    expect(after.role === 'user' && JSON.stringify(after).includes('image_url'), JSON.stringify(after).slice(0, 300));
    const src = await page.$eval('details.tool img.tool-image', (img) => img.src.length);
    expect(src > 1000, 'no image in the tool card');
  });

  await check('file_info, search_file and character reads navigate a 50,000-line file', async () => {
    expect(/50,000 lines/.test(toolText(2)) && /longest line 120,016/.test(toolText(2)), toolText(2));
    expect(/line 25000, column 12/.test(toolText(3)) && /NEEDLE/.test(toolText(3)), toolText(3));
    expect(/line continues: 118016 more characters; char_start \d+/.test(toolText(4)), toolText(4).slice(0, 300));
    expect(toolText(5).includes('END-OF-LONG-LINE'), toolText(5));
  });

  await check('the agent checks an imported app by its folder', async () => {
    const tool = toolText(6);
    expect(tool.includes('App: Imported/'), tool);
    if (softnInstalled) expect(tool.includes('rendered without reported errors'), tool);
  });

  await check('the agent unpacks the original .softn again into a folder of its choosing', async () => {
    const tool = toolText(7);
    expect(tool.includes('into copies/Imported/ (3 files)') && tool.includes('ui/main.ui'), tool);
    const apps = await page.$$eval('.preview-app option', (els) => els.map((e) => e.value));
    expect(apps.includes('copies/Imported'), `apps: ${apps}`);
  });

  await check('a project holds several apps: the preview picks one, export takes a folder', async () => {
    await page.type('.chat-input', '/softn new apps/second');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelectorAll('.preview-app option').length === 3, { timeout: 20_000 });
    const options = await page.$$eval('.preview-app option', (o) => o.map((x) => x.value));
    expect(options.includes('Imported') && options.includes('apps/second'), `${options}`);
    if (softnInstalled) {
      await page.select('.preview-app', 'Imported');
      await page.waitForFunction(() => /live/.test(document.querySelector('.preview-status')?.textContent ?? ''), { timeout: 40_000 });
      const frame = page.frames().find((f) => f.url().includes('/softn/index.html'));
      expect((await frame.evaluate(() => document.body.innerText)).includes('hello from an imported app'), 'the imported app did not render');
    }
    await page.type('.chat-input', '/softn export Imported');
    await page.keyboard.press('Enter');
    let file;
    for (let i = 0; i < 50 && !file; i++) {
      await new Promise((r) => setTimeout(r, 200));
      file = readdirSync(downloads).find((f) => f.endsWith('.softn'));
    }
    expect(file === 'Imported.softn', `downloads: ${readdirSync(downloads)}`);
    const entries = Object.keys(unzipSync(readFileSync(join(downloads, file)))).sort();
    expect(JSON.stringify(entries) === JSON.stringify(['logic/main.logic', 'manifest.json', 'ui/main.ui']), `entries: ${entries}`);
  });
} finally {
  await browser.close();
  await server.close();
  model.close();
}
console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
