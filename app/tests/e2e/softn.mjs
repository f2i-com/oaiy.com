// SoftN in bot.computer: a new app renders live, reacts to input, follows
// edits, reports errors, is checked by the agent, and exports as .softn.
// Needs the SoftN runtime (npm run fetch:softn).   node tests/e2e/softn.mjs
import { existsSync, mkdtempSync, readdirSync, readFileSync } from 'node:fs';
import { createServer as createHttpServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { unzipSync } from 'fflate';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

if (!existsSync('public/softn/index.html')) {
  console.log('skipped: the SoftN runtime is not installed (npm run fetch:softn)');
  process.exit(0);
}
const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].filter(Boolean).find((p) => existsSync(p));

// A scripted model that reads the guide, changes the heading, and checks.
const requests = [];
const script = [
  { calls: [{ name: 'softn_docs', input: { section: 'mistakes' } }] },
  { calls: [{ name: 'read_file', input: { path: 'ui/main.ui' } }] },
  { calls: [{ name: 'edit_file', input: { path: 'ui/main.ui', old_string: '<Heading level={1}>Groceries</Heading>', new_string: '<Heading level={1}>Shopping list</Heading>' } }] },
  { calls: [{ name: 'softn_check', input: {} }] },
  { text: 'Renamed the heading and checked the app.' },
];
const model = createHttpServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', '*');
  if (req.method === 'OPTIONS') return res.end();
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', () => {
    requests.push(JSON.parse(body));
    const step = script[requests.length - 1] ?? { text: 'out of script' };
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
const appFrame = (page) => page.frames().find((f) => f.url().includes('/softn/index.html'));
async function waitStatus(page, pattern, label) {
  try {
    await page.waitForFunction((p) => new RegExp(p).test(document.querySelector('.preview-status')?.textContent ?? ''), { timeout: 40_000 }, pattern.source);
  } catch {
    throw new Error(`${label}: the preview status stayed "${await page.$eval('.preview-status', (e) => e.textContent)}"`);
  }
}
/** Run `action`, then wait for the preview to finish a new render (live or error). */
async function afterRender(page, action, label) {
  const before = await page.$eval('.preview-status', (e) => e.textContent);
  await action();
  try {
    await page.waitForFunction((b) => {
      const t = document.querySelector('.preview-status')?.textContent ?? '';
      return t !== b && /live|error/.test(t);
    }, { timeout: 40_000 }, before);
  } catch {
    throw new Error(`${label}: no new render; the status stayed "${await page.$eval('.preview-status', (e) => e.textContent)}"`);
  }
  return page.$eval('.preview-status', (e) => e.textContent);
}
async function frameText(page) {
  const frame = appFrame(page);
  return frame ? frame.evaluate(() => document.body.innerText) : '';
}
async function terminal(page, command) {
  await page.click('.term-input');
  await page.type('.term-input', command);
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => !document.querySelector('.term-prompt')?.classList.contains('busy'), { timeout: 30_000 });
}

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  page.on('dialog', (d) => d.accept(d.defaultValue()));
  const downloads = mkdtempSync(join(tmpdir(), 'botc-softn-'));
  const cdp = await page.createCDPSession();
  await cdp.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: downloads });
  await page.goto(base);
  await page.waitForSelector('.tree-row', { timeout: 60_000 });

  await check('a new SoftN app renders live in the sandboxed preview', async () => {
    await page.type('.chat-input', '/softn new');
    await page.keyboard.press('Enter');
    await waitStatus(page, /live/, 'first render');
    const text = await frameText(page);
    expect(text.includes('Tasks') && text.includes('0 remaining'), text);
    const sandbox = await page.$eval('.preview-frame iframe', (f) => f.getAttribute('sandbox'));
    expect(!/allow-same-origin/.test(sandbox), `sandbox: ${sandbox}`);
  });

  await check('the app runs its logic: adding a task updates the page', async () => {
    const frame = appFrame(page);
    await frame.type('input', 'Buy milk');
    await frame.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Add').click());
    await frame.waitForFunction(() => document.body.innerText.includes('1 remaining') && document.body.innerText.includes('Buy milk'), { timeout: 10_000 });
  });

  await check('an edit from the terminal re-renders the preview', async () => {
    const status = await afterRender(page, () => terminal(page, 'sed -i "s/>Tasks</>Groceries</" ui/main.ui'), 'after edit');
    expect(/live/.test(status), status);
    const text = await frameText(page);
    expect(text.includes('Groceries'), text);
  });

  await check('a broken app is reported in the preview and by /softn check', async () => {
    const status = await afterRender(page, () => terminal(page, 'cp logic/main.logic /tmp.bak && echo "function broken( {" >> logic/main.logic'), 'broken logic');
    expect(/error/.test(status), `status: ${status}`);
    await page.type('.chat-input', '/softn check');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => /Render: (?!ok)/.test(document.querySelector('.chat-log')?.textContent ?? ''), { timeout: 40_000 });
    const repaired = await afterRender(page, () => terminal(page, 'mv /tmp.bak logic/main.logic'), 'repaired');
    expect(/live/.test(repaired), repaired);
  });

  await check('the agent reads the guide, edits the page and checks it', async () => {
    await page.click('button[title="AI providers"]');
    await page.waitForSelector('dialog.settings[open]');
    await page.evaluate(() => [...document.querySelectorAll('dialog.settings button')].find((b) => b.textContent.includes('Add a provider')).click());
    await page.select('dialog.settings select', 'local-other');
    await page.evaluate((url) => {
      const input = [...document.querySelectorAll('dialog.settings label')].find((l) => l.firstChild.textContent === 'Address').querySelector('input');
      input.value = url;
      input.dispatchEvent(new Event('input'));
      const custom = document.querySelector('.model-picker input');
      document.querySelector('.model-picker select').value = '\u0000other';
      document.querySelector('.model-picker select').dispatchEvent(new Event('change'));
      custom.value = 'mock';
      custom.dispatchEvent(new Event('input'));
      [...document.querySelectorAll('dialog.settings button')].find((b) => b.textContent === 'Save').click();
    }, `http://127.0.0.1:${model.address().port}`);
    await page.type('.chat-input', 'Call it a shopping list.');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Renamed the heading'), { timeout: 60_000 });
    expect(JSON.stringify(requests[1]).includes('Common mistakes'), 'the guide section did not reach the model');
    expect(JSON.stringify(requests[4]).includes('rendered without reported errors'), `softn_check said: ${JSON.stringify(requests[4]).slice(-400)}`);
    await waitStatus(page, /live/, 'after the agent');
    expect((await frameText(page)).includes('Shopping list'), await frameText(page));
  });

  await check('Export .softn downloads a flat zip with the manifest at its root', async () => {
    await page.evaluate(() => [...document.querySelectorAll('.actions button')].find((b) => b.textContent === 'Export .softn').click());
    let file;
    for (let i = 0; i < 50 && !file; i++) {
      await new Promise((r) => setTimeout(r, 200));
      file = readdirSync(downloads).find((f) => f.endsWith('.softn'));
    }
    expect(file === 'Tasks.softn', `downloaded: ${readdirSync(downloads)}`);
    const entries = unzipSync(readFileSync(join(downloads, file)));
    const manifest = JSON.parse(new TextDecoder().decode(entries['manifest.json']));
    expect(manifest.main === 'ui/main.ui' && manifest.files.ui.includes('ui/main.ui') && manifest.files.logic.includes('logic/main.logic'), JSON.stringify(manifest));
    expect(Object.keys(entries).every((n) => !n.startsWith('Tasks/')), `entries: ${Object.keys(entries)}`);
    expect(new TextDecoder().decode(entries['ui/main.ui']).includes('Shopping list'), 'stale ui/main.ui');
  });
} finally {
  await browser.close();
  await server.close();
  model.close();
}
console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
