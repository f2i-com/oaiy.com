// The production build on a plain static server that sends NO isolation
// headers: the service worker must add them (one automatic reload), and the
// app must keep working offline.   node tests/e2e/static.mjs
import { existsSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import { execSync } from 'node:child_process';
import puppeteer from 'puppeteer-core';

const root = join(process.cwd(), 'dist');
if (!existsSync(join(root, 'index.html')) || process.argv.includes('--build')) execSync('npx vite build', { stdio: 'inherit' });
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm', '.svg': 'image/svg+xml', '.json': 'application/json', '.webmanifest': 'application/manifest+json' };
const server = createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^([/\\])+/, '');
  let file = join(root, path);
  if (!file.startsWith(root) || !existsSync(file) || statSync(file).isDirectory()) file = join(root, 'index.html');
  res.setHeader('content-type', types[extname(file)] ?? 'application/octet-stream');
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/`;

const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].filter(Boolean).find((p) => existsSync(p));
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
async function terminal(page, command, waitFor) {
  await page.click('.term-input');
  await page.type('.term-input', command);
  await page.keyboard.press('Enter');
  try {
    await page.waitForFunction((w) => document.querySelector('.term-out')?.textContent.includes(w), { timeout: 30_000 }, waitFor);
  } catch {
    throw new Error(`the terminal never showed "${waitFor}":\n${(await page.$eval('.term-out', (el) => el.textContent)).slice(-400)}`);
  }
}

try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  await page.goto(base);
  await check('the service worker makes the page cross-origin isolated (one reload)', async () => {
    await page.waitForFunction(() => globalThis.crossOriginIsolated === true, { timeout: 30_000, polling: 250 });
  });
  await page.waitForSelector('.tree-row', { timeout: 30_000 });
  await check('the sandbox runs in the built app', async () => {
    await terminal(page, 'python hello.py', 'mean temperature: 19.8');
  });
  await check('with the server gone, the app opens from the cache and the sandbox runs', async () => {
    await new Promise((r) => setTimeout(r, 1500));
    // Really offline: nothing answers at that address any more.
    server.closeAllConnections();
    server.close();
    await page.close();
    const offline = await browser.newPage();
    await offline.goto(base);
    await offline.waitForSelector('.tree-row', { timeout: 30_000 });
    if (!(await offline.evaluate(() => globalThis.crossOriginIsolated))) throw new Error('not isolated offline');
    await terminal(offline, 'node hello.js', 'warmest: Cairo');
  });
} finally {
  await browser.close();
  server.close();
}
console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
